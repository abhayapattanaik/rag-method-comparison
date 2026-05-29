"""LLM chunk contextualization for the RAG Comparison System.

Two-pass strategy (Strategy 5 from contextualization-optimization.md):

  Pass 1 — Document Summarization
    For each unique document, send the full text to the LLM and request a
    ~1500-token summary covering main thesis, methodology, key findings, and
    section structure. Summaries are cached in data/cache/summaries/{doc_id}.txt.
    On restart the cached file is loaded instead of calling the LLM.

  Pass 2 — Batch Contextualization
    Chunks are grouped by doc_id and processed in batches of 5. Each batch
    call receives: the document summary, the prev/next chunk texts as local
    context, and the 5 current chunks. The LLM returns a 1-2 sentence
    context prefix per chunk. Prefixes are prepended to chunk.text to
    produce a contextualized Chunk. Each result is cached in
    data/cache/contextualized/{chunk_id}.json for crash recovery.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from src.ingestion.chunker import Chunk

if TYPE_CHECKING:
    from src.config import AppConfig
    from src.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pass 1 — Document summarization
# ---------------------------------------------------------------------------

_SUMMARIZE_SYSTEM = (
    "You are a research assistant that summarizes academic papers concisely."
)

_SUMMARIZE_PROMPT_TMPL = """\
Summarize the following research paper in approximately 1500 tokens.

Your summary MUST cover:
1. Main thesis / research question
2. Methodology (datasets, models, experimental setup)
3. Key findings and results
4. Section structure overview (briefly — what each major section covers)

The summary will be used to give context to individual passages from this paper,
so preserve enough topical and structural detail to situate any paragraph within
the paper's overall argument.

---BEGIN PAPER---
{doc_text}
---END PAPER---

Summary:"""


def summarize_document(
    doc_text: str,
    doc_id: str,
    provider: "BaseLLMProvider",
    config: "AppConfig",
) -> str:
    """Summarize a single document. Returns ~1500-token summary string.

    Checks cache first (data/cache/summaries/{doc_id}.txt). If the cache file
    exists, loads and returns it without making an LLM call.

    Args:
        doc_text:  Full Markdown text of the document.
        doc_id:    Document identifier (stem of source filename).
        provider:  LLM provider to call if cache is absent.
        config:    AppConfig — used for cache_dir path.

    Returns:
        Summary string (from cache or freshly generated).
    """
    summaries_dir = Path(config.paths.cache_dir) / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    cache_path = summaries_dir / f"{doc_id}.txt"

    # Cache hit — skip LLM call
    if cache_path.exists():
        summary = cache_path.read_text(encoding="utf-8").strip()
        logger.info("summarize_document: cache hit doc_id=%s (%d chars)", doc_id, len(summary))
        return summary

    logger.info("summarize_document: calling LLM for doc_id=%s (%d chars)", doc_id, len(doc_text))
    prompt = _SUMMARIZE_PROMPT_TMPL.format(doc_text=doc_text)
    messages = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    response = provider.complete(messages, max_tokens=2000)
    summary = response.text.strip()

    # Persist to cache
    cache_path.write_text(summary, encoding="utf-8")
    logger.info(
        "summarize_document: cached doc_id=%s summary_chars=%d "
        "input_tokens=%d output_tokens=%d",
        doc_id, len(summary), response.input_tokens, response.output_tokens,
    )
    return summary


def summarize_all_documents(
    chunks: list[Chunk],
    extracted_dir: str,
    provider: "BaseLLMProvider",
    config: "AppConfig",
) -> dict[str, str]:
    """Summarize every unique document referenced by the chunk list.

    Groups chunks by doc_id to discover the document set, then reads each
    document's .md file from *extracted_dir* and calls summarize_document().

    Args:
        chunks:        Full list of Chunk objects (used to discover doc_ids).
        extracted_dir: Directory containing the extracted .md files (one per doc).
        provider:      LLM provider.
        config:        AppConfig.

    Returns:
        Dict mapping doc_id -> summary string.
    """
    logger.info("summarize_all_documents: pass 1 start — discovering documents")

    # Collect unique doc_ids from the chunk list
    doc_ids: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.doc_id not in seen:
            seen.add(chunk.doc_id)
            doc_ids.append(chunk.doc_id)

    logger.info("summarize_all_documents: found %d unique documents", len(doc_ids))

    extracted_path = Path(extracted_dir)
    summaries: dict[str, str] = {}

    for doc_id in doc_ids:
        # Try to find the matching .md file (may have different prefix conventions)
        md_candidates = list(extracted_path.glob(f"{doc_id}.md"))
        if not md_candidates:
            # Try glob with doc_id as a substring — handles filename variants
            md_candidates = [p for p in extracted_path.glob("*.md") if p.stem == doc_id]
        if not md_candidates:
            logger.warning(
                "summarize_all_documents: no .md file found for doc_id=%s in %s — skipping",
                doc_id, extracted_dir,
            )
            continue

        md_path = md_candidates[0]
        doc_text = md_path.read_text(encoding="utf-8")

        summary = summarize_document(doc_text, doc_id, provider, config)
        summaries[doc_id] = summary

    logger.info(
        "summarize_all_documents: pass 1 complete — %d/%d documents summarized",
        len(summaries), len(doc_ids),
    )
    return summaries


# ---------------------------------------------------------------------------
# Pass 2 — Batch contextualization
# ---------------------------------------------------------------------------

_BATCH_SYSTEM = (
    "You are a research assistant that writes precise, minimal context labels for "
    "passages from academic papers."
)

_BATCH_PROMPT_TMPL = """\
You are given a summary of a research paper and a batch of {n} text chunks from that paper.
For each chunk write a 1-2 sentence context prefix that situates it within the paper's
overall argument. The prefix should help a reader understand what part of the paper the
chunk comes from and why it matters.

Do NOT rephrase or reproduce the chunk text — only provide the situating context.

Respond with a JSON array of exactly {n} strings, one per chunk, in the same order.
Example (for n=2): ["Context for chunk 0.", "Context for chunk 1."]

---DOCUMENT SUMMARY---
{doc_summary}

---LOCAL CONTEXT (chunks immediately before/after the batch)---
Previous chunk (index {prev_idx}):
{prev_text}

Next chunk (index {next_idx}):
{next_text}

---BATCH CHUNKS TO CONTEXTUALIZE---
{chunks_block}

JSON array of {n} context strings:"""


def _build_chunks_block(chunks: list[Chunk]) -> str:
    """Format a list of chunks for inclusion in the batch prompt."""
    lines: list[str] = []
    for i, chunk in enumerate(chunks):
        lines.append(f"Chunk {i} (chunk_index={chunk.chunk_index}, section={chunk.section}):")
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines)


def _parse_batch_response(response_text: str, expected_n: int) -> list[str]:
    """Parse the LLM's JSON-array response into a list of prefix strings.

    Attempts to extract a JSON array from the response. Falls back to an
    empty-string list of length *expected_n* if parsing fails, so a parse
    error in one batch never aborts the entire run.
    """
    text = response_text.strip()

    # Find JSON array boundaries — the response may include preamble text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        json_str = text[start : end + 1]
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list) and len(parsed) == expected_n:
                return [str(item) for item in parsed]
            # Length mismatch — pad or trim
            result = [str(item) for item in parsed]
            if len(result) < expected_n:
                result.extend([""] * (expected_n - len(result)))
            return result[:expected_n]
        except json.JSONDecodeError:
            logger.warning(
                "_parse_batch_response: JSON decode failed on: %.200s", json_str
            )

    logger.warning(
        "_parse_batch_response: could not find JSON array in response (len=%d) — "
        "using empty prefixes for this batch",
        len(text),
    )
    return [""] * expected_n


def contextualize_batch(
    chunks: list[Chunk],
    doc_summary: str,
    all_doc_chunks: list[Chunk],
    provider: "BaseLLMProvider",
    config: "AppConfig",
) -> list[str]:
    """Contextualize a batch of up to 5 chunks from the same document.

    Builds a prompt containing the document summary, the text of the chunk
    immediately before the first chunk in the batch and immediately after the
    last chunk (local sliding-window context), and all chunks in the batch.
    Calls the LLM once and parses the returned JSON array.

    Args:
        chunks:         Batch of Chunk objects (max 5, all same doc_id).
        doc_summary:    Pre-generated summary for the document.
        all_doc_chunks: All chunks for this document (for prev/next lookup).
        provider:       LLM provider.
        config:         AppConfig.

    Returns:
        List of context prefix strings, one per chunk in *chunks*.
    """
    if not chunks:
        return []

    n = len(chunks)
    # Build an index-keyed lookup for prev/next chunk resolution
    by_index: dict[int, Chunk] = {c.chunk_index: c for c in all_doc_chunks}
    min_batch_idx = min(c.chunk_index for c in chunks)
    max_batch_idx = max(c.chunk_index for c in chunks)

    prev_chunk = by_index.get(min_batch_idx - 1)
    next_chunk = by_index.get(max_batch_idx + 1)

    prev_text = prev_chunk.text if prev_chunk else "(none — this is the first chunk)"
    next_text = next_chunk.text if next_chunk else "(none — this is the last chunk)"
    prev_idx = (min_batch_idx - 1) if prev_chunk else "N/A"
    next_idx = (max_batch_idx + 1) if next_chunk else "N/A"

    chunks_block = _build_chunks_block(chunks)

    prompt = _BATCH_PROMPT_TMPL.format(
        n=n,
        doc_summary=doc_summary,
        prev_idx=prev_idx,
        prev_text=prev_text,
        next_idx=next_idx,
        next_text=next_text,
        chunks_block=chunks_block,
    )

    messages = [
        {"role": "system", "content": _BATCH_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    response = provider.complete(messages, max_tokens=config.llm.max_tokens)
    prefixes = _parse_batch_response(response.text, n)

    logger.debug(
        "contextualize_batch: doc_id=%s batch_size=%d "
        "input_tokens=%d output_tokens=%d",
        chunks[0].doc_id, n,
        response.input_tokens, response.output_tokens,
    )
    return prefixes


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _contextualized_cache_path(chunk_id: str, config: "AppConfig") -> Path:
    cache_dir = Path(config.paths.cache_dir) / "contextualized"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{chunk_id}.json"


def _load_contextualized_cache(chunk_id: str, config: "AppConfig") -> str | None:
    """Return cached contextualized text for *chunk_id*, or None if absent."""
    path = _contextualized_cache_path(chunk_id, config)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("contextualized_text")
        except (json.JSONDecodeError, KeyError):
            logger.warning("_load_contextualized_cache: corrupt cache at %s — will recompute", path)
    return None


def _save_contextualized_cache(
    chunk: Chunk,
    contextualized_text: str,
    config: "AppConfig",
) -> None:
    """Persist the contextualized text for a chunk to the JSON cache."""
    path = _contextualized_cache_path(chunk.chunk_id, config)
    payload = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "chunk_index": chunk.chunk_index,
        "source_file": chunk.source_file,
        "contextualized_text": contextualized_text,
    }
    # Atomic write: write to a temp file then rename
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Main pass-2 orchestrator
# ---------------------------------------------------------------------------


def contextualize_all_chunks(
    chunks: list[Chunk],
    summaries: dict[str, str],
    extracted_dir: str,
    provider: "BaseLLMProvider",
    config: "AppConfig",
) -> list[Chunk]:
    """Contextualize all chunks using the two-pass strategy.

    Groups chunks by doc_id, skips chunks with a valid cache entry, and
    processes remaining chunks in batches of 5. For each chunk a context
    prefix is prepended to chunk.text to produce a new Chunk with updated
    text. Cache entries are written after each batch for crash recovery.

    Stall detection: if any single batch call takes longer than
    config.concurrency.stall_timeout_seconds, logs a warning and continues.

    Progress: logs every completed batch.

    Args:
        chunks:        Full list of raw Chunk objects.
        summaries:     Doc-level summaries from pass 1, keyed by doc_id.
        extracted_dir: Directory with .md source files (not used here, passed
                       through for future extensibility).
        provider:      LLM provider.
        config:        AppConfig.

    Returns:
        List of contextualized Chunks (one per input chunk), in the same order
        as *chunks*. Chunks whose doc_id has no summary are returned unchanged.
    """
    logger.info("contextualize_all_chunks: pass 2 start — %d total chunks", len(chunks))

    stall_threshold = config.concurrency.stall_timeout_seconds
    batch_size = 5

    # Group chunks by doc_id, preserving original order
    doc_chunks: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        doc_chunks.setdefault(chunk.doc_id, []).append(chunk)

    # Build a flat result map keyed by chunk_id (preserves original ordering at end)
    result_map: dict[str, Chunk] = {}

    # Cache-hit pass: populate result_map for already-cached chunks
    cache_hits = 0
    cache_misses = 0
    for chunk in chunks:
        cached_text = _load_contextualized_cache(chunk.chunk_id, config)
        if cached_text is not None:
            result_map[chunk.chunk_id] = Chunk(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                text=cached_text,
                section=chunk.section,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                source_file=chunk.source_file,
                token_count=chunk.token_count,
            )
            cache_hits += 1
        else:
            cache_misses += 1

    logger.info(
        "contextualize_all_chunks: cache scan complete — hits=%d misses=%d",
        cache_hits, cache_misses,
    )

    # Process each document's chunks
    total_batches = math.ceil(cache_misses / batch_size) if cache_misses > 0 else 0
    completed_batches = 0

    for doc_id, doc_chunk_list in doc_chunks.items():
        # Filter to only chunks that need processing
        pending = [c for c in doc_chunk_list if c.chunk_id not in result_map]
        if not pending:
            logger.debug("contextualize_all_chunks: doc_id=%s — all chunks cached, skipping", doc_id)
            continue

        doc_summary = summaries.get(doc_id)
        if doc_summary is None:
            logger.warning(
                "contextualize_all_chunks: no summary for doc_id=%s — "
                "returning chunks unchanged",
                doc_id,
            )
            for chunk in pending:
                result_map[chunk.chunk_id] = chunk
            continue

        # Split pending chunks for this doc into batches of 5
        batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]

        for batch in batches:
            batch_start = time.monotonic()

            try:
                prefixes = contextualize_batch(
                    chunks=batch,
                    doc_summary=doc_summary,
                    all_doc_chunks=doc_chunk_list,
                    provider=provider,
                    config=config,
                )
            except Exception as exc:
                logger.error(
                    "contextualize_all_chunks: batch failed doc_id=%s "
                    "chunk_indices=%s error=%s — using empty prefixes",
                    doc_id,
                    [c.chunk_index for c in batch],
                    exc,
                )
                prefixes = [""] * len(batch)

            batch_elapsed = time.monotonic() - batch_start

            # Stall warning
            if batch_elapsed > stall_threshold:
                logger.warning(
                    "contextualize_all_chunks: stall detected — batch for doc_id=%s "
                    "took %.1fs (threshold=%ds)",
                    doc_id, batch_elapsed, stall_threshold,
                )

            # Build contextualized chunks and cache each one
            for chunk, prefix in zip(batch, prefixes):
                if prefix:
                    ctx_text = f"{prefix}\n\n{chunk.text}"
                else:
                    ctx_text = chunk.text

                ctx_chunk = Chunk(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    text=ctx_text,
                    section=chunk.section,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    source_file=chunk.source_file,
                    token_count=chunk.token_count,
                )
                result_map[chunk.chunk_id] = ctx_chunk
                _save_contextualized_cache(chunk, ctx_text, config)

            completed_batches += 1
            logger.info(
                "contextualize_all_chunks: batch %d/%d complete "
                "doc_id=%s chunks=%d elapsed=%.1fs",
                completed_batches, total_batches,
                doc_id, len(batch), batch_elapsed,
            )

    # Reconstruct the output list in the same order as input *chunks*
    output: list[Chunk] = []
    for chunk in chunks:
        if chunk.chunk_id in result_map:
            output.append(result_map[chunk.chunk_id])
        else:
            # Fallback: should not happen, but guard to avoid silent data loss
            logger.warning(
                "contextualize_all_chunks: chunk_id=%s missing from result_map — appending unchanged",
                chunk.chunk_id,
            )
            output.append(chunk)

    logger.info(
        "contextualize_all_chunks: pass 2 complete — %d chunks returned "
        "(cache_hits=%d newly_contextualized=%d)",
        len(output), cache_hits, completed_batches * batch_size,
    )
    return output
