"""Document-level section-aware chunker for the RAG Comparison System.

Splits extracted Markdown documents into overlapping chunks, respecting
section boundaries (# / ## / ###). Overlap is applied only within the
same section, never carried across section boundaries.

Token counting uses tiktoken when available, falling back to a whitespace
approximation calibrated for academic text (1 word ≈ 1.3 tokens).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token counter (tiktoken optional)
# ---------------------------------------------------------------------------

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))

except ImportError:
    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        """Whitespace-split approximation: 1 word ≈ 1.3 tokens (academic text)."""
        words = len(text.split())
        return max(1, round(words * 1.3))


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    doc_id: str          # source filename without extension
    chunk_id: str        # deterministic hash for idempotency
    chunk_index: int
    text: str
    section: str         # detected heading or "unknown"
    page_start: int
    page_end: int
    source_file: str
    token_count: int


# ---------------------------------------------------------------------------
# Chunk ID generation
# ---------------------------------------------------------------------------


def _config_version(chunk_size: int, chunk_overlap: int) -> str:
    """Short hash of the chunking parameters — baked into chunk_id."""
    raw = f"{chunk_size}:{chunk_overlap}"
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


def _make_chunk_id(doc_id: str, chunk_index: int, cfg_version: str) -> str:
    raw = f"{doc_id}:{chunk_index}:{cfg_version}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Return list of (heading, body) pairs.

    The first section may have no heading (preamble before first #-line).
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("unknown", text)]

    sections: list[tuple[str, str]] = []

    # Preamble before first heading
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("unknown", preamble))

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((heading, body))

    return sections


# ---------------------------------------------------------------------------
# Page mapping helpers
# ---------------------------------------------------------------------------


def _load_page_map(chunks_json_path: str) -> list[tuple[int, int, int]]:
    """Load page boundary data from pymupdf4llm *_chunks.json.

    Returns list of (page_number, char_start, char_end) tuples sorted by
    page_number, where char_start/char_end are cumulative character offsets
    in the *concatenated* document text.

    pymupdf4llm emits one entry per page with a ``text`` field. We
    reconstruct cumulative offsets by summing text lengths.
    """
    try:
        with open(chunks_json_path, "r", encoding="utf-8") as fh:
            pages: list[dict] = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    mapping: list[tuple[int, int, int]] = []
    cursor = 0
    for page_entry in pages:
        page_text: str = page_entry.get("text", "")
        page_num: int = page_entry.get("metadata", {}).get("page_number", 1)
        char_start = cursor
        char_end = cursor + len(page_text)
        mapping.append((page_num, char_start, char_end))
        cursor = char_end

    return mapping


def _char_offset_to_page(
    char_offset: int,
    page_map: list[tuple[int, int, int]],
    total_chars: int,
) -> int:
    """Return page number for a character offset in the full document text."""
    if not page_map:
        # Fallback: rough estimate (~2000 chars/page for academic PDFs)
        chars_per_page = 2000
        return max(1, char_offset // chars_per_page + 1)

    for page_num, start, end in page_map:
        if start <= char_offset < end:
            return page_num

    # Beyond last recorded page — return last page number
    return page_map[-1][0]


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------


def _chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split *text* into token-bounded overlapping windows.

    Uses word-level sliding window so chunk boundaries fall on word edges.
    Overlap is in tokens, applied only backward within the same text block.
    """
    words = text.split()
    if not words:
        return []

    # Convert word list to token-cumulative counts for fast boundary lookup.
    # We track cumulative token count per word using the counter.
    tok_counts: list[int] = []
    running = 0
    for w in words:
        running += _count_tokens(w + " ")
        tok_counts.append(running)

    total_tokens = tok_counts[-1]
    if total_tokens <= chunk_size:
        return [text]

    chunks: list[str] = []
    start_word = 0

    while start_word < len(words):
        # Determine end word: first word that pushes cumulative tokens beyond chunk_size
        base_tokens = tok_counts[start_word - 1] if start_word > 0 else 0
        end_word = start_word
        while end_word < len(words):
            chunk_tokens = tok_counts[end_word] - base_tokens
            if chunk_tokens > chunk_size:
                break
            end_word += 1

        if end_word == start_word:
            # Single word exceeds chunk_size — include it anyway
            end_word = start_word + 1

        chunk_words = words[start_word:end_word]
        chunks.append(" ".join(chunk_words))

        if end_word >= len(words):
            break

        # Advance start by (chunk_size - overlap) tokens
        stride_tokens = chunk_size - chunk_overlap
        stride_tokens = max(1, stride_tokens)
        new_base = base_tokens + stride_tokens
        # Find first word whose cumulative count exceeds new_base
        next_start = end_word
        for idx in range(start_word, end_word):
            if tok_counts[idx] >= new_base:
                next_start = idx
                break
        else:
            next_start = end_word

        if next_start <= start_word:
            next_start = start_word + 1  # guard infinite loop

        start_word = next_start

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_document(
    text: str,
    source_file: str,
    config,  # AppConfig
    page_map: Optional[list[tuple[int, int, int]]] = None,
) -> list[Chunk]:
    """Chunk a single document's Markdown text into Chunk objects.

    Args:
        text: Full Markdown text of the document.
        source_file: Filename (basename) of the source file, e.g. "01_rag_lewis_2020.md".
        config: AppConfig instance (reads config.chunking.chunk_size / chunk_overlap).
        page_map: Optional pre-loaded page boundary map from _load_page_map().
                  If None, page numbers are estimated from char offsets.

    Returns:
        Ordered list of Chunk objects with all metadata populated.
    """
    chunk_size: int = config.chunking.chunk_size
    chunk_overlap: int = config.chunking.chunk_overlap

    doc_id = Path(source_file).stem
    cfg_ver = _config_version(chunk_size, chunk_overlap)
    total_chars = len(text)

    if page_map is None:
        page_map = []

    sections = _split_into_sections(text)
    logger.debug(
        "chunk_document: doc_id=%s sections=%d chunk_size=%d overlap=%d",
        doc_id, len(sections), chunk_size, chunk_overlap,
    )

    chunks: list[Chunk] = []
    global_chunk_index = 0

    for section_heading, section_body in sections:
        text_chunks = _chunk_text(section_body, chunk_size, chunk_overlap)

        # Track char offset within full document for page mapping.
        # We locate where section_body starts in the full text.
        section_start_char = text.find(section_body[:min(80, len(section_body))])
        if section_start_char < 0:
            section_start_char = 0

        # Running char offset within section for page estimation
        section_char_cursor = 0

        for chunk_text in text_chunks:
            chunk_char_start = section_start_char + section_char_cursor
            chunk_char_end = chunk_char_start + len(chunk_text)

            page_start = _char_offset_to_page(chunk_char_start, page_map, total_chars)
            page_end = _char_offset_to_page(
                max(chunk_char_start, chunk_char_end - 1), page_map, total_chars
            )

            chunk_id = _make_chunk_id(doc_id, global_chunk_index, cfg_ver)
            token_count = _count_tokens(chunk_text)

            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    chunk_index=global_chunk_index,
                    text=chunk_text,
                    section=section_heading,
                    page_start=page_start,
                    page_end=page_end,
                    source_file=source_file,
                    token_count=token_count,
                )
            )

            global_chunk_index += 1
            # Advance cursor by stride (chunk - overlap) to mirror token-level windowing
            stride_chars = max(1, len(chunk_text) - round(len(chunk_text) * chunk_overlap / chunk_size))
            section_char_cursor += stride_chars

    logger.info(
        "chunk_document: doc_id=%s produced %d chunks from %d sections",
        doc_id, len(chunks), len(sections),
    )
    return chunks


def chunk_all_documents(
    extracted_dir: str,
    config,  # AppConfig
) -> dict[str, list[Chunk]]:
    """Process all .md files in *extracted_dir*.

    For each .md file, looks for a sibling *_chunks.json* file produced by
    pymupdf4llm for page metadata. Falls back to char-offset estimation when
    missing.

    Returns:
        Dict mapping doc_id (stem of .md filename) to list of Chunk objects.
    """
    extracted_path = Path(extracted_dir)
    md_files = sorted(extracted_path.glob("*.md"))

    logger.info("chunk_all_documents: found %d .md files in %s", len(md_files), extracted_dir)
    result: dict[str, list[Chunk]] = {}

    for md_file in md_files:
        doc_id = md_file.stem
        text = md_file.read_text(encoding="utf-8")

        # Locate companion _chunks.json
        chunks_json = extracted_path / f"{doc_id}_chunks.json"
        page_map = _load_page_map(str(chunks_json))

        chunks = chunk_document(
            text=text,
            source_file=md_file.name,
            config=config,
            page_map=page_map,
        )
        result[doc_id] = chunks

    total_chunks = sum(len(v) for v in result.values())
    logger.info(
        "chunk_all_documents: processed %d documents → %d total chunks",
        len(result), total_chunks,
    )
    return result
