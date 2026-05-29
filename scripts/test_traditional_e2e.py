"""End-to-end test for the TRADITIONAL RAG pipeline.

Stages
------
1. Load AppConfig from config/default.yaml
2. Open ChromaDB collection rag_traditional_v1
3. Verify document count == 266 (fail fast if not)
4. Retrieval-only smoke test (no LLM): embed query, get top-5 chunks, print
5. Full pipeline run: Pipeline.run(query, PipelineMethod.TRADITIONAL)
   - Print retrieved chunks, generated answer, and latency
   - Handles missing `claude` binary and timeout gracefully

Usage
-----
    python scripts/test_traditional_e2e.py

Run from the project root so that `src/` is on the Python path, or add the
project root to PYTHONPATH:
    PYTHONPATH=. python scripts/test_traditional_e2e.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when invoked directly.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "rag_traditional_v1"
EXPECTED_DOC_COUNT = 266
QUERY = "What is retrieval augmented generation?"
TOP_K_SMOKE = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _separator(title: str = "") -> None:
    width = 72
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "=" * pad + f" {title} " + "=" * pad)
    else:
        print("\n" + "=" * width)


def _print_chunks(chunks: list[dict], top_k: int = TOP_K_SMOKE) -> None:
    for i, chunk in enumerate(chunks[:top_k], start=1):
        score = chunk.get("score", float("nan"))
        source = chunk.get("source_file", "unknown")
        page_start = chunk.get("page_start", "?")
        page_end = chunk.get("page_end", "?")
        text_preview = chunk.get("text", "").strip().replace("\n", " ")[:200]
        print(
            f"  [{i}] score={score:.4f}  source={source}  pages={page_start}-{page_end}\n"
            f"      {text_preview}{'...' if len(chunk.get('text','')) > 200 else ''}"
        )


# ---------------------------------------------------------------------------
# Stage 1 – Load config
# ---------------------------------------------------------------------------


def stage_load_config():
    _separator("Stage 1: Load config")
    from src.config import load_config

    config = load_config()
    print(f"  chroma_dir : {config.paths.chroma_dir}")
    print(f"  embed model: {config.models.embedding_model}")
    print(f"  llm.provider: {config.llm.provider}")
    print(f"  retrieval.top_k_dense: {config.retrieval.top_k_dense}")
    return config


# ---------------------------------------------------------------------------
# Stage 2 – Open collection
# ---------------------------------------------------------------------------


def stage_open_collection(config):
    _separator("Stage 2: Open ChromaDB collection")
    from src.ingestion.store import get_or_create_collection

    collection = get_or_create_collection(COLLECTION_NAME, config)
    count = collection.count()
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Doc count  : {count}")
    return collection, count


# ---------------------------------------------------------------------------
# Stage 3 – Verify document count
# ---------------------------------------------------------------------------


def stage_verify_count(count: int) -> None:
    _separator("Stage 3: Verify document count")
    if count != EXPECTED_DOC_COUNT:
        print(
            f"  FAIL — expected {EXPECTED_DOC_COUNT} documents, found {count}.\n"
            "  Re-run ingestion before testing."
        )
        sys.exit(1)
    print(f"  PASS — {count} documents confirmed.")


# ---------------------------------------------------------------------------
# Stage 4 – Retrieval-only smoke test
# ---------------------------------------------------------------------------


def stage_retrieval_smoke(config, collection) -> list[dict]:
    _separator("Stage 4: Retrieval-only smoke test (no LLM)")
    from src.retrieval.dense import dense_retrieve

    print(f"  Query: {QUERY!r}")
    t0 = time.perf_counter()
    chunks = dense_retrieve(QUERY, collection, config, top_k=TOP_K_SMOKE)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"  Retrieved {len(chunks)} chunks in {elapsed_ms:.1f}ms\n")
    _print_chunks(chunks, top_k=TOP_K_SMOKE)
    return chunks


# ---------------------------------------------------------------------------
# Stage 5 – Full pipeline run
# ---------------------------------------------------------------------------


def stage_full_pipeline(config) -> None:
    _separator("Stage 5: Full pipeline run (retrieval + LLM)")
    from src.retrieval.pipeline import Pipeline, PipelineMethod

    pipeline = Pipeline(config=config, telemetry=None)

    print(f"  Query: {QUERY!r}")
    print("  Calling Pipeline.run() — this shells out to `claude` CLI ...\n")

    try:
        result = pipeline.run(QUERY, PipelineMethod.TRADITIONAL)
    except RuntimeError as exc:
        # Covers: claude not found, timeout, non-zero exit code
        print(f"  ERROR during LLM call: {exc}")
        print(
            "\n  Retrieval portion may still have succeeded. "
            "Check that `claude` is installed and on PATH, or set "
            "llm.provider to 'anthropic' with a valid ANTHROPIC_API_KEY."
        )
        return

    # --- Retrieved chunks ---
    _separator("Retrieved Chunks")
    _print_chunks(result.retrieved_chunks, top_k=TOP_K_SMOKE)

    # --- Source citations ---
    _separator("Source Citations")
    if result.sources:
        for src in result.sources:
            print(f"  doc={src['doc']}  page={src['page']}")
    else:
        print("  (no sources extracted)")

    # --- Generated answer ---
    _separator("Generated Answer")
    print(result.answer)

    # --- Latency summary ---
    _separator("Latency")
    print(f"  Total pipeline latency: {result.latency_ms:.1f} ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("  Traditional RAG Pipeline — End-to-End Test")
    print("=" * 72)

    config = stage_load_config()
    collection, count = stage_open_collection(config)
    stage_verify_count(count)
    stage_retrieval_smoke(config, collection)
    stage_full_pipeline(config)

    _separator()
    print("  Test complete.")
    _separator()


if __name__ == "__main__":
    main()
