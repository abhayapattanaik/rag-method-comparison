"""embed_contextualized.py — Embed contextualized chunks and upsert into ChromaDB.

Idempotent: skips embedding if the target collection already has the expected
chunk count. Loads chunks from data/cache/contextualized_chunks.json (produced
by the contextualizer phase).

Usage:
    cd /path/to/claude_rc_2
    python3 scripts/embed_contextualized.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src` imports resolve
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from src.config import load_config
from src.ingestion.chunker import Chunk
from src.ingestion import embedder as _embedder
from src.ingestion import store as _store

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "rag_contextualized_v1"
CHUNKS_JSON_PATH = _PROJECT_ROOT / "data" / "cache" / "contextualized_chunks.json"
CONFIG_YAML_PATH = _PROJECT_ROOT / "config" / "default.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_chunks(path: Path) -> list[Chunk]:
    """Deserialise contextualized chunks from the JSON cache."""
    print(f"Loading contextualized chunks from {path} ...")
    with path.open("r", encoding="utf-8") as fh:
        raw: list[dict] = json.load(fh)

    chunks = [
        Chunk(
            doc_id=item["doc_id"],
            chunk_id=item["chunk_id"],
            chunk_index=int(item["chunk_index"]),
            text=item["text"],
            section=item["section"],
            page_start=int(item["page_start"]),
            page_end=int(item["page_end"]),
            source_file=item["source_file"],
            token_count=int(item["token_count"]),
        )
        for item in raw
    ]
    print(f"  Loaded {len(chunks)} chunks")
    return chunks


def _embed_in_batches(
    chunks: list[Chunk],
    config,
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed all chunk texts, printing progress."""
    texts = [c.text for c in chunks]
    total = len(texts)
    all_embeddings: list[list[float]] = []

    print(f"Embedding {total} chunks (batch_size={batch_size}) ...")
    t0 = time.perf_counter()

    for batch_start in range(0, total, batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        batch_embeddings = _embedder.embed_texts(batch, config, batch_size=batch_size)
        all_embeddings.extend(batch_embeddings)

        done = min(batch_start + batch_size, total)
        elapsed = time.perf_counter() - t0
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  {done}/{total} ({rate:.0f} chunks/s)", end="\r", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\n  Done — {total} embeddings in {elapsed:.1f}s ({total/elapsed:.0f} chunks/s)")
    return all_embeddings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Check prerequisites
    # ------------------------------------------------------------------
    if not CHUNKS_JSON_PATH.exists():
        print(f"ERROR: contextualized_chunks.json not found at {CHUNKS_JSON_PATH}")
        print("Run the contextualizer script first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load config
    # ------------------------------------------------------------------
    config_path = str(CONFIG_YAML_PATH) if CONFIG_YAML_PATH.exists() else None
    config = load_config(config_path)
    print(f"Config loaded — embedding model: {config.models.embedding_model}")
    print(f"ChromaDB dir: {config.paths.chroma_dir}")

    # ------------------------------------------------------------------
    # 3. Load chunks
    # ------------------------------------------------------------------
    chunks = _load_chunks(CHUNKS_JSON_PATH)
    expected_count = len(chunks)

    # ------------------------------------------------------------------
    # 4. Idempotency check — skip if collection already fully populated
    # ------------------------------------------------------------------
    collection = _store.get_or_create_collection(COLLECTION_NAME, config)
    existing_count = _store.get_collection_count(collection)

    if existing_count == expected_count and expected_count > 0:
        print(
            f"\nCollection '{COLLECTION_NAME}' already has {existing_count} chunks "
            f"(matches expected {expected_count}). Skipping embedding."
        )
        t_elapsed = time.perf_counter() - t_start
        print(f"\nSummary:")
        print(f"  Total chunks embedded : 0 (skipped — already populated)")
        print(f"  Collection count      : {existing_count}")
        print(f"  Time taken            : {t_elapsed:.1f}s")
        return

    if existing_count > 0:
        print(
            f"Collection '{COLLECTION_NAME}' has {existing_count}/{expected_count} chunks. "
            "Proceeding with full upsert (idempotent)."
        )

    # ------------------------------------------------------------------
    # 5. Embed
    # ------------------------------------------------------------------
    t_embed_start = time.perf_counter()
    embeddings = _embed_in_batches(chunks, config, batch_size=32)
    embed_elapsed = time.perf_counter() - t_embed_start

    # ------------------------------------------------------------------
    # 6. Upsert into ChromaDB
    # ------------------------------------------------------------------
    print(f"\nUpserting {len(chunks)} chunks into '{COLLECTION_NAME}' ...")
    upserted = _store.upsert_chunks(collection, chunks, embeddings)
    print(f"  Upserted {upserted} chunks")

    # ------------------------------------------------------------------
    # 7. Verify
    # ------------------------------------------------------------------
    final_count = _store.get_collection_count(collection)
    if final_count != expected_count:
        print(
            f"WARNING: collection count {final_count} != expected {expected_count}. "
            "Some chunks may have failed to upsert."
        )
        sys.exit(1)
    else:
        print(f"  Verification OK: collection count = {final_count}")

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    t_elapsed = time.perf_counter() - t_start
    print(f"\nSummary:")
    print(f"  Total chunks embedded : {len(chunks)}")
    print(f"  Embedding time        : {embed_elapsed:.1f}s")
    print(f"  Collection count      : {final_count}")
    print(f"  Total time            : {t_elapsed:.1f}s")


if __name__ == "__main__":
    main()
