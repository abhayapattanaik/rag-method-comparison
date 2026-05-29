"""Chunk all extracted papers and save to data/cache/chunks.json.

Usage:
    cd /path/to/claude_rc_2
    python3 scripts/chunk_papers.py

Reads .md files from data/papers/extracted/, applies section-aware chunking
per config/default.yaml, prints per-document counts, and writes all chunks
to data/cache/chunks.json as a JSON array.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.ingestion.chunker import chunk_all_documents


def main() -> None:
    config = load_config()

    extracted_dir = PROJECT_ROOT / "data" / "papers" / "extracted"
    if not extracted_dir.exists():
        print(f"ERROR: extracted dir not found: {extracted_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Chunking documents in: {extracted_dir}")
    print(
        f"Config: chunk_size={config.chunking.chunk_size}, "
        f"chunk_overlap={config.chunking.chunk_overlap}"
    )
    print()

    all_chunks = chunk_all_documents(str(extracted_dir), config)

    total = 0
    all_serializable: list[dict] = []

    print(f"{'Document':<45} {'Chunks':>6}")
    print("-" * 53)

    for doc_id in sorted(all_chunks.keys()):
        doc_chunks = all_chunks[doc_id]
        count = len(doc_chunks)
        total += count
        print(f"{doc_id:<45} {count:>6}")
        all_serializable.extend(asdict(c) for c in doc_chunks)

    print("-" * 53)
    print(f"{'TOTAL':<45} {total:>6}")
    print()

    # Persist
    cache_dir = PROJECT_ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / "chunks.json"

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(all_serializable, fh, ensure_ascii=False, indent=2)

    print(f"Saved {total} chunks to: {output_path}")


if __name__ == "__main__":
    main()
