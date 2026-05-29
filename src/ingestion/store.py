"""ChromaDB wrapper for the RAG Comparison System.

Provides persistent storage for chunk embeddings. All writes use upsert
(not add) so re-running ingestion is idempotent — existing chunk_ids are
overwritten rather than duplicated.

Embeddings are generated externally (by embedder.py) and passed in at upsert
time. ChromaDB is NOT configured with its own embedding function; the choice
of embedding model is centralised in AppConfig.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chromadb as _chromadb_type
    from src.config import AppConfig
    from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client factory (persistent)
# ---------------------------------------------------------------------------


def _get_client(config: "AppConfig") -> "_chromadb_type.PersistentClient":
    """Return a ChromaDB PersistentClient rooted at config.paths.chroma_dir."""
    import chromadb

    return chromadb.PersistentClient(path=config.paths.chroma_dir)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_or_create_collection(
    name: str,
    config: "AppConfig",
) -> "_chromadb_type.Collection":
    """Get an existing ChromaDB collection or create it if absent.

    The collection is configured without an embedding function because
    embeddings are provided externally.

    Args:
        name: Collection name (e.g. "rag_traditional_v1").
        config: AppConfig — provides chroma_dir path.

    Returns:
        chromadb.Collection instance.
    """
    import chromadb
    from chromadb.config import Settings

    client = _get_client(config)
    collection = client.get_or_create_collection(
        name=name,
        # embedding_function=None tells ChromaDB we'll supply embeddings ourselves
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Collection '%s' ready (count=%d)", name, collection.count())
    return collection


def upsert_chunks(
    collection: "_chromadb_type.Collection",
    chunks: "list[Chunk]",
    embeddings: list[list[float]],
) -> int:
    """Upsert chunks with their embeddings into a ChromaDB collection.

    Uses chunk.chunk_id as the ChromaDB document ID, ensuring idempotency.
    All Chunk metadata fields are stored so downstream retrieval can surface
    source citations.

    Args:
        collection: Target ChromaDB collection.
        chunks: List of Chunk objects (must match length of embeddings).
        embeddings: Parallel list of embedding vectors.

    Returns:
        Number of chunks upserted.

    Raises:
        ValueError: If chunks and embeddings lengths differ.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have equal length"
        )

    if not chunks:
        return 0

    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "doc_id": c.doc_id,
            "chunk_index": c.chunk_index,
            "section": c.section,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "source_file": c.source_file,
            "token_count": c.token_count,
        }
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info("Upserted %d chunks into collection '%s'", len(chunks), collection.name)
    return len(chunks)


def query_collection(
    collection: "_chromadb_type.Collection",
    query_embedding: list[float],
    top_k: int,
) -> list[dict]:
    """Dense similarity search against a ChromaDB collection.

    Args:
        collection: ChromaDB collection to search.
        query_embedding: Query vector (must match dimension of stored embeddings).
        top_k: Number of results to return.

    Returns:
        List of dicts, each containing:
            - "id": chunk_id
            - "text": chunk document text
            - "metadata": dict of stored metadata fields
            - "distance": cosine distance (lower = more similar)
    """
    logger.debug(
        "query_collection: collection=%s top_k=%d", collection.name, top_k
    )
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict] = []
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, text, meta, dist in zip(ids, documents, metadatas, distances):
        hits.append(
            {
                "id": chunk_id,
                "text": text,
                "metadata": meta,
                "distance": dist,
            }
        )

    top_score = round(1.0 - distances[0], 6) if distances else 0.0
    logger.info(
        "query_collection: collection=%s returned %d results top_score=%.4f",
        collection.name, len(hits), top_score,
    )
    return hits


def get_collection_count(collection: "_chromadb_type.Collection") -> int:
    """Return the number of documents stored in the collection."""
    return collection.count()
