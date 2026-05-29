"""Dense (vector similarity) retrieval from ChromaDB.

Embeds the query with the BGE embedder, then issues a cosine-similarity
search against the given ChromaDB collection. Returns normalised result
dicts suitable for the pipeline layer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chromadb as _chromadb_type
    from src.config import AppConfig

logger = logging.getLogger(__name__)


def dense_retrieve(
    query: str,
    collection: "_chromadb_type.Collection",
    config: "AppConfig",
    top_k: int | None = None,
) -> list[dict]:
    """Embed *query* and return the top-K most similar chunks from *collection*.

    The function uses :func:`src.ingestion.embedder.embed_query` for query
    embedding and :func:`src.ingestion.store.query_collection` for the
    ChromaDB search. Results are converted from raw ChromaDB hit dicts into
    a flattened schema that the pipeline layer can work with directly.

    Relevance scores are derived from ChromaDB cosine *distances*:
        score = 1 - distance
    A score of 1.0 is a perfect match; 0.0 is maximally dissimilar.

    Args:
        query:      Raw query text (the BGE prefix is added automatically
                    inside :func:`embed_query`).
        collection: ChromaDB collection to search (e.g. the object returned
                    by :func:`src.ingestion.store.get_or_create_collection`).
        config:     AppConfig — used for embedding model selection and the
                    default ``retrieval.top_k_dense`` value.
        top_k:      Number of results to return. Defaults to
                    ``config.retrieval.top_k_dense``.

    Returns:
        List of result dicts, each containing:
            - ``text``        (str)   chunk text
            - ``doc_id``      (str)   source document identifier
            - ``section``     (str)   section label if stored, else empty string
            - ``page_start``  (int)   first page this chunk spans
            - ``page_end``    (int)   last page this chunk spans
            - ``source_file`` (str)   original PDF filename
            - ``score``       (float) relevance score in [0, 1]; higher is better

        Results are already ordered from highest to lowest score (i.e. most
        relevant first) as returned by ChromaDB.
    """
    from src.ingestion.embedder import embed_query
    from src.ingestion.store import query_collection

    effective_top_k = top_k if top_k is not None else config.retrieval.top_k_dense

    logger.debug(
        "dense_retrieve: query=%r top_k=%d collection=%s",
        query,
        effective_top_k,
        collection.name,
    )

    query_embedding = embed_query(query, config)
    raw_hits = query_collection(collection, query_embedding, effective_top_k)

    results: list[dict] = []
    for hit in raw_hits:
        meta = hit.get("metadata") or {}
        distance = hit.get("distance", 1.0)
        score = max(0.0, 1.0 - distance)
        results.append(
            {
                "text": hit.get("text", ""),
                "doc_id": meta.get("doc_id", ""),
                "section": meta.get("section", ""),
                "page_start": meta.get("page_start", 0),
                "page_end": meta.get("page_end", 0),
                "source_file": meta.get("source_file", ""),
                "score": round(score, 6),
            }
        )

    top_score = results[0]["score"] if results else 0.0
    logger.info(
        "dense_retrieve: query=%r collection=%s returned %d results top_score=%.4f",
        query[:80], collection.name, len(results), top_score,
    )
    return results
