"""BM25 sparse retrieval for the RAG Comparison System.

Builds an in-memory BM25Okapi index from a ChromaDB collection and provides
keyword-based retrieval. Results are returned in the same flat-dict schema used
by dense.py so the pipeline layer can treat them interchangeably.

Typical usage::

    index = BM25Index.build_from_collection(collection, config)
    results = index.search("what is reciprocal rank fusion", top_k=20)
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chromadb as _chromadb_type
    from src.config import AppConfig

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on whitespace — simple, fast, reproducible."""
    return text.lower().split()


class BM25Index:
    """BM25Okapi index built over a corpus of chunk dicts.

    Args:
        chunks: List of chunk dicts, each must contain at least a ``text`` key.
                All other metadata keys are carried through to search results.
    """

    def __init__(self, chunks: list[dict]) -> None:
        from rank_bm25 import BM25Okapi

        if not chunks:
            raise ValueError("Cannot build BM25Index from an empty chunk list.")

        logger.info("Building BM25 index over %d chunks", len(chunks))

        self._chunks = chunks
        tokenized_corpus = [_tokenize(c.get("text", "")) for c in chunks]

        logger.debug("Tokenizing corpus complete, fitting BM25Okapi")
        self._bm25 = BM25Okapi(tokenized_corpus)

        logger.info("BM25 index built successfully: %d documents indexed", len(chunks))

    # ------------------------------------------------------------------
    # Class-method factory
    # ------------------------------------------------------------------

    @classmethod
    def build_from_collection(
        cls,
        collection: "_chromadb_type.Collection",
        config: "AppConfig",
    ) -> "BM25Index":
        """Load all documents from a ChromaDB collection and build a BM25 index.

        Retrieves every document (no limit) via ``collection.get()``, flattens
        each hit into the same schema as :func:`src.retrieval.dense.dense_retrieve`
        returns, then constructs a :class:`BM25Index` from the full corpus.

        Args:
            collection: ChromaDB collection object (e.g. ``rag_contextualized_v1``).
            config:     AppConfig — currently unused but kept for symmetry with
                        other factory methods and future tokeniser options.

        Returns:
            A ready-to-use :class:`BM25Index` instance.

        Raises:
            ValueError: If the collection is empty.
        """
        logger.info(
            "BM25Index.build_from_collection: loading all docs from '%s'",
            collection.name,
        )

        result = collection.get(include=["documents", "metadatas"])

        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        if not ids:
            raise ValueError(
                f"Collection '{collection.name}' is empty — cannot build BM25 index."
            )

        logger.info(
            "Loaded %d documents from collection '%s'", len(ids), collection.name
        )

        chunks: list[dict] = []
        for doc_text, meta in zip(documents, metadatas):
            meta = meta or {}
            chunks.append(
                {
                    "text": doc_text or "",
                    "doc_id": meta.get("doc_id", ""),
                    "section": meta.get("section", ""),
                    "page_start": meta.get("page_start", 0),
                    "page_end": meta.get("page_end", 0),
                    "source_file": meta.get("source_file", ""),
                }
            )

        return cls(chunks)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int) -> list[dict]:
        """BM25 keyword search.

        Tokenizes *query*, scores every indexed document, then returns the
        *top_k* highest-scoring results. Scores are normalised to [0, 1] by
        dividing by the maximum score in the result set. Documents with a
        raw score of 0 (no term overlap) are excluded before normalisation.

        Args:
            query:  Raw query text.
            top_k:  Maximum number of results to return.

        Returns:
            List of result dicts (same schema as dense_retrieve):
                - ``text``        (str)
                - ``doc_id``      (str)
                - ``section``     (str)
                - ``page_start``  (int)
                - ``page_end``    (int)
                - ``source_file`` (str)
                - ``score``       (float) normalised BM25 score in [0, 1]

            Ordered from highest to lowest score. May return fewer than
            *top_k* items if fewer documents have any query-term overlap.
        """
        tokenized_query = _tokenize(query)

        logger.debug(
            "BM25Index.search: query=%r tokens=%d top_k=%d",
            query[:80],
            len(tokenized_query),
            top_k,
        )

        raw_scores = self._bm25.get_scores(tokenized_query)

        # Pair each score with its corpus index; keep only positive scores.
        scored = [
            (float(score), idx)
            for idx, score in enumerate(raw_scores)
            if score > 0.0
        ]

        if not scored:
            logger.warning(
                "BM25Index.search: no results with positive score for query=%r",
                query[:80],
            )
            return []

        # Sort descending by score, take top_k.
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        # Normalise scores to [0, 1].
        max_score = top[0][0]
        results: list[dict] = []
        for raw_score, idx in top:
            chunk = self._chunks[idx]
            normalised = round(raw_score / max_score, 6) if max_score > 0.0 else 0.0
            results.append(
                {
                    "text": chunk.get("text", ""),
                    "doc_id": chunk.get("doc_id", ""),
                    "section": chunk.get("section", ""),
                    "page_start": chunk.get("page_start", 0),
                    "page_end": chunk.get("page_end", 0),
                    "source_file": chunk.get("source_file", ""),
                    "score": normalised,
                }
            )

        logger.info(
            "BM25Index.search: query=%r returned %d results top_score=%.4f (normalised)",
            query[:80],
            len(results),
            results[0]["score"] if results else 0.0,
        )
        return results
