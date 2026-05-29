"""Reciprocal Rank Fusion (RRF) for the RAG Comparison System.

Merges multiple ranked result lists — e.g. dense retrieval + BM25 — into a
single fused ranking without requiring score normalisation across retrieval
methods. Documents are identified by a SHA-256 hash of their text content so
duplicate entries across lists are collapsed correctly.

Reference: Cormack, Clarke & Buettcher (2009) "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods".

RRF formula::

    rrf_score(d) = sum_over_lists( 1 / (k + rank(d, list)) )

where ``rank`` is 1-indexed and ``k`` is a smoothing constant (default 60).

Typical usage::

    from src.retrieval.fusion import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion([dense_results, bm25_results], k=60)
    top_chunks = fused[:top_k_fusion]
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # no runtime-only imports needed

logger = logging.getLogger(__name__)


def _chunk_key(chunk: dict) -> str:
    """Return a stable deduplication key for a chunk dict.

    Uses a SHA-256 digest of the chunk's text content.  Chunks that are
    identical in text (even from different retrieval methods) map to the same
    key and are merged during fusion.
    """
    text = chunk.get("text", "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Merge ranked result lists using Reciprocal Rank Fusion.

    Each document is identified by a hash of its text.  If the same document
    appears in multiple lists its RRF contributions are summed.  The returned
    list is deduplicated and sorted by descending RRF score.

    The ``score`` field of each returned dict is the RRF score (not normalised
    — values are in the range (0, n_lists/k] where n_lists is len(ranked_lists)
    and k is the smoothing constant).

    Args:
        ranked_lists: One or more ranked lists of chunk dicts.  Each inner list
                      should be ordered from most-relevant (index 0) to
                      least-relevant.  An empty outer list or empty inner lists
                      are handled gracefully.
        k:            RRF smoothing constant.  Default 60 follows the original
                      paper's recommendation.  Higher values reduce the
                      influence of rank position; lower values amplify it.

    Returns:
        Deduplicated list of chunk dicts sorted by descending RRF score.  The
        ``score`` field is replaced with the computed RRF score.  All other
        fields (``text``, ``doc_id``, ``source_file``, ``page_start``,
        ``page_end``, ``section``) are preserved from the first ranked list in
        which the document appeared.
    """
    if not ranked_lists:
        logger.warning("reciprocal_rank_fusion: received empty ranked_lists, returning []")
        return []

    non_empty = [lst for lst in ranked_lists if lst]
    if not non_empty:
        logger.warning("reciprocal_rank_fusion: all ranked lists are empty, returning []")
        return []

    logger.info(
        "reciprocal_rank_fusion: merging %d list(s) with sizes %s k=%d",
        len(non_empty),
        [len(lst) for lst in non_empty],
        k,
    )

    # rrf_scores: key -> accumulated RRF score
    rrf_scores: dict[str, float] = {}
    # first_seen: key -> representative chunk dict (from first occurrence)
    first_seen: dict[str, dict] = {}

    for list_idx, ranked_list in enumerate(non_empty):
        for rank_0based, chunk in enumerate(ranked_list):
            rank_1based = rank_0based + 1
            key = _chunk_key(chunk)
            contribution = 1.0 / (k + rank_1based)

            if key not in rrf_scores:
                rrf_scores[key] = 0.0
                first_seen[key] = chunk
                logger.debug(
                    "reciprocal_rank_fusion: new doc key=%s... list=%d rank=%d contrib=%.6f",
                    key[:12],
                    list_idx,
                    rank_1based,
                    contribution,
                )
            else:
                logger.debug(
                    "reciprocal_rank_fusion: merge doc key=%s... list=%d rank=%d contrib=%.6f",
                    key[:12],
                    list_idx,
                    rank_1based,
                    contribution,
                )

            rrf_scores[key] += contribution

    # Build result list, replacing score with RRF score.
    fused: list[dict] = []
    for key, rrf_score in rrf_scores.items():
        chunk = dict(first_seen[key])  # shallow copy so we don't mutate callers' data
        chunk["score"] = round(rrf_score, 8)
        fused.append(chunk)

    fused.sort(key=lambda c: c["score"], reverse=True)

    logger.info(
        "reciprocal_rank_fusion: fused %d unique documents (from %d total entries); "
        "top_score=%.6f",
        len(fused),
        sum(len(lst) for lst in non_empty),
        fused[0]["score"] if fused else 0.0,
    )

    return fused
