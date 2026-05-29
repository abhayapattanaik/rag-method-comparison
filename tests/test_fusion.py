"""Tests for src/retrieval/fusion.py"""

from __future__ import annotations

import pytest

from src.retrieval.fusion import _chunk_key, reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str, score: float = 1.0) -> dict:
    return {
        "text": text,
        "doc_id": "doc_1",
        "section": "intro",
        "page_start": 1,
        "page_end": 1,
        "source_file": "doc_1.pdf",
        "score": score,
    }


# ---------------------------------------------------------------------------
# RRF score calculation with known values
# ---------------------------------------------------------------------------


def test_rrf_score_single_list_rank_1():
    """Rank-1 doc in single list with k=60: score = 1/(60+1) ≈ 0.016393."""
    chunks = [_make_chunk("alpha text"), _make_chunk("beta text"), _make_chunk("gamma text")]
    fused = reciprocal_rank_fusion([chunks], k=60)
    # Top result is the first-ranked doc
    top_score = fused[0]["score"]
    expected = 1.0 / (60 + 1)
    assert abs(top_score - expected) < 1e-6


def test_rrf_score_two_lists_same_doc():
    """Same doc at rank-1 in two lists should accumulate 2/(60+1) ≈ 0.032787."""
    doc = _make_chunk("shared document")
    list_a = [doc]
    list_b = [doc]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    assert len(fused) == 1
    expected = 2.0 / (60 + 1)
    assert abs(fused[0]["score"] - expected) < 1e-6


def test_rrf_score_rank_position_matters():
    """Lower rank should produce lower RRF score."""
    chunks_a = [_make_chunk(f"doc {i}") for i in range(5)]
    fused = reciprocal_rank_fusion([chunks_a], k=60)
    scores = [r["score"] for r in fused]
    # Scores should be strictly decreasing
    assert all(scores[i] > scores[i + 1] for i in range(len(scores) - 1))


# ---------------------------------------------------------------------------
# Deduplication across lists
# ---------------------------------------------------------------------------


def test_deduplication_same_text():
    """Same text appearing in both lists should be merged into one result."""
    doc = _make_chunk("shared chunk text")
    other_a = _make_chunk("unique to list a")
    other_b = _make_chunk("unique to list b")
    fused = reciprocal_rank_fusion([[doc, other_a], [doc, other_b]], k=60)
    texts = [r["text"] for r in fused]
    assert texts.count("shared chunk text") == 1


def test_deduplication_preserves_unique_docs():
    """Docs that appear in only one list should be included."""
    list_a = [_make_chunk("only in a")]
    list_b = [_make_chunk("only in b")]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    texts = {r["text"] for r in fused}
    assert "only in a" in texts
    assert "only in b" in texts


# ---------------------------------------------------------------------------
# Single list passthrough
# ---------------------------------------------------------------------------


def test_single_list_passthrough():
    """With a single ranked list, output should contain same docs re-scored."""
    chunks = [_make_chunk(f"document {i}") for i in range(3)]
    fused = reciprocal_rank_fusion([chunks], k=60)
    assert len(fused) == 3
    # Order should be preserved (rank-1 stays rank-1)
    assert fused[0]["text"] == "document 0"


def test_single_list_scores_replaced():
    """Original score fields should be replaced with RRF scores."""
    chunks = [_make_chunk("doc a", score=0.99), _make_chunk("doc b", score=0.50)]
    fused = reciprocal_rank_fusion([chunks], k=60)
    for r in fused:
        # Scores should now be RRF values (much smaller than 0.99)
        assert r["score"] < 0.1


# ---------------------------------------------------------------------------
# Empty list handling
# ---------------------------------------------------------------------------


def test_empty_outer_list_returns_empty():
    result = reciprocal_rank_fusion([], k=60)
    assert result == []


def test_all_inner_lists_empty_returns_empty():
    result = reciprocal_rank_fusion([[], []], k=60)
    assert result == []


def test_one_empty_one_populated():
    """One empty + one populated list → results from populated list only."""
    chunks = [_make_chunk("doc a"), _make_chunk("doc b")]
    fused = reciprocal_rank_fusion([[], chunks], k=60)
    assert len(fused) == 2


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


def test_result_has_score_field():
    chunks = [_make_chunk("hello world")]
    fused = reciprocal_rank_fusion([chunks], k=60)
    assert "score" in fused[0]


def test_result_preserves_metadata():
    chunk = _make_chunk("some text")
    chunk["doc_id"] = "specific_doc"
    fused = reciprocal_rank_fusion([[chunk]], k=60)
    assert fused[0]["doc_id"] == "specific_doc"


# ---------------------------------------------------------------------------
# chunk_key
# ---------------------------------------------------------------------------


def test_chunk_key_same_text_same_key():
    c1 = _make_chunk("same text")
    c2 = _make_chunk("same text")
    assert _chunk_key(c1) == _chunk_key(c2)


def test_chunk_key_different_text_different_key():
    c1 = _make_chunk("text a")
    c2 = _make_chunk("text b")
    assert _chunk_key(c1) != _chunk_key(c2)
