"""Tests for src/retrieval/bm25.py"""

from __future__ import annotations

import pytest

from src.retrieval.bm25 import BM25Index, _tokenize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunks(texts: list[str]) -> list[dict]:
    return [
        {
            "text": t,
            "doc_id": f"doc_{i}",
            "section": "intro",
            "page_start": 1,
            "page_end": 1,
            "source_file": f"doc_{i}.pdf",
        }
        for i, t in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# Score normalisation (0-1 range)
# ---------------------------------------------------------------------------


def test_top_score_is_one():
    """The highest-scoring result should always have score == 1.0."""
    chunks = _make_chunks([
        "machine learning neural network deep learning",
        "retrieval augmented generation transformer",
        "random text about cooking pasta",
    ])
    index = BM25Index(chunks)
    results = index.search("machine learning", top_k=10)
    assert results[0]["score"] == 1.0


def test_all_scores_in_range():
    chunks = _make_chunks([
        "the quick brown fox",
        "lazy dog sleeping",
        "quick fox jumps",
    ])
    index = BM25Index(chunks)
    results = index.search("quick fox", top_k=10)
    for r in results:
        assert 0.0 <= r["score"] <= 1.0


# ---------------------------------------------------------------------------
# Top-k filtering
# ---------------------------------------------------------------------------


def test_top_k_limits_results():
    texts = [f"document number {i} retrieval" for i in range(20)]
    index = BM25Index(_make_chunks(texts))
    results = index.search("document retrieval", top_k=5)
    assert len(results) <= 5


def test_results_sorted_descending():
    chunks = _make_chunks([
        "retrieval augmented generation",
        "retrieval only",
        "augmented generation systems retrieval information",
    ])
    index = BM25Index(chunks)
    results = index.search("retrieval augmented generation", top_k=10)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Empty query handling
# ---------------------------------------------------------------------------


def test_empty_query_returns_empty():
    chunks = _make_chunks(["some text here", "more text there"])
    index = BM25Index(chunks)
    results = index.search("", top_k=5)
    assert results == []


def test_no_overlap_query_returns_empty():
    """A query with zero term overlap with corpus returns empty list."""
    chunks = _make_chunks(["machine learning", "deep neural networks"])
    index = BM25Index(chunks)
    results = index.search("zzzznotawordxxx", top_k=5)
    assert results == []


# ---------------------------------------------------------------------------
# Known document retrieval
# ---------------------------------------------------------------------------


def test_known_document_retrieved():
    """The document containing the exact query terms should be top-1."""
    chunks = _make_chunks([
        "reciprocal rank fusion combines ranked lists",
        "random text about something else entirely",
        "neural networks for image classification",
    ])
    index = BM25Index(chunks)
    results = index.search("reciprocal rank fusion", top_k=5)
    assert len(results) >= 1
    assert "reciprocal" in results[0]["text"].lower()


def test_result_schema():
    """Every result dict contains all required schema keys."""
    required_keys = {"text", "doc_id", "section", "page_start", "page_end", "source_file", "score"}
    chunks = _make_chunks(["hello world test"])
    index = BM25Index(chunks)
    results = index.search("hello world", top_k=5)
    for r in results:
        assert required_keys.issubset(r.keys())


# ---------------------------------------------------------------------------
# Empty corpus
# ---------------------------------------------------------------------------


def test_empty_corpus_raises():
    with pytest.raises(ValueError, match="empty"):
        BM25Index([])


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenize_lowercases():
    assert _tokenize("Hello World") == ["hello", "world"]


def test_tokenize_splits_whitespace():
    assert _tokenize("a b  c") == ["a", "b", "c"]


def test_tokenize_empty():
    assert _tokenize("") == []
