"""Tests for src/ingestion/chunker.py"""

from __future__ import annotations

import pytest

from src.ingestion.chunker import (
    Chunk,
    _chunk_text,
    _config_version,
    _make_chunk_id,
    _split_into_sections,
    chunk_document,
)


# ---------------------------------------------------------------------------
# Deterministic chunk IDs
# ---------------------------------------------------------------------------


def test_chunk_id_deterministic():
    """Same doc_id + index + config_version always produces the same chunk_id."""
    cfg_ver = _config_version(400, 50)
    id1 = _make_chunk_id("my_doc", 0, cfg_ver)
    id2 = _make_chunk_id("my_doc", 0, cfg_ver)
    assert id1 == id2


def test_chunk_id_differs_for_different_index():
    cfg_ver = _config_version(400, 50)
    id0 = _make_chunk_id("my_doc", 0, cfg_ver)
    id1 = _make_chunk_id("my_doc", 1, cfg_ver)
    assert id0 != id1


def test_chunk_id_differs_for_different_config():
    cfg_ver_a = _config_version(400, 50)
    cfg_ver_b = _config_version(200, 25)
    id_a = _make_chunk_id("my_doc", 0, cfg_ver_a)
    id_b = _make_chunk_id("my_doc", 0, cfg_ver_b)
    assert id_a != id_b


def test_chunk_ids_stable_across_calls(sample_config):
    """chunk_document called twice on same input produces identical chunk_ids."""
    text = "# Introduction\n\nThis is a short test document for chunking."
    chunks1 = chunk_document(text, "test_doc.md", sample_config)
    chunks2 = chunk_document(text, "test_doc.md", sample_config)
    ids1 = [c.chunk_id for c in chunks1]
    ids2 = [c.chunk_id for c in chunks2]
    assert ids1 == ids2


# ---------------------------------------------------------------------------
# Token count accuracy
# ---------------------------------------------------------------------------


def test_token_count_positive(sample_config):
    text = "# Section\n\n" + "word " * 50
    chunks = chunk_document(text, "doc.md", sample_config)
    for chunk in chunks:
        assert chunk.token_count > 0


def test_token_count_within_chunk_size(sample_config):
    """No chunk should have a token count more than 2x chunk_size (generous bound)."""
    text = "# Section\n\n" + "word " * 300
    chunks = chunk_document(text, "doc.md", sample_config)
    for chunk in chunks:
        assert chunk.token_count <= sample_config.chunking.chunk_size * 2


# ---------------------------------------------------------------------------
# Section-aware splitting
# ---------------------------------------------------------------------------


def test_section_detection_with_markdown_headings(sample_config):
    text = "# Introduction\n\nSome intro text.\n\n## Methods\n\nSome methods text."
    chunks = chunk_document(text, "doc.md", sample_config)
    sections = {c.section for c in chunks}
    assert "Introduction" in sections or "Methods" in sections


def test_section_unknown_for_plain_text(sample_config):
    """Text with no markdown headings should produce 'unknown' section."""
    text = "This is plain text with no headings. " * 20
    chunks = chunk_document(text, "doc.md", sample_config)
    for chunk in chunks:
        assert chunk.section == "unknown"


def test_split_into_sections_detects_headings():
    text = "# Alpha\n\nContent A.\n\n## Beta\n\nContent B."
    sections = _split_into_sections(text)
    headings = [s[0] for s in sections]
    assert "Alpha" in headings
    assert "Beta" in headings


def test_split_into_sections_preamble():
    """Text before first heading is labelled 'unknown'."""
    text = "Preamble text here.\n\n# Section One\n\nBody."
    sections = _split_into_sections(text)
    assert sections[0][0] == "unknown"
    assert "Preamble" in sections[0][1]


def test_split_into_sections_no_headings():
    text = "Just some plain text."
    sections = _split_into_sections(text)
    assert len(sections) == 1
    assert sections[0][0] == "unknown"


# ---------------------------------------------------------------------------
# Chunk overlap
# ---------------------------------------------------------------------------


def test_chunk_overlap_produces_multiple_chunks():
    """Long text with overlap should produce multiple chunks, not one."""
    # chunk_size=10 with overlap=3 should split easily
    long_text = "one two three four five six seven eight nine ten eleven twelve thirteen"
    chunks = _chunk_text(long_text, chunk_size=10, chunk_overlap=3)
    assert len(chunks) > 1


def test_chunk_overlap_consecutive_chunks_share_words():
    """With overlap > 0, consecutive chunks should share some token content."""
    long_text = " ".join([f"word{i}" for i in range(40)])
    chunks = _chunk_text(long_text, chunk_size=10, chunk_overlap=3)
    if len(chunks) >= 2:
        words0 = set(chunks[0].split())
        words1 = set(chunks[1].split())
        # There should be some overlap in words between adjacent chunks
        assert len(words0 & words1) >= 1


# ---------------------------------------------------------------------------
# Empty input handling
# ---------------------------------------------------------------------------


def test_empty_text_returns_no_chunks(sample_config):
    chunks = chunk_document("", "empty.md", sample_config)
    assert chunks == []


def test_whitespace_only_text_returns_no_chunks(sample_config):
    chunks = chunk_document("   \n\n\t  ", "empty.md", sample_config)
    assert chunks == []


def test_chunk_text_empty_returns_empty():
    result = _chunk_text("", chunk_size=100, chunk_overlap=10)
    assert result == []


# ---------------------------------------------------------------------------
# Chunk metadata
# ---------------------------------------------------------------------------


def test_chunk_has_correct_doc_id(sample_config):
    text = "# Section\n\nSome text here."
    chunks = chunk_document(text, "my_paper.md", sample_config)
    for chunk in chunks:
        assert chunk.doc_id == "my_paper"


def test_chunk_has_correct_source_file(sample_config):
    text = "# Section\n\nSome text here."
    chunks = chunk_document(text, "my_paper.md", sample_config)
    for chunk in chunks:
        assert chunk.source_file == "my_paper.md"


def test_chunk_indices_are_sequential(sample_config):
    text = "# Section\n\n" + " ".join([f"word{i}" for i in range(200)])
    chunks = chunk_document(text, "doc.md", sample_config)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))
