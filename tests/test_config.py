"""Tests for src/config.py"""

from __future__ import annotations

import os

import pytest
import yaml

from src.config import AppConfig, load_config, _set_nested


# ---------------------------------------------------------------------------
# Default config loads
# ---------------------------------------------------------------------------


def test_load_config_returns_app_config():
    """load_config() with no arguments should load the project default config."""
    config = load_config()
    assert isinstance(config, AppConfig)


def test_default_config_has_expected_models():
    config = load_config()
    assert config.models.embedding_model == "BAAI/bge-base-en-v1.5"
    assert config.models.reranker_model == "BAAI/bge-reranker-v2-m3"


def test_default_config_has_paths():
    config = load_config()
    assert config.paths.data_dir is not None
    assert config.paths.chroma_dir is not None
    assert config.paths.results_dir is not None


def test_default_config_has_chunking():
    config = load_config()
    assert config.chunking.chunk_size > 0
    assert config.chunking.chunk_overlap >= 0
    assert config.chunking.chunk_overlap < config.chunking.chunk_size


def test_default_config_has_four_metrics():
    config = load_config()
    assert len(config.evaluation.metrics) == 4
    expected_metrics = {
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    }
    assert set(config.evaluation.metrics) == expected_metrics


def test_default_config_retrieval_defaults():
    config = load_config()
    assert config.retrieval.rrf_k == 60
    assert config.retrieval.top_k_dense > 0
    assert config.retrieval.top_k_rerank > 0


# ---------------------------------------------------------------------------
# CLI overrides
# ---------------------------------------------------------------------------


def test_cli_override_chunk_size():
    config = load_config(cli_overrides={"chunking.chunk_size": 200})
    assert config.chunking.chunk_size == 200


def test_cli_override_chunk_overlap():
    config = load_config(cli_overrides={"chunking.chunk_overlap": 25})
    assert config.chunking.chunk_overlap == 25


def test_cli_override_string_converted_to_int():
    """String values from CLI should be auto-coerced to int."""
    config = load_config(cli_overrides={"chunking.chunk_size": "350"})
    assert config.chunking.chunk_size == 350
    assert isinstance(config.chunking.chunk_size, int)


def test_cli_override_top_k():
    config = load_config(cli_overrides={"retrieval.top_k_dense": 15})
    assert config.retrieval.top_k_dense == 15


def test_cli_override_temperature():
    config = load_config(cli_overrides={"llm.temperature": 0.5})
    assert config.llm.temperature == pytest.approx(0.5)


def test_cli_override_does_not_affect_other_fields():
    config = load_config(cli_overrides={"chunking.chunk_size": 999})
    # Other fields should be unaffected
    assert config.models.embedding_model == "BAAI/bge-base-en-v1.5"


# ---------------------------------------------------------------------------
# Missing config file error
# ---------------------------------------------------------------------------


def test_missing_config_file_raises_file_not_found(tmp_path):
    nonexistent = str(tmp_path / "nonexistent_config.yaml")
    with pytest.raises(FileNotFoundError):
        load_config(config_path=nonexistent)


def test_custom_config_file_loaded(tmp_path):
    """A custom YAML config file at a given path should be loaded correctly."""
    config_data = {
        "paths": {
            "data_dir": "data",
            "chroma_dir": "data/chroma",
            "cache_dir": "data/cache",
            "results_dir": "data/results",
            "papers_dir": "data/papers",
        },
        "chunking": {
            "chunk_size": 123,
            "chunk_overlap": 12,
        },
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as fh:
        yaml.dump(config_data, fh)

    config = load_config(config_path=str(config_file))
    assert config.chunking.chunk_size == 123
    assert config.chunking.chunk_overlap == 12


# ---------------------------------------------------------------------------
# _set_nested helper
# ---------------------------------------------------------------------------


def test_set_nested_simple_key():
    d = {}
    _set_nested(d, "key", "value")
    assert d["key"] == "value"


def test_set_nested_two_levels():
    d = {}
    _set_nested(d, "outer.inner", 42)
    assert d["outer"]["inner"] == 42


def test_set_nested_three_levels():
    d = {}
    _set_nested(d, "a.b.c", True)
    assert d["a"]["b"]["c"] is True


def test_set_nested_int_coercion():
    d = {}
    _set_nested(d, "chunking.chunk_size", "400")
    assert d["chunking"]["chunk_size"] == 400
    assert isinstance(d["chunking"]["chunk_size"], int)


def test_set_nested_float_coercion():
    d = {}
    _set_nested(d, "llm.temperature", "0.7")
    assert d["llm"]["temperature"] == pytest.approx(0.7)


def test_set_nested_bool_coercion_true():
    d = {}
    _set_nested(d, "flag", "true")
    assert d["flag"] is True


def test_set_nested_bool_coercion_false():
    d = {}
    _set_nested(d, "flag", "false")
    assert d["flag"] is False
