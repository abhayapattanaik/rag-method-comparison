"""Tests for src/evaluation/runner.py"""

from __future__ import annotations

import json
import os

import pytest

from src.evaluation.judge import MetricType
from src.evaluation.question_gen import EvalQuestion
from src.evaluation.runner import EvalResult, EvaluationRunner
from tests.conftest import MockLLMProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_result(
    question_id: str = "q001",
    pipeline: str = "traditional",
    metric: str = "context_precision",
    score: float = 0.75,
) -> EvalResult:
    return EvalResult(
        question_id=question_id,
        pipeline=pipeline,
        metric=metric,
        score=score,
        justification="Good retrieval quality.",
        answer="The answer text.",
        retrieved_chunks=["chunk one text", "chunk two text"],
        cost_usd=0.000120,
        latency_ms=300.0,
    )


def _make_runner(sample_config, tmp_path):
    """Build a minimal EvaluationRunner pointing at tmp_path."""
    # Override results_dir in config
    from src.config import AppConfig
    import copy

    # Patch results_dir to tmp_path
    cfg_data = sample_config.model_dump()
    cfg_data["paths"]["results_dir"] = str(tmp_path)
    patched_config = AppConfig.model_validate(cfg_data)
    return EvaluationRunner(pipelines={}, judge=None, config=patched_config)


# ---------------------------------------------------------------------------
# Result persistence (save + load roundtrip)
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(sample_config, tmp_path):
    runner = _make_runner(sample_config, tmp_path)
    result = _make_eval_result()
    runner._save_result(result)

    loaded = runner._load_result("q001", "traditional", "context_precision")
    assert loaded is not None
    assert loaded.question_id == result.question_id
    assert loaded.pipeline == result.pipeline
    assert loaded.metric == result.metric
    assert loaded.score == pytest.approx(result.score)
    assert loaded.justification == result.justification
    assert loaded.answer == result.answer
    assert loaded.retrieved_chunks == result.retrieved_chunks


def test_save_creates_json_file(sample_config, tmp_path):
    runner = _make_runner(sample_config, tmp_path)
    result = _make_eval_result()
    runner._save_result(result)
    expected_path = runner._result_path("q001", "traditional", "context_precision")
    assert os.path.isfile(expected_path)


def test_load_missing_returns_none(sample_config, tmp_path):
    runner = _make_runner(sample_config, tmp_path)
    result = runner._load_result("nonexistent", "pipeline", "metric")
    assert result is None


def test_saved_json_is_valid(sample_config, tmp_path):
    runner = _make_runner(sample_config, tmp_path)
    result = _make_eval_result()
    runner._save_result(result)
    path = runner._result_path("q001", "traditional", "context_precision")
    with open(path, "r") as fh:
        data = json.load(fh)
    assert data["question_id"] == "q001"
    assert data["score"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# _is_completed check
# ---------------------------------------------------------------------------


def test_is_completed_false_before_save(sample_config, tmp_path):
    runner = _make_runner(sample_config, tmp_path)
    assert runner._is_completed("q001", "traditional", "context_precision") is False


def test_is_completed_true_after_save(sample_config, tmp_path):
    runner = _make_runner(sample_config, tmp_path)
    result = _make_eval_result()
    runner._save_result(result)
    assert runner._is_completed("q001", "traditional", "context_precision") is True


def test_is_completed_false_for_different_metric(sample_config, tmp_path):
    runner = _make_runner(sample_config, tmp_path)
    result = _make_eval_result(metric="context_precision")
    runner._save_result(result)
    assert runner._is_completed("q001", "traditional", "faithfulness") is False


def test_is_completed_false_for_llm_call_failed_justification(sample_config, tmp_path):
    """A result with 'LLM call failed' in justification must NOT be treated as complete."""
    runner = _make_runner(sample_config, tmp_path)
    failed_result = EvalResult(
        question_id="q001",
        pipeline="traditional",
        metric="context_precision",
        score=0.0,
        justification="LLM call failed: connection timeout",
        answer="",
        retrieved_chunks=[],
        cost_usd=0.0,
        latency_ms=0.0,
    )
    runner._save_result(failed_result)
    assert runner._is_completed("q001", "traditional", "context_precision") is False


def test_is_completed_false_for_generic_failed_justification(sample_config, tmp_path):
    """A result whose justification contains 'failed' must NOT be treated as complete."""
    runner = _make_runner(sample_config, tmp_path)
    failed_result = EvalResult(
        question_id="q002",
        pipeline="hybrid",
        metric="faithfulness",
        score=0.0,
        justification="Judge call failed: unexpected error",
        answer="",
        retrieved_chunks=[],
        cost_usd=0.0,
        latency_ms=0.0,
    )
    runner._save_result(failed_result)
    assert runner._is_completed("q002", "hybrid", "faithfulness") is False


def test_is_completed_true_for_zero_score_valid_justification(sample_config, tmp_path):
    """score=0.0 with a legitimate (non-failure) justification IS complete."""
    runner = _make_runner(sample_config, tmp_path)
    low_score_result = EvalResult(
        question_id="q003",
        pipeline="traditional",
        metric="answer_relevancy",
        score=0.0,
        justification="The answer did not address the question at all.",
        answer="42",
        retrieved_chunks=["irrelevant chunk"],
        cost_usd=0.001,
        latency_ms=200.0,
    )
    runner._save_result(low_score_result)
    assert runner._is_completed("q003", "traditional", "answer_relevancy") is True


def test_is_completed_false_for_corrupt_json(sample_config, tmp_path):
    """A result file with invalid JSON content must NOT be treated as complete."""
    runner = _make_runner(sample_config, tmp_path)
    # Manually write a corrupt JSON file at the expected path.
    path = runner._result_path("q004", "modern", "context_recall")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{this is not valid json")
    assert runner._is_completed("q004", "modern", "context_recall") is False


# ---------------------------------------------------------------------------
# Deterministic result paths
# ---------------------------------------------------------------------------


def test_result_path_deterministic(sample_config, tmp_path):
    runner = _make_runner(sample_config, tmp_path)
    path1 = runner._result_path("q001", "traditional", "context_precision")
    path2 = runner._result_path("q001", "traditional", "context_precision")
    assert path1 == path2


def test_result_path_differs_for_different_pipeline(sample_config, tmp_path):
    runner = _make_runner(sample_config, tmp_path)
    path_a = runner._result_path("q001", "traditional", "faithfulness")
    path_b = runner._result_path("q001", "modern", "faithfulness")
    assert path_a != path_b


def test_result_path_contains_all_components(sample_config, tmp_path):
    runner = _make_runner(sample_config, tmp_path)
    path = runner._result_path("q123", "hybrid", "answer_relevancy")
    fname = os.path.basename(path)
    assert "q123" in fname
    assert "hybrid" in fname
    assert "answer_relevancy" in fname
    assert fname.endswith(".json")


# ---------------------------------------------------------------------------
# Resumability — skip existing results
# ---------------------------------------------------------------------------


def test_load_all_results_empty_dir(sample_config, tmp_path):
    runner = _make_runner(sample_config, tmp_path)
    results = runner.load_all_results()
    assert results == []


def test_load_all_results_loads_saved(sample_config, tmp_path):
    runner = _make_runner(sample_config, tmp_path)
    r1 = _make_eval_result("q001", "traditional", "faithfulness", 0.8)
    r2 = _make_eval_result("q001", "hybrid", "context_recall", 0.6)
    runner._save_result(r1)
    runner._save_result(r2)

    loaded = runner.load_all_results()
    assert len(loaded) == 2


def test_load_all_results_ignores_non_json(sample_config, tmp_path):
    runner = _make_runner(sample_config, tmp_path)
    # Create a non-JSON file in the results directory
    eval_dir = runner.results_dir
    with open(os.path.join(eval_dir, "not_a_result.txt"), "w") as f:
        f.write("dummy")
    results = runner.load_all_results()
    assert results == []
