"""Tests for src/evaluation/judge.py"""

from __future__ import annotations

import json

import pytest

from src.evaluation.judge import Judge, JudgeResult, MetricType
from src.llm.base import LLMResponse
from tests.conftest import MockLLMProvider, make_llm_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_judge(mock_provider, sample_config, telemetry=None):
    return Judge(provider=mock_provider, config=sample_config, telemetry=telemetry)


def _call_args():
    return dict(
        question="What is RAG?",
        answer="RAG stands for Retrieval-Augmented Generation.",
        contexts=["Context about RAG systems.", "More context on retrieval."],
        ground_truth="RAG is a technique that augments LLMs with retrieved documents.",
    )


# ---------------------------------------------------------------------------
# Valid JSON response parsing
# ---------------------------------------------------------------------------


def test_valid_json_score_parsed(mock_provider, sample_config):
    mock_provider._canned_text = '{"score": 0.75, "justification": "Good retrieval."}'
    judge = _make_judge(mock_provider, sample_config)
    result = judge.score(MetricType.CONTEXT_PRECISION, **_call_args())
    assert result.score == pytest.approx(0.75)
    assert result.justification == "Good retrieval."


def test_valid_json_score_zero(mock_provider, sample_config):
    mock_provider._canned_text = '{"score": 0.0, "justification": "No relevant chunks."}'
    judge = _make_judge(mock_provider, sample_config)
    result = judge.score(MetricType.FAITHFULNESS, **_call_args())
    assert result.score == 0.0


def test_valid_json_score_one(mock_provider, sample_config):
    mock_provider._canned_text = '{"score": 1.0, "justification": "Perfect score."}'
    judge = _make_judge(mock_provider, sample_config)
    result = judge.score(MetricType.ANSWER_RELEVANCY, **_call_args())
    assert result.score == 1.0


# ---------------------------------------------------------------------------
# Malformed JSON fallback → score = 0.0
# ---------------------------------------------------------------------------


def test_malformed_json_returns_zero(mock_provider, sample_config):
    mock_provider._canned_text = "This is not valid JSON at all!"
    judge = _make_judge(mock_provider, sample_config)
    result = judge.score(MetricType.CONTEXT_RECALL, **_call_args())
    assert result.score == 0.0
    assert "JSON parse error" in result.justification or "parse" in result.justification.lower()


def test_empty_response_returns_zero(mock_provider, sample_config):
    mock_provider._canned_text = ""
    judge = _make_judge(mock_provider, sample_config)
    result = judge.score(MetricType.CONTEXT_PRECISION, **_call_args())
    assert result.score == 0.0


def test_partial_json_returns_zero(mock_provider, sample_config):
    mock_provider._canned_text = '{"score": 0.8'  # truncated
    judge = _make_judge(mock_provider, sample_config)
    result = judge.score(MetricType.FAITHFULNESS, **_call_args())
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# Score clamping
# ---------------------------------------------------------------------------


def test_score_above_one_clamped_to_one(mock_provider, sample_config):
    mock_provider._canned_text = '{"score": 1.5, "justification": "Over-scored."}'
    judge = _make_judge(mock_provider, sample_config)
    result = judge.score(MetricType.CONTEXT_PRECISION, **_call_args())
    assert result.score == 1.0


def test_score_below_zero_clamped_to_zero(mock_provider, sample_config):
    mock_provider._canned_text = '{"score": -0.3, "justification": "Negative."}'
    judge = _make_judge(mock_provider, sample_config)
    result = judge.score(MetricType.ANSWER_RELEVANCY, **_call_args())
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# All 4 metric types work
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", list(MetricType))
def test_all_metrics_return_judge_result(mock_provider, sample_config, metric):
    mock_provider._canned_text = '{"score": 0.6, "justification": "Adequate."}'
    judge = _make_judge(mock_provider, sample_config)
    result = judge.score(metric, **_call_args())
    assert isinstance(result, JudgeResult)
    assert result.metric == metric
    assert 0.0 <= result.score <= 1.0


def test_score_all_metrics_returns_four_results(mock_provider, sample_config):
    mock_provider._canned_text = '{"score": 0.7, "justification": "OK."}'
    judge = _make_judge(mock_provider, sample_config)
    results = judge.score_all_metrics(**_call_args())
    assert len(results) == 4
    metrics = {r.metric for r in results}
    assert metrics == set(MetricType)


# ---------------------------------------------------------------------------
# Missing fields in response
# ---------------------------------------------------------------------------


def test_missing_score_field_returns_zero(mock_provider, sample_config):
    mock_provider._canned_text = '{"justification": "No score key here."}'
    judge = _make_judge(mock_provider, sample_config)
    result = judge.score(MetricType.CONTEXT_RECALL, **_call_args())
    assert result.score == 0.0
    assert "Missing" in result.justification or "score" in result.justification.lower()


def test_missing_justification_uses_fallback(mock_provider, sample_config):
    mock_provider._canned_text = '{"score": 0.5}'
    judge = _make_judge(mock_provider, sample_config)
    result = judge.score(MetricType.FAITHFULNESS, **_call_args())
    assert result.score == 0.5
    assert result.justification  # should have some fallback text


def test_non_numeric_score_returns_zero(mock_provider, sample_config):
    mock_provider._canned_text = '{"score": "high", "justification": "Not a number."}'
    judge = _make_judge(mock_provider, sample_config)
    result = judge.score(MetricType.ANSWER_RELEVANCY, **_call_args())
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# Markdown fence stripping
# ---------------------------------------------------------------------------


def test_markdown_fenced_json_parsed_correctly(mock_provider, sample_config):
    mock_provider._canned_text = "```json\n{\"score\": 0.9, \"justification\": \"Fenced.\"}\n```"
    judge = _make_judge(mock_provider, sample_config)
    result = judge.score(MetricType.CONTEXT_PRECISION, **_call_args())
    assert result.score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# LLMResponse attached
# ---------------------------------------------------------------------------


def test_result_has_llm_response(mock_provider, sample_config):
    mock_provider._canned_text = '{"score": 0.8, "justification": "Good."}'
    judge = _make_judge(mock_provider, sample_config)
    result = judge.score(MetricType.FAITHFULNESS, **_call_args())
    assert isinstance(result.llm_response, LLMResponse)
    assert result.llm_response.input_tokens == 100  # from make_llm_response
