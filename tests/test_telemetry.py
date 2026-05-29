"""Tests for src/telemetry.py"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pytest

from src.telemetry import LLMCallRecord, TelemetryTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    operation: str = "generate_answer",
    pipeline: str | None = "traditional",
    input_tokens: int = 500,
    output_tokens: int = 100,
    latency_ms: float = 300.0,
    cost_usd: float = 0.001,
) -> LLMCallRecord:
    return LLMCallRecord(
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        operation=operation,
        pipeline=pipeline,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        success=True,
        error=None,
    )


# ---------------------------------------------------------------------------
# record + get_pipeline_summary
# ---------------------------------------------------------------------------


def test_record_and_get_pipeline_summary(sample_config):
    tracker = TelemetryTracker(sample_config)
    tracker.record(_make_record(pipeline="traditional", input_tokens=500, output_tokens=100, cost_usd=0.002))
    tracker.record(_make_record(pipeline="traditional", input_tokens=600, output_tokens=150, cost_usd=0.003))
    summary = tracker.get_pipeline_summary("traditional")
    assert summary["total_calls"] == 2
    assert summary["total_input_tokens"] == 1100
    assert summary["total_output_tokens"] == 250
    assert summary["total_cost"] == pytest.approx(0.005)


def test_pipeline_summary_excludes_other_pipelines(sample_config):
    tracker = TelemetryTracker(sample_config)
    tracker.record(_make_record(pipeline="traditional"))
    tracker.record(_make_record(pipeline="hybrid"))
    summary = tracker.get_pipeline_summary("traditional")
    assert summary["total_calls"] == 1


def test_pipeline_summary_empty_returns_zeros(sample_config):
    tracker = TelemetryTracker(sample_config)
    summary = tracker.get_pipeline_summary("nonexistent")
    assert summary["total_calls"] == 0
    assert summary["total_cost"] == 0.0
    assert summary["avg_latency"] == 0.0


def test_pipeline_summary_none_pipeline(sample_config):
    """Records with pipeline=None (shared ops) can be queried."""
    tracker = TelemetryTracker(sample_config)
    tracker.record(_make_record(pipeline=None))
    summary = tracker.get_pipeline_summary(None)
    assert summary["total_calls"] == 1


# ---------------------------------------------------------------------------
# get_operation_summary
# ---------------------------------------------------------------------------


def test_operation_summary(sample_config):
    tracker = TelemetryTracker(sample_config)
    tracker.record(_make_record(operation="judge", pipeline="traditional"))
    tracker.record(_make_record(operation="judge", pipeline="hybrid"))
    tracker.record(_make_record(operation="contextualize", pipeline="contextual"))
    summary = tracker.get_operation_summary("judge")
    assert summary["total_calls"] == 2


def test_operation_summary_empty(sample_config):
    tracker = TelemetryTracker(sample_config)
    summary = tracker.get_operation_summary("unknown_op")
    assert summary["total_calls"] == 0


# ---------------------------------------------------------------------------
# avg_latency calculation
# ---------------------------------------------------------------------------


def test_avg_latency_computed_correctly(sample_config):
    tracker = TelemetryTracker(sample_config)
    tracker.record(_make_record(pipeline="modern", latency_ms=200.0))
    tracker.record(_make_record(pipeline="modern", latency_ms=400.0))
    summary = tracker.get_pipeline_summary("modern")
    assert summary["avg_latency"] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Export JSON roundtrip
# ---------------------------------------------------------------------------


def test_export_json_creates_file(sample_config, tmp_path):
    tracker = TelemetryTracker(sample_config)
    tracker.record(_make_record())
    out_path = str(tmp_path / "telemetry.json")
    tracker.export_json(out_path)
    assert os.path.isfile(out_path)


def test_export_json_valid_structure(sample_config, tmp_path):
    tracker = TelemetryTracker(sample_config)
    tracker.record(_make_record(operation="generate_answer", pipeline="traditional"))
    out_path = str(tmp_path / "telemetry.json")
    tracker.export_json(out_path)

    with open(out_path, "r") as fh:
        data = json.load(fh)

    assert "exported_at" in data
    assert "total_records" in data
    assert "records" in data
    assert data["total_records"] == 1


def test_export_json_record_fields(sample_config, tmp_path):
    tracker = TelemetryTracker(sample_config)
    tracker.record(_make_record(operation="judge", pipeline="hybrid", input_tokens=300))
    out_path = str(tmp_path / "telemetry.json")
    tracker.export_json(out_path)

    with open(out_path, "r") as fh:
        data = json.load(fh)

    record = data["records"][0]
    assert record["operation"] == "judge"
    assert record["pipeline"] == "hybrid"
    assert record["input_tokens"] == 300
    assert "timestamp" in record


def test_export_json_multiple_records(sample_config, tmp_path):
    tracker = TelemetryTracker(sample_config)
    for i in range(5):
        tracker.record(_make_record(input_tokens=100 * (i + 1)))
    out_path = str(tmp_path / "telemetry.json")
    tracker.export_json(out_path)

    with open(out_path, "r") as fh:
        data = json.load(fh)

    assert data["total_records"] == 5
    assert len(data["records"]) == 5


def test_export_json_empty_tracker(sample_config, tmp_path):
    tracker = TelemetryTracker(sample_config)
    out_path = str(tmp_path / "telemetry_empty.json")
    tracker.export_json(out_path)

    with open(out_path, "r") as fh:
        data = json.load(fh)

    assert data["total_records"] == 0
    assert data["records"] == []


# ---------------------------------------------------------------------------
# get_all_summaries
# ---------------------------------------------------------------------------


def test_get_all_summaries_structure(sample_config):
    tracker = TelemetryTracker(sample_config)
    tracker.record(_make_record(operation="judge", pipeline="traditional"))
    summaries = tracker.get_all_summaries()
    assert "by_pipeline" in summaries
    assert "by_operation" in summaries
    assert "totals" in summaries


def test_get_all_summaries_totals(sample_config):
    tracker = TelemetryTracker(sample_config)
    tracker.record(_make_record(input_tokens=100, output_tokens=50, cost_usd=0.001))
    tracker.record(_make_record(input_tokens=200, output_tokens=80, cost_usd=0.002))
    summaries = tracker.get_all_summaries()
    totals = summaries["totals"]
    assert totals["total_calls"] == 2
    assert totals["total_input_tokens"] == 300
    assert totals["total_cost"] == pytest.approx(0.003)
