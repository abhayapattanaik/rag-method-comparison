"""Tests for src/cost_gate.py"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from src.cost_gate import CostEstimate, CostGate


# ---------------------------------------------------------------------------
# Blocks without approval (sys.exit(0))
# ---------------------------------------------------------------------------


def test_require_approval_exits_when_not_approved(sample_config):
    gate = CostGate(sample_config, approved=False)
    estimate = gate.estimate("test_op", num_items=10, avg_input_tokens=1000, avg_output_tokens=200)
    with pytest.raises(SystemExit) as exc_info:
        gate.require_approval(estimate)
    assert exc_info.value.code == 0


def test_require_approval_exits_even_for_low_cost(sample_config):
    gate = CostGate(sample_config, approved=False)
    estimate = gate.estimate("cheap_op", num_items=1, avg_input_tokens=1, avg_output_tokens=1)
    with pytest.raises(SystemExit):
        gate.require_approval(estimate)


# ---------------------------------------------------------------------------
# Passes with approval
# ---------------------------------------------------------------------------


def test_require_approval_passes_when_approved(sample_config):
    gate = CostGate(sample_config, approved=True)
    estimate = gate.estimate("test_op", num_items=10, avg_input_tokens=1000, avg_output_tokens=200)
    # Should NOT raise SystemExit
    gate.require_approval(estimate)  # no exception = pass


# ---------------------------------------------------------------------------
# Estimate calculation
# ---------------------------------------------------------------------------


def test_estimate_total_tokens(sample_config):
    gate = CostGate(sample_config, approved=True)
    estimate = gate.estimate(
        "contextualize",
        num_items=5,
        avg_input_tokens=1000,
        avg_output_tokens=100,
    )
    assert estimate.estimated_input_tokens == 5 * 1000
    assert estimate.estimated_output_tokens == 5 * 100
    assert estimate.estimated_calls == 5


def test_estimate_cost_by_model(sample_config):
    gate = CostGate(sample_config, approved=True)
    estimate = gate.estimate(
        "evaluate",
        num_items=10,
        avg_input_tokens=1000,
        avg_output_tokens=200,
    )
    # All known models should be present in cost_by_model
    assert len(estimate.cost_by_model) > 0
    for model_name, cost in estimate.cost_by_model.items():
        assert cost >= 0.0


def test_estimate_cost_greater_than_zero(sample_config):
    """Non-trivial token counts should produce non-zero cost for known models."""
    gate = CostGate(sample_config, approved=True)
    estimate = gate.estimate(
        "generate",
        num_items=100,
        avg_input_tokens=2000,
        avg_output_tokens=500,
    )
    # At least one model should have a cost > 0
    assert any(cost > 0 for cost in estimate.cost_by_model.values())


def test_estimate_operation_label(sample_config):
    gate = CostGate(sample_config, approved=True)
    estimate = gate.estimate("my_operation", num_items=1, avg_input_tokens=100, avg_output_tokens=50)
    assert estimate.operation == "my_operation"


def test_estimate_zero_items_produces_zero_cost(sample_config):
    gate = CostGate(sample_config, approved=True)
    estimate = gate.estimate("zero_op", num_items=0, avg_input_tokens=1000, avg_output_tokens=200)
    for cost in estimate.cost_by_model.values():
        assert cost == 0.0


# ---------------------------------------------------------------------------
# display_estimate produces output
# ---------------------------------------------------------------------------


def test_display_estimate_prints_output(sample_config, capsys):
    gate = CostGate(sample_config, approved=True)
    estimate = gate.estimate("display_test", num_items=5, avg_input_tokens=1000, avg_output_tokens=100)
    gate.display_estimate(estimate)
    captured = capsys.readouterr()
    assert "display_test" in captured.out
    assert "Cost Estimate" in captured.out


# ---------------------------------------------------------------------------
# CostEstimate dataclass
# ---------------------------------------------------------------------------


def test_cost_estimate_dataclass():
    estimate = CostEstimate(
        operation="test",
        estimated_calls=10,
        estimated_input_tokens=10000,
        estimated_output_tokens=2000,
        cost_by_model={"model_a": 0.05},
    )
    assert estimate.operation == "test"
    assert estimate.estimated_calls == 10
    assert estimate.cost_by_model["model_a"] == 0.05
