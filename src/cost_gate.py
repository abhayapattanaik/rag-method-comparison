"""
cost_gate.py -- Cost Estimation & Approval Gate

Estimates LLM costs before execution and blocks unless --approve flag is set.
Every CLI subcommand that makes LLM calls must instantiate CostGate and call
require_approval() before proceeding.
"""

import logging
import sys
from dataclasses import dataclass, field

from src.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class CostEstimate:
    operation: str                      # e.g. "contextualize", "evaluate"
    estimated_calls: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    cost_by_model: dict[str, float]     # model_name -> estimated USD


class CostGate:
    def __init__(self, config: AppConfig, approved: bool = False):
        self._config = config
        self._approved = approved

    def estimate(
        self,
        operation: str,
        num_items: int,
        avg_input_tokens: int,
        avg_output_tokens: int,
    ) -> CostEstimate:
        """Calculate estimated cost across all configured models in pricing config."""
        total_input_tokens = num_items * avg_input_tokens
        total_output_tokens = num_items * avg_output_tokens

        cost_by_model: dict[str, float] = {}
        for model_name, pricing in self._config.cost.pricing.items():
            input_cost = (total_input_tokens / 1000) * pricing["input_per_1k"]
            output_cost = (total_output_tokens / 1000) * pricing["output_per_1k"]
            cost_by_model[model_name] = round(input_cost + output_cost, 6)

        estimate = CostEstimate(
            operation=operation,
            estimated_calls=num_items,
            estimated_input_tokens=total_input_tokens,
            estimated_output_tokens=total_output_tokens,
            cost_by_model=cost_by_model,
        )
        logger.debug(
            "Cost estimate calculated: operation=%s calls=%d input_tokens=%d output_tokens=%d",
            operation, num_items, total_input_tokens, total_output_tokens,
        )
        return estimate

    def display_estimate(self, estimate: CostEstimate) -> None:
        """Print formatted cost table to stdout."""
        header = f"Cost Estimate: {estimate.operation} ({estimate.estimated_calls:,} calls)"
        separator = "=" * 63
        row_sep = "-" * 63

        print(f"\n{header}")
        print(separator)
        print(f"{'Model':<30} {'Input Tokens':>13} {'Output Tokens':>14} {'Est. Cost':>10}")
        print(row_sep)

        for model_name, cost in estimate.cost_by_model.items():
            print(
                f"{model_name:<30}"
                f" {estimate.estimated_input_tokens:>13,}"
                f" {estimate.estimated_output_tokens:>14,}"
                f" ${cost:>9.2f}"
            )

        print(separator)
        print()

    def require_approval(self, estimate: CostEstimate) -> None:
        """Display estimate and exit with code 0 if not approved.

        If approved=True, this is a no-op and execution proceeds.
        """
        if not self._approved:
            self.display_estimate(estimate)
            logger.info("Cost gate: approval required for operation=%s — exiting", estimate.operation)
            print("Run with --approve to proceed.")
            sys.exit(0)
        logger.info("Cost gate: approved for operation=%s", estimate.operation)

    def refine_estimate(
        self,
        operation: str,
        sample_records: list,
        total_items: int,
    ) -> CostEstimate:
        """Extrapolate cost from actual sample telemetry to total_items.

        sample_records: list of LLMCallRecord (or any object with
            input_tokens and output_tokens attributes) gathered from a
            small sample run.
        total_items: the full number of items to extrapolate to.
        """
        if not sample_records:
            raise ValueError("sample_records must not be empty")

        sample_size = len(sample_records)
        total_sample_input = sum(r.input_tokens for r in sample_records)
        total_sample_output = sum(r.output_tokens for r in sample_records)

        avg_input_tokens = total_sample_input // sample_size
        avg_output_tokens = total_sample_output // sample_size

        return self.estimate(
            operation=operation,
            num_items=total_items,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
        )
