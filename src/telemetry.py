"""
src/telemetry.py -- Token usage, latency, and cost tracking per LLM call.

Accumulates LLMCallRecords in memory. Summaries computed on demand.
Exports full record list to JSON at end of each CLI run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.config import AppConfig


@dataclass
class LLMCallRecord:
    timestamp: datetime
    provider: str        # "anthropic" | "openai" | "claude_cli"
    model: str           # e.g. "claude-sonnet-4-20250514"
    operation: str       # "contextualize" | "generate_answer" | "judge" | "question_gen"
    pipeline: str | None # "traditional" | "contextual" | "hybrid" | "modern" | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    success: bool
    error: str | None = None


class TelemetryTracker:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._records: list[LLMCallRecord] = []

    # ------------------------------------------------------------------
    # Core record API
    # ------------------------------------------------------------------

    def record(self, call: LLMCallRecord) -> None:
        """Append a completed LLM call record."""
        self._records.append(call)
        logger.debug(
            "Telemetry record: provider=%s model=%s operation=%s pipeline=%s "
            "input_tokens=%d output_tokens=%d latency_ms=%.1f cost_usd=%.6f success=%s",
            call.provider, call.model, call.operation, call.pipeline,
            call.input_tokens, call.output_tokens, call.latency_ms, call.cost_usd, call.success,
        )

    # ------------------------------------------------------------------
    # Summary helpers (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise(records: list[LLMCallRecord]) -> dict:
        """Compute aggregated stats over an arbitrary list of records."""
        total_calls = len(records)
        if total_calls == 0:
            return {
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0,
                "avg_latency": 0.0,
            }
        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)
        total_cost = sum(r.cost_usd for r in records)
        avg_latency = sum(r.latency_ms for r in records) / total_calls
        return {
            "total_calls": total_calls,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost": round(total_cost, 6),
            "avg_latency": round(avg_latency, 2),
        }

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_pipeline_summary(self, pipeline: str) -> dict:
        """
        Return stats for all calls that served a specific pipeline.

        Keys: total_calls, total_input_tokens, total_output_tokens,
              total_cost, avg_latency
        """
        subset = [r for r in self._records if r.pipeline == pipeline]
        return self._summarise(subset)

    def get_operation_summary(self, operation: str) -> dict:
        """
        Return stats for all calls of a specific operation type.

        Keys: total_calls, total_input_tokens, total_output_tokens,
              total_cost, avg_latency
        """
        subset = [r for r in self._records if r.operation == operation]
        return self._summarise(subset)

    def get_all_summaries(self) -> dict:
        """
        Return summaries keyed by pipeline and by operation.

        Structure:
            {
                "by_pipeline": { pipeline_name: {...}, ... },
                "by_operation": { operation_name: {...}, ... },
                "totals": {...},
            }
        """
        pipelines = {r.pipeline for r in self._records}
        operations = {r.operation for r in self._records}

        summaries = {
            "by_pipeline": {
                p: self.get_pipeline_summary(p)
                for p in sorted(pipelines, key=lambda x: (x is None, x))
            },
            "by_operation": {
                op: self.get_operation_summary(op)
                for op in sorted(operations)
            },
            "totals": self._summarise(self._records),
        }
        totals = summaries["totals"]
        logger.info(
            "Telemetry summary: total_calls=%d total_cost_usd=%.6f avg_latency_ms=%.1f",
            totals["total_calls"], totals["total_cost"], totals["avg_latency"],
        )
        return summaries

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display_summary(self) -> None:
        """Print a formatted summary table to stdout."""
        summaries = self.get_all_summaries()

        header = f"{'Source':<20} {'Calls':>6} {'In Tokens':>12} {'Out Tokens':>12} {'Cost':>10} {'Avg Latency':>12}"
        sep = "-" * len(header)

        print()
        print("Pipeline Cost Summary")
        print("=" * len(header))
        print(header)
        print(sep)

        # Per-pipeline rows
        pipeline_rows = summaries["by_pipeline"]
        for label, stats in pipeline_rows.items():
            display_label = str(label) if label is not None else "(shared)"
            print(
                f"{display_label:<20} "
                f"{stats['total_calls']:>6,} "
                f"{stats['total_input_tokens']:>12,} "
                f"{stats['total_output_tokens']:>12,} "
                f"${stats['total_cost']:>9.2f} "
                f"{stats['avg_latency']:>10.0f}ms"
            )

        print(sep)

        # Per-operation rows
        operation_rows = summaries["by_operation"]
        for label, stats in operation_rows.items():
            print(
                f"{label:<20} "
                f"{stats['total_calls']:>6,} "
                f"{stats['total_input_tokens']:>12,} "
                f"{stats['total_output_tokens']:>12,} "
                f"${stats['total_cost']:>9.2f} "
                f"{stats['avg_latency']:>10.0f}ms"
            )

        print("=" * len(header))

        totals = summaries["totals"]
        print(
            f"{'Total':<20} "
            f"{totals['total_calls']:>6,} "
            f"{totals['total_input_tokens']:>12,} "
            f"{totals['total_output_tokens']:>12,} "
            f"${totals['total_cost']:>9.2f} "
            f"{totals['avg_latency']:>10.0f}ms"
        )
        print()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def export_json(self, path: str) -> None:
        """Persist all records to a JSON file at the given path."""
        def _serialise(record: LLMCallRecord) -> dict:
            d = asdict(record)
            # datetime is not JSON-serialisable by default
            d["timestamp"] = record.timestamp.isoformat()
            return d

        payload = {
            "exported_at": datetime.utcnow().isoformat(),
            "total_records": len(self._records),
            "records": [_serialise(r) for r in self._records],
        }

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
