"""Abstract LLM provider interface and shared types.

Defines:
    LLMResponse       -- dataclass returned by every provider.complete() call
    BaseLLMProvider   -- ABC that all concrete providers implement
    get_provider      -- factory: selects provider from config.llm.provider
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.config import AppConfig
    from src.telemetry import TelemetryTracker


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Structured result from a single LLM call."""

    text: str           # Generated text
    input_tokens: int   # Prompt tokens consumed
    output_tokens: int  # Completion tokens generated
    model: str          # Model identifier, e.g. "claude-sonnet-4-20250514"
    latency_ms: float   # Wall-clock time for the call (milliseconds)
    cost_usd: float     # Calculated cost based on pricing table


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class BaseLLMProvider(ABC):
    """Common interface implemented by all LLM providers."""

    def __init__(self, config: AppConfig, telemetry: TelemetryTracker) -> None:
        self.config = config
        self.telemetry = telemetry
        logger.info("Provider instantiated: %s model=%s", self.__class__.__name__, config.models.llm_model)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a list of chat messages to the LLM, return a structured response.

        Args:
            messages:    OpenAI-style message list, e.g.
                         [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            temperature: Sampling temperature override (uses config default when None).
            max_tokens:  Max completion tokens override (uses config default when None).

        Returns:
            LLMResponse with text, token counts, model, latency, and cost.
        """
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier string (e.g. 'claude-sonnet-4-20250514')."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Look up per-token pricing from config and compute the USD cost.

        Falls back to 0.0 if the model is not present in the pricing table
        (e.g. for claude_cli where token counts are estimates).
        """
        pricing = self.config.cost.pricing
        if model not in pricing:
            return 0.0
        rates = pricing[model]
        input_cost = (input_tokens / 1000) * rates.get("input_per_1k", 0.0)
        output_cost = (output_tokens / 1000) * rates.get("output_per_1k", 0.0)
        return round(input_cost + output_cost, 8)

    def _record_call(
        self,
        response: LLMResponse,
        operation: str,
        pipeline: str | None,
    ) -> None:
        """Record a completed LLM call to telemetry.

        Args:
            response:  The LLMResponse to record.
            operation: Logical operation label, e.g. "generate_answer", "judge".
            pipeline:  Pipeline name if applicable, else None.
        """
        from src.telemetry import LLMCallRecord  # local import to avoid circular dep

        record = LLMCallRecord(
            timestamp=datetime.utcnow(),
            provider=self.__class__.__name__.replace("Provider", "").lower(),
            model=response.model,
            operation=operation,
            pipeline=pipeline,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            cost_usd=response.cost_usd,
            success=True,
            error=None,
        )
        self.telemetry.record(record)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_provider(config: AppConfig, telemetry: TelemetryTracker) -> BaseLLMProvider:
    """Instantiate and return the LLM provider specified in config.llm.provider.

    Supported values: "anthropic", "openai", "claude_cli".

    Raises:
        ValueError: If config.llm.provider is not a recognised value.
    """
    # Deferred imports so that optional SDKs are only loaded when selected.
    match config.llm.provider:
        case "anthropic":
            from src.llm.anthropic_provider import AnthropicProvider
            return AnthropicProvider(config, telemetry)
        case "openai":
            from src.llm.openai_provider import OpenAIProvider
            return OpenAIProvider(config, telemetry)
        case "claude_cli":
            from src.llm.claude_cli_provider import ClaudeCLIProvider
            return ClaudeCLIProvider(config, telemetry)
        case _:
            raise ValueError(
                f"Unknown LLM provider: '{config.llm.provider}'. "
                "Expected one of: 'anthropic', 'openai', 'claude_cli'."
            )
