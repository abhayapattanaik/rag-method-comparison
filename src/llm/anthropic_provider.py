"""Anthropic SDK LLM provider.

Uses the official `anthropic` Python SDK. API key is read from the environment
variable named in config.llm.anthropic_api_key_env (default: ANTHROPIC_API_KEY).

Token counts come directly from response.usage (exact).
Cost is calculated from config.cost.pricing.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

try:
    import anthropic as _anthropic_sdk
except ImportError as _anthropic_import_err:
    _anthropic_sdk = None  # type: ignore[assignment]
    _ANTHROPIC_IMPORT_ERROR = _anthropic_import_err
else:
    _ANTHROPIC_IMPORT_ERROR = None

from src.llm.base import BaseLLMProvider, LLMResponse

if TYPE_CHECKING:
    from src.config import AppConfig
    from src.telemetry import TelemetryTracker


class AnthropicProvider(BaseLLMProvider):
    """LLM provider backed by the Anthropic messages API."""

    def __init__(self, config: AppConfig, telemetry: TelemetryTracker) -> None:
        if _anthropic_sdk is None:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicProvider. "
                "Install it with: pip install anthropic"
            ) from _ANTHROPIC_IMPORT_ERROR

        super().__init__(config, telemetry)

        api_key_env = config.llm.anthropic_api_key_env
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Anthropic API key not found. "
                f"Set the '{api_key_env}' environment variable."
            )

        self.client: _anthropic_sdk.Anthropic = _anthropic_sdk.Anthropic(api_key=api_key)

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Call client.messages.create(), return LLMResponse with exact token counts.

        Args:
            messages:    Chat messages in OpenAI-style format. A message with
                         role "system" is extracted and passed as the Anthropic
                         `system` parameter; remaining messages are forwarded as-is.
            temperature: Sampling temperature (falls back to config default).
            max_tokens:  Max output tokens (falls back to config default).

        Returns:
            LLMResponse with text, token counts, latency, and calculated cost.
        """
        effective_temperature = temperature if temperature is not None else self.config.llm.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self.config.llm.max_tokens
        model = self.get_model_name()

        # Split system message from the rest (Anthropic API takes it separately).
        system_content: str | _anthropic_sdk.NotGiven = _anthropic_sdk.NOT_GIVEN
        chat_messages: list[dict] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg["content"]
            else:
                chat_messages.append(msg)

        logger.debug(
            "Anthropic API call start: model=%s max_tokens=%d temperature=%s",
            model, effective_max_tokens, effective_temperature,
        )
        start = time.perf_counter()
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=effective_max_tokens,
                temperature=effective_temperature,
                system=system_content,
                messages=chat_messages,  # type: ignore[arg-type]
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "Anthropic API call failed: model=%s latency_ms=%.1f error=%s",
                model, latency_ms, exc,
            )
            raise
        latency_ms = (time.perf_counter() - start) * 1000

        input_tokens: int = response.usage.input_tokens
        output_tokens: int = response.usage.output_tokens
        text: str = response.content[0].text if response.content else ""
        cost = self._calculate_cost(input_tokens, output_tokens, model)

        logger.info(
            "Anthropic API call complete: model=%s input_tokens=%d output_tokens=%d "
            "latency_ms=%.1f cost_usd=%.6f",
            model, input_tokens, output_tokens, round(latency_ms, 2), cost,
        )

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            latency_ms=round(latency_ms, 2),
            cost_usd=cost,
        )

    def get_model_name(self) -> str:
        return self.config.models.llm_model
