"""OpenAI SDK LLM provider.

Uses the official `openai` Python SDK. API key is read from the environment
variable named in config.llm.openai_api_key_env (default: OPENAI_API_KEY).

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
    import openai as _openai_sdk
except ImportError as _openai_import_err:
    _openai_sdk = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = _openai_import_err
else:
    _OPENAI_IMPORT_ERROR = None

from src.llm.base import BaseLLMProvider, LLMResponse

if TYPE_CHECKING:
    from src.config import AppConfig
    from src.telemetry import TelemetryTracker


class OpenAIProvider(BaseLLMProvider):
    """LLM provider backed by the OpenAI chat completions API."""

    def __init__(self, config: AppConfig, telemetry: TelemetryTracker) -> None:
        if _openai_sdk is None:
            raise ImportError(
                "The 'openai' package is required for OpenAIProvider. "
                "Install it with: pip install openai"
            ) from _OPENAI_IMPORT_ERROR

        super().__init__(config, telemetry)

        api_key_env = config.llm.openai_api_key_env
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"OpenAI API key not found. "
                f"Set the '{api_key_env}' environment variable."
            )

        self.client: _openai_sdk.OpenAI = _openai_sdk.OpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Call client.chat.completions.create(), return LLMResponse with exact token counts.

        Args:
            messages:    Chat messages in OpenAI-style format. System messages
                         are passed through in messages[0] with role "system"
                         as the OpenAI API natively supports this.
            temperature: Sampling temperature (falls back to config default).
            max_tokens:  Max output tokens (falls back to config default).

        Returns:
            LLMResponse with text, token counts, latency, and calculated cost.
        """
        effective_temperature = temperature if temperature is not None else self.config.llm.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self.config.llm.max_tokens
        model = self.get_model_name()

        logger.debug(
            "OpenAI API call start: model=%s max_tokens=%d temperature=%s",
            model, effective_max_tokens, effective_temperature,
        )
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "OpenAI API call failed: model=%s latency_ms=%.1f error=%s",
                model, latency_ms, exc,
            )
            raise
        latency_ms = (time.perf_counter() - start) * 1000

        usage = response.usage
        input_tokens: int = usage.prompt_tokens if usage else 0
        output_tokens: int = usage.completion_tokens if usage else 0

        choice = response.choices[0] if response.choices else None
        text: str = choice.message.content or "" if choice and choice.message else ""

        cost = self._calculate_cost(input_tokens, output_tokens, model)

        logger.info(
            "OpenAI API call complete: model=%s input_tokens=%d output_tokens=%d "
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
