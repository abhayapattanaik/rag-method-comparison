"""Claude CLI subprocess provider.

Shells out to the `claude` command-line tool rather than calling an API
directly. Useful when an Anthropic API key is unavailable or when the
caller wants to reuse an existing `claude` CLI session.

Limitations (accepted per architecture decision):
  - Slower than direct API calls due to subprocess overhead.
  - System messages are prepended to the user prompt as plain text since the
    CLI does not have a dedicated --system flag in all versions.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from typing import TYPE_CHECKING

from src.llm.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.config import AppConfig
    from src.telemetry import TelemetryTracker

_CLI_MODEL_NAME = "claude-cli"
_CHARS_PER_TOKEN = 4  # rough estimate: 1 token ≈ 4 characters


def _estimate_tokens(text: str) -> int:
    """Estimate token count from character length."""
    return max(0, len(text) // _CHARS_PER_TOKEN)


class ClaudeCLIProvider(BaseLLMProvider):
    """LLM provider that shells out to the `claude` CLI tool via subprocess."""

    def __init__(self, config: AppConfig, telemetry: TelemetryTracker) -> None:
        super().__init__(config, telemetry)

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Run `claude` as a subprocess, capture stdout, return LLMResponse.

        The full conversation is serialised as plain text and passed via stdin.
        System messages are prefixed before the user messages.

        Uses --output-format json to obtain exact token counts, real cost, and
        CLI-measured latency from the structured response. Falls back to plain
        text parsing with estimated counts if JSON parsing fails.

        Args:
            messages:    Chat messages in OpenAI-style format.
            temperature: Ignored (claude CLI does not expose a --temperature flag;
                         accepted and silently dropped).
            max_tokens:  Ignored (claude CLI does not expose a --max-tokens flag;
                         accepted and silently dropped for interface compatibility).

        Returns:
            LLMResponse with exact token counts and actual cost from CLI output.

        Raises:
            RuntimeError: If the subprocess exits with a non-zero return code.
        """
        # Build the prompt text from all messages.
        system_parts: list[str] = []
        user_parts: list[str] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            else:
                # Include role label for multi-turn clarity.
                user_parts.append(f"{role.upper()}: {content}")

        prompt_lines: list[str] = []
        if system_parts:
            prompt_lines.append("SYSTEM:\n" + "\n".join(system_parts))
            prompt_lines.append("")
        prompt_lines.extend(user_parts)
        prompt_text = "\n".join(prompt_lines)

        # Build the CLI command.
        # --output-format json returns a structured envelope with exact token
        # counts, real cost, and CLI-measured duration.
        # Note: the claude CLI does not support --max-tokens; token limits are
        # managed internally by the CLI. The max_tokens parameter is accepted by
        # this method for interface compatibility but is intentionally not passed
        # to the subprocess.
        cmd: list[str] = ["claude", "--print", "--output-format", "json"]

        logger.debug("Claude CLI command: %s (prompt_chars=%d)", cmd, len(prompt_text))
        start = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                input=prompt_text,
                capture_output=True,
                text=True,
                timeout=self.config.concurrency.stall_timeout_seconds,
            )
        except FileNotFoundError as exc:
            logger.error("Claude CLI not found in PATH")
            raise RuntimeError(
                "The 'claude' CLI was not found in PATH. "
                "Install it from https://github.com/anthropics/anthropic-sdk-python "
                "or set a different provider in config.llm.provider."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "Claude CLI timed out after %.1fs (limit=%ds)",
                latency_ms / 1000, self.config.concurrency.stall_timeout_seconds,
            )
            raise RuntimeError(
                f"claude CLI call timed out after "
                f"{self.config.concurrency.stall_timeout_seconds}s."
            ) from exc

        wall_latency_ms = (time.perf_counter() - start) * 1000

        if result.returncode != 0:
            stderr_excerpt = (result.stderr or "").strip()[:500]
            logger.error(
                "Claude CLI exited with code %d latency_ms=%.1f stderr=%s",
                result.returncode, wall_latency_ms, stderr_excerpt,
            )
            raise RuntimeError(
                f"claude CLI exited with code {result.returncode}. "
                f"stderr: {stderr_excerpt}"
            )

        model = self.get_model_name()
        stdout_raw = result.stdout.strip()

        # Attempt to parse the JSON envelope returned by --output-format json.
        try:
            payload = json.loads(stdout_raw)
            response_text = payload["result"]

            usage = payload.get("usage", {})
            input_tokens = (
                usage.get("input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
            )
            output_tokens = usage.get("output_tokens", 0)

            cost = float(payload.get("total_cost_usd", 0.0))
            latency_ms = float(payload.get("duration_ms", wall_latency_ms))

            logger.info(
                "Claude CLI call complete (json): exit_code=0 latency_ms=%.1f "
                "input_tokens=%d output_tokens=%d cost_usd=%.6f",
                round(latency_ms, 2), input_tokens, output_tokens, cost,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            # Defensive fallback: treat stdout as plain text and estimate counts.
            logger.warning(
                "Claude CLI JSON parse failed (%s); falling back to plain-text mode",
                exc,
            )
            response_text = stdout_raw
            input_tokens = _estimate_tokens(prompt_text)
            output_tokens = _estimate_tokens(response_text)
            cost = self._calculate_cost(input_tokens, output_tokens, model)
            latency_ms = wall_latency_ms

            logger.info(
                "Claude CLI call complete (text fallback): exit_code=0 latency_ms=%.1f "
                "input_tokens=%d output_tokens=%d cost_usd=%.6f",
                round(latency_ms, 2), input_tokens, output_tokens, cost,
            )

        return LLMResponse(
            text=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            latency_ms=round(latency_ms, 2),
            cost_usd=cost,
        )

    def get_model_name(self) -> str:
        return _CLI_MODEL_NAME
