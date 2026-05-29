"""Interactive REPL for the RAG Comparison System.

Provides a terminal read-eval-print loop that allows users to ask questions
against one or more pipeline methods and inspect retrieved chunks, generated
answers, per-pipeline latency, and token usage.

Usage (from Pipeline layer):
    from src.interactive.repl import run_repl
    run_repl(config, [PipelineMethod.TRADITIONAL])
"""

from __future__ import annotations

import logging
import sys
import time
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.config import AppConfig
    from src.retrieval.pipeline import PipelineMethod

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WIDE_SEP  = "=" * 72
_NARROW_SEP = "-" * 72
_CHUNK_PREVIEW_CHARS = 300   # max chars to show per chunk in REPL output
_MAX_CHUNKS_DISPLAY  = 5     # max retrieved chunks to print per pipeline

_SPECIAL_COMMANDS: dict[str, str] = {
    "help":      "Show this help message.",
    "pipelines": "List active pipeline methods.",
    "history":   "Show question history for this session.",
    "quit":      "Exit the REPL.",
    "exit":      "Exit the REPL.",
}

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_repl(
    config: "AppConfig",
    pipelines: "list[PipelineMethod]",
) -> None:
    """Start the interactive Q&A REPL loop.

    Prompts the user for a question, runs each pipeline in *pipelines* against
    it, and prints structured results (chunks + scores, answer, latency,
    token counts). The loop continues until the user types "quit" / "exit", or
    sends EOF (Ctrl-D).

    Args:
        config:    AppConfig — passed through to Pipeline.
        pipelines: Ordered list of PipelineMethod values to run per question.
    """
    from src.retrieval.pipeline import Pipeline

    pipeline_obj = Pipeline(config=config)
    pipeline_names = [m.value for m in pipelines]
    logger.info("REPL session start: pipelines=%s", pipeline_names)

    question_history: list[str] = []

    _print_banner(pipelines)

    while True:
        # ---- prompt -------------------------------------------------------
        try:
            raw = input("\nQuestion> ").strip()
        except EOFError:
            logger.info("REPL session end: EOF received")
            print("\nEOF received. Exiting.")
            break
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'quit' to exit.")
            continue

        if not raw:
            continue

        lower = raw.lower()

        # ---- special commands ---------------------------------------------
        if lower in ("quit", "exit"):
            logger.info("REPL session end: user exited")
            print("Goodbye.")
            break

        if lower == "help":
            _print_help()
            continue

        if lower == "pipelines":
            _print_pipelines(pipelines)
            continue

        if lower == "history":
            _print_history(question_history)
            continue

        # ---- run each pipeline --------------------------------------------
        query = raw
        question_history.append(query)
        logger.info("REPL query received: %r", query[:120])
        print()

        for method in pipelines:
            _print_pipeline_header(method)

            try:
                result = pipeline_obj.run(query, method)
            except NotImplementedError as exc:
                logger.warning("Pipeline %s not implemented: %s", method.value, exc)
                print(f"[NOT IMPLEMENTED] {exc}")
                print(_NARROW_SEP)
                continue
            except RuntimeError as exc:
                # Covers Claude CLI timeout, non-zero exit, CLI not found.
                logger.error("Pipeline %s runtime error: %s", method.value, exc)
                print(f"[ERROR] {exc}")
                print(_NARROW_SEP)
                continue
            except Exception as exc:  # noqa: BLE001
                logger.exception("Pipeline %s unexpected error", method.value)
                print(f"[UNEXPECTED ERROR] {type(exc).__name__}: {exc}")
                print(_NARROW_SEP)
                continue

            _print_result(result)

        print(_WIDE_SEP)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_banner(pipelines: "list[PipelineMethod]") -> None:
    """Print the REPL welcome banner."""
    print(_WIDE_SEP)
    print("  RAG Comparison -- Interactive Q&A")
    print(_WIDE_SEP)
    _print_pipelines(pipelines)
    print()
    print("Type a question to query all active pipelines.")
    print("Special commands: " + "  |  ".join(_SPECIAL_COMMANDS.keys()))
    print(_WIDE_SEP)


def _print_pipelines(pipelines: "list[PipelineMethod]") -> None:
    """Print the list of active pipeline methods."""
    print("Active pipelines:")
    for i, m in enumerate(pipelines, start=1):
        print(f"  {i}. {m.value}")


def _print_help() -> None:
    """Print available REPL commands."""
    print()
    print(_NARROW_SEP)
    print("  Available commands")
    print(_NARROW_SEP)
    for cmd, desc in _SPECIAL_COMMANDS.items():
        print(f"  {cmd:<12}  {desc}")
    print(_NARROW_SEP)
    print("  Any other input is treated as a question and run through all")
    print("  active pipelines.")
    print(_NARROW_SEP)


def _print_history(history: list[str]) -> None:
    """Print question history for this session."""
    print()
    if not history:
        print("No questions asked yet this session.")
        return
    print(_NARROW_SEP)
    print(f"  Question history ({len(history)} question(s))")
    print(_NARROW_SEP)
    for i, q in enumerate(history, start=1):
        # Truncate very long questions for display.
        preview = q[:120] + ("..." if len(q) > 120 else "")
        print(f"  [{i:2d}] {preview}")
    print(_NARROW_SEP)


def _print_pipeline_header(method: "PipelineMethod") -> None:
    """Print the bold section header for a single pipeline result."""
    label = method.value.upper()
    print(_WIDE_SEP)
    print(f"  Pipeline: {label}")
    print(_WIDE_SEP)


def _print_result(result) -> None:  # result: RetrievalResult
    """Render a RetrievalResult to stdout in a human-readable format.

    Sections printed (in order):
        1. Retrieved chunks (top _MAX_CHUNKS_DISPLAY, with score/source/pages)
        2. Sources list
        3. Answer text (indented)
        4. Latency + token counts
    """
    # ---- Retrieved chunks -------------------------------------------------
    all_chunks = result.retrieved_chunks
    display_chunks = all_chunks[:_MAX_CHUNKS_DISPLAY]

    print(
        f"Retrieved chunks: showing {len(display_chunks)} of {len(all_chunks)}"
        f" (top {_MAX_CHUNKS_DISPLAY})"
    )
    print(_NARROW_SEP)

    for i, chunk in enumerate(display_chunks, start=1):
        score      = chunk.get("score", 0.0)
        source     = chunk.get("source_file", "unknown")
        page_start = chunk.get("page_start", "?")
        page_end   = chunk.get("page_end", "?")
        text       = chunk.get("text", "").strip()

        preview  = text[:_CHUNK_PREVIEW_CHARS]
        ellipsis = "..." if len(text) > _CHUNK_PREVIEW_CHARS else ""

        print(f"  [{i}] score={score:.4f}  source={source}  pages={page_start}-{page_end}")
        print(f"       {preview}{ellipsis}")
        if i < len(display_chunks):
            print()

    # ---- Sources ----------------------------------------------------------
    print(_NARROW_SEP)
    sources = result.sources
    if sources:
        source_strs = [f"{s['doc']} p.{s['page']}" for s in sources]
        print("Sources: " + " | ".join(source_strs))
    else:
        print("Sources: (none)")

    # ---- Answer -----------------------------------------------------------
    print(_NARROW_SEP)
    print("Answer:")
    print()
    # Indent each answer line for visual separation.
    for line in result.answer.splitlines():
        print(f"  {line}")
    print()

    # ---- Telemetry: latency + token counts --------------------------------
    print(_NARROW_SEP)
    print(f"Latency : {result.latency_ms:.0f} ms")

    # Extract token counts from the most recent telemetry record for this
    # pipeline, if available.
    tok_in, tok_out = _extract_tokens(result)
    if tok_in is not None:
        print(f"Tokens  : input={tok_in:,}  output={tok_out:,}  total={tok_in + tok_out:,}")
    else:
        print("Tokens  : (not available)")

    print(_NARROW_SEP)


# ---------------------------------------------------------------------------
# Telemetry helper
# ---------------------------------------------------------------------------


def _extract_tokens(result) -> "tuple[int | None, int | None]":
    """Return (input_tokens, output_tokens) from the result's telemetry records.

    Looks for the last 'generate_answer' record belonging to this pipeline.
    Falls back to the last record of any operation if none is found.
    Returns (None, None) when no telemetry is present.
    """
    records = getattr(result, "telemetry_records", None)
    if not records:
        return None, None

    pipeline_val = result.method.value if hasattr(result.method, "value") else str(result.method)

    # Prefer a generate_answer record for this pipeline.
    for rec in reversed(records):
        op       = getattr(rec, "operation", "")
        pipeline = getattr(rec, "pipeline", None)
        if op == "generate_answer" and pipeline == pipeline_val:
            return rec.input_tokens, rec.output_tokens

    # Fallback: last record that belongs to this pipeline.
    for rec in reversed(records):
        pipeline = getattr(rec, "pipeline", None)
        if pipeline == pipeline_val:
            return rec.input_tokens, rec.output_tokens

    # Last resort: most recent record of any kind.
    last = records[-1]
    tok_in  = getattr(last, "input_tokens", None)
    tok_out = getattr(last, "output_tokens", None)
    if tok_in is not None:
        return tok_in, tok_out

    return None, None
