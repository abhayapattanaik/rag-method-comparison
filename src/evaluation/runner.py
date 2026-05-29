"""Evaluation runner for the RAG Comparison System.

Runs all pipelines against all questions, collects judge scores, and persists
results incrementally. Each result is stored as an individual JSON file at:

    data/results/eval/{question_id}_{pipeline}_{metric}.json

On restart, completed result files are detected and skipped, making the
runner fully resumable after interruption.

Pipeline answers are cached in-memory per (question_id, pipeline) so each
pipeline runs at most once per question regardless of the number of metrics.

Typical usage:
    runner = EvaluationRunner(
        pipelines={"traditional": pipeline_obj, ...},
        judge=judge,
        config=config,
        telemetry=telemetry,
    )
    results = runner.run(questions)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import AppConfig
    from src.evaluation.judge import Judge, MetricType
    from src.evaluation.question_gen import EvalQuestion
    from src.retrieval.pipeline import Pipeline
    from src.telemetry import TelemetryTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EvalResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """A single scored result: one question x one pipeline x one metric."""

    question_id: str
    pipeline: str           # pipeline name string, e.g. "traditional"
    metric: str             # metric name string, e.g. "context_precision"
    score: float            # 0.0 to 1.0
    justification: str
    answer: str
    retrieved_chunks: list[str]   # chunk text strings
    cost_usd: float         # judge LLM call cost
    latency_ms: float       # judge LLM call latency
    input_tokens: int = 0           # judge LLM call prompt tokens
    output_tokens: int = 0          # judge LLM call completion tokens
    answer_input_tokens: int = 0    # pipeline answer LLM call prompt tokens
    answer_output_tokens: int = 0   # pipeline answer LLM call completion tokens


# ---------------------------------------------------------------------------
# EvaluationRunner
# ---------------------------------------------------------------------------


class EvaluationRunner:
    """Runs evaluation across all question x pipeline x metric combinations.

    Args:
        pipelines:  Dict mapping pipeline name strings to Pipeline instances.
        judge:      Judge instance for scoring.
        config:     AppConfig — uses config.paths.results_dir.
        telemetry:  Optional TelemetryTracker (unused directly here; judge
                    and pipeline handle their own telemetry recording).
    """

    def __init__(
        self,
        pipelines: "dict[str, Pipeline]",
        judge: "Judge",
        config: "AppConfig",
        telemetry: "TelemetryTracker | None" = None,
    ) -> None:
        self._pipelines = pipelines
        self._judge = judge
        self._config = config
        self._telemetry = telemetry

        self.results_dir = os.path.join(config.paths.results_dir, "eval")
        os.makedirs(self.results_dir, exist_ok=True)

        logger.info(
            "EvaluationRunner initialised: pipelines=%s results_dir=%s",
            list(pipelines.keys()),
            self.results_dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        questions: "list[EvalQuestion]",
        pipeline_filter: "list[str] | None" = None,
        question_filter: "list[str] | None" = None,
    ) -> "list[EvalResult]":
        """Run evaluation and return all EvalResult objects.

        Iterates over questions (filtered by question_filter) and pipelines
        (filtered by pipeline_filter). For each (question, pipeline) pair,
        runs the pipeline once to get an answer, then scores all 4 metrics.
        Each metric result is persisted immediately after scoring.

        Already-completed results (those with an existing JSON file on disk)
        are loaded from disk and skipped — the pipeline is not re-run for those.

        Args:
            questions:        All evaluation questions to run.
            pipeline_filter:  If provided, only run these pipeline names.
            question_filter:  If provided, only run questions with these IDs.

        Returns:
            List of EvalResult objects (both newly scored and loaded from cache).
        """
        # Determine the active pipelines and questions after filtering.
        active_pipelines = self._apply_pipeline_filter(pipeline_filter)
        active_questions = self._apply_question_filter(questions, question_filter)

        n_q = len(active_questions)
        n_p = len(active_pipelines)

        # Determine metrics from config.
        metrics = self._config.evaluation.metrics

        logger.info(
            "EvaluationRunner.run: %d questions x %d pipelines x %d metrics = %d total results",
            n_q, n_p, len(metrics), n_q * n_p * len(metrics),
        )

        all_results: list[EvalResult] = []

        for q_idx, question in enumerate(active_questions, start=1):
            for pipeline_name in active_pipelines:
                # Cache pipeline answer per (question_id, pipeline_name) so we
                # only call the pipeline once regardless of metric count.
                answer_cache: dict[tuple[str, str], tuple[str, list[str], float, int, int]] = {}

                for metric_name in metrics:
                    # Check if result already persisted on disk.
                    if self._is_completed(question.question_id, pipeline_name, metric_name):
                        logger.info(
                            "Evaluating question %d/%d, pipeline %s, metric %s — CACHED (skipping)",
                            q_idx, n_q, pipeline_name, metric_name,
                        )
                        existing = self._load_result(question.question_id, pipeline_name, metric_name)
                        if existing is not None:
                            all_results.append(existing)
                        continue

                    logger.info(
                        "Evaluating question %d/%d, pipeline %s, metric %s",
                        q_idx, n_q, pipeline_name, metric_name,
                    )

                    # Get (or compute and cache) pipeline answer for this (question, pipeline).
                    cache_key = (question.question_id, pipeline_name)
                    if cache_key not in answer_cache:
                        answer, retrieved_chunks, answer_latency_ms, ans_in_tok, ans_out_tok = self._run_pipeline(
                            pipeline_name=pipeline_name,
                            question=question,
                        )
                        answer_cache[cache_key] = (answer, retrieved_chunks, answer_latency_ms, ans_in_tok, ans_out_tok)
                    else:
                        logger.debug(
                            "run: reusing cached answer for question=%s pipeline=%s",
                            question.question_id, pipeline_name,
                        )

                    answer, retrieved_chunks, _, answer_input_tokens, answer_output_tokens = answer_cache[cache_key]

                    # Score the metric.
                    result = self._score_metric(
                        question=question,
                        pipeline_name=pipeline_name,
                        metric_name=metric_name,
                        answer=answer,
                        retrieved_chunks=retrieved_chunks,
                        answer_input_tokens=answer_input_tokens,
                        answer_output_tokens=answer_output_tokens,
                    )

                    # Persist immediately.
                    self._save_result(result)
                    all_results.append(result)
                    logger.debug(
                        "run: persisted result question=%s pipeline=%s metric=%s score=%.3f",
                        question.question_id, pipeline_name, metric_name, result.score,
                    )

        logger.info(
            "EvaluationRunner.run: complete — %d total results collected",
            len(all_results),
        )
        return all_results

    def load_all_results(self) -> "list[EvalResult]":
        """Load all persisted EvalResult JSON files from results_dir.

        Returns:
            List of EvalResult objects sorted by (question_id, pipeline, metric).
            Returns [] if results_dir does not exist or is empty.
        """
        if not os.path.isdir(self.results_dir):
            logger.info("load_all_results: results_dir does not exist — returning []")
            return []

        results: list[EvalResult] = []
        for fname in sorted(os.listdir(self.results_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(self.results_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                # Back-compat: fill in token fields absent from older result files.
                data.setdefault("input_tokens", 0)
                data.setdefault("output_tokens", 0)
                data.setdefault("answer_input_tokens", 0)
                data.setdefault("answer_output_tokens", 0)
                results.append(EvalResult(**data))
            except Exception as exc:
                logger.warning("load_all_results: failed to load %s: %s", fpath, exc)

        logger.info("load_all_results: loaded %d results from %s", len(results), self.results_dir)
        return results

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _result_path(self, question_id: str, pipeline: str, metric: str) -> str:
        """Return the deterministic file path for a (question, pipeline, metric) result.

        Format: {results_dir}/{question_id}_{pipeline}_{metric}.json
        """
        fname = f"{question_id}_{pipeline}_{metric}.json"
        return os.path.join(self.results_dir, fname)

    def _is_completed(self, question_id: str, pipeline: str, metric: str) -> bool:
        """Return True if a valid result file exists for this (question, pipeline, metric).

        A result is considered incomplete (returns False) if:
        - The file does not exist.
        - The file cannot be parsed as JSON.
        - The justification field contains "LLM call failed" or "failed", indicating
          the result was persisted from a failed LLM call and must be retried.

        Note: score=0.0 alone is NOT grounds for rejection — a genuinely poor score
        is a valid result. Only the justification text is checked.
        """
        path = self._result_path(question_id, pipeline, metric)
        if not os.path.isfile(path):
            return False

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.warning(
                "_is_completed: could not parse %s as JSON — treating as incomplete: %s",
                path, exc,
            )
            return False

        justification = data.get("justification", "")
        if "LLM call failed" in justification or "failed" in justification.lower():
            logger.info(
                "_is_completed: result at %s has failure justification %r — treating as incomplete",
                path, justification,
            )
            return False

        return True

    def _save_result(self, result: EvalResult) -> None:
        """Persist an EvalResult to its deterministic JSON file.

        Uses atomic write (write to temp file in same directory, then rename)
        to prevent partial writes on crash.
        """
        path = self._result_path(result.question_id, result.pipeline, result.metric)
        data = asdict(result)
        dir_path = os.path.dirname(path)

        try:
            # Write to a temp file in the same directory, then rename atomically.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=dir_path,
                suffix=".tmp",
                delete=False,
            ) as tmp_fh:
                json.dump(data, tmp_fh, indent=2)
                tmp_path = tmp_fh.name

            os.replace(tmp_path, path)
            logger.debug("_save_result: wrote %s", path)
        except Exception as exc:
            logger.error("_save_result: failed to write %s: %s", path, exc)
            # Clean up orphaned temp file if possible.
            try:
                if "tmp_path" in dir() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
            raise

    def _load_result(
        self, question_id: str, pipeline: str, metric: str
    ) -> "EvalResult | None":
        """Load an EvalResult from disk. Returns None if file is missing or invalid.

        Handles old result files that pre-date token fields by defaulting missing
        fields to 0 so existing results remain loadable.
        """
        path = self._result_path(question_id, pipeline, metric)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Back-compat: fill in token fields absent from older result files.
            data.setdefault("input_tokens", 0)
            data.setdefault("output_tokens", 0)
            data.setdefault("answer_input_tokens", 0)
            data.setdefault("answer_output_tokens", 0)
            return EvalResult(**data)
        except Exception as exc:
            logger.warning("_load_result: failed to load %s: %s", path, exc)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_pipeline_filter(self, pipeline_filter: "list[str] | None") -> "list[str]":
        """Return the list of pipeline names to evaluate, respecting the filter."""
        all_names = list(self._pipelines.keys())
        if pipeline_filter is None:
            return all_names

        filtered = [name for name in all_names if name in pipeline_filter]
        excluded = set(all_names) - set(filtered)
        if excluded:
            logger.info("_apply_pipeline_filter: excluding pipelines %s", sorted(excluded))
        logger.info("_apply_pipeline_filter: active pipelines = %s", filtered)
        return filtered

    def _apply_question_filter(
        self,
        questions: "list[EvalQuestion]",
        question_filter: "list[str] | None",
    ) -> "list[EvalQuestion]":
        """Return the list of questions to evaluate, respecting the filter."""
        if question_filter is None:
            return questions

        filtered = [q for q in questions if q.question_id in question_filter]
        excluded_count = len(questions) - len(filtered)
        if excluded_count:
            logger.info(
                "_apply_question_filter: excluded %d of %d questions",
                excluded_count, len(questions),
            )
        logger.info(
            "_apply_question_filter: active questions = %d", len(filtered)
        )
        return filtered

    def _run_pipeline(
        self,
        pipeline_name: str,
        question: "EvalQuestion",
    ) -> "tuple[str, list[str], float, int, int]":
        """Run a single pipeline for a question and return
        (answer, chunk_texts, latency_ms, input_tokens, output_tokens).

        On failure, returns ("", [], 0.0, 0, 0) and logs an error so the caller
        can still attempt judge scoring with empty context.
        """
        pipeline = self._pipelines[pipeline_name]
        logger.info(
            "_run_pipeline: running pipeline=%s question=%s",
            pipeline_name, question.question_id,
        )

        try:
            from src.retrieval.pipeline import PipelineMethod
            method = PipelineMethod(pipeline_name)
            result = pipeline.run(question.question, method)
            answer = result.answer
            # Extract plain text from each retrieved chunk dict.
            chunk_texts: list[str] = [
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in result.retrieved_chunks
            ]
            latency_ms = result.latency_ms
            answer_input_tokens = result.input_tokens
            answer_output_tokens = result.output_tokens
            logger.info(
                "_run_pipeline: pipeline=%s question=%s answer_len=%d chunks=%d "
                "latency_ms=%.1f input_tokens=%d output_tokens=%d",
                pipeline_name, question.question_id,
                len(answer), len(chunk_texts), latency_ms,
                answer_input_tokens, answer_output_tokens,
            )
            return answer, chunk_texts, latency_ms, answer_input_tokens, answer_output_tokens
        except Exception as exc:
            logger.error(
                "_run_pipeline: pipeline=%s question=%s failed: %s",
                pipeline_name, question.question_id, exc,
            )
            return "", [], 0.0, 0, 0

    def _score_metric(
        self,
        question: "EvalQuestion",
        pipeline_name: str,
        metric_name: str,
        answer: str,
        retrieved_chunks: "list[str]",
        answer_input_tokens: int = 0,
        answer_output_tokens: int = 0,
    ) -> EvalResult:
        """Score a single (question, pipeline, metric) tuple via the judge.

        On judge failure, the EvalResult will have score=0.0 and a descriptive
        justification containing the error. This prevents a single LLM failure
        from aborting the entire evaluation run.
        """
        from src.evaluation.judge import MetricType

        try:
            metric = MetricType(metric_name)
        except ValueError:
            logger.error(
                "_score_metric: unknown metric name=%s — returning score=0.0",
                metric_name,
            )
            return EvalResult(
                question_id=question.question_id,
                pipeline=pipeline_name,
                metric=metric_name,
                score=0.0,
                justification=f"Unknown metric: {metric_name}",
                answer=answer,
                retrieved_chunks=retrieved_chunks,
                cost_usd=0.0,
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                answer_input_tokens=answer_input_tokens,
                answer_output_tokens=answer_output_tokens,
            )

        try:
            judge_result = self._judge.score(
                metric=metric,
                question=question.question,
                answer=answer,
                contexts=retrieved_chunks,
                ground_truth=question.ground_truth,
            )
            return EvalResult(
                question_id=question.question_id,
                pipeline=pipeline_name,
                metric=metric_name,
                score=judge_result.score,
                justification=judge_result.justification,
                answer=answer,
                retrieved_chunks=retrieved_chunks,
                cost_usd=judge_result.llm_response.cost_usd,
                latency_ms=judge_result.llm_response.latency_ms,
                input_tokens=judge_result.input_tokens,
                output_tokens=judge_result.output_tokens,
                answer_input_tokens=answer_input_tokens,
                answer_output_tokens=answer_output_tokens,
            )
        except Exception as exc:
            logger.error(
                "_score_metric: judge failed for question=%s pipeline=%s metric=%s: %s",
                question.question_id, pipeline_name, metric_name, exc,
            )
            return EvalResult(
                question_id=question.question_id,
                pipeline=pipeline_name,
                metric=metric_name,
                score=0.0,
                justification=f"Judge call failed: {exc}",
                answer=answer,
                retrieved_chunks=retrieved_chunks,
                cost_usd=0.0,
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                answer_input_tokens=answer_input_tokens,
                answer_output_tokens=answer_output_tokens,
            )
