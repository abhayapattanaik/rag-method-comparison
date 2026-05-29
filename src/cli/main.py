"""CLI entry point for the RAG Comparison System.

Subcommands:
  ingest              Ingest PDFs into traditional ChromaDB collection
  contextualize       LLM-contextualize chunks into contextualized collection
  generate-questions  Generate evaluation questions from papers
  evaluate            Run evaluation across pipelines
  interactive         Start interactive Q&A REPL
  compare             Generate comparison report from evaluation results

Run as: python -m src.cli.main <subcommand> [OPTIONS]
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from src.config import load_config, AppConfig
from src.cost_gate import CostGate
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cli_overrides_from(args: argparse.Namespace) -> dict[str, Any]:
    """Build a dot-notation override dict from parsed CLI args."""
    overrides: dict[str, Any] = {}
    if getattr(args, "provider", None) is not None:
        overrides["llm.provider"] = args.provider
    if getattr(args, "model", None) is not None:
        overrides["models.llm_model"] = args.model
    return overrides


# ---------------------------------------------------------------------------
# Subcommand handlers (stubs)
# ---------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> None:
    """Stub: ingest PDFs into the traditional ChromaDB collection."""
    overrides = _cli_overrides_from(args)
    if getattr(args, "papers_dir", None) is not None:
        overrides["paths.papers_dir"] = args.papers_dir

    config: AppConfig = load_config(args.config, overrides)
    cost_gate = CostGate(config, approved=args.approve)

    print("Running ingest...")
    print(f"  papers_dir : {config.paths.papers_dir}")
    print(f"  chroma_dir : {config.paths.chroma_dir}")

    # Placeholder cost estimate
    estimate = cost_gate.estimate(
        operation="ingest",
        num_items=0,
        avg_input_tokens=0,
        avg_output_tokens=0,
    )
    cost_gate.display_estimate(estimate)
    cost_gate.require_approval(estimate)

    print("[stub] ingest implementation pending")


def cmd_contextualize(args: argparse.Namespace) -> None:
    """LLM-contextualize chunks: pass 1 (doc summaries) + pass 2 (batch contextualization)."""
    import json
    import math
    import os
    import time
    from dataclasses import asdict

    from src.ingestion.contextualizer import (
        summarize_all_documents,
        contextualize_all_chunks,
    )
    from src.llm.base import get_provider
    from src.telemetry import TelemetryTracker

    overrides = _cli_overrides_from(args)
    config: AppConfig = load_config(args.config, overrides)

    # ---- Load chunks from cache ------------------------------------------
    chunks_cache_path = os.path.join(config.paths.cache_dir, "chunks.json")
    if not os.path.exists(chunks_cache_path):
        print(
            f"chunks.json not found at {chunks_cache_path}\n"
            "Run 'ingest' first to generate chunks.",
            file=sys.stderr,
        )
        sys.exit(1)

    from src.ingestion.chunker import Chunk

    with open(chunks_cache_path, "r", encoding="utf-8") as fh:
        raw_chunks = json.load(fh)

    all_chunks: list[Chunk] = [Chunk(**c) for c in raw_chunks]
    logger.info("cmd_contextualize: loaded %d chunks from %s", len(all_chunks), chunks_cache_path)

    # ---- Apply --sample limit --------------------------------------------
    sample_n: int | None = args.sample
    if sample_n is not None:
        working_chunks = all_chunks[:sample_n]
        print(f"Sample mode: processing first {sample_n} of {len(all_chunks)} chunks.")
    else:
        working_chunks = all_chunks

    # ---- Cost estimate ---------------------------------------------------
    # Discover unique doc count from working_chunks
    unique_docs = len({c.doc_id for c in working_chunks})
    total_chunks = len(working_chunks)
    num_batches = math.ceil(total_chunks / 5)

    est = config.cost_estimation

    # Pass 1: 1 call per document
    # Input: avg_doc_tokens per doc; Output: ~1500 tokens per summary
    pass1_input = unique_docs * est.avg_doc_tokens
    pass1_output = unique_docs * 1500

    # Pass 2: ceil(total_chunks/5) calls
    # Input per call: 1500 (summary) + 900 (prev+next) + 5*avg_chunk (batch) + 300 (prompt overhead)
    pass2_input_per_call = 1500 + 900 + (5 * est.avg_chunk_tokens) + 300
    pass2_input = num_batches * pass2_input_per_call
    pass2_output = num_batches * 500  # ~100 tokens per prefix x 5 chunks

    total_input = pass1_input + pass2_input
    total_output = pass1_output + pass2_output
    total_calls = unique_docs + num_batches

    cost_gate = CostGate(config, approved=args.approve)

    # Build a combined estimate object
    estimate = cost_gate.estimate(
        operation="contextualize",
        num_items=total_calls,
        avg_input_tokens=total_input // max(total_calls, 1),
        avg_output_tokens=total_output // max(total_calls, 1),
    )

    print()
    print("Contextualization cost estimate")
    print("=" * 60)
    print(f"  Documents to summarize (pass 1) : {unique_docs}")
    print(f"  Chunks to contextualize (pass 2): {total_chunks}")
    print(f"  Batches (batch_size=5)           : {num_batches}")
    print(f"  Total LLM calls                  : {total_calls}")
    print(f"  Est. input tokens                : {total_input:,}")
    print(f"  Est. output tokens               : {total_output:,}")
    cost_gate.display_estimate(estimate)
    cost_gate.require_approval(estimate)  # exits if not --approve

    # ---- Locate extracted .md directory ----------------------------------
    extracted_dir = os.path.join(config.paths.data_dir, "papers", "extracted")
    if not os.path.isdir(extracted_dir):
        # Try papers dir as fallback (older layout)
        extracted_dir = config.paths.papers_dir
    logger.info("cmd_contextualize: using extracted_dir=%s", extracted_dir)

    # ---- Build provider + telemetry --------------------------------------
    telemetry = TelemetryTracker(config)
    provider = get_provider(config, telemetry)

    start_time = time.monotonic()

    # ---- Pass 1: summarize documents ------------------------------------
    print("\nPass 1 — Summarizing documents...")
    summaries = summarize_all_documents(
        chunks=working_chunks,
        extracted_dir=extracted_dir,
        provider=provider,
        config=config,
    )

    print(f"\nPass 1 complete. {len(summaries)} document summaries:")
    print("-" * 60)
    for doc_id, summary in summaries.items():
        preview = summary[:300].replace("\n", " ")
        print(f"\n[{doc_id}]")
        print(f"  {preview}...")
    print("-" * 60)
    print("\nReview summaries above. Pass 2 will proceed using these summaries.")

    # ---- Pass 2: contextualize chunks -----------------------------------
    print("\nPass 2 — Contextualizing chunks in batches of 5...")
    ctx_chunks = contextualize_all_chunks(
        chunks=working_chunks,
        summaries=summaries,
        extracted_dir=extracted_dir,
        provider=provider,
        config=config,
    )

    elapsed = time.monotonic() - start_time

    # Count cache hits: chunks whose text is identical to the original
    # (proxy — true hits were logged inside contextualize_all_chunks)
    op_summary = telemetry.get_operation_summary("contextualize")

    # ---- Save contextualized chunks -------------------------------------
    ctx_chunks_path = os.path.join(config.paths.cache_dir, "contextualized_chunks.json")
    os.makedirs(config.paths.cache_dir, exist_ok=True)

    def _chunk_to_dict(c: Chunk) -> dict:
        return {
            "doc_id": c.doc_id,
            "chunk_id": c.chunk_id,
            "chunk_index": c.chunk_index,
            "text": c.text,
            "section": c.section,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "source_file": c.source_file,
            "token_count": c.token_count,
        }

    with open(ctx_chunks_path, "w", encoding="utf-8") as fh:
        json.dump([_chunk_to_dict(c) for c in ctx_chunks], fh, ensure_ascii=False, indent=2)

    logger.info("cmd_contextualize: saved %d contextualized chunks to %s", len(ctx_chunks), ctx_chunks_path)

    # ---- Export telemetry -----------------------------------------------
    telemetry_path = os.path.join(config.paths.results_dir, "telemetry_contextualize.json")
    os.makedirs(config.paths.results_dir, exist_ok=True)
    telemetry.export_json(telemetry_path)

    # ---- Summary --------------------------------------------------------
    print()
    print("=" * 60)
    print("Contextualization complete")
    print("=" * 60)
    print(f"  Total chunks contextualized : {len(ctx_chunks)}")
    print(f"  Output saved to             : {ctx_chunks_path}")
    print(f"  Elapsed time                : {elapsed:.1f}s")
    print()
    telemetry.display_summary()


def cmd_generate_questions(args: argparse.Namespace) -> None:
    """Generate evaluation questions from extracted paper documents."""
    import json
    import os

    from src.llm.base import get_provider
    from src.telemetry import TelemetryTracker

    overrides = _cli_overrides_from(args)
    config: AppConfig = load_config(args.config, overrides)

    # ---- Cost estimate ---------------------------------------------------
    # Formula (from architecture section 9.4):
    #   input  = num_papers × avg_paper_tokens
    #   output = count × question_output_tokens
    # Use 8 as a reasonable default paper count; actual count determined at
    # generation time but we need the estimate before executing.
    NUM_PAPERS = 8
    est = config.cost_estimation
    total_input = NUM_PAPERS * est.avg_paper_tokens
    total_output = args.count * est.question_output_tokens
    total_calls = NUM_PAPERS  # one LLM call per paper

    cost_gate = CostGate(config, approved=args.approve)
    estimate = cost_gate.estimate(
        operation="generate-questions",
        num_items=total_calls,
        avg_input_tokens=total_input // max(total_calls, 1),
        avg_output_tokens=total_output // max(total_calls, 1),
    )

    print()
    print("generate-questions cost estimate")
    print("=" * 60)
    print(f"  Papers (estimate)   : {NUM_PAPERS}")
    print(f"  Candidate questions : {args.count}")
    print(f"  Est. input tokens   : {total_input:,}")
    print(f"  Est. output tokens  : {total_output:,}")
    cost_gate.display_estimate(estimate)
    cost_gate.require_approval(estimate)  # exits here if not --approve

    # ---- Locate extracted documents directory ----------------------------
    documents_dir = os.path.join(config.paths.data_dir, "papers", "extracted")
    if not os.path.isdir(documents_dir):
        # Fallback to raw papers dir
        documents_dir = config.paths.papers_dir

    logger.info("cmd_generate_questions: documents_dir=%s", documents_dir)

    if not os.path.isdir(documents_dir):
        print(
            f"Documents directory not found: {documents_dir}\n"
            "Run 'ingest' first to extract papers.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Build provider + telemetry --------------------------------------
    telemetry = TelemetryTracker(config)
    provider = get_provider(config, telemetry)

    # ---- Instantiate QuestionGenerator and generate candidates -----------
    from src.evaluation.question_gen import QuestionGenerator

    generator = QuestionGenerator(provider=provider, config=config)

    logger.info(
        "cmd_generate_questions: generating %d candidates from %s",
        args.count,
        documents_dir,
    )
    print(f"\nGenerating {args.count} candidate questions from {documents_dir} ...")

    questions = generator.generate_candidates(documents_dir=documents_dir, count=args.count, approved=args.approve)

    # ---- Persist to data/questions.json ----------------------------------
    questions_path = os.path.join(config.paths.data_dir, "questions.json")
    os.makedirs(config.paths.data_dir, exist_ok=True)
    generator.save(questions, questions_path)

    logger.info(
        "cmd_generate_questions: saved %d questions to %s",
        len(questions),
        questions_path,
    )

    # ---- Export telemetry ------------------------------------------------
    telemetry_path = os.path.join(config.paths.results_dir, "telemetry_generate_questions.json")
    os.makedirs(config.paths.results_dir, exist_ok=True)
    telemetry.export_json(telemetry_path)

    # ---- Summary ---------------------------------------------------------
    print()
    print("=" * 60)
    print("generate-questions complete")
    print("=" * 60)
    print(f"  Questions generated : {len(questions)}")
    print(f"  Output saved to     : {questions_path}")
    print()
    print("Generated questions:")
    print("-" * 60)
    for i, q in enumerate(questions, start=1):
        print(f"  [{i:02d}] {q.question}")
    print()
    telemetry.display_summary()


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation across pipelines with LLM-as-judge scoring."""
    import json
    import os
    import time

    from src.llm.base import get_provider
    from src.telemetry import TelemetryTracker

    overrides = _cli_overrides_from(args)
    config: AppConfig = load_config(args.config, overrides)

    # ---- Resolve pipeline filter -----------------------------------------
    raw_pipelines: str = args.pipelines or "all"
    if raw_pipelines.strip().lower() == "all":
        pipeline_filter: list[str] | None = None
    else:
        pipeline_filter = [t.strip().lower() for t in raw_pipelines.split(",") if t.strip()]

    # ---- Resolve question filter -----------------------------------------
    if args.questions:
        question_filter: list[str] | None = [
            q.strip() for q in args.questions.split(",") if q.strip()
        ]
    else:
        question_filter = None

    # ---- Cost estimate ---------------------------------------------------
    # Counts:
    #   answer generation: num_questions × num_pipelines (1 LLM call per Q/P pair)
    #   judge calls      : num_questions × num_pipelines × num_metrics
    # Use architecture constant: 32 questions × 4 pipelines = 128 answer + 512 judge = 640 total
    NUM_QUESTIONS = 32
    NUM_PIPELINES = len(pipeline_filter) if pipeline_filter is not None else 4
    NUM_METRICS = len(config.evaluation.metrics)

    num_answer_calls = NUM_QUESTIONS * NUM_PIPELINES
    num_judge_calls = NUM_QUESTIONS * NUM_PIPELINES * NUM_METRICS
    total_calls = num_answer_calls + num_judge_calls

    est = config.cost_estimation

    # Answer call tokens: context (~4K input from paper) + query overhead; ~500 output
    answer_input_per_call = est.avg_chunk_tokens * config.retrieval.top_k_rerank + est.query_overhead_tokens
    answer_output_per_call = est.answer_max_tokens

    # Judge call tokens: judge prompt + context + question/answer/ground_truth
    judge_input_per_call = (
        est.judge_prompt_tokens
        + est.avg_chunk_tokens * config.retrieval.top_k_rerank
        + est.query_overhead_tokens
    )
    judge_output_per_call = est.judge_output_tokens

    total_input = num_answer_calls * answer_input_per_call + num_judge_calls * judge_input_per_call
    total_output = num_answer_calls * answer_output_per_call + num_judge_calls * judge_output_per_call

    cost_gate = CostGate(config, approved=args.approve)
    estimate = cost_gate.estimate(
        operation="evaluate",
        num_items=total_calls,
        avg_input_tokens=total_input // max(total_calls, 1),
        avg_output_tokens=total_output // max(total_calls, 1),
    )

    print()
    print("evaluate cost estimate")
    print("=" * 60)
    print(f"  Questions (estimate)   : {NUM_QUESTIONS}")
    print(f"  Pipelines              : {NUM_PIPELINES}")
    print(f"  Metrics per result     : {NUM_METRICS}")
    print(f"  Answer generation calls: {num_answer_calls}")
    print(f"  Judge calls            : {num_judge_calls}")
    print(f"  Total LLM calls        : {total_calls}")
    print(f"  Est. input tokens      : {total_input:,}")
    print(f"  Est. output tokens     : {total_output:,}")
    if pipeline_filter is not None:
        print(f"  Pipeline filter        : {', '.join(pipeline_filter)}")
    if question_filter is not None:
        print(f"  Question filter        : {', '.join(question_filter)}")
    cost_gate.display_estimate(estimate)
    cost_gate.require_approval(estimate)  # exits here if not --approve

    # ---- Load questions from data/questions.json -------------------------
    questions_path = os.path.join(config.paths.data_dir, "questions.json")
    if not os.path.exists(questions_path):
        print(
            f"questions.json not found at {questions_path}\n"
            "Run 'generate-questions' first to generate evaluation questions.",
            file=sys.stderr,
        )
        sys.exit(1)

    from src.evaluation.question_gen import QuestionGenerator

    telemetry = TelemetryTracker(config)
    provider = get_provider(config, telemetry)

    generator = QuestionGenerator(provider=provider, config=config)
    questions = generator.load(questions_path)
    logger.info("cmd_evaluate: loaded %d questions from %s", len(questions), questions_path)

    if not questions:
        print("No questions found in questions.json. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ---- Instantiate Judge -----------------------------------------------
    from src.evaluation.judge import Judge

    judge = Judge(provider=provider, config=config)

    # ---- Instantiate pipelines and EvaluationRunner (deferred import) ----
    from src.evaluation.runner import EvaluationRunner
    from src.retrieval.pipeline import Pipeline, PipelineMethod

    all_methods = list(PipelineMethod)
    if pipeline_filter is None:
        selected_methods = all_methods
    else:
        selected_methods = []
        for token in pipeline_filter:
            try:
                method = PipelineMethod(token)
            except ValueError:
                valid = ", ".join(m.value for m in all_methods)
                print(f"Unknown pipeline '{token}'. Valid options: {valid}", file=sys.stderr)
                sys.exit(1)
            selected_methods.append(method)

    pipeline_obj = Pipeline(config)
    # Runner expects string keys (pipeline.value), not enum objects.
    # Using enum objects as keys causes f-string formatting to produce
    # "PipelineMethod.TRADITIONAL" instead of "traditional" in filenames.
    pipelines = {method.value: pipeline_obj for method in selected_methods}

    runner = EvaluationRunner(pipelines=pipelines, judge=judge, config=config)

    # ---- Run evaluation --------------------------------------------------
    start_time = time.monotonic()
    print(f"\nRunning evaluation on {len(questions)} questions × {len(selected_methods)} pipelines...")

    results = runner.run(
        questions=questions,
        pipeline_filter=pipeline_filter,  # already list[str] or None
        question_filter=question_filter,
    )

    elapsed = time.monotonic() - start_time

    # ---- Export telemetry -----------------------------------------------
    os.makedirs(config.paths.results_dir, exist_ok=True)
    telemetry_path = os.path.join(config.paths.results_dir, "telemetry_evaluate.json")
    telemetry.export_json(telemetry_path)

    # ---- Print summary table --------------------------------------------
    print()
    print("=" * 60)
    print("Evaluation complete")
    print("=" * 60)
    print(f"  Results collected : {len(results)}")
    print(f"  Elapsed time      : {elapsed:.1f}s")
    print(f"  Telemetry saved   : {telemetry_path}")
    print()

    # Aggregate scores per pipeline per metric
    if results:
        from collections import defaultdict

        scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in results:
            pipeline_name = r.pipeline.value if hasattr(r.pipeline, "value") else str(r.pipeline)
            metric_name = r.metric.value if hasattr(r.metric, "value") else str(r.metric)
            scores[pipeline_name][metric_name].append(r.score)

        # Header
        metrics_list = list(config.evaluation.metrics)
        col_w = 12
        header = f"{'Pipeline':<16}" + "".join(f"{m[:col_w]:>{col_w}}" for m in metrics_list)
        print(header)
        print("-" * len(header))

        for pipeline_name in sorted(scores.keys()):
            row = f"{pipeline_name:<16}"
            for metric in metrics_list:
                vals = scores[pipeline_name].get(metric, [])
                avg = sum(vals) / len(vals) if vals else 0.0
                row += f"{avg:>{col_w}.3f}"
            print(row)

        print()

    telemetry.display_summary()


def cmd_interactive(args: argparse.Namespace) -> None:
    """Start interactive Q&A REPL with cost gate."""
    from src.retrieval.pipeline import PipelineMethod
    from src.interactive.repl import run_repl

    overrides = _cli_overrides_from(args)
    config: AppConfig = load_config(args.config, overrides)
    cost_gate = CostGate(config, approved=args.approve)

    # ---- Resolve pipeline list -------------------------------------------
    all_methods = list(PipelineMethod)
    implemented = [PipelineMethod.TRADITIONAL]

    raw_pipelines: str = args.pipelines or "all"
    if raw_pipelines.strip().lower() == "all":
        selected = implemented
    else:
        selected = []
        for token in raw_pipelines.split(","):
            token = token.strip().lower()
            try:
                method = PipelineMethod(token)
            except ValueError:
                valid = ", ".join(m.value for m in all_methods)
                print(f"Unknown pipeline '{token}'. Valid options: {valid}", file=sys.stderr)
                sys.exit(1)
            selected.append(method)

    if not selected:
        print("No pipelines selected.", file=sys.stderr)
        sys.exit(1)

    # ---- Cost gate: 1 LLM call per question per pipeline ------------------
    print(f"\nSelected pipeline(s): {', '.join(m.value for m in selected)}")
    print("Each question costs ~1 LLM call per pipeline.")

    est = config.cost_estimation
    # Use 1 as num_items because the REPL runs indefinitely; we show per-question cost.
    estimate = cost_gate.estimate(
        operation="interactive (per question)",
        num_items=len(selected),
        avg_input_tokens=(
            est.avg_chunk_tokens * config.retrieval.top_k_rerank
            + est.query_overhead_tokens
        ),
        avg_output_tokens=est.answer_max_tokens,
    )
    cost_gate.display_estimate(estimate)

    if not args.approve:
        print(
            "Each question costs ~1 LLM call per pipeline. "
            "Approve to start interactive session.\n"
            "Run with --approve to proceed."
        )
        sys.exit(0)

    # ---- Verify ChromaDB collections have data ----------------------------
    from src.ingestion.store import get_or_create_collection, get_collection_count

    _TRADITIONAL_COLLECTION = "rag_traditional_v1"
    needed_collections = []
    from src.retrieval.pipeline import PipelineMethod as PM
    if PM.TRADITIONAL in selected:
        needed_collections.append(_TRADITIONAL_COLLECTION)

    for coll_name in needed_collections:
        try:
            coll = get_or_create_collection(coll_name, config)
            count = get_collection_count(coll)
        except Exception as exc:
            print(
                f"Failed to open ChromaDB collection '{coll_name}': {exc}\n"
                "Run 'ingest' first to populate the collection.",
                file=sys.stderr,
            )
            sys.exit(1)

        if count == 0:
            print(
                f"ChromaDB collection '{coll_name}' is empty.\n"
                "Run 'ingest' first to populate it.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Collection '{coll_name}': {count:,} chunks ready.")

    # ---- Launch REPL -----------------------------------------------------
    run_repl(config, selected)


def cmd_query(args: argparse.Namespace) -> None:
    """Run a single question non-interactively through selected pipeline(s)."""
    import hashlib
    import json
    import os
    import time
    from datetime import datetime, timezone
    from src.retrieval.pipeline import Pipeline, PipelineMethod

    overrides = _cli_overrides_from(args)
    config: AppConfig = load_config(args.config, overrides)
    cost_gate = CostGate(config, approved=args.approve)

    # ---- Resolve pipeline list -------------------------------------------
    all_methods = list(PipelineMethod)
    raw_pipelines: str = args.pipelines or "traditional"
    selected = []
    for token in raw_pipelines.split(","):
        token = token.strip().lower()
        try:
            method = PipelineMethod(token)
        except ValueError:
            valid = ", ".join(m.value for m in all_methods)
            print(f"Unknown pipeline '{token}'. Valid options: {valid}", file=sys.stderr)
            sys.exit(1)
        selected.append(method)

    if not selected:
        print("No pipelines selected.", file=sys.stderr)
        sys.exit(1)

    # ---- Cost estimate ---------------------------------------------------
    est = config.cost_estimation
    estimate = cost_gate.estimate(
        operation="query (per question)",
        num_items=len(selected),
        avg_input_tokens=(
            est.avg_chunk_tokens * config.retrieval.top_k_rerank
            + est.query_overhead_tokens
        ),
        avg_output_tokens=est.answer_max_tokens,
    )
    cost_gate.display_estimate(estimate)

    if not args.approve:
        print(
            f"Question : {args.question}\n"
            f"Pipelines: {', '.join(m.value for m in selected)}\n"
            "Run with --approve to execute."
        )
        sys.exit(0)

    # ---- Verify ChromaDB collections have data ---------------------------
    from src.ingestion.store import get_or_create_collection, get_collection_count

    _TRADITIONAL_COLLECTION = "rag_traditional_v1"
    needed_collections = []
    if PipelineMethod.TRADITIONAL in selected:
        needed_collections.append(_TRADITIONAL_COLLECTION)

    for coll_name in needed_collections:
        try:
            coll = get_or_create_collection(coll_name, config)
            count = get_collection_count(coll)
        except Exception as exc:
            print(
                f"Failed to open ChromaDB collection '{coll_name}': {exc}\n"
                "Run 'ingest' first to populate the collection.",
                file=sys.stderr,
            )
            sys.exit(1)

        if count == 0:
            print(
                f"ChromaDB collection '{coll_name}' is empty.\n"
                "Run 'ingest' first to populate it.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Collection '{coll_name}': {count:,} chunks ready.")

    # ---- Run pipelines ---------------------------------------------------
    pipeline = Pipeline(config)

    for method in selected:
        print(f"\n{'=' * 63}")
        print(f"Pipeline: {method.value.upper()}")
        print(f"{'=' * 63}")

        result = pipeline.run(args.question, method)

        # ---- Persist result to disk --------------------------------------
        ts_obj = datetime.now(timezone.utc)
        ts_str = ts_obj.strftime("%Y%m%dT%H%M%SZ")
        question_hash = hashlib.md5(args.question.encode()).hexdigest()[:8]

        queries_dir = os.path.join(config.paths.results_dir, "queries")
        os.makedirs(queries_dir, exist_ok=True)

        result_filename = f"{ts_str}_{method.value}_{question_hash}.json"
        result_filepath = os.path.join(queries_dir, result_filename)

        result_payload = {
            "query": result.query,
            "method": result.method.value,
            "retrieved_chunks": [
                {
                    "text": c.get("text", ""),
                    "score": c.get("score", 0.0),
                    "source": c.get("source_file", ""),
                    "page_start": c.get("page_start"),
                    "page_end": c.get("page_end"),
                }
                for c in result.retrieved_chunks
            ],
            "answer": result.answer,
            "latency_ms": result.latency_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "timestamp": ts_obj.isoformat(),
        }

        with open(result_filepath, "w", encoding="utf-8") as fh:
            json.dump(result_payload, fh, indent=2)

        # ---- Append summary line to query_log.jsonl ----------------------
        log_path = os.path.join(config.paths.results_dir, "query_log.jsonl")
        log_entry = {
            "timestamp": ts_obj.isoformat(),
            "question": args.question,
            "pipeline": method.value,
            "answer_preview": result.answer[:200],
            "latency_ms": result.latency_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "num_chunks": len(result.retrieved_chunks),
            "result_file": result_filepath,
        }
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(log_entry) + "\n")

        # ---- Print result file path --------------------------------------
        print(f"\nResult saved: {result_filepath}")

        # Retrieved chunks (top 5)
        print(f"\nRetrieved chunks (top {min(5, len(result.retrieved_chunks))}):")
        print("-" * 63)
        for i, chunk in enumerate(result.retrieved_chunks[:5], start=1):
            src = chunk.get("source_file", "unknown")
            pg_start = chunk.get("page_start", "?")
            pg_end = chunk.get("page_end", "?")
            score = chunk.get("score", 0.0)
            text_preview = chunk.get("text", "").strip()[:200].replace("\n", " ")
            print(f"[{i}] score={score:.4f}  {src}  pp.{pg_start}-{pg_end}")
            print(f"    {text_preview}...")

        # Generated answer
        print(f"\nAnswer:")
        print("-" * 63)
        print(result.answer)

        # Sources cited
        print(f"\nSources cited ({len(result.sources)}):")
        print("-" * 63)
        for src in result.sources:
            print(f"  - {src.get('doc', 'unknown')}  p.{src.get('page', '?')}")

        # Latency
        print(f"\nLatency: {result.latency_ms:.1f} ms")


def cmd_compare(args: argparse.Namespace) -> None:
    """Generate comparison report from evaluation results."""
    import json
    import os
    from collections import defaultdict

    overrides: dict[str, Any] = {}
    if getattr(args, "results_dir", None) is not None:
        overrides["paths.results_dir"] = args.results_dir

    config: AppConfig = load_config(args.config, overrides)
    output_format: str = args.output_format

    logger.info("cmd_compare: results_dir=%s output_format=%s", config.paths.results_dir, output_format)

    # ---- Load all eval results -------------------------------------------
    eval_dir = os.path.join(config.paths.results_dir, "eval")
    logger.info("cmd_compare: loading results from %s", eval_dir)

    if not os.path.isdir(eval_dir):
        print(
            f"No eval results directory found at {eval_dir}\n"
            "Run 'evaluate' first to generate results.",
            file=sys.stderr,
        )
        sys.exit(1)

    from src.evaluation.runner import EvalResult, EvaluationRunner

    # Instantiate runner with minimal args just for load_all_results().
    # load_all_results() only reads files — pipelines/judge are not used.
    runner = EvaluationRunner(pipelines={}, judge=None, config=config)  # type: ignore[arg-type]
    results = runner.load_all_results()

    if not results:
        print(
            f"No evaluation result files found in {eval_dir}\n"
            "Run 'evaluate' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    logger.info("cmd_compare: loaded %d results", len(results))

    # ---- Aggregate: avg score per pipeline per metric --------------------
    # scores[pipeline][metric] -> list[float]
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # cost_totals[pipeline] -> total USD
    cost_totals: dict[str, float] = defaultdict(float)
    # latency_totals[pipeline] -> list[ms] (judge latency per result)
    latency_totals: dict[str, list[float]] = defaultdict(list)
    # token_totals[pipeline] -> {judge_in, judge_out, answer_in, answer_out}
    token_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"judge_in": 0, "judge_out": 0, "answer_in": 0, "answer_out": 0}
    )
    # question IDs seen per pipeline (for breakdown count)
    question_ids: set[str] = set()

    for r in results:
        scores[r.pipeline][r.metric].append(r.score)
        cost_totals[r.pipeline] += r.cost_usd
        latency_totals[r.pipeline].append(r.latency_ms)
        token_totals[r.pipeline]["judge_in"] += r.input_tokens
        token_totals[r.pipeline]["judge_out"] += r.output_tokens
        token_totals[r.pipeline]["answer_in"] += r.answer_input_tokens
        token_totals[r.pipeline]["answer_out"] += r.answer_output_tokens
        question_ids.add(r.question_id)

    # Ordered pipelines and metrics
    pipeline_order = ["traditional", "contextual", "hybrid", "modern"]
    pipelines_found = sorted(scores.keys(), key=lambda p: pipeline_order.index(p) if p in pipeline_order else 999)
    metrics_list = list(config.evaluation.metrics)

    # avg_scores[pipeline][metric] -> float
    avg_scores: dict[str, dict[str, float]] = {}
    for pipeline in pipelines_found:
        avg_scores[pipeline] = {}
        for metric in metrics_list:
            vals = scores[pipeline].get(metric, [])
            avg_scores[pipeline][metric] = sum(vals) / len(vals) if vals else float("nan")

    # ---- Per-question breakdown (if <= 10 questions) ---------------------
    do_breakdown = len(question_ids) <= 10
    breakdown: dict[str, dict[str, dict[str, float]]] = {}  # qid -> pipeline -> metric -> score
    if do_breakdown:
        for r in results:
            breakdown.setdefault(r.question_id, {}).setdefault(r.pipeline, {})[r.metric] = r.score

    # ---- Output ----------------------------------------------------------
    if output_format == "json":
        payload = {
            "summary": {
                pipeline: {
                    "avg_scores": avg_scores[pipeline],
                    "total_cost_usd": cost_totals[pipeline],
                    "avg_latency_ms": (
                        sum(latency_totals[pipeline]) / len(latency_totals[pipeline])
                        if latency_totals[pipeline] else 0.0
                    ),
                    "total_judge_input_tokens": token_totals[pipeline]["judge_in"],
                    "total_judge_output_tokens": token_totals[pipeline]["judge_out"],
                    "total_answer_input_tokens": token_totals[pipeline]["answer_in"],
                    "total_answer_output_tokens": token_totals[pipeline]["answer_out"],
                }
                for pipeline in pipelines_found
            },
            "num_questions": len(question_ids),
            "metrics": metrics_list,
        }
        if do_breakdown:
            payload["per_question"] = breakdown
        print(json.dumps(payload, indent=2))

    elif output_format == "md":
        _render_markdown(
            pipelines_found=pipelines_found,
            metrics_list=metrics_list,
            avg_scores=avg_scores,
            cost_totals=cost_totals,
            latency_totals=latency_totals,
            token_totals=token_totals,
            question_ids=question_ids,
            do_breakdown=do_breakdown,
            breakdown=breakdown,
        )

    else:  # table (default)
        _render_rich_table(
            pipelines_found=pipelines_found,
            metrics_list=metrics_list,
            avg_scores=avg_scores,
            cost_totals=cost_totals,
            latency_totals=latency_totals,
            token_totals=token_totals,
            question_ids=question_ids,
            do_breakdown=do_breakdown,
            breakdown=breakdown,
        )


# ---------------------------------------------------------------------------
# compare output helpers
# ---------------------------------------------------------------------------


def _score_color(score: float) -> str:
    """Return rich color markup string for a score value."""
    if score != score:  # nan
        return "dim"
    if score > 0.8:
        return "green"
    if score > 0.5:
        return "yellow"
    return "red"


def _render_rich_table(
    pipelines_found: list[str],
    metrics_list: list[str],
    avg_scores: dict,
    cost_totals: dict,
    latency_totals: dict,
    token_totals: dict,
    question_ids: set,
    do_breakdown: bool,
    breakdown: dict,
) -> None:
    """Render rich terminal table with colored scores."""
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()

    # ---- Summary table --------------------------------------------------
    table = Table(
        title=f"RAG Pipeline Comparison  ({len(question_ids)} questions)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Pipeline", style="bold", min_width=14)
    for metric in metrics_list:
        table.add_column(metric.replace("_", " ").title(), justify="center", min_width=10)
    table.add_column("Total Cost", justify="right", min_width=11)
    table.add_column("Avg Latency", justify="right", min_width=12)
    table.add_column("Ans In Tok", justify="right", min_width=11)
    table.add_column("Ans Out Tok", justify="right", min_width=12)
    table.add_column("Judge In Tok", justify="right", min_width=13)
    table.add_column("Judge Out Tok", justify="right", min_width=14)

    for pipeline in pipelines_found:
        row: list[str] = [pipeline]
        for metric in metrics_list:
            val = avg_scores[pipeline].get(metric, float("nan"))
            if val != val:
                cell = "[dim]n/a[/dim]"
            else:
                color = _score_color(val)
                cell = f"[{color}]{val:.3f}[/{color}]"
            row.append(cell)
        cost = cost_totals.get(pipeline, 0.0)
        lats = latency_totals.get(pipeline, [])
        avg_lat = sum(lats) / len(lats) if lats else 0.0
        tok = token_totals.get(pipeline, {"judge_in": 0, "judge_out": 0, "answer_in": 0, "answer_out": 0})
        row.append(f"${cost:.4f}")
        row.append(f"{avg_lat:.0f} ms")
        row.append(str(tok["answer_in"]))
        row.append(str(tok["answer_out"]))
        row.append(str(tok["judge_in"]))
        row.append(str(tok["judge_out"]))
        table.add_row(*row)

    console.print(table)

    # ---- Per-question breakdown ------------------------------------------
    if do_breakdown and breakdown:
        for qid in sorted(breakdown.keys()):
            bk_table = Table(
                title=f"Question: {qid}",
                box=box.SIMPLE,
                show_header=True,
                header_style="bold",
            )
            bk_table.add_column("Pipeline", style="bold", min_width=14)
            for metric in metrics_list:
                bk_table.add_column(metric.replace("_", " ").title(), justify="center", min_width=10)

            for pipeline in pipelines_found:
                q_scores = breakdown.get(qid, {}).get(pipeline, {})
                row = [pipeline]
                for metric in metrics_list:
                    val = q_scores.get(metric, float("nan"))
                    if val != val:
                        cell = "[dim]n/a[/dim]"
                    else:
                        color = _score_color(val)
                        cell = f"[{color}]{val:.3f}[/{color}]"
                    row.append(cell)
                bk_table.add_row(*row)

            console.print(bk_table)


def _render_markdown(
    pipelines_found: list[str],
    metrics_list: list[str],
    avg_scores: dict,
    cost_totals: dict,
    latency_totals: dict,
    token_totals: dict,
    question_ids: set,
    do_breakdown: bool,
    breakdown: dict,
) -> None:
    """Render Markdown table suitable for docs/comparison-analysis.md."""
    lines: list[str] = []

    lines.append(f"## RAG Pipeline Comparison ({len(question_ids)} questions)")
    lines.append("")

    # Header row
    header_cols = (
        ["Pipeline"]
        + [m.replace("_", " ").title() for m in metrics_list]
        + ["Total Cost", "Avg Latency", "Ans In Tok", "Ans Out Tok", "Judge In Tok", "Judge Out Tok"]
    )
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")

    for pipeline in pipelines_found:
        cells: list[str] = [pipeline]
        for metric in metrics_list:
            val = avg_scores[pipeline].get(metric, float("nan"))
            cells.append("n/a" if val != val else f"{val:.3f}")
        cost = cost_totals.get(pipeline, 0.0)
        lats = latency_totals.get(pipeline, [])
        avg_lat = sum(lats) / len(lats) if lats else 0.0
        tok = token_totals.get(pipeline, {"judge_in": 0, "judge_out": 0, "answer_in": 0, "answer_out": 0})
        cells.append(f"${cost:.4f}")
        cells.append(f"{avg_lat:.0f} ms")
        cells.append(str(tok["answer_in"]))
        cells.append(str(tok["answer_out"]))
        cells.append(str(tok["judge_in"]))
        cells.append(str(tok["judge_out"]))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")

    if do_breakdown and breakdown:
        lines.append("### Per-Question Breakdown")
        lines.append("")
        for qid in sorted(breakdown.keys()):
            lines.append(f"#### {qid}")
            lines.append("")
            lines.append("| Pipeline | " + " | ".join(m.replace("_", " ").title() for m in metrics_list) + " |")
            lines.append("|" + "|".join(["---"] * (1 + len(metrics_list))) + "|")
            for pipeline in pipelines_found:
                q_scores = breakdown.get(qid, {}).get(pipeline, {})
                cells = [pipeline]
                for metric in metrics_list:
                    val = q_scores.get(metric, float("nan"))
                    cells.append("n/a" if val != val else f"{val:.3f}")
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-compare",
        description="RAG Comparison System — compare retrieval methods on answer quality.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="SUBCOMMAND")
    subparsers.required = True

    # ------------------------------------------------------------------
    # ingest
    # ------------------------------------------------------------------
    p_ingest = subparsers.add_parser(
        "ingest",
        help="Ingest PDFs into the traditional ChromaDB collection.",
    )
    p_ingest.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to YAML config file (default: config/default.yaml).",
    )
    p_ingest.add_argument(
        "--approve",
        action="store_true",
        default=False,
        help="Execute the operation (without this flag, only prints cost estimate).",
    )
    p_ingest.add_argument(
        "--papers-dir",
        metavar="PATH",
        default=None,
        dest="papers_dir",
        help="Override papers directory.",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # ------------------------------------------------------------------
    # contextualize
    # ------------------------------------------------------------------
    p_ctx = subparsers.add_parser(
        "contextualize",
        help="LLM-contextualize chunks into the contextualized collection.",
    )
    p_ctx.add_argument("--config", metavar="PATH", default=None)
    p_ctx.add_argument("--approve", action="store_true", default=False)
    p_ctx.add_argument(
        "--provider",
        metavar="TEXT",
        default=None,
        help="LLM provider override (anthropic/openai/claude_cli).",
    )
    p_ctx.add_argument(
        "--model",
        metavar="TEXT",
        default=None,
        help="Model override.",
    )
    p_ctx.add_argument(
        "--sample",
        metavar="N",
        type=int,
        default=None,
        help="Run on N chunks only (for quality review before full run).",
    )
    p_ctx.set_defaults(func=cmd_contextualize)

    # ------------------------------------------------------------------
    # generate-questions
    # ------------------------------------------------------------------
    p_gen = subparsers.add_parser(
        "generate-questions",
        help="Generate evaluation questions from papers.",
    )
    p_gen.add_argument("--config", metavar="PATH", default=None)
    p_gen.add_argument("--approve", action="store_true", default=False)
    p_gen.add_argument(
        "--count",
        metavar="N",
        type=int,
        default=30,
        help="Number of candidate questions to generate (default: 30).",
    )
    p_gen.add_argument("--provider", metavar="TEXT", default=None)
    p_gen.add_argument("--model", metavar="TEXT", default=None)
    p_gen.set_defaults(func=cmd_generate_questions)

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------
    p_eval = subparsers.add_parser(
        "evaluate",
        help="Run evaluation across pipelines.",
    )
    p_eval.add_argument("--config", metavar="PATH", default=None)
    p_eval.add_argument("--approve", action="store_true", default=False)
    p_eval.add_argument(
        "--pipelines",
        metavar="TEXT",
        default="all",
        help="Comma-separated pipeline list (default: all).",
    )
    p_eval.add_argument(
        "--questions",
        metavar="TEXT",
        default=None,
        help="Comma-separated question IDs to evaluate (default: all).",
    )
    p_eval.add_argument("--provider", metavar="TEXT", default=None)
    p_eval.add_argument("--model", metavar="TEXT", default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    # ------------------------------------------------------------------
    # interactive
    # ------------------------------------------------------------------
    p_int = subparsers.add_parser(
        "interactive",
        help="Start interactive Q&A REPL.",
    )
    p_int.add_argument("--config", metavar="PATH", default=None)
    p_int.add_argument(
        "--approve",
        action="store_true",
        default=False,
        help="Approve LLM costs and start the REPL (without this flag, only prints cost estimate).",
    )
    p_int.add_argument(
        "--pipelines",
        metavar="TEXT",
        default="all",
        help="Comma-separated pipeline list (default: all).",
    )
    p_int.add_argument("--provider", metavar="TEXT", default=None)
    p_int.add_argument("--model", metavar="TEXT", default=None)
    p_int.set_defaults(func=cmd_interactive)

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------
    p_query = subparsers.add_parser(
        "query",
        help="Run a single question non-interactively through selected pipeline(s).",
    )
    p_query.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to YAML config file (default: config/default.yaml).",
    )
    p_query.add_argument(
        "--approve",
        action="store_true",
        default=False,
        help="Execute the query (without this flag, only prints cost estimate).",
    )
    p_query.add_argument(
        "--pipelines",
        metavar="TEXT",
        default="traditional",
        help="Comma-separated pipeline list (default: traditional).",
    )
    p_query.add_argument(
        "--question",
        metavar="TEXT",
        required=True,
        help="The question to answer.",
    )
    p_query.add_argument("--provider", metavar="TEXT", default=None)
    p_query.add_argument("--model", metavar="TEXT", default=None)
    p_query.set_defaults(func=cmd_query)

    # ------------------------------------------------------------------
    # compare
    # ------------------------------------------------------------------
    p_cmp = subparsers.add_parser(
        "compare",
        help="Generate comparison report from evaluation results.",
    )
    p_cmp.add_argument("--config", metavar="PATH", default=None)
    p_cmp.add_argument(
        "--results-dir",
        metavar="PATH",
        default=None,
        dest="results_dir",
        help="Override results directory.",
    )
    p_cmp.add_argument(
        "--output-format",
        metavar="TEXT",
        default="table",
        dest="output_format",
        choices=["table", "json", "md"],
        help="Output format: table, json, md (default: table).",
    )
    p_cmp.set_defaults(func=cmd_compare)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging()
    parser = _build_parser()
    args = parser.parse_args()

    logger.info("CLI subcommand dispatched: %s", args.command)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
