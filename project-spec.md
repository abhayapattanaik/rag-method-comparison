# RAG Comparison Project -- Build Specification

> **Audience**: An AI assistant tasked with building this project from scratch.
> **Tone**: Prescriptive. Requirements, constraints, rules.
> **Priorities**: Cost-efficient development — validate incrementally with minimal LLM calls before scaling. Verify outcomes at each step before proceeding to the next.

---

## 1. Project Overview & Goals

**Goals:**
1. Learn RAG retrieval methods by building a working RAG system end-to-end
2. Understand how retrieval technique choices affect answer quality, cost, and latency through concrete comparison
3. Implement efficiently with AI assistance — minimize iterations, cost, and rework

**The 4 retrieval methods:**

1. **Traditional** -- chunk, embed, dense retrieval, LLM answer
2. **Contextual** -- chunk, LLM-contextualize each chunk, embed, dense retrieval, LLM answer
3. **Hybrid** -- chunk, LLM-contextualize, embed, dense + BM25 with RRF fusion, LLM answer
4. **Modern** -- chunk, LLM-contextualize, embed, dense + BM25, RRF, cross-encoder rerank, LLM answer

### 1.1 Functional Requirements

1. Ingest PDF documents into per-method vector stores
2. 4 retrieval methods (traditional, contextual, hybrid, modern)
3. Evaluate answers with LLM-as-judge (4 metrics)
4. Interactive Q&A — query all methods, display answers side by side with source citations
5. Cost estimation before any LLM execution — user approves before spending
6. Combined comparison view across all methods

### 1.2 Non-Functional Requirements

1. **Scientific control** — only the retrieval technique varies; all other variables (embedding model, chunk size, LLM, judge) held constant
2. **Idempotent operations** — safe to re-run at any stage
3. **Crash recovery** — resumable after interruption without re-doing completed work
4. **Telemetry** — token usage, cost, and latency tracked per call
5. **Configurable** — all parameters via YAML config or CLI override
6. **Stall detection** — abort on lack of progress
7. **Cost-efficient development** — validate incrementally before scaling
8. **Structured logging** — all modules must use Python `logging` with file + console handlers. Log config loading, pipeline steps, LLM calls, errors, and timing.

---

## 2. Architecture

### 2.1 Data Flow

The system has two phases: **ingestion** and **evaluation**.

Ingestion is a one-time operation per configuration. Evaluation and interactive Q&A run repeatedly against the ingested stores.

### 2.2 LLM Backend

Support multiple LLM backends — Anthropic API, OpenAI API, Claude CLI, and Ollama — through a common interface, configurable at runtime.

### 2.3 Cost Control: Approval Gate

Every CLI entry point follows two phases:
1. **Estimate phase** (no `--approve`): count chunks/questions, calculate LLM calls,
   print cost estimate for haiku/sonnet/opus, exit.
2. **Execute phase** (`--approve`): build provider with `approved=True`, run the work.

This prevents accidental large LLM bills. Build this into every CLI from day 1.

---

## 3. Module Design

Produce a modular design that satisfies the functional and non-functional requirements. Get the design reviewed and approved by the author before implementing.

If shared resources exist across modules, discuss thread safety and concurrency strategy with the author. When scaling parallelism, monitor CPU and memory — do not exceed 80% utilization.

---

## 4. Configuration

Every tunable parameter — paths, model names, thresholds, worker counts — must be configurable via YAML config with CLI override. Nothing hardcoded.

---

## 5. Chunking

Chunk at document level, not per-page. Clean PDF extraction artifacts without losing content. Research chunk sizing best practices for the chosen embedding model — recommend with tradeoffs, agree with author before implementing. Generated answers must cite source document and page number.

---

## 6. Evaluation

Evaluate each method's answers using LLM-as-judge on four dimensions: context precision, context recall, faithfulness, and answer relevancy (each 0.0–1.0). Do not use the RAGAS library. Persist results incrementally for crash recovery. Each run produces per-method scores, cross-method comparison, and a combined side-by-side view.

---

## 7. Telemetry & Cost Tracking

Every LLM call (query, judge, contextualization) must track token usage, latency, and cost. Results must show per-pipeline cost breakdown so pipelines can be compared on efficiency, not just quality.

---

## 8. CLI Interface

User must be able to provide configuration through command-line arguments or YAML config. User has explicit control over when LLM calls are made (ingestion, contextualization, evaluation are separate invocations, not automatic). Pipelines must be individually selectable and questions filterable.

---

## 9. Concurrency & Resilience

- **Idempotency**: All operations (ingestion, caching, evaluation) must be safe to re-run without duplicating data.
- **Stall detection**: Long-running operations must detect lack of progress and time out gracefully.
- **Verification**: After critical operations, verify results programmatically. Do not trust log output alone.

---

## 10. Testing Strategy

Validate incrementally: confirm basic end-to-end pipeline functionality with a single question and single pipeline before scaling to full evaluation across all pipelines and questions. Do not run expensive full evaluations without first verifying that results are structurally correct. Document tests and review with the author before executing.

---

## 11. Acceptance Criteria

Create a detailed acceptance criteria checklist for author review. The checklist must cover:

1. **Functional completeness** — all 4 pipelines ingest, retrieve, and generate answers; evaluation produces scored results; combined comparison view works
2. **Scientific validity** — only the retrieval technique varies between pipelines; all other variables (embedding model, chunk size, LLM, judge) are controlled
3. **Cost observability** — token usage and cost tracked and visible per pipeline
4. **Resilience** — operations are idempotent and resumable after interruption
5. **Usability** — user can estimate cost before committing to expensive operations

---

## 12. Expected Outputs

1. **Architecture document** — system design with component responsibilities and interactions
2. **Design decisions log** — key decisions made during implementation and why
3. **Test plan** — verification strategy and acceptance test results
4. **Comparison analysis** — findings from running all 4 methods, what performed better and why
5. **Cost-quality tradeoff summary** — which method gives best quality per dollar
6. **RAG concepts glossary** — key terms encountered during build (dense retrieval, RRF, cross-encoder, BM25, contextual chunking, etc.)
