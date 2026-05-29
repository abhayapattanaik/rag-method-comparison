# Getting Started — RAG Comparison System

Step-by-step guide from clone to results. Covers all ingestion scripts, the CLI,
evaluation, comparison reporting, and interactive mode.

---

## Table of Contents

1. [Hardware Requirements](#1-hardware-requirements)
2. [Prerequisites](#2-prerequisites)
3. [Clone and Install](#3-clone-and-install)
4. [Data Setup](#4-data-setup)
5. [Ingestion Pipeline](#5-ingestion-pipeline)
   - 5a. Extract PDFs
   - 5b. Chunk Papers
   - 5c. Embed and Store (Traditional)
   - 5d. Contextualize Chunks (CLI)
   - 5e. Embed and Store (Contextualized)
6. [Generate Evaluation Questions](#6-generate-evaluation-questions)
7. [Run Evaluation](#7-run-evaluation)
8. [Compare Results](#8-compare-results)
9. [Interactive Mode](#9-interactive-mode)
10. [One-off Query](#10-one-off-query)
11. [Cost Gate Explanation](#11-cost-gate-explanation)
12. [Crash Recovery](#12-crash-recovery)
13. [Configuration Reference](#13-configuration-reference)
14. [Timelines](#14-timelines)
15. [Known Quirks](#15-known-quirks)

---

## 1. Hardware Requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| RAM | 16 GB | Embedding model (~416 MB) and reranker (~1.1 GB) are loaded sequentially, never together |
| GPU | Apple Silicon MPS recommended | PyTorch MPS backend; CPU fallback works but is 3-5x slower |
| Disk | 4 GB free | ChromaDB, model weights cache (~1.5 GB), eval results |
| Python | 3.11+ | f-string syntax and match statements used throughout |

The project was developed and tested on an M2 MacBook Pro 16 GB. If running on CPU, embedding
a 266-chunk corpus takes ~5-10 minutes instead of ~1 minute.

---

## 2. Prerequisites

- Python 3.11 or later
- `pip` (or a virtual environment manager of your choice)
- At least one of the following for LLM calls:
  - `claude` CLI installed and authenticated (default provider — `llm.provider: claude_cli`)
  - `ANTHROPIC_API_KEY` environment variable set (provider `anthropic`)
  - `OPENAI_API_KEY` environment variable set (provider `openai`)

Check which provider is active in `config/default.yaml` under `llm.provider`. To override on the
fly, pass `--provider anthropic` (or `openai`, `claude_cli`) to any CLI subcommand.

---

## 3. Clone and Install

```bash
git clone <repo-url>
cd claude_rc_2

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

The first `import` of the embedding model (`BAAI/bge-base-en-v1.5`) and reranker
(`BAAI/bge-reranker-v2-m3`) will trigger an automatic download from Hugging Face (~1.5 GB total).
This happens once; subsequent runs load from the local cache.

Verify the install:

```bash
python -m src.cli.main --help
```

---

## 4. Data Setup

See `DATA_SETUP.md` for full instructions on downloading papers.

The system expects 8 arXiv PDFs in `data/papers/`. The filenames are prefixed with a
two-digit index:

```
data/papers/
  01_rag_lewis_2020.pdf
  02_dpr_karpukhin_2020.pdf
  03_late_chunking_gunther_2024.pdf
  04_raptor_sarthi_2024.pdf
  05_hybrid_retrieval_kuzi_2020.pdf
  06_fusion_functions_bruch_2022.pdf
  07_bert_reranking_nogueira_2019.pdf
  08_multistage_ranking_nogueira_2019.pdf
```

All downstream scripts expect files in exactly this location. Do not rename the PDFs.

---

## 5. Ingestion Pipeline

Ingestion is a one-time operation. All steps are idempotent — safe to re-run if interrupted.
The sequence is:

```
PDF files  ->  extract  ->  chunk  ->  embed (traditional)
                                  \->  contextualize (LLM)  ->  embed (contextualized)
```

### 5a. Extract PDFs

Converts each PDF to Markdown using `pymupdf4llm`. Outputs one `.md` file and one
`_chunks.json` file per paper into `data/papers/extracted/`.

```bash
python3 scripts/extract_papers.py
```

Expected output: 8 `.md` files and 8 `_chunks.json` files in `data/papers/extracted/`.
This step has no LLM calls and costs nothing.

### 5b. Chunk Papers

Splits each extracted Markdown into overlapping chunks using section-aware logic.
Reads from `data/papers/extracted/` and writes all chunks to `data/cache/chunks.json`.

```bash
python3 scripts/chunk_papers.py
```

Expected output: 266 chunks written to `data/cache/chunks.json`.
Parameters controlled by `config/default.yaml`:
- `chunking.chunk_size`: 450 tokens
- `chunking.chunk_overlap`: 68 tokens

### 5c. Embed and Store (Traditional)

Embeds all 266 chunks with `BAAI/bge-base-en-v1.5` and upserts them into the ChromaDB
collection `rag_traditional_v1`. Reads from `data/cache/chunks.json`.

```bash
python3 scripts/embed_and_store.py
```

Idempotency: if the collection already has 266 chunks, the script prints a skip message
and exits without re-embedding.

Expected time: ~1 minute on MPS, ~5-10 minutes on CPU.

### 5d. Contextualize Chunks (CLI)

This is the only ingestion step that makes LLM calls. It runs in two passes:

- Pass 1: One LLM call per document to generate a document-level summary (8 calls total).
- Pass 2: Batches of 5 chunks contextualized in a single LLM call
  (~54 batches for 266 chunks).

The result is a new text for each chunk that prepends a context prefix describing where
the chunk sits within its document.

Always run the estimate first (no `--approve`):

```bash
python -m src.cli.main contextualize
```

Review the token and call counts printed. Then execute:

```bash
python -m src.cli.main contextualize --approve
```

Output: `data/cache/contextualized_chunks.json` (266 entries with context-prefixed text).
Telemetry saved to `data/results/telemetry_contextualize.json`.

The `--sample N` flag processes only the first N chunks — useful for inspecting output
quality before committing to the full run:

```bash
python -m src.cli.main contextualize --sample 10
```

Per-chunk results are cached in `data/cache/contextualized/` (one JSON file per chunk).
If the run is interrupted, restarting skips chunks that already have cached results.

### 5e. Embed and Store (Contextualized)

Embeds the 266 contextualized chunks and upserts them into the ChromaDB collection
`rag_contextualized_v1`. Reads from `data/cache/contextualized_chunks.json`.

```bash
python3 scripts/embed_contextualized.py
```

This collection is shared by the Contextual, Hybrid, and Modern pipelines. Embedding
once keeps the vectors scientifically identical across those three methods.

Idempotency: same skip logic as step 5c.

---

## 6. Generate Evaluation Questions

Generates evaluation questions from the extracted paper Markdown files. Each question
includes a `ground_truth` answer used by the LLM judge.

Estimate first:

```bash
python -m src.cli.main generate-questions --count 32
```

Then execute:

```bash
python -m src.cli.main generate-questions --count 32 --approve
```

Output: `data/questions.json` (32 question/ground-truth pairs).
Telemetry saved to `data/results/telemetry_generate_questions.json`.

The `--count` flag controls how many candidate questions are requested (default: 30).
The actual number in `questions.json` may be slightly lower if the LLM returns fewer
usable pairs than requested.

To use a different provider for generation:

```bash
python -m src.cli.main generate-questions --count 32 --provider anthropic --model claude-haiku-4-5-20241022 --approve
```

---

## 7. Run Evaluation

Evaluates all four pipelines across all questions using LLM-as-judge scoring on four
metrics: `context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`.

Each pipeline/question/metric combination produces one JSON file in `data/results/eval/`.
Results are written incrementally — if the run is interrupted, restarting skips
already-scored combinations.

Estimate first (32 questions x 4 pipelines x 4 metrics = 512 judge calls + 128 answer calls):

```bash
python -m src.cli.main evaluate
```

Then execute:

```bash
python -m src.cli.main evaluate --approve
```

Filtering options:

```bash
# Run only two pipelines
python -m src.cli.main evaluate --pipelines traditional,modern --approve

# Run specific questions by ID (hex prefix from questions.json)
python -m src.cli.main evaluate --questions 175d393e,ab12cd34 --approve

# Combine both filters
python -m src.cli.main evaluate --pipelines hybrid --questions 175d393e --approve
```

A summary table is printed at the end showing average scores per pipeline per metric.
Full telemetry saved to `data/results/telemetry_evaluate.json`.

---

## 8. Compare Results

Loads all scored results from `data/results/eval/` and renders a cross-pipeline
comparison table with quality scores, token counts, cost, and latency.

```bash
# Rich terminal table (default)
python -m src.cli.main compare

# Markdown table (pipe to file for docs)
python -m src.cli.main compare --output-format md

# Machine-readable JSON
python -m src.cli.main compare --output-format json
```

The table shows, per pipeline:
- Average score for each of the 4 metrics (color-coded: green > 0.8, yellow > 0.5, red otherwise)
- Total cost in USD
- Average judge latency in ms
- Total answer input/output tokens
- Total judge input/output tokens

If there are 10 or fewer questions, a per-question breakdown table is also printed below
the summary.

The `--results-dir` flag overrides where results are read from, useful for comparing
runs from different config variants:

```bash
python -m src.cli.main compare --results-dir data/results_experiment_2
```

---

## 9. Interactive Mode

Starts a REPL where you can type questions and receive answers from one or more pipelines
side by side in real time.

Estimate cost per question first:

```bash
python -m src.cli.main interactive
```

Then start the session:

```bash
python -m src.cli.main interactive --approve
```

Select specific pipelines:

```bash
python -m src.cli.main interactive --pipelines traditional,contextual --approve
```

Within the REPL, type any question and press Enter. Type `quit` or `exit` to end the
session. Each answer includes retrieved chunk previews with source document and page
numbers.

Note: the interactive REPL currently enables only the Traditional pipeline by default
(`implemented = [PipelineMethod.TRADITIONAL]` in `cmd_interactive`). Passing
`--pipelines contextual,hybrid,modern` will work once those pipeline paths are fully
verified end-to-end.

---

## 10. One-off Query

Runs a single question non-interactively. Useful for spot-checking a pipeline without
starting a full REPL session.

Estimate:

```bash
python -m src.cli.main query --question "What is Reciprocal Rank Fusion?"
```

Execute:

```bash
python -m src.cli.main query --question "What is Reciprocal Rank Fusion?" --approve
```

Specify pipeline(s):

```bash
python -m src.cli.main query \
  --question "How does DPR compare to BM25 on Natural Questions?" \
  --pipelines traditional,modern \
  --approve
```

Results are saved to `data/results/queries/<timestamp>_<pipeline>_<hash>.json` and a
summary line is appended to `data/results/query_log.jsonl`.

---

## 11. Cost Gate Explanation

Every CLI subcommand that makes LLM calls follows a two-phase pattern:

1. **Estimate phase** (default, no `--approve`): the CLI counts how many chunks,
   questions, and batches are involved, estimates input/output tokens, and prints a
   cost table for the configured model. The program then exits. No LLM calls are made.

2. **Execute phase** (`--approve`): the CLI runs the actual work. Cost is tracked per
   call via the telemetry module.

This design prevents accidental large API bills. Always run without `--approve` first
to review the estimate, then re-run with `--approve` when you are ready.

The pricing table in `config/default.yaml` under `cost_pricing` controls the rates used
for estimation. Update it if provider prices change.

---

## 12. Crash Recovery

All long-running operations are designed to resume safely after interruption:

| Operation | Resume mechanism |
|-----------|-----------------|
| `contextualize` | Per-chunk JSON cache in `data/cache/contextualized/`. On restart, any chunk with an existing cache file is skipped. |
| `embed_and_store.py` | Idempotency check: if the collection count matches the expected chunk count, the script exits immediately. Upserts are otherwise idempotent by chunk ID. |
| `embed_contextualized.py` | Same idempotency logic against `rag_contextualized_v1`. |
| `evaluate` | Per-result JSON files in `data/results/eval/`. The runner checks for an existing file before scoring each question/pipeline/metric combination. If the file exists, that combination is skipped. |

If you suspect partial state (e.g., a corrupt file), delete the relevant cache file or
result file and re-run. The system will regenerate only that entry.

---

## 13. Configuration Reference

All parameters live in `config/default.yaml`. Key sections:

| Section | Key parameters |
|---------|---------------|
| `paths` | `data_dir`, `chroma_dir`, `cache_dir`, `results_dir`, `papers_dir` |
| `models` | `embedding_model`, `reranker_model`, `llm_model`, `judge_model` |
| `chunking` | `chunk_size` (450 tokens), `chunk_overlap` (68 tokens) |
| `retrieval` | `top_k_dense` (20), `top_k_bm25` (20), `top_k_fusion` (20), `top_k_rerank` (10), `rrf_k` (60) |
| `llm` | `provider` (claude_cli/anthropic/openai), `temperature` (0.0), `max_tokens` (2048) |
| `evaluation` | `metrics` (4 metrics), `judge_temperature` (0.0) |
| `concurrency` | `max_workers` (4), `stall_timeout_seconds` (120) |

Override any parameter on the CLI using dot-notation:

```bash
# Use Anthropic API with Haiku for judging
python -m src.cli.main evaluate --provider anthropic --model claude-haiku-4-5-20241022 --approve

# Override top-k for retrieval (not yet a named CLI flag — use config override)
# Edit config/default.yaml directly or pass a custom config:
python -m src.cli.main evaluate --config config/my_variant.yaml --approve
```

---

## 14. Timelines

Approximate wall-clock times on an M2 MacBook Pro 16 GB with Apple MPS:

| Step | Time | LLM calls | Notes |
|------|------|-----------|-------|
| PDF extraction | < 1 min | 0 | One-time |
| Chunking | < 30 s | 0 | One-time |
| Embed traditional (266 chunks) | ~1 min | 0 | One-time |
| Contextualize (266 chunks, claude_cli) | 30-60 min | ~62 | One-time; depends on provider latency |
| Embed contextualized (266 chunks) | ~1 min | 0 | One-time |
| Generate 32 questions | 5-10 min | 8 | One-time |
| Evaluate all 4 pipelines x 32 questions | 2-4 hours | 640 | Main experiment |
| Compare (report generation) | < 5 s | 0 | Repeatable |
| Interactive (per question, 1 pipeline) | 5-30 s | 1 | |

The evaluate step is the most expensive. Running a single pipeline at a time
(`--pipelines traditional`) lets you stagger costs and verify output quality before
scaling to all four.

---

## 15. Known Quirks

**claude_cli provider token counts are estimated, not exact.**
The `claude_cli` provider shells out to the `claude` binary and parses stdout. Token
counts are estimated as `len(text) / 4` because the CLI does not expose token counts.
Cost estimates for this provider are approximate. Use `--provider anthropic` if you
need precise token tracking.

**Reranker and embedder are never loaded together.**
The `bge-reranker-v2-m3` model (~1.1 GB on MPS) and `bge-base-en-v1.5` (~416 MB) are
loaded in separate processes to avoid OOM on 16 GB systems. The Modern pipeline loads
the reranker after embedding is complete and releases it before returning.

**Stall detection aborts at 120 seconds without progress.**
The `concurrency.stall_timeout_seconds` setting controls how long a batch operation
can go without any task completing before it raises a stall error. If you are on a slow
network or CPU, increase this value in `config/default.yaml` before running
contextualize or evaluate.

**ChromaDB lock on concurrent processes.**
ChromaDB uses a filesystem lock on `data/chroma_db/`. Do not run two CLI commands that
write to ChromaDB simultaneously. Read-only operations (query, interactive) are safe
to run while evaluate is running, but only if they target different collections.

**extract_papers.py has a hardcoded path.**
`scripts/extract_papers.py` contains a hardcoded absolute path
(`/Users/abhayapat/git-repos/rag-comparison-2/claude_rc_2`). If you clone to a different
location, update the `BASE` variable at the top of that file. The preferred script is
`scripts/extract_papers.py` (project root relative). The copy at `data/papers/extract_pdfs.py`
is an earlier standalone version with the same limitation.

**Interactive mode defaults to Traditional pipeline only.**
Until all four pipelines have been validated end-to-end, `cmd_interactive` defaults
`implemented` to `[PipelineMethod.TRADITIONAL]`. Override with `--pipelines` to select
others.
