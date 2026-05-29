# RAG Comparison System

A controlled experiment comparing four RAG retrieval methods on answer quality, token usage,
and latency. Each method builds on the previous one, adding exactly one technique at a time,
so the contribution of each component can be isolated and measured.

Corpus: 20 arXiv RAG research papers.
Evaluation: LLM-as-judge scoring on 4 metrics across 32 questions per pipeline.

---

## The 4 Retrieval Methods

```
Traditional
-----------
PDF --> Extract --> Chunk --> Embed --> Dense Search --> LLM Answer

Contextual
----------
PDF --> Extract --> Chunk --> LLM Contextualize --> Embed --> Dense Search --> LLM Answer

Hybrid
------
PDF --> Extract --> Chunk --> LLM Contextualize --> Embed --> Dense Search --|
                                                                             |--> RRF --> LLM Answer
                                                             BM25 Search ----|

Modern
------
PDF --> Extract --> Chunk --> LLM Contextualize --> Embed --> Dense Search --|
                                                                             |--> RRF --> Rerank --> LLM Answer
                                                             BM25 Search ----|
```

| Method | New Technique Added | Collection |
|--------|---------------------|------------|
| Traditional | Baseline: dense retrieval only | `rag_traditional_v1` |
| Contextual | LLM-enriched chunk text before embedding | `rag_contextualized_v1` |
| Hybrid | BM25 sparse retrieval + RRF fusion | `rag_contextualized_v1` |
| Modern | Cross-encoder reranking after RRF | `rag_contextualized_v1` |

Contextual, Hybrid, and Modern share one ChromaDB collection. Their differences are in
post-retrieval processing, not stored data.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set LLM provider credentials (or use claude_cli — no key needed)
export ANTHROPIC_API_KEY=sk-...
# or
export OPENAI_API_KEY=sk-...

# 3. Put arXiv PDFs in data/papers/
#    See DATA_SETUP.md for the paper list and download instructions.

# 4. Ingest (estimate cost first, then approve)
python -m src.cli.main ingest
python -m src.cli.main ingest --approve

# 5. Contextualize (LLM enriches each chunk — sample 5 first to review quality)
python -m src.cli.main contextualize --sample 5 --approve
python -m src.cli.main contextualize --approve

# 6. Generate evaluation questions (produces data/questions.json)
python -m src.cli.main generate-questions --approve --count 30

# 7. Run evaluation
python -m src.cli.main evaluate --approve

# 8. View comparison table
python -m src.cli.main compare

# 9. Interactive Q&A (side-by-side answers from all pipelines)
python -m src.cli.main interactive --approve

# 10. Single non-interactive query
python -m src.cli.main query --question "What is RAG?" --approve
```

Every command without `--approve` prints a token/cost estimate and exits without spending any
LLM calls. See [GETTING_STARTED.md](GETTING_STARTED.md) for a step-by-step walkthrough.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | |
| PyTorch | 2.0+ | MPS backend used on Apple Silicon |
| Embedding model | BAAI/bge-base-en-v1.5 | ~416 MB, auto-downloaded by sentence-transformers |
| Reranker model | BAAI/bge-reranker-v2-m3 | ~1.1 GB, auto-downloaded, used by Modern pipeline only |
| LLM provider | Anthropic, OpenAI, or claude CLI | Configured via `config/default.yaml` |
| ChromaDB | 0.5+ | Persistent local storage |

Models are downloaded on first use. On a 16 GB Apple M2, the embedding model and reranker
are never loaded simultaneously — each is loaded on demand and unloaded after use.

---

## File Tree

```
claude_rc_2/
  config/
    default.yaml              All tunable parameters (models, paths, chunk size, top-k, pricing)
  src/
    config.py                 Pydantic config loader — YAML + CLI overrides
    cost_gate.py              Estimate/approve pattern for every LLM operation
    telemetry.py              Token usage, latency, and cost tracking per LLM call
    logging_config.py         Structured logging setup (file + console)
    llm/
      base.py                 Abstract LLM provider + LLMResponse dataclass
      anthropic_provider.py   Anthropic SDK implementation
      openai_provider.py      OpenAI SDK implementation
      claude_cli_provider.py  Subprocess to `claude` CLI
    ingestion/
      pdf_extractor.py        pymupdf4llm — PDF to Markdown with page metadata
      chunker.py              Document-level chunking (configurable size/overlap)
      contextualizer.py       LLM chunk contextualization + per-chunk JSON cache
      embedder.py             bge-base-en-v1.5 via sentence-transformers + MPS
      store.py                ChromaDB collection management (create, upsert, query)
    retrieval/
      dense.py                Vector similarity search against ChromaDB
      bm25.py                 BM25 sparse retrieval via rank_bm25
      fusion.py               Reciprocal Rank Fusion (RRF) — merges dense + BM25
      reranker.py             bge-reranker-v2-m3 cross-encoder reranking
      pipeline.py             Pipeline orchestrator — routes queries through each method
    evaluation/
      judge.py                LLM-as-judge — 4-metric scoring prompts
      question_gen.py         Question + ground-truth generation from papers
      runner.py               Evaluation runner — incremental persistence per result
      comparison.py           Cross-method comparison tables and summaries
    interactive/
      repl.py                 Terminal REPL — side-by-side answers with citations
    cli/
      main.py                 argparse entry point with subcommands
  data/
    papers/                   Downloaded arXiv PDFs (20 papers)
    chroma_db/                ChromaDB persistent storage (2 collections)
    cache/                    Contextualization cache (per-chunk JSON files)
    results/                  Evaluation results (per question-pipeline-metric JSON)
    questions.json            Generated questions + ground truth
  docs/
    architecture.md           System design with component responsibilities
    design-decisions.md       Key decisions made during implementation
    test-plan.md              Verification strategy and acceptance criteria
    evaluation-questions.md   The 32 evaluation questions and ground truth
    comparison-analysis.md    Findings from running all 4 methods
    cost-quality-tradeoff.md  Which method gives best quality per token spent
    rag-glossary.md           Key RAG terms (dense retrieval, RRF, cross-encoder, BM25, etc.)
    chunk-sizing-research.md  Chunk size research and rationale for chosen parameters
  tests/                      Unit and integration tests
  requirements.txt
  project-spec.md             Full build specification
  project-plan.md             Implementation phases
```

---

## Methods Explained

### Traditional

The baseline. Chunks are embedded as-is using `BAAI/bge-base-en-v1.5`. Queries are embedded
the same way. ChromaDB cosine similarity returns the top-K most similar chunks, which are
passed to the LLM to generate an answer.

Limitation: embeddings encode only the local chunk text, with no signal about what paper the
chunk comes from or how it fits the broader argument.

### Contextual

Before embedding, each chunk is sent to an LLM which prepends a brief document-level context:
what the paper is about and where this chunk fits within it. The contextualized text is then
embedded and stored in `rag_contextualized_v1`.

Contextualization is cached per chunk (one JSON file per `chunk_id`). Re-running is safe —
already-contextualized chunks are skipped.

Trade-off: one LLM call per chunk at ingestion time. Retrieval cost at query time is identical
to Traditional.

### Hybrid

Runs both dense retrieval (same as Contextual) and BM25 keyword search over the same
contextualized chunks. The two ranked lists are merged using Reciprocal Rank Fusion (RRF),
which combines rank positions without requiring score normalization.

RRF score for chunk d = sum over all lists of: 1 / (k + rank(d))

Default k = 60. Dense and BM25 each return top-20 candidates; RRF merges them to top-20.

Advantage over Contextual: keyword-heavy queries (specific terms, paper names, acronyms)
get boosted by BM25 even when the dense embedding similarity is mediocre.

### Modern

Identical to Hybrid up through RRF. The RRF-merged set is then re-scored by
`BAAI/bge-reranker-v2-m3`, a cross-encoder that performs full joint attention over the query
and each chunk. Unlike bi-encoder embeddings (which encode query and chunk independently),
the cross-encoder sees both together and scores relevance more accurately.

Only the top-10 reranked chunks are passed to the LLM.

Trade-off: the reranker adds ~200-400 ms per query on Apple M2 MPS. The model (~1.1 GB) is
loaded on demand and unloaded after use to avoid simultaneous MPS pressure with the embedder.

---

## Design Principles

**Scientific control.** Only the retrieval technique varies between pipelines. Embedding model,
chunk size, LLM, judge model, and prompt templates are identical across all four methods.

**Cost gate.** Every CLI subcommand that makes LLM calls prints a token/cost estimate first
and exits. Pass `--approve` to execute. There is no way to accidentally trigger large LLM
spend.

**Idempotent operations.** Chunk IDs are deterministic hashes. ChromaDB uses `upsert`. The
contextualization cache and evaluation results are stored as individual JSON files keyed by
chunk ID or `question_id_pipeline_metric`. Re-running any step is safe.

**Crash recovery.** Long batch operations (contextualization, evaluation) check existing cache
files before each item and skip completed work. An interrupted run resumes where it left off.

**Telemetry.** Every LLM call records input tokens, output tokens, latency, and cost. Per-pipeline
token breakdowns are available at the end of every `evaluate` run and in
`data/results/telemetry_evaluate.json`.

**Configurable.** All parameters — chunk size, top-k values, model names, pricing — live in
`config/default.yaml`. CLI flags override YAML. Nothing is hardcoded.

---

## CLI Reference

```
python -m src.cli.main <subcommand> [OPTIONS]

Subcommands:
  ingest              Ingest PDFs into rag_traditional_v1 ChromaDB collection
  contextualize       LLM-contextualize chunks into rag_contextualized_v1
  generate-questions  Generate evaluation questions from extracted papers
  evaluate            Run all 4 pipelines x 32 questions x 4 judge metrics
  interactive         Start REPL: type questions, see side-by-side answers
  query               Single non-interactive question through selected pipeline(s)
  compare             Print comparison table from completed evaluation results

Common flags (most subcommands):
  --config PATH       YAML config file (default: config/default.yaml)
  --approve           Execute (without this flag: estimate only, then exit)
  --provider TEXT     anthropic | openai | claude_cli
  --model TEXT        Override llm_model in config
  --pipelines TEXT    Comma-separated: traditional,contextual,hybrid,modern
```

Key overrides:

```bash
# Use OpenAI for contextualization instead of default
python -m src.cli.main contextualize --approve --provider openai --model gpt-4o-mini

# Evaluate only the traditional pipeline on specific questions
python -m src.cli.main evaluate --approve --pipelines traditional --questions q1,q5,q10

# Export comparison as Markdown
python -m src.cli.main compare --output-format md > docs/comparison-analysis.md
```

---

## Configuration

All parameters are in `config/default.yaml`. Key sections:

```yaml
models:
  embedding_model: "BAAI/bge-base-en-v1.5"   # local, MPS, ~416 MB
  reranker_model: "BAAI/bge-reranker-v2-m3"  # local, MPS, ~1.1 GB (Modern only)
  llm_model: "claude-sonnet-4-20250514"       # answer generation
  judge_model: "claude-haiku-4-5-20241022"    # evaluation scoring

chunking:
  chunk_size: 450      # tokens (max 512 for bge-base-en-v1.5)
  chunk_overlap: 68    # tokens

retrieval:
  top_k_dense: 20
  top_k_bm25: 20
  top_k_fusion: 20
  top_k_rerank: 10     # final chunks passed to LLM (Modern pipeline)
  rrf_k: 60

llm:
  provider: "claude_cli"   # default; no API key needed
```

---

## Evaluation Metrics

Each pipeline answer is scored 0.0-1.0 on four dimensions by an LLM judge
(default: `claude-haiku-4-5-20241022`):

| Metric | Definition |
|--------|------------|
| Context Precision | Are the retrieved chunks relevant to the question? |
| Context Recall | Do the retrieved chunks cover the ground truth? |
| Faithfulness | Is the answer grounded in the retrieved chunks (not hallucinated)? |
| Answer Relevancy | Does the answer directly address the question asked? |

Total judge calls for a full run: 32 questions x 4 pipelines x 4 metrics = 512.
Each result is persisted immediately as `data/results/eval/{qid}_{pipeline}_{metric}.json`.

---

## Further Reading

- [GETTING_STARTED.md](GETTING_STARTED.md) — step-by-step first-run walkthrough
- [DATA_SETUP.md](DATA_SETUP.md) — paper list, download instructions, directory layout
- [docs/evaluation-questions.md](docs/evaluation-questions.md) — the 32 evaluation questions
- [docs/architecture.md](docs/architecture.md) — full system design with data flow diagrams
- [docs/design-decisions.md](docs/design-decisions.md) — rationale for key technology choices
- [docs/comparison-analysis.md](docs/comparison-analysis.md) — results from running all 4 methods
- [docs/cost-quality-tradeoff.md](docs/cost-quality-tradeoff.md) — quality per token analysis
- [docs/rag-glossary.md](docs/rag-glossary.md) — glossary of RAG concepts used in this project
