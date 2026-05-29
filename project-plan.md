# RAG Comparison Project — Implementation Plan

Status: **Phase 4 — Ready to embed contextualized chunks**

---

## Phases

### Phase 0: Foundation (No LLM calls)
- **Goal:** Skeleton project — config, cost gate, telemetry, LLM providers, CLI skeleton
- **Produces:** default.yaml, config loader (Pydantic), cost gate (estimate/approve), telemetry tracker, LLM provider interface + 3 implementations (Anthropic, OpenAI, Claude CLI), CLI with subcommands
- **Complexity:** Medium | **LLM Cost:** ~0 (1 test call)
- **Author approval:** Module design, directory structure, config schema
- **Validation:**
  - [ ] `python -m src.cli.main --help` prints all subcommands
  - [ ] Config loads from YAML, CLI overrides work
  - [ ] Cost gate: running without `--approve` prints estimate, exits
  - [ ] LLM provider test call succeeds, telemetry records tokens
- **Status: COMPLETE**
- **Results:** All 6 modules created. Config loads, CLI shows 6 subcommands, cost gate blocks without --approve. LLM provider test call succeeds.

### Phase 1: PDF Ingestion (No LLM calls)
- **Goal:** Download papers, extract text, chunk, embed into `rag_traditional_v1`
- **Produces:** PDF extractor (pymupdf4llm), chunker (configurable size/overlap), embedder (bge-base-en-v1.5 on MPS), ChromaDB store wrapper, 20 arXiv papers downloaded
- **Complexity:** Medium | **LLM Cost:** 0
- **Author approval:** Paper list (top 20 RAG papers), chunk size/overlap
- **Sub-steps:**
  - [ ] 1a: Research + propose top 20 arXiv RAG papers → author approves
  - [ ] 1b: Download PDFs, extract to markdown, verify quality
  - [ ] 1c: Research chunk sizing for bge-base-en-v1.5 (max 512 tokens) → author approves
  - [ ] 1d: Chunk all documents, log counts
  - [ ] 1e: Embed + store in `rag_traditional_v1`, verify counts
- **Validation:**
  - [ ] All 20 PDFs extracted successfully
  - [ ] Chunk counts reasonable (~50-150 per paper)
  - [ ] ChromaDB collection count matches total chunks
  - [ ] Similarity query returns relevant chunks (spot-check)
  - [ ] Re-run does not duplicate data (idempotency)
- **Dependencies:** Phase 0
- **Status: COMPLETE**
- **Results:** 8 arXiv papers downloaded + extracted (132 pages, 476K chars). 266 chunks at 450 tokens / 68 overlap. ChromaDB `rag_traditional_v1` populated with 266 documents. Embedding took 55s on MPS.

### Phase 2: Traditional Pipeline E2E (Critical Gate)
- **Goal:** Simplest complete pipeline — question in, cited answer out
- **Produces:** Dense retrieval module, pipeline orchestrator, answer generation prompt with citations, minimal REPL
- **Complexity:** Low-Medium | **LLM Cost:** ~1 question
- **Author approval:** Pipeline design
- **Validation:**
  - [ ] 1 question → coherent answer with source citations (doc + page)
  - [ ] Telemetry shows token usage and cost
  - [ ] Cost gate works (estimate without `--approve`)
- **Dependencies:** Phase 0 + Phase 1
- **Status: COMPLETE**
- **Results:** Traditional pipeline works end-to-end. Test question 'What is RAG?' returned coherent answer with source citations. Latency 18.4s. Claude CLI provider updated to use --output-format json for real token tracking. Results persisted to data/results/queries/.

### Phase 3: Contextualization (LLM-heavy)
- **Goal:** LLM-contextualize all chunks, embed into `rag_contextualized_v1`
- **Produces:** Contextualizer with caching (JSON per chunk_id), contextualized ChromaDB collection
- **Complexity:** High | **LLM Cost:** $$$ (all chunks × LLM call each)
- **Author approval:** Quality review after 5-sample trial
- **Cost control:**
  - [ ] Cost estimate before starting
  - [ ] Run on 5 chunks first → author reviews quality → approve full run
  - [ ] Progress bar + ETA
  - [ ] Stall detection: abort if no chunk completes in N seconds
- **Validation:**
  - [ ] Contextualized chunks visibly improve on raw (5-sample review)
  - [ ] Full collection populated, count matches `rag_traditional_v1`
  - [ ] Cache complete and resumable (kill-restart test)
  - [ ] Actual cost within 20% of estimate
- **Dependencies:** Phase 0 + Phase 1
- **Status: COMPLETE**
- **Results:** Two-pass contextualization with batch=5 optimization (DD-014). Pass 1: 8 doc summaries cached. Pass 2: 39 batches, 266 chunks contextualized. Actual tokens: ~897K input / ~10.9K output. Cost: ~$3.30. Time: 7 min. Zero errors. Saved to data/cache/contextualized_chunks.json.

### Phase 4: Contextual Pipeline
- **Goal:** Second pipeline — dense retrieval from contextualized collection
- **Produces:** Contextual pipeline in orchestrator, REPL supports Traditional + Contextual
- **Complexity:** Low | **LLM Cost:** ~1 question
- **Validation:**
  - [ ] Same question to both pipelines produces answers
  - [ ] Contextual retrieves different chunks than Traditional
  - [ ] Telemetry tracks pipelines separately
- **Dependencies:** Phase 2 + Phase 3
- **Status: READY**

### Phase 5: Hybrid Pipeline (BM25 + RRF)
- **Goal:** Third pipeline — dense + BM25 with RRF fusion
- **Produces:** BM25 retrieval module (rank_bm25), RRF fusion module, Hybrid pipeline
- **Complexity:** Medium | **LLM Cost:** ~1 question
- **Validation:**
  - [ ] BM25 returns results for keyword-heavy queries
  - [ ] RRF merges dense + BM25 correctly (spot-check ranks)
  - [ ] Hybrid pipeline returns cited answers
  - [ ] 3 pipelines produce different retrieval sets for same question
- **Dependencies:** Phase 4
- **Status:** Not started

### Phase 6: Modern Pipeline (+ Reranker)
- **Goal:** Fourth pipeline — adds cross-encoder reranking after RRF
- **Produces:** Reranker module (bge-reranker-v2-m3 on MPS), Modern pipeline
- **Complexity:** Medium | **LLM Cost:** ~1 question
- **Validation:**
  - [ ] Reranker loads, scores query-chunk pairs
  - [ ] Modern reorders chunks differently from Hybrid
  - [ ] All 4 pipelines callable from REPL
  - [ ] Reranker latency measured
- **Dependencies:** Phase 5
- **Status:** Not started

### Phase 7: Question Generation (Parallel with 3-6)
- **Goal:** Generate 20 evaluation questions + ground truth from papers
- **Produces:** question_gen module, `data/questions.json` with questions + ground truth + page citations
- **Complexity:** Medium | **LLM Cost:** $$
- **Author approval:** Curate final question set from candidates
- **Process:**
  - [ ] Estimate cost → approval gate
  - [ ] Generate ~30 candidate questions
  - [ ] Author reviews, selects 20
  - [ ] Ground truth answers stored with citations
- **Validation:**
  - [ ] 20 questions covering diverse RAG topics
  - [ ] Each has ground truth with page citations
  - [ ] All answerable from corpus
- **Dependencies:** Phase 1 (extracted papers only)
- **Status:** Not started

### Phase 8: Evaluation (LLM-as-judge)
- **Goal:** Score all 4 pipelines on 4 metrics × 20 questions
- **Produces:** Judge module (4 metric prompts), evaluation runner (incremental persistence), comparison module
- **Complexity:** High | **LLM Cost:** $$$ (320 judge calls + 80 answer calls)
- **Author approval:** Judge prompts, 1-question validation before full run
- **Cost control:**
  - [ ] Full cost estimate (20 × 4 × 4 = 320 judge + 80 generation)
  - [ ] Run 1 question × 1 pipeline × 4 metrics first → validate structure
  - [ ] Incremental persistence to `data/results/` (resumable)
- **Validation:**
  - [ ] Single question produces 4 scores in [0.0, 1.0] with justifications
  - [ ] Full evaluation completes
  - [ ] Results persist incrementally (kill-resume test)
  - [ ] Comparison table shows meaningful differences
  - [ ] Total cost matches estimate
- **Dependencies:** Phase 6 + Phase 7
- **Status:** Not started

### Phase 9: REPL + Reports
- **Goal:** Polish interactive Q&A, generate all output documents
- **Produces:** Full REPL (rich terminal tables, side-by-side 4-method answers), 6 output docs
- **Complexity:** Medium | **LLM Cost:** Minimal
- **Output documents:**
  - [ ] docs/architecture.md
  - [ ] docs/design-decisions.md
  - [ ] docs/test-plan.md
  - [ ] docs/comparison-analysis.md
  - [ ] docs/cost-quality-tradeoff.md
  - [ ] docs/rag-glossary.md
- **Validation:**
  - [ ] REPL handles arbitrary questions, displays 4 answers side-by-side
  - [ ] Comparison report complete with per-metric, per-pipeline breakdowns
  - [ ] All 6 documents produced
- **Dependencies:** Phase 8
- **Status:** Not started

---

## Parallelization Map

```
Phase 0 → Phase 1 → Phase 2 (critical gate)
                         |
                         ├→ Phase 3 → Phase 4 → Phase 5 → Phase 6 ─┐
                         |                                           |
                         └→ Phase 7 (parallel with 3-6) ────────────┤
                                                                     |
                                                                  Phase 8 → Phase 9
```

---

## Architecture Decisions (Confirmed)

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python | — |
| Vector store | ChromaDB (2 collections) | Scientific validity, cost efficiency |
| Embedding | bge-base-en-v1.5 + sentence-transformers + MPS | 109M params, free local, fits M2 16GB |
| Reranker | bge-reranker-v2-m3 + MPS | 570M params, 8K context, free local |
| BM25 | rank_bm25 | — |
| PDF extraction | pymupdf4llm | Fast, page metadata, Markdown output |
| Store layout | `rag_traditional_v1` (raw) + `rag_contextualized_v1` (shared) | Identical vectors guaranteed, embed once |
| LLM backends | Anthropic API, OpenAI API, Claude CLI | Ollama deferred |
| Interactive | Terminal REPL with rich tables | — |
| MPS memory | Load models on demand, not simultaneously | ~1.5GB combined, 16GB total |

---

## Directory Structure (Proposed — Needs Author Approval)

```
claude_rc_2/
  config/
    default.yaml
  src/
    __init__.py
    config.py                  # Pydantic config loader (YAML + CLI)
    cost_gate.py               # Estimate/approve pattern
    telemetry.py               # Token usage, latency, cost tracking
    llm/
      __init__.py
      base.py                  # Abstract LLM provider
      anthropic_provider.py
      openai_provider.py
      claude_cli_provider.py
    ingestion/
      __init__.py
      pdf_extractor.py         # pymupdf4llm extraction
      chunker.py               # Document-level chunking
      contextualizer.py        # LLM chunk contextualization + cache
      embedder.py              # bge-base-en-v1.5 via sentence-transformers
      store.py                 # ChromaDB collection management
    retrieval/
      __init__.py
      dense.py                 # Dense retrieval from ChromaDB
      bm25.py                  # BM25 via rank_bm25
      fusion.py                # Reciprocal Rank Fusion
      reranker.py              # bge-reranker-v2-m3 cross-encoder
      pipeline.py              # Pipeline orchestrator (4 methods)
    evaluation/
      __init__.py
      judge.py                 # LLM-as-judge (4 metrics)
      question_gen.py          # Question + ground truth generation
      runner.py                # Evaluation runner (incremental persistence)
      comparison.py            # Cross-method comparison + side-by-side
    interactive/
      __init__.py
      repl.py                  # Terminal REPL
    cli/
      __init__.py
      main.py                  # argparse with subcommands
  data/
    papers/                    # arXiv PDFs
    chroma_db/                 # ChromaDB persistent storage
    cache/                     # Contextualization cache
    results/                   # Evaluation results (JSON)
  docs/                        # 6 output documents
  tests/
```

---

## Actual vs Estimated

| Phase | Estimated Tokens | Actual Tokens | Estimated Time | Actual Time |
|-------|-----------------|---------------|----------------|-------------|
| Phase 2 (1 query) | 4,700 | ~4,700 (CLI estimate) | 10-15s | 18.4s |
| Phase 3 (contextualize) | 336K input / 39K output | 897K input / 10.9K output | 12-15 min | 7 min |

Note: Phase 3 actual input higher than estimate because Claude CLI includes system prompt + cache tokens in reported input_tokens. Output lower because contextualization prefixes averaged ~280 tokens per batch (56/chunk) vs estimated 100/chunk.

---

## Risk Mitigation

- **PDF extraction:** Test quality in Phase 1b before committing to chunk sizing. Marker fallback for formula-heavy papers.
- **Contextualization cost:** 5-chunk trial catches bad prompts before full spend.
- **MPS memory:** Load embedding + reranker on demand, not simultaneously.
- **ChromaDB idempotency:** Deterministic chunk IDs (hash of source_file + chunk_index + config version).
- **Stall detection:** Timeout on all long-running LLM operations.

---

## Pending Author Decisions

- [ ] Approve directory structure above
- [ ] Approve CLI subcommand structure: `ingest`, `contextualize`, `generate-questions`, `evaluate`, `interactive`, `compare`
- [ ] Approve config approach (Pydantic + YAML + CLI overrides)
