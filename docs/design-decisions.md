# Design Decisions Log — RAG Comparison Project

All decisions recorded with rationale and references. Status: living document — update when decisions change or are superseded.

---

## DD-001: Language — Python

**Date:** 2026-05-26
**Status:** Decided

### Decision
Python is the implementation language for all project components.

### Alternatives Considered
- No alternatives evaluated — Python was confirmed directly.

### Rationale
Ecosystem standard for ML/RAG tooling. All target libraries (sentence-transformers, ChromaDB, rank_bm25, pymupdf4llm, LangChain, etc.) have first-class Python support. Author confirmed.

### References
- N/A

---

## DD-002: Vector Store — ChromaDB

**Date:** 2026-05-26
**Status:** Decided

### Decision
ChromaDB as the vector store for all embedding collections.

### Alternatives Considered
- FAISS — lower-level, no built-in metadata filtering, no persistence without extra wrapper
- Qdrant — more production-oriented, heavier operational footprint for a local comparison project
- Weaviate — heavyweight, schema-heavy setup cost

### Rationale
ChromaDB is lightweight, Python-native, supports local persistent storage, and has built-in metadata filtering. Sufficient for a local comparison project with ~420-450 chunks. Minimal operational overhead.

### References
- ChromaDB docs: https://docs.trychroma.com

---

## DD-003: Embedding Model — BAAI/bge-base-en-v1.5 via sentence-transformers + PyTorch MPS

**Date:** 2026-05-26
**Status:** Decided

### Decision
Use `BAAI/bge-base-en-v1.5` loaded via `sentence-transformers`, running on PyTorch with Apple MPS backend on M2 MacBook.

### Alternatives Considered
| Model | Reason Rejected |
|-------|----------------|
| `text-embedding-3-small` (OpenAI API) | Per-call API cost accumulates; requires network; not free local |
| `bge-large-en-v1.5` (335M params, MTEB ~63-65) | Higher quality but larger memory footprint; base chosen to preserve headroom on 16GB M2 |
| `all-MiniLM-L6-v2` | Lower MTEB scores; less suitable for academic/technical retrieval |
| `Cohere embed-english-v3.0` | API cost; external dependency |
| MLX-accelerated inference | No pre-converted MLX model exists for bge-base; MLX optimization primarily benefits 7B+ scale models; PyTorch MPS sufficient at 109M |

### Rationale
- 109M params, ~416MB memory — fits M2 16GB with substantial headroom alongside reranker (570M)
- Free, fully local — no per-embedding API cost for re-indexing experiments
- MTEB retrieval score ~53-55 — sufficient for relative method comparison (goal is comparing retrieval techniques, not maximizing absolute quality)
- MLX dropped because no pre-converted model exists and PyTorch MPS handles 109M without bottleneck
- bge-large considered seriously but base chosen to preserve memory margin for reranker + LLM calls

### References
- MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
- HuggingFace model card: https://huggingface.co/BAAI/bge-base-en-v1.5
- OpenAI embedding pricing: https://openai.com/pricing

---

## DD-004: Cross-Encoder Reranker — BAAI/bge-reranker-v2-m3

**Date:** 2026-05-26
**Status:** Decided

### Decision
Use `BAAI/bge-reranker-v2-m3` as the cross-encoder reranker in the Modern pipeline, loaded via `sentence-transformers` + PyTorch MPS.

### Alternatives Considered
| Model | Reason Rejected |
|-------|----------------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 512-token context limit — too short for academic PDF chunks which routinely exceed 512 tokens |
| `jinaai/jina-reranker-v3` | Evaluated but bge-reranker-v2-m3 preferred for BEIR benchmark performance and local availability |
| `Cohere Rerank 3.5` | API cost; external dependency; adds network latency to each reranking call |

### Rationale
- 8192-token context window — critical for academic paper chunks that can be long (abstracts, dense methodology sections)
- 570M params — fits M2 16GB alongside embedding model (~416MB) with headroom
- Free, fully local — no per-call reranking API cost
- Strong BEIR benchmark performance for passage reranking tasks

### References
- BEIR Benchmark: https://github.com/beir-cellar/beir
- HuggingFace model card: https://huggingface.co/BAAI/bge-reranker-v2-m3
- Reranker comparison overview: https://huggingface.co/blog/llm-reranking

---

## DD-005: BM25 — rank_bm25 Library

**Date:** 2026-05-26
**Status:** Decided

### Decision
Use the `rank_bm25` Python library for BM25 sparse retrieval in the Hybrid and Modern pipelines.

### Alternatives Considered
- Elasticsearch/OpenSearch BM25 — production-grade but requires running a separate service; unnecessary overhead for local comparison project
- Whoosh — pure Python but less actively maintained
- Custom BM25 implementation — unnecessary complexity

### Rationale
`rank_bm25` is lightweight, pure Python, no external services required, and provides standard BM25 and BM25+ implementations. Sufficient for in-memory sparse retrieval over ~420-450 chunks.

### References
- rank_bm25 PyPI: https://pypi.org/project/rank-bm25/
- rank_bm25 GitHub: https://github.com/dorianbrown/rank_bm25

---

## DD-006: PDF Extraction — pymupdf4llm

**Date:** 2026-05-26
**Status:** Decided

### Decision
Use `pymupdf4llm` for PDF text extraction, producing Markdown output with page metadata.

### Alternatives Considered
| Tool | Reason Rejected / Deferred |
|------|---------------------------|
| PyMuPDF (raw) | Lower-level; no built-in Markdown conversion; more manual post-processing |
| pdfplumber | Good for tables but slower; no Markdown output; page metadata less convenient |
| Marker | Higher quality for formula-heavy/complex layouts; slower; deferred as fallback only |
| Docling | Heavier dependency; overkill for mostly-text arXiv survey papers |
| Unstructured.io | Cloud-dependent or heavy local setup; unnecessary for clean arXiv PDFs |

### Rationale
- Fast extraction with minimal setup
- Page metadata built-in — required for page citation tracking in answers
- Markdown output aligns naturally with section-aware chunking strategy
- arXiv RAG survey papers are mostly text (limited tables/formulas) — pymupdf4llm handles this class well
- Marker retained as named fallback for formula-heavy papers if encountered

### References
- pymupdf4llm docs: https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/
- pymupdf4llm PyPI: https://pypi.org/project/pymupdf4llm/

---

## DD-007: Store Layout — 2 Shared ChromaDB Collections

**Date:** 2026-05-26
**Status:** Decided

### Decision
Use 2 shared ChromaDB collections:
- `rag_traditional_v1` — Traditional pipeline only (raw, uncontextualized chunks)
- `rag_contextualized_v1` — Shared by Contextual, Hybrid, and Modern pipelines (LLM-contextualized chunks)

### Alternatives Considered
| Option | Description | Reason Rejected |
|--------|-------------|----------------|
| A — 4 Isolated Collections | One collection per pipeline; each pipeline embeds its own copy | 3x embedding cost (Contextual, Hybrid, Modern embed identical text separately); scientifically weaker (introduces embedding variance from re-runs) |
| B — 2 Shared (chosen) | Traditional isolated; Contextual/Hybrid/Modern share | Chosen — see rationale |
| C — Shared embed + separate collections | Embed once, copy vectors into separate collections | Needless duplication of stored vectors; same embedding cost as Option B but more storage and management overhead |

### Rationale
- **Scientific validity**: Contextual, Hybrid, and Modern pipelines differ only in post-retrieval processing (BM25 fusion, reranking). Sharing the same collection guarantees provably identical vectors — no embedding variance between methods.
- **Cost**: Contextualization LLM calls and embeddings run once, not three times.
- **Simplicity**: Two collections are trivial to manage. Post-retrieval differentiation (RRF fusion, reranking) lives in application code, not data layer.

### References
- ChromaDB collections docs: https://docs.trychroma.com/usage-guide

---

## DD-008: MLX — Not Needed

**Date:** 2026-05-26
**Status:** Decided

### Decision
MLX (Apple's ML framework for Apple Silicon) will not be used. PyTorch with MPS backend is sufficient for all local model inference.

### Alternatives Considered
- MLX inference for embedding model and/or reranker

### Rationale
- MLX optimization primarily targets large model inference (7B+ parameter models) where PyTorch MPS has overhead
- Embedding model (bge-base-en-v1.5, 109M) and reranker (bge-reranker-v2-m3, 570M) are both well within PyTorch MPS capability without bottleneck
- No pre-converted MLX model exists for bge-base-en-v1.5 — MLX requires model conversion, adding setup friction for no measurable gain at this scale
- Project does not perform local LLM inference (LLM calls go to Anthropic API, OpenAI API, or Claude CLI)

### References
- MLX GitHub: https://github.com/ml-explore/mlx
- PyTorch MPS backend docs: https://pytorch.org/docs/stable/notes/mps.html

---

## DD-009: LLM Backends — Anthropic API, OpenAI API, Claude CLI; Ollama Deferred

**Date:** 2026-05-26
**Status:** Decided

### Decision
Support three LLM backends in v1:
1. Anthropic API (via `anthropic` Python SDK)
2. OpenAI API (via `openai` Python SDK)
3. Claude CLI (shell out to `claude` command, parse stdout)

Ollama is deferred to v2.

### Alternatives Considered
- Ollama — local open-source LLM serving; deferred

### Rationale
- **Anthropic API + OpenAI API**: Core backends for the comparison. Full token tracking, structured responses, cost telemetry.
- **Claude CLI**: Simplest path to using Claude without a direct API key setup. Tradeoff accepted — limited token tracking (stdout parsing), slower than SDK. Author confirmed this tradeoff is acceptable.
- **Ollama deferred**: Adds model download management, port/service dependencies, and testing complexity. Not needed for v1 comparison goals. Will be added in v2 if needed.

### References
- Anthropic Python SDK: https://github.com/anthropic/anthropic-sdk-python
- OpenAI Python SDK: https://github.com/openai/openai-python
- Claude CLI docs: https://docs.anthropic.com/en/docs/claude-cli

---

## DD-010: Interactive Mode — Terminal REPL with Rich Tables

**Date:** 2026-05-26
**Status:** Decided

### Decision
Interactive Q&A mode implemented as a terminal REPL. Side-by-side comparison displayed using `rich` library terminal tables.

### Alternatives Considered
- Web UI (e.g., Streamlit, Gradio) — richer display, more development overhead

### Rationale
- Terminal REPL is sufficient for the comparison use case — 4 pipelines, tabular output
- Significantly lower implementation cost than a web UI
- `rich` library provides professional terminal tables, color coding, and progress indicators without a web server
- Author confirmed terminal is sufficient

### References
- rich library docs: https://rich.readthedocs.io

---

## DD-011: Paper Corpus — 8 arXiv Papers (2 Per Retrieval Method)

**Date:** 2026-05-26
**Status:** Decided

### Decision
Use 8 arXiv papers as the document corpus (approximately 2 papers per retrieval method topic area), instead of the initially discussed 20.

### Alternatives Considered
- 20 papers — originally proposed; rejected as unnecessarily large for a 4-method comparison

### Rationale
- 8 papers (~132 pages, ~476K chars) yields ~420-450 chunks — sufficient for meaningful retrieval evaluation across 4 pipelines
- 20 papers would produce ~1,000+ chunks and significantly increase contextualization LLM cost (Contextual/Hybrid/Modern pipelines contextualize every chunk)
- Retrieval method comparison requires relative quality differences, not corpus breadth — 8 well-chosen papers provide that
- Papers are arXiv-only (freely downloadable, no paywall)

### References
- arXiv: https://arxiv.org

---

## DD-012: Cost Estimation — Static Estimate (Phase A) then Sample-Based Refinement (Phase B)

**Date:** 2026-05-26
**Status:** Decided

### Decision
Expensive operations (contextualization, evaluation) use a two-phase cost gate:
- **Phase A (Static)**: Calculate estimated cost from known parameters (chunk count, token estimates, model pricing) before any LLM calls. Gives ±50% order-of-magnitude awareness.
- **Phase B (Sample-based)**: Run on 5 chunks / 1 question, measure actual token usage, extrapolate to full run. Gives accurate cost projection before committing to full spend.

Both phases require `--approve` flag confirmation before proceeding.

### Alternatives Considered
- Static estimate only — insufficient accuracy; risk of 2x cost surprises
- Sample-based only — skips fast preliminary check that can catch obvious misconfigurations

### Rationale
- Static estimate is cheap (pure arithmetic) and catches order-of-magnitude errors (e.g., wrong model selected, chunk count 10x expected)
- Sample-based refinement dramatically improves accuracy before committing full spend — actual prompt templates and model responses can differ significantly from theoretical estimates
- Two-phase approach aligns with build rule #2 (cost gate): estimate first, execute only with `--approve`

### References
- Anthropic pricing: https://www.anthropic.com/pricing
- OpenAI pricing: https://openai.com/pricing

---

## DD-013: Chunk Sizing — 450 Tokens / 68-Token Overlap (15%)

**Date:** 2026-05-26
**Status:** APPROVED

### Decision
Chunk size: 450 tokens. Overlap: 68 tokens (15%). Section-aware splitting recommended (split at section boundaries first, then apply token cap within each section, no overlap across section boundaries).

### Alternatives Considered
| Option | Tokens | % of Max | Reason Rejected |
|--------|--------|----------|----------------|
| Small | 200 | 39% | Splits equations and multi-clause arguments; breaks section coherence; high lost-context risk for academic text |
| Medium (chosen) | 450 | 88% | See rationale |
| Near-max | 480 | 94% | Near embedding model saturation; fact dilution risk; chunks more likely to span pages, degrading citation accuracy |

### Rationale
- **Model fit**: 88% of bge-base-en-v1.5's 512-token maximum — well within the quality window before embedding compression degrades
- **Academic text alignment**: arXiv paragraphs average 120-200 words (~160-270 tokens). 450 tokens captures 1-2 full paragraphs, which is the natural argument unit in academic writing
- **Precision vs context balance**: 200-token chunks split equations and multi-clause arguments; 480-token chunks dilute specific facts and frequently span page boundaries
- **Page citation accuracy**: 450-token chunks rarely span more than 1 page in two-column arXiv format — important for answer citations
- **Overlap at 15% (68 tokens)**: Covers ~2-3 carry-over sentences at chunk boundaries. Industry consensus sweet spot — sufficient continuity without meaningfully ballooning index size or embedding cost
- **Estimated chunk count**: ~380-400 chunks from 8-paper corpus (~476K chars)

### References
- Rethinking Chunk Size for Long-Document Retrieval: https://arxiv.org/abs/2505.21700
- Evaluating Chunking Strategies for RAG: https://arxiv.org/abs/2603.24556
- Systematic Analysis of Chunking for QA: https://arxiv.org/abs/2601.14123
- NVIDIA Technical Blog — Finding Best Chunking Strategy: https://developer.nvidia.com/blog/best-practices-for-using-nvidia-nemo-retrieval-question-answering/
- Milvus RAG Chunking Guide: https://milvus.io/ai-quick-reference/what-is-the-optimal-chunk-size-for-rag

---

### DD-014: Contextualization Optimization — Summary + Local Context + Batch=5
**Date:** 2026-05-26
**Status:** APPROVED

**Decision:** Use two-pass contextualization with batching:
- Pass 1: Summarize each document once (8 LLM calls) → cache summaries
- Pass 2: For each chunk, send doc summary (1.5K) + local context (prev+next chunks, 900) + current chunk (450) + prompt (200) = ~3,050 tokens per chunk, batched 5 at a time

**Alternatives Considered:**

| Strategy | Total Tokens | Savings | Quality Risk | Complexity |
|----------|-------------|---------|-------------|------------|
| Baseline (full doc per chunk) | 2.3M | — | None | Low |
| Batch=5 with full doc | 570K | 75% | Low | Low |
| Doc summary only | 736K | 68% | Medium | Medium |
| Sliding window (local only) | 412K | 82% | High | Very Low |
| **Summary + local + batch=5** | **485K** | **79%** | **Low** | **Medium** |

**Why alternatives were rejected:**

- **Baseline (full doc per chunk):** No optimization. 2.3M tokens, 266 LLM calls, ~60 min. Same document resent for every chunk (e.g., RAG paper sent 44 times). Unacceptable token waste.
- **Batch=5 with full doc:** 75% savings but still resends full document 9× per paper (44 chunks / 5 = 9 batches). 570K tokens is good but not optimal when summary caching eliminates resends entirely.
- **Doc summary only:** 68% savings, but loses local context. Chunk contextualization without neighboring chunks produces vaguer prefixes — e.g., "discusses retrieval" instead of "continues the analysis from the previous section on dense retrieval training."
- **Sliding window (local only):** Best token savings (82%) but drops all document-level awareness. A chunk from the Results section gets no signal that it belongs to a paper about RAG. Fatal for academic papers where section context matters for disambiguation.
- **Summary + local + batch=5 (chosen):** Best balance — 79% savings, zero doc resends (summaries cached), local context preserves positional awareness, batch=5 keeps output quality high. Only medium complexity (two-pass pipeline with caching).

**Rationale:**
- 79% token reduction (2.3M → 485K) — best cost optimization with acceptable quality
- Zero document resends — each doc summarized once, summary cached
- Local context (neighboring chunks) preserves positional awareness
- Batch=5 keeps output quality high (degradation starts at 10+)
- Quality risk is low because: contextualization only adds 1-2 sentence prefix (doesn't need formula-level detail), chunk text itself unchanged, retrieval depends more on embedding than contextualization precision
- Mitigation: author reviews 8 doc summaries before Pass 2 proceeds

**References:**
- Internal analysis comparing 5 optimization strategies for chunk contextualization
- Anthropic contextual retrieval blog post (concept reference)
