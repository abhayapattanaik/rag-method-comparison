# Clarifying Questions — RAG Comparison Project

Status: **In Progress** — answers will be filled in as discussed.

---

## Documents & Data

### Q1: What PDFs?
Specific documents ready, or design for arbitrary PDF input? Approximate count and total pages?

**Answer:**
Reaseach Internet and find top-20 literature on RAG methods. Discuss with the author. Use them as input PDFs.

### Q2: Ground truth
Reference answers available for evaluation questions, or purely LLM-as-judge with no gold standard?

**Answer:**
Look at Q3 for the answer.

### Q3: Question set
Pre-defined question set, ad-hoc user questions, or both?

**Answer:**
Read the PDFs. generate 20 top questions on RAG and find ground truth from PDF documents.

---

## Technology Choices

### Q4: Language
Python assumed. Confirm?

**Answer:**
Python

### Q5: Vector store
Preference? ChromaDB, FAISS, Qdrant, Weaviate? Or should I recommend with tradeoffs?

**Answer:**
ChromeDB

### Q6: Embedding model
Preference? OpenAI `text-embedding-3-small`, local `sentence-transformers`, other? Affects cost and offline capability.

**Answer:**
`BAAI/bge-base-en-v1.5` via `sentence-transformers` + PyTorch MPS. 109M params, ~416MB memory — fits M2 16GB easily. MTEB retrieval ~53-55. Sufficient for comparing retrieval methods (relative differences matter, not absolute quality). MLX considered but dropped — no pre-converted MLX model exists for bge-base, and at 109M params PyTorch runs fine without MLX optimization (MLX mainly helps at 7B+ scale). bge-large-en-v1.5 (335M, MTEB ~63-65) considered but base chosen to reduce memory pressure on 16GB system.

### Q7: Cross-encoder for reranking
Local model (e.g., `cross-encoder/ms-marco-MiniLM`) or API-based reranker (e.g., Cohere)?

**Answer:**
`BAAI/bge-reranker-v2-m3` via `sentence-transformers` + PyTorch MPS. 570M params, 8192-token context (critical for academic chunks). Free, local. MiniLM variants have 512-token limit — too short for academic PDFs.

### Q8: BM25 implementation
`rank_bm25` library, or something else?

**Answer:**
rank_bm25

### Q9: PDF extraction
`PyMuPDF`, `pdfplumber`, `pymupdf4llm`? Each has tradeoffs on table/layout handling.

**Answer:**
`pymupdf4llm` — fast, page metadata built-in, Markdown output for chunking. Sufficient for RAG survey papers (mostly text). Add `Marker` later only if formula-heavy papers need LaTeX extraction.

---

## Design Decisions

### Q10: Shared vs isolated vector stores
Traditional uses raw chunks. Contextual/Hybrid/Modern use contextualized chunks. Hybrid and Modern share same contextualized embeddings. Should Hybrid and Modern share a store, or strict isolation per method?

**Answer:**
Option B — 2 shared ChromaDB collections:
- `rag_traditional_v1` → Traditional only (raw chunks)
- `rag_contextualized_v1` → Contextual, Hybrid, Modern share (contextualized chunks)
Rationale: scientifically strongest (provably identical vectors), cheapest (embed once), simplest. Post-retrieval differences (BM25, reranking) live in application code.

### Q11: "Claude CLI" as backend
Shell out to `claude` CLI tool? Or Claude API via SDK? Clarify scope.

**Answer:**
For now, keep it simple CLI too.
### Q12: Interactive Q&A mode
Terminal-based REPL sufficient? Or web UI? "Side by side" display — rich terminal tables OK?

**Answer:**
Terminal is sufficinet.
---

## Scope & Priorities

### Q13: Build order
Start with Traditional pipeline end-to-end, then layer on others? Or shared infrastructure first, then all 4 in parallel?

**Answer:**
Whatever is efficient.

### Q14: Ollama priority
Need Ollama support from day 1, or add later?

**Answer:**
Later

### Q15: Concurrency
Parallel chunk contextualization and pipeline evaluation? Or sequential fine for v1?

**Answer:**
Find th emost cost efficient method unless it introduces large complexity.

### Q16: RAG literature sourcing
Scope for top-20 RAG papers?

**Answer:**
Academic papers only (arXiv) preferred. Must be freely downloadable (no paywalled journals).

### Q17: Question generation from PDFs
LLM reads PDFs → generates 20 questions → extracts ground truth with page citations. This costs LLM tokens.
- Should this go through same approval gate?
- Review/curate generated questions before using for evaluation?

**Answer:**
Yes, question generation goes through approval gate. Author will review/curate generated questions before using for evaluation.

### Q18: Claude CLI as LLM backend
Shell out to `claude` command, parse stdout. Slower, less structured than API. Limits token tracking. Acceptable tradeoff?

**Answer:**
Yes, shell out to `claude` CLI. Accepted tradeoff — limited token tracking OK.
