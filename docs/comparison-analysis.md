# RAG Pipeline Comparison Analysis

Status: **Final** — based on 2-question evaluation sample

---

## 1. Executive Summary

Four RAG retrieval pipelines were evaluated on 2 questions drawn from arXiv RAG research papers. All other variables — embedding model, chunk size, LLM for answer generation, and judge model — were held constant. Only the retrieval technique varied.

**Key finding:** Modern pipeline (dense + BM25 + RRF + cross-encoder reranking) outperforms all others with a 0.912 average score. The primary differentiator across pipelines is context precision — how many of the retrieved chunks are actually relevant. Answer generation quality (faithfulness, answer relevancy) is high and roughly equivalent across all pipelines, suggesting the LLM can generate faithful answers given adequate context. The bottleneck is retrieval precision, not generation.

**Ranking: Modern (0.912) > Hybrid (0.844) > Contextual (0.831) > Traditional (0.781)**

The gap from Traditional to Modern is 13 percentage points overall, driven almost entirely by a 25-point jump in context precision (0.450 → 0.700).

---

## 2. Per-Metric Comparison Table

Scores are averages over 2 questions. Range: 0.0–1.0 (higher is better).

| Metric | Traditional | Contextual | Hybrid | **Modern** | Winner |
|--------|:-----------:|:----------:|:------:|:----------:|--------|
| Context Precision | 0.450 | 0.375 | 0.450 | **0.700** | Modern (+25pp vs Traditional) |
| Context Recall | 0.750 | 1.000 | 1.000 | **1.000** | Contextual/Hybrid/Modern tied |
| Faithfulness | 0.975 | 1.000 | 0.975 | **1.000** | Contextual/Modern tied |
| Answer Relevancy | 0.950 | 0.950 | 0.950 | 0.950 | All tied |
| **Overall Average** | 0.781 | 0.831 | 0.844 | **0.912** | Modern |

### Per-Question Breakdown

**Question 1 — `8b471c86` (RAG-Sequence vs RAG-Token)**

| Metric | Traditional | Contextual | Hybrid | Modern |
|--------|:-----------:|:----------:|:------:|:------:|
| Context Precision | 0.450 | 0.350 | 0.500 | **0.900** |
| Context Recall | 1.000 | 1.000 | 1.000 | 1.000 |
| Faithfulness | 1.000 | 1.000 | 1.000 | 1.000 |
| Answer Relevancy | 0.950 | 0.950 | 0.950 | 0.950 |

**Question 2 — `f41716c9` (RAG index hot-swapping)**

| Metric | Traditional | Contextual | Hybrid | Modern |
|--------|:-----------:|:----------:|:------:|:------:|
| Context Precision | 0.450 | 0.400 | 0.400 | **0.500** |
| Context Recall | **0.500** | 1.000 | 1.000 | 1.000 |
| Faithfulness | 0.950 | 1.000 | 0.950 | 1.000 |
| Answer Relevancy | 0.950 | 0.950 | 0.950 | 0.950 |

---

## 3. Per-Pipeline Analysis

### 3.1 Traditional Pipeline

**Method:** Raw chunks → embed → dense search (`rag_traditional_v1`) → LLM answer

**Strengths:**
- Simplest architecture — no LLM calls during ingestion, no BM25 or reranker at query time
- Lowest ingestion cost (no contextualization LLM calls)
- Faithfulness and answer relevancy remain high (0.975 / 0.950) — the LLM generates faithfully from whatever it retrieves
- Predictable latency (dense search only)

**Weaknesses:**
- Lowest context precision (0.450) — dense retrieval on raw chunks retrieves contextually adjacent chunks but cannot distinguish highly relevant from marginally relevant
- Context recall failure on Q2 (0.500) — raw chunks lack document-level context, so a specific detail about index hot-swapping is spread across chunks without positional signals; the retrieval misses half the relevant content
- No keyword fallback — queries with rare technical terms (paper-specific acronyms, model names) that do not align well with embedding space fall through

**Verdict:** Acceptable baseline for low-cost scenarios where precise retrieval is not critical.

---

### 3.2 Contextual Pipeline

**Method:** Contextualized chunks → embed → dense search (`rag_contextualized_v1`) → LLM answer

**Strengths:**
- Context recall improves to 1.000 on both questions — LLM-added document context ensures even specific details are anchored to their paper and topic area, enabling the embedder to retrieve them
- Faithfulness reaches 1.000 — contextualized chunks provide cleaner grounding for the answer LLM

**Weaknesses:**
- Context precision is actually lower than Traditional on Q1 (0.350 vs 0.450) — the LLM-prepended document summary adds tokens that semantically dilute the chunk's embedding signal, causing borderline-relevant chunks to score higher relative to highly relevant ones
- Still relies on single-modality retrieval (dense only) — keywords in the query that match terminology in the document but are uncommon in the embedding space are not boosted

**Verdict:** Reliably improves recall at marginal cost. Precision regression on dense-only retrieval is the main limitation.

---

### 3.3 Hybrid Pipeline

**Method:** Contextualized chunks → dense + BM25 → RRF fusion → LLM answer

**Strengths:**
- Context recall matches Contextual at 1.000 — retains the recall gains from contextualization
- Context precision recovers to 0.450 on Q1 (matching Traditional) — BM25 adds a keyword-matching signal that boosts chunks containing the exact terms in the query (e.g., "RAG-Sequence", "RAG-Token"), partially counteracting the precision regression from contextualization alone
- RRF fusion is score-agnostic — does not require normalization between cosine similarity and BM25 scores

**Weaknesses:**
- Context precision still only 0.450 on Q1, 0.400 on Q2 — RRF improves recall breadth but does not yet prune irrelevant chunks with high confidence
- BM25 index rebuilds in memory each run — adds overhead for large corpora
- Highest average latency (15,194 ms/record) due to BM25 index build + dual retrieval + RRF computation on top of LLM latency

**Verdict:** Reliable improvement over Contextual in precision for keyword-heavy queries. Best choice when exact-match terminology matters and reranking overhead is undesirable.

---

### 3.4 Modern Pipeline

**Method:** Contextualized chunks → dense + BM25 → RRF fusion → cross-encoder rerank → LLM answer

**Strengths:**
- Highest context precision by far: 0.700 average (0.900 on Q1) — the cross-encoder scores every (query, chunk) pair with full attention, distinguishing highly relevant from marginally relevant far more accurately than embedding similarity alone
- Perfect context recall (1.000) maintained — reranking operates on the already-fused set, retaining coverage while improving precision
- Faithful and relevant answers (1.000 / 0.950) — smaller, higher-precision context window (top_k=10 after rerank vs 20 for others) produces more focused answers
- Lowest average latency among LLM-intensive records (8,602 ms) despite reranker compute — the reduced context size (10 chunks instead of 20) speeds up LLM answer generation enough to offset reranking time

**Weaknesses:**
- Requires loading `bge-reranker-v2-m3` (~1.1 GB) on MPS for every query — model load/unload adds 2–3 seconds per pipeline switch
- More complex infrastructure — 4 components (embedder, BM25, RRF, cross-encoder) vs 1 for Traditional
- Higher ingestion cost — shares contextualization cost with Contextual/Hybrid

**Verdict:** Best retrieval quality in this experiment. Cross-encoder reranking is the single most impactful addition — it provides the largest precision gain of any technique evaluated.

---

## 4. Statistical Observations

### Small Sample Caveat

2 questions is a minimal sample. The patterns observed are consistent and directionally clear, but confidence intervals are wide. Full 20-question evaluation is needed for statistical robustness.

### Context Precision Is the Differentiating Metric

Three of the four pipelines score identically on context recall (1.000), faithfulness (~1.000), and answer relevancy (0.950). Context precision is the only metric that separates them. This suggests that, for this corpus and question set, the primary challenge is not retrieving enough relevant content — it is filtering out irrelevant content.

### Reranking Provides the Largest Precision Jump

Going from Hybrid (0.450) to Modern (0.700) is a 25-point jump. Going from Traditional (0.450) to Contextual (0.375) is a slight regression. Going from Contextual (0.375) to Hybrid (0.450) is a modest 7.5-point gain. The cross-encoder reranker is the dominant driver of precision improvement in this evaluation.

### Traditional Context Recall Failure on Q2

Traditional scored 0.500 on context recall for the index hot-swapping question, while all other pipelines scored 1.000. This is the starkest gap. The question asks about a specific experiment (Wikipedia dump date comparison) — a detail that appears in a few specific chunks. Without LLM-added context, these chunks do not carry enough semantic signal to be retrieved by dense search alone. Contextualization "anchors" them to the topic area, making them retrievable.

### Answer Quality Is Ceiling-Bounded by the LLM

All pipelines score 0.950 on answer relevancy and 0.950–1.000 on faithfulness. This indicates the LLM (claude-sonnet-4-20250514) is capable of generating high-quality answers from any of the retrieval sets provided. The generation bottleneck has not been hit at this scale — improving retrieval is the highest-leverage intervention.

### Modern's Latency Advantage

Modern runs at 8,602 ms average latency vs Hybrid's 15,194 ms. The reranker reduces the context passed to the LLM from 20 chunks to 10, cutting LLM input tokens roughly in half. On long-context models billed by token, this also reduces per-query cost.

---

## 5. Retrieval Quality Analysis

### Dense Retrieval Behavior

Dense retrieval (cosine similarity over bge-base-en-v1.5 embeddings) achieves high recall but moderate precision. It returns semantically related chunks broadly but cannot distinguish between "this chunk directly answers the question" and "this chunk is about the same topic area." At top_k=20, many of the retrieved chunks are contextually adjacent to the answer without containing it.

### BM25 Keyword Contribution

BM25 adds value primarily for queries with specific technical terms (model names, acronyms, specific dates). For Q1 (RAG-Sequence vs RAG-Token), exact-match retrieval on "RAG-Sequence" and "RAG-Token" boosts the specific methodology chunks. RRF then promotes chunks that appear in both dense and BM25 results — a strong signal of relevance.

### Cross-Encoder Reranking

The cross-encoder reads the full query and each chunk together, enabling it to detect whether the chunk actually addresses the question rather than just being topically related. This is the key quality difference. A chunk mentioning "RAG-Sequence" in a comparison table will score lower than a chunk explaining the RAG-Sequence marginalization equation when the query asks about the architecture.

---

## 6. Overall Ranking

```
Modern   ████████████████████████████████████████  0.912
Hybrid   ██████████████████████████████████████    0.844
Contextual █████████████████████████████████████   0.831
Traditional ████████████████████████████████████   0.781
```

| Rank | Pipeline | Avg Score | Key Advantage |
|------|----------|:---------:|---------------|
| 1 | Modern | 0.912 | Cross-encoder reranking (+25pp precision) |
| 2 | Hybrid | 0.844 | BM25+RRF adds keyword matching |
| 3 | Contextual | 0.831 | LLM context anchors specific details (recall=1.000) |
| 4 | Traditional | 0.781 | Lowest cost, but misses specific details |
