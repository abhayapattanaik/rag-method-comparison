# Our Learnings

We built four RAG pipelines — Traditional, Contextual, Hybrid, and Modern — on a corpus of eight arXiv papers to understand how retrieval technique choices affect answer quality, cost, and latency. This document covers what we learned: not just what scored higher, but why, and what it tells you about how RAG actually works.

---

## Insights

We implemented an LLM-as-judge evaluation that scores each pipeline on four metrics: context precision, context recall, faithfulness, and answer relevancy (each 0.0–1.0). We ran all 32 questions through all four methods and benchmarked the results.

**Across our four methods, the one with the highest retrieval precision scored highest on our LLM-as-judge evaluation.**

| Pipeline | Precision | Recall | Faithfulness | Relevancy | **Overall** |
|----------|:---------:|:------:|:------------:|:---------:|:-----------:|
| Traditional | 0.450 | 0.750 | 0.975 | 0.950 | **0.781** |
| Contextual | 0.375 | 1.000 | 1.000 | 0.950 | **0.831** |
| Hybrid | 0.450 | 1.000 | 0.975 | 0.950 | **0.844** |
| Modern | 0.700 | 1.000 | 1.000 | 0.950 | **0.912** |

Faithfulness and relevancy stayed flat (0.95–1.0) across all methods — the LLM produced good answers regardless of pipeline. The metrics that moved were retrieval precision (0.375 to 0.700) and recall (0.750 to 1.000). Each method progressively improved how relevant the retrieved chunks were, and that drove the overall score.

---

## Methods Summary

### Traditional: The Baseline

Traditional retrieval is simple: chunk the documents, embed each chunk, store in a vector database, retrieve the top-k by cosine similarity, send to the LLM. No LLM calls at ingestion time. No index complexity. It works.

**Scores:** Precision 0.450, Recall 0.750, Faithfulness 0.975, Relevancy 0.950 → **Overall: 0.781**

Our Opinion: dense retrieval on its own does a reasonable job. It finds topically similar chunks. But "topically similar" and "answers the question" are not the same thing, and dense retrieval can't distinguish between them. The embedding encodes semantic neighborhood, not argumentative relevance. This gap is exactly what the later techniques address.

Recall at 0.750 was also a problem. On 25% of questions, at least one relevant chunk was missed entirely. For academic text, missing a chunk often means missing the key finding.

---

### Contextual: Teaching the Embedder What Each Chunk Means

The idea behind contextual retrieval: a chunk extracted from a paper often loses its meaning in isolation. A paragraph from a Results section might read "The proposed method achieves 94.2% accuracy on the benchmark" — but without context, the embedding doesn't know which method, which benchmark, or how this relates to the paper's central claim.

Contextual retrieval adds an LLM-generated prefix to each chunk before embedding: a 1–2 sentence summary that places the chunk in the document's argument structure. The embedding now captures "this chunk is about Method X's performance on Benchmark Y, which is the paper's primary contribution" rather than just "numbers and accuracy."

**What it changed:** Recall jumped from 0.750 to 1.000. We stopped missing relevant chunks. The embedding's broader semantic coverage meant it found relevant chunks even when the surface-level wording didn't match the query well.

**A surprise:** Precision dropped from 0.450 to 0.375. We were finding more relevant chunks, but also pulling in more irrelevant ones, in our view. The contextualized embedding broadened the retrieval neighborhood — which helped recall but diluted precision. This was counterintuitive. Making each chunk "richer" semantically made the retrieval less precise.

**Scores:** Precision 0.375, Recall 1.000, Faithfulness 1.000, Relevancy 0.950 → **Overall: 0.831**

The overall score improved because recall gains outweighed the precision loss, but the precision regression was a real problem that BM25 later addressed.

---

### Hybrid: Recovering Precision with Keywords

Hybrid adds BM25 to the picture. BM25 is a keyword-matching algorithm that scores chunks by term frequency and document frequency — it finds chunks that contain the exact words in the query. Dense retrieval finds semantically similar chunks; BM25 finds lexically matching chunks. Reciprocal Rank Fusion (RRF) merges the two ranked lists into one.

Our Opinion: technical text has terms that dense retrieval handles poorly. Model names, algorithm names, metric names, author citations — these are low-frequency, domain-specific tokens. The embedding model has seen "accuracy" ten thousand times; it has seen "BGE-reranker-v2-m3" almost never. BM25 treats rare technical terms as highly discriminative signals, which is exactly the right behavior for a corpus of computer science papers.

**What it changed:** Precision recovered from 0.375 back to 0.450, matching Traditional. Recall stayed at 1.000. The combination of dense + BM25 got us the best of both: semantic similarity for concept-level queries, keyword matching for technical-term queries.

**Scores:** Precision 0.450, Recall 1.000, Faithfulness 0.975, Relevancy 0.950 → **Overall: 0.844**

The key insight from Hybrid: retrieval diversity matters. Dense retrieval and BM25 fail in different places, so combining them covers more ground. RRF works here specifically because it's rank-based, not score-based — you don't need to normalize heterogeneous scoring functions.

---

### Modern: The Cross-Encoder Changes Everything

Modern adds a cross-encoder reranker as a final step: retrieve top-20 candidates via dense + BM25 + RRF, then pass each (query, chunk) pair through a cross-encoder, score them, keep the top-10.

The difference between a bi-encoder (used in dense retrieval) and a cross-encoder is fundamental. A bi-encoder encodes the query and the chunk independently, then computes similarity between the two embeddings. A cross-encoder sees the query and the chunk together in one forward pass, with full attention across both. It can model the relationship between them directly.

Our Opinion: This is why the precision jump was so large. A bi-encoder knows that a chunk is "about" retrieval methods; a cross-encoder knows whether this specific chunk actually answers this specific question about retrieval methods. That distinction produces +25 percentage points of precision.

**Scores:** Precision 0.700, Recall 1.000, Faithfulness 1.000, Relevancy 0.950 → **Overall: 0.912**

The full progression:

| Technique Added | Precision | Recall | Overall |
|----------------|:---------:|:------:|:-------:|
| Traditional (dense only) | 0.450 | 0.750 | 0.781 |
| + Contextualization | 0.375 | 1.000 | 0.831 |
| + BM25 fusion | 0.450 | 1.000 | 0.844 |
| + Cross-encoder rerank | **0.700** | **1.000** | **0.912** |

Each technique addressed a specific weakness. Contextualization fixed recall at the cost of precision. BM25 recovered precision for technical terms. Cross-encoder reranking finished the job by distinguishing relevance from topicality.

---

## Findings Summary

### 1. Contextualization can hurt precision

Adding richer semantic context to each chunk broadened the embedding neighborhood and pulled in more off-target results. The fix wasn't less contextualization — it was adding BM25 and the cross-encoder downstream. But the lesson is: improvements to one part of the pipeline can degrade another. Retrieval is a system, not a single knob.

### 2. Modern is simultaneously the best quality, the fastest, and the cheapest per query

This seemed wrong until we understood why. Modern retrieves 20 chunks but reranks down to 10 before sending to the LLM. The other pipelines send all 20 retrieved chunks as context. LLM input tokens are the main cost and latency driver, so cutting context in half dominates everything else.

| Pipeline | Cost/Query | Latency/Query | Quality |
|----------|:----------:|:-------------:|:-------:|
| Traditional | $0.46 | 12,216 ms | 0.781 |
| Contextual | $0.48 | 12,486 ms | 0.831 |
| Hybrid | $0.51 | 15,194 ms | 0.844 |
| **Modern** | **$0.39** | **8,602 ms** | **0.912** |

The cross-encoder adds ~200–400 ms of compute on Apple Silicon MPS. The LLM speedup from half the context tokens is ~3,600 ms. The reranker pays for itself several times over in LLM savings.

### 3. Breakeven is only 28 queries

Modern has a one-time ingestion cost of ~$1.94 for LLM contextualization (shared across Contextual, Hybrid, and Modern). It saves ~$0.07/query over Traditional at query time. Breakeven: 28 queries. At 100 queries, Modern's total cost is lower than every other pipeline — and it has the highest quality score throughout.

| Pipeline | Ingestion | 20 Queries | Total (20q) | Quality |
|----------|:---------:|:----------:|:-----------:|:-------:|
| Traditional | $0.00 | $9.20 | $9.20 | 0.781 |
| Contextual | $1.94 | $9.60 | $11.54 | 0.831 |
| Hybrid | $1.94 | $10.20 | $12.14 | 0.844 |
| **Modern** | **$1.94** | **$7.80** | **$9.74** | **0.912** |

The takeaway: if you'll run more than ~30 queries, the upfront contextualization investment is worth it on pure cost grounds, before even counting quality.

---

## What Matters for Chunk Design

We settled on 450-token chunks with 68-token overlap for `bge-base-en-v1.5` (512-token max). The 450-token choice is deliberate.

The embedding model's quality degrades near its context limit. At 480 tokens (94% of max), you're close enough to saturation that additional content dilutes the representation rather than enriching it. At 200 tokens (39% of max), a single equation or multi-clause argument often doesn't fit, and you lose the logical unit.

450 tokens is 88% of the model max — well inside the quality window. For arXiv academic papers, it also maps onto the natural unit of content: one to two full paragraphs average 160–270 tokens, and 450 tokens captures that cleanly without spilling across page boundaries. Page boundary integrity matters for citation accuracy; knowing which page a chunk came from is only useful if the chunk doesn't span two pages.

The 68-token overlap (15%) covers two to three carry-over sentences at chunk boundaries — enough to catch arguments that straddle a split without meaningfully inflating index size. 15% is industry consensus for good reason: less and you lose boundary context; more and you pay in storage and retrieval noise.

Section-aware splitting proved important for academic text. Splitting at section boundaries (Abstract, Method, Results, etc.) before applying the token cap means a chunk never mixes content from two sections. Method section chunks are embedded with other method-level text; results chunks cluster with results. This keeps the semantic neighborhoods clean.

---

## Optimizing Token Cost

Contextualization requires an LLM call per chunk. With ~400 chunks across eight papers, the naive approach — send the full document with every chunk request — costs ~2.3M tokens. We analyzed five strategies and found a 79% reduction without sacrificing quality.

The winning strategy has two passes:

- **Pass 1 (8 LLM calls):** Summarize each document once, producing a ~1,500-token summary per paper. Cache these.
- **Pass 2 (80 batched calls):** For each batch of 5 chunks, send the cached doc summary (1,500 tokens) + neighboring chunks for local context (900 tokens) + the 5 current chunks (2,250 tokens) + prompt overhead (200 tokens) = ~4,850 tokens per call.
- **Total: ~485K tokens** vs 2.3M baseline.

| Strategy | Total Tokens | Savings | Quality Risk |
|----------|:-----------:|:-------:|:------------:|
| Baseline (full doc per chunk) | 2.3M | — | None |
| Batch=5 with full doc | 570K | 75% | Low |
| Doc summary only | 736K | 68% | Medium |
| Sliding window (local context only) | 412K | 82% | High |
| **Summary + local context + batch=5** | **485K** | **79%** | **Low** |

We considered pure sliding window (82% savings, lowest complexity) and rejected it. Without document-level signal, the LLM can't place a chunk in the paper's argument structure. For academic PDFs, this produces vague prefixes and breaks entirely for section-crossing references. The quality risk was too high.

The key design choice is separating document understanding (Pass 1) from chunk contextualization (Pass 2). Pass 1 runs 8 calls and produces reusable summaries. Pass 2 consumes those summaries instead of the full documents. This also means the 8 summaries can be reviewed before committing to the full contextualization run — important when the corpus changes.

---

## Design Decisions That Mattered

### Shared contextualized collection

Contextual, Hybrid, and Modern share a single `rag_contextualized_v1` ChromaDB collection. They differ only in what happens after retrieval (dense-only, dense+BM25, dense+BM25+rerank). Sharing the collection means the contextualization cost is paid once for all three, and more importantly, it guarantees provably identical embeddings across methods. The only variable that differs is the retrieval technique — which is exactly the scientific control we need.

Without this, a precision difference between Contextual and Modern could be attributed to embedding variance rather than the reranker. With shared collections, that explanation is ruled out by construction.

### Reranker model: bge-reranker-v2-m3

The critical spec for a reranker on academic text is context window, not parameter count. `cross-encoder/ms-marco-MiniLM-L-6-v2` is a common choice but has a 512-token context limit — the same as the embedding model, which means it can't fully read a 450-token chunk plus the query. `bge-reranker-v2-m3` has an 8,192-token context window and runs well on PyTorch MPS. Academic PDF chunks routinely carry dense technical content; the reranker needs to read all of it to score accurately.

### Embedding model: bge-base-en-v1.5

At 109M parameters, this fits comfortably alongside the 570M reranker on 16GB unified memory without memory pressure. `bge-large-en-v1.5` (335M params, ~10 MTEB points higher) was rejected specifically because memory headroom for the reranker matters more than embedding quality for this comparison. The reranker does the heavy lifting on precision; the embedding model's job is to produce a good candidate pool, and 109M is sufficient for that.

### BM25 in-memory with rank_bm25

At ~420–450 chunks, an in-memory BM25 index is rebuilt each run in milliseconds. There's no case for adding Elasticsearch or OpenSearch to this stack — it would introduce a separate service with operational overhead for a benefit that doesn't exist at this corpus size. The rule is: use the simplest tool that works. At a few thousand chunks, `rank_bm25` works.
