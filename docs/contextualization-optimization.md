# Contextualization Optimization Analysis

**Date:** 2026-05-26
**Status:** APPROVED — see DD-014 in design-decisions.md

---

## Problem

Contextual retrieval requires prepending a short context prefix to each chunk before embedding. The naive approach sends the full document with every chunk request, which is expensive at scale.

**Baseline parameters:**
- 8 documents, ~50 chunks per doc = ~400 chunks total
- Average document size: ~5,750 tokens
- Average chunk size: 450 tokens
- Prompt overhead per LLM call: ~200 tokens

**Baseline cost (full doc per chunk, no batching):**
- Tokens per call: 5,750 (doc) + 450 (chunk) + 200 (prompt) = ~6,400 tokens
- Total calls: 400
- Total input tokens: 6,400 × 400 = **2.56M tokens** (~2.3M rounded in analysis)

---

## 5 Strategies Analyzed

### Strategy 1: Baseline — Full Document Per Chunk

Send the entire document with every chunk context call. One call per chunk.

- Tokens per call: ~6,400
- Total calls: 400
- **Total tokens: ~2.3M**
- Savings: —
- Quality risk: None (gold standard)
- Complexity: Low

### Strategy 2: Batch=5 with Full Document

Send the full document once per batch of 5 chunks. Ask the LLM to contextualize all 5 chunks in a single call.

- Tokens per call: 5,750 (doc) + 5 × 450 (chunks) + 200 (prompt) = ~8,200 tokens
- Total calls: 400 / 5 = 80
- **Total tokens: ~570K** (document repeated across 80 batches = 80 × 5,750 = 460K doc tokens + 180K chunk tokens)
- Savings: 75%
- Quality risk: Low (batch=5 stays within reliable output range)
- Complexity: Low

### Strategy 3: Doc Summary Only

Pre-summarize each document once (target: ~1,500 tokens). Send the summary instead of the full doc with each chunk.

- Summarization cost: 8 × 5,750 = 46,000 tokens input + 8 × 1,500 = 12,000 tokens output
- Contextualization: (1,500 + 450 + 200) × 400 = 2,150 × 400 = 860,000 tokens
- Minus batching savings not applied: no batching assumed in this strategy
- **Total tokens: ~736K** (summarization overhead + contextualization)
- Savings: 68%
- Quality risk: Medium — summary loses formula-level and table-level detail
- Complexity: Medium (two-pass, summary caching)

### Strategy 4: Sliding Window — Local Context Only

No document-level context. For each chunk, include only the previous chunk and next chunk as local context.

- Tokens per call: 900 (prev + next chunks) + 450 (current) + 200 (prompt) = ~1,550 tokens
- Total calls: 400
- **Total tokens: ~412K**
- Savings: 82%
- Quality risk: High — LLM has no document-level context; cannot situate chunk within paper's argument structure
- Complexity: Very Low

### Strategy 5: Summary + Local Context + Batch=5 (CHOSEN)

Two-pass approach combining document summary, local neighboring chunks, and batching.

**Pass 1 — Summarization (8 LLM calls):**
- Input per call: ~5,750 tokens
- Output per call: ~1,500 tokens
- Total Pass 1 cost: 8 × 5,750 = **46,000 input tokens**

**Pass 2 — Contextualization (80 batched calls, batch size = 5):**
- Tokens per call: 1,500 (doc summary) + 900 (prev + next chunks) + 5 × 450 (batch of 5 current chunks) + 200 (prompt) = ~4,850 tokens
- Total calls: 400 / 5 = 80
- Total Pass 2 cost: 80 × 4,850 = **388,000 input tokens**

**Total: 46K + 388K ≈ 485K tokens**
- Savings: **79%** vs baseline
- Quality risk: **Low**
- Complexity: Medium

---

## Comparison Table

| Strategy | Total Tokens | Savings | Quality Risk | Complexity |
|----------|-------------|---------|-------------|------------|
| Baseline (full doc per chunk) | 2.3M | — | None | Low |
| Batch=5 with full doc | 570K | 75% | Low | Low |
| Doc summary only | 736K | 68% | Medium | Medium |
| Sliding window (local only) | 412K | 82% | High | Very Low |
| **Summary + local + batch=5** | **485K** | **79%** | **Low** | **Medium** |

---

## Recommendation: Strategy 5 — Summary + Local Context + Batch=5

### Why not sliding window (best token savings, lowest complexity)?

Sliding window saves 82% but introduces High quality risk. The LLM cannot place a chunk in the document's larger argument — it only sees neighbors. For academic RAG where a chunk might reference a concept introduced 10 pages earlier, this produces weak or misleading context prefixes.

### Why not batch=5 with full doc (simpler, similar savings)?

Batch=5 with full doc is simpler (no summarization pass) but saves only 75% vs 79% for Strategy 5. More importantly, it resends the full document 80 times. Strategy 5 sends the full document only 8 times (for summarization), then reuses the cached summaries. This reduces total LLM I/O and is faster in wall-clock time.

### Why Strategy 5 is acceptable quality at lower token cost

Contextualization adds only a 1-2 sentence prefix to each chunk. The prefix does not need formula-level precision — it only needs to situate the chunk topically (e.g., "This chunk is from Section 3 of a survey on dense retrieval, discussing bi-encoder training objectives."). A 1,500-token document summary is sufficient for that level of situational context. The chunk text itself is unchanged; retrieval quality depends primarily on the chunk embedding, not the contextualization prefix quality.

### Mitigation

Author reviews all 8 document summaries (Pass 1 output) before Pass 2 proceeds. If any summary is poor quality, that document's summary is manually corrected before contextualization runs. This review gate is feasible because 8 summaries is a small human-reviewable set.

---

## Implementation Notes

**Pass 1 prompt (per document):**
Summarize this document in ~1,500 tokens. Cover: main thesis, methodology, key findings, section structure. Preserve enough detail to situate any individual passage within the document's argument.

**Pass 2 prompt (per batch of 5 chunks):**
Given the document summary and local context below, write a 1-2 sentence context prefix for each chunk that situates it within the document. Output JSON array of 5 strings.

**Batch size rationale:**
Batch=5 was chosen based on output reliability. Empirically, LLMs maintain accurate per-item correspondence at batch sizes up to ~10, but quality degrades noticeably at 10+ items (position confusion, merged outputs). Batch=5 provides a conservative safety margin.

**Caching:**
Pass 1 summaries stored as JSON keyed by document filename/hash. Pass 2 reads from cache — re-running contextualization does not re-summarize.

---

## References

- Anthropic Contextual Retrieval blog post (concept reference)
- DD-014 in design-decisions.md
- DD-013 (chunk sizing: 450 tokens, 15% overlap)
- DD-011 (corpus: 8 arXiv papers)
