# Chunk Sizing Research — bge-base-en-v1.5

## Context
- Embedding model: BAAI/bge-base-en-v1.5 (max 512 tokens)
- Corpus: 8 arXiv academic papers (~476K chars, 132 pages)
- Need: page number citations in answers

## What is a Token?

A token is the unit of text that embedding/LLM models process. For academic/technical text:
- ~3.5-3.8 characters per token (shorter than general English ~4.0 due to specialized vocabulary)
- Working constant: **3.6 chars/token** for arXiv papers

| Tokens | Approx. Characters | Approx. Content |
|--------|-------------------|-----------------|
| 100 | ~360 chars | ~2-3 sentences |
| 200 | ~720 chars | ~1 short paragraph |
| 400 | ~1,440 chars | ~1-2 full paragraphs |
| 512 | ~1,840 chars | Model maximum — anything beyond is truncated |

## Chunk Size Options

| | Option A — Small | Option B — Medium | Option C — Near-Max |
|---|---|---|---|
| **Token size** | ~200 tokens | ~400 tokens | ~480 tokens |
| **Approx. chars** | ~720 chars | ~1,440 chars | ~1,728 chars |
| **% of model max** | 39% | 78% | 94% |
| **Retrieval precision** | High — tight semantic match | Moderate-high | Lower — embedding compresses more meaning |
| **Context per chunk** | Low — often mid-sentence | Good — full paragraph(s) | High — full section fragment |
| **Fact dilution risk** | Low | Low-moderate | Moderate-high |
| **Lost-context risk** | High — splits arguments, equations | Low | Very low |
| **Page citation accuracy** | High (fine grain) | High | Moderate (may span pages) |
| **bge-base fitness** | Under-utilized | Strong fit (78% capacity) | Near saturation, quality degrades |
| **Best for** | Fact lookup | Conceptual + factual mixed | Broad analytical queries |
| **Academic paper fit** | Poor — breaks section coherence | Good | Acceptable with section-aware only |

## Overlap Options

Overlap = how many tokens from the end of chunk N appear at the start of chunk N+1. Prevents losing information at chunk boundaries.

| Overlap % | For 200-tok | For 400-tok | For 480-tok | Effect |
|---|---|---|---|---|
| **10%** | 20 tokens | 40 tokens | 48 tokens | Minimal boundary bleed; risk of missing cross-boundary facts |
| **15%** | 30 tokens | 60 tokens | 72 tokens | Industry consensus sweet spot; covers ~1-2 sentences carry-over |
| **20%** | 40 tokens | 80 tokens | 96 tokens | Better continuity; ~11% more chunks (storage + cost increase) |

## Section-Aware Chunking

Fixed-size chunking across arXiv PDFs breaks logical units (theorems, derivations, result tables). Research shows structure-aware chunking achieves higher retrieval accuracy.

Recommended approach:
1. Split first at section boundaries (Abstract, Introduction, Method, Results, Conclusion, References)
2. Apply token-capped chunking within each section
3. Overlap only within same section (don't carry overlap across section boundaries)
4. Store metadata per chunk: `{doc_id, section, chunk_index, page_start, page_end}`

## Recommendation

**450 tokens / 68-token overlap (15%)**

Rationale:
- **Model fit**: 88% of bge-base 512 max — well within quality window before compression degrades
- **Academic text**: arXiv paragraphs avg 120-200 words (~160-270 tokens). 450 tokens captures 1-2 full paragraphs — natural argument unit
- **Precision vs context**: 200 tokens splits equations and multi-clause arguments. 480 tokens dilutes specific facts and spans pages.
- **Page citations**: 450-token chunks rarely span more than 1 page in two-column format
- **Overlap at 15%**: Covers ~2-3 carry-over sentences. Sufficient without ballooning index size.

## Estimated Chunk Counts

With 450 tokens (~1,620 chars) and 68-token overlap (~245 chars):
- Effective stride: 382 tokens (~1,375 chars)
- Total corpus: ~476K chars
- Estimated chunks: ~347 chunks (before section-aware adjustments)
- With section-aware splitting: ~380-400 chunks (slightly more due to section boundaries)

## Sources

- Rethinking Chunk Size for Long-Document Retrieval (arXiv 2505.21700)
- Optimizing RAG Chunk Size — Machine Learning Plus
- Best Chunking Strategies for RAG 2026 — Firecrawl
- Optimal chunk size for RAG — Milvus
- Evaluation of RAG Retrieval Chunking Methods — Superlinked VectorHub
- Finding Best Chunking Strategy — NVIDIA Technical Blog
- Chunk Size is Query-Dependent — AI21
- Evaluating Chunking Strategies for RAG (arXiv 2603.24556)
- Systematic Analysis of Chunking for QA (arXiv 2601.14123)

## Status: ✓ APPROVED
