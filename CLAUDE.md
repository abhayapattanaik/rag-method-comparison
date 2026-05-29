# RAG Comparison Project

## Project Summary

RAG comparison system — 4 retrieval methods (Traditional, Contextual, Hybrid, Modern) compared on answer quality, cost, and latency. Full spec in `project-spec.md`.

## Build Rules

1. **Coordinator pattern** — main thread coordinates only, all work through spawned agents
2. **Cost gate** — every CLI entry point: estimate first (no `--approve`), execute only with `--approve`
3. **Incremental validation** — single question + single pipeline before scaling
4. **Scientific control** — only retrieval technique varies; embedding model, chunk size, LLM, judge constant
5. **Idempotent** — all operations safe to re-run without duplicating data
6. **Crash recovery** — resumable after interruption, no re-doing completed work
7. **No hardcoding** — all parameters via YAML config + CLI override
8. **Author approval required** — module design, chunk sizing, test plans reviewed before implementing

## Agent Rules

1. **Check before doing** — every agent prompt that creates, downloads, or writes must include "check if X exists first, skip if already done." Never redo completed work.
2. **Main thread verifies state** — before spawning work agents, verify current state (via memory or lightweight check). If desired state already exists, skip the agent.
3. **Ask on rejection** — when user rejects an agent, main thread asks why. Never silently retry rejected work later.
4. **No one-off scripts** — all LLM operations go through CLI with cost gate. Never create scripts that bypass estimate/approve.
5. **Logging from day 1** — every module gets Python `logging`. Never ship code without it.

## 4 Retrieval Methods

1. **Traditional** — chunk → embed → dense retrieval → LLM answer
2. **Contextual** — chunk → LLM-contextualize → embed → dense retrieval → LLM answer
3. **Hybrid** — chunk → LLM-contextualize → embed → dense + BM25 + RRF fusion → LLM answer
4. **Modern** — chunk → LLM-contextualize → embed → dense + BM25 + RRF + cross-encoder rerank → LLM answer

## Technology Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python | — |
| Vector store | ChromaDB | — |
| Embedding | `BAAI/bge-base-en-v1.5` via `sentence-transformers` + PyTorch MPS | 109M params, ~416MB, fits M2 16GB. Free local. MTEB ~53-55 — sufficient for method comparison |
| Reranker | `BAAI/bge-reranker-v2-m3` via PyTorch MPS | 570M params, 8K context (critical for academic chunks). Free local |
| BM25 | `rank_bm25` | — |
| PDF extraction | `pymupdf4llm` | Fast, page metadata, Markdown output. Marker as fallback for formula-heavy papers |
| Store layout | 2 shared collections: `rag_traditional_v1` (raw), `rag_contextualized_v1` (shared by Contextual/Hybrid/Modern) | Scientific validity (identical vectors), cost (embed once), simplicity |
| MLX | Not needed | Models small enough for PyTorch directly. MLX only helps at 7B+ scale |
| Ollama | Deferred | Not needed for v1 |

## LLM Backends

Support: Anthropic API, OpenAI API, Claude CLI (shell out to `claude` command). Ollama deferred.

## Evaluation

LLM-as-judge on 4 dimensions (0.0–1.0 each):
- Context precision
- Context recall
- Faithfulness
- Answer relevancy

No RAGAS library. Persist results incrementally.

## Telemetry

Every LLM call tracks: token usage, latency, cost. Per-pipeline cost breakdown in results.

## Expected Outputs

1. Architecture document
2. Design decisions log
3. Test plan + acceptance test results
4. Comparison analysis (4 methods)
5. Cost-quality tradeoff summary
6. RAG concepts glossary
