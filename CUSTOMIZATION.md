# CUSTOMIZATION.md — Adapting the RAG Comparison System to a New Corpus

This guide covers everything needed to swap in a different document corpus and tune the system for a new use case. The retrieval code, evaluation framework, pipeline logic, and scoring system require no changes.

---

## 1. Use Your Own Corpus

### Step 1: Replace the PDFs

Remove the existing papers and add your own:

```bash
rm data/papers/*.pdf
cp /path/to/your/documents/*.pdf data/papers/
```

Any PDF works — academic papers, technical reports, documentation. The system is not tied to arXiv content.

### Step 2: Full Reset

Delete all derived artifacts so the pipeline re-processes from scratch. Do not delete the papers themselves.

```bash
rm -rf data/chroma_db/
rm -rf data/cache/
rm -rf data/results/
rm -f  data/questions.json
# data/papers/ stays intact
```

### Step 3: Run the Full Pipeline in Order

Each command requires `--approve` to execute. Without `--approve`, it prints a token/cost estimate and exits. Review the estimate before approving.

**1. Ingest — extract, chunk, embed into the traditional collection**
```bash
python -m src.cli.main ingest
python -m src.cli.main ingest --approve
```

**2. Contextualize — LLM-adds context prefixes, embeds into the contextualized collection**
```bash
python -m src.cli.main contextualize
python -m src.cli.main contextualize --approve
```

**3. Generate questions — auto-generates Q+A pairs from your papers**
```bash
python -m src.cli.main generate-questions --count 30
python -m src.cli.main generate-questions --count 30 --approve
```

**4. Evaluate — runs all 4 pipelines on all questions, scores with LLM judge**
```bash
python -m src.cli.main evaluate
python -m src.cli.main evaluate --approve
```

**5. Compare — produces the final comparison report**
```bash
python -m src.cli.main compare
python -m src.cli.main compare --output-format md
```

The pipeline is idempotent at every step. If a run is interrupted, re-run the same command — completed work is skipped automatically.

---

## 2. Configuration Tuning

All parameters live in `config/default.yaml`. CLI flags override the file at runtime without editing it.

### chunk_size / chunk_overlap

```yaml
chunking:
  chunk_size: 450    # tokens per chunk (hard ceiling: 512 — the embedding model's max)
  chunk_overlap: 68  # tokens of overlap between consecutive chunks
```

- **Smaller chunks** (150–300 tokens): higher precision, but each chunk carries less context. Useful for dense, fact-rich documents where a single sentence answers the question.
- **Larger chunks** (400–512 tokens): more context per retrieval hit, but relevance signal is diluted. Useful for documents where answers span multiple sentences.
- **Overlap** prevents answer fragments from being split across boundaries. Increase for long continuous prose; decrease for short, self-contained paragraphs.

CLI override:
```bash
python -m src.cli.main ingest --approve  # re-chunk after changing chunk_size
```
Note: changing `chunk_size` invalidates the existing ChromaDB collections. Run a full reset (Step 2) before re-ingesting.

### top_k values

```yaml
retrieval:
  top_k_dense: 20    # candidates from vector search
  top_k_bm25: 20     # candidates from BM25
  top_k_fusion: 20   # after RRF merging dense + BM25
  top_k_rerank: 10   # final set passed to LLM (Modern pipeline only)
```

- Increase `top_k_dense` / `top_k_bm25` for higher recall (more candidates, higher token cost per query).
- Decrease `top_k_rerank` to reduce the context window sent to the LLM for answer generation.

CLI override:
```bash
python -m src.cli.main evaluate --approve  # no re-ingestion needed; retrieval params apply at query time
```

### LLM provider and model

```yaml
llm:
  provider: "claude_cli"   # anthropic | openai | claude_cli

models:
  llm_model: "claude-sonnet-4-20250514"
  judge_model: "claude-haiku-4-5-20241022"
```

- `llm_model` is used for answer generation across all 4 pipelines and for contextualization.
- `judge_model` is used for evaluation scoring (the LLM-as-judge calls). Keep it cheap — Haiku or gpt-4o-mini.
- Changing the LLM does not require re-ingestion. It only affects contextualization and evaluation runs.

CLI override:
```bash
python -m src.cli.main evaluate --provider openai --model gpt-4o --approve
```

### Number of questions

Default is 30. Distributed roughly evenly across papers (~4 per paper for a 7-paper corpus).

```bash
python -m src.cli.main generate-questions --count 50 --approve
```

More questions = more reliable evaluation scores, more judge token cost. For a quick sanity check on a new corpus, start with 10–15.

### Evaluation metrics

The 4 metrics are defined in `config/default.yaml` under `evaluation.metrics`. They are:

- `context_precision` — are the retrieved chunks relevant to the question?
- `context_recall` — do the retrieved chunks cover the ground truth?
- `faithfulness` — is the answer grounded in retrieved content?
- `answer_relevancy` — does the answer directly address the question?

To drop a metric (reduces judge token cost proportionally), remove its entry from the list. The scoring and comparison code reads this list dynamically.

---

## 3. Switching LLM Provider

### OpenAI

Set the API key in your environment:

```bash
export OPENAI_API_KEY="sk-..."
```

Override provider and model at the CLI:

```bash
# Contextualization with GPT-4o
python -m src.cli.main contextualize --provider openai --model gpt-4o --approve

# Evaluation with GPT-4o for answers, gpt-4o-mini for judging
python -m src.cli.main evaluate --provider openai --model gpt-4o --approve
```

To make OpenAI the persistent default, edit `config/default.yaml`:

```yaml
llm:
  provider: "openai"

models:
  llm_model: "gpt-4o"
  judge_model: "gpt-4o-mini"
```

Pricing for OpenAI models is already in `config/default.yaml` under `cost_pricing`. Token estimates and cost gates work identically across providers.

### Anthropic API (direct, not claude_cli)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

```yaml
llm:
  provider: "anthropic"
```

The `claude_cli` provider shells out to the local `claude` CLI and estimates token counts from character counts (chars / 4). The `anthropic` provider uses the SDK and returns exact token counts from the API response.

---

## 4. Question Generation

The `generate-questions` command removes the need for manual question curation. For each PDF in `data/papers/`, the system:

1. Reads the extracted Markdown text of the paper.
2. Sends the full paper to the LLM with instructions to produce `N` question/ground-truth pairs.
3. Each pair includes:
   - The evaluation question (non-trivial, specific, diverse across paper sections)
   - A 2-4 sentence ground truth answer grounded in the paper text
   - Page number citations for where the supporting evidence appears
4. Saves all pairs to `data/questions.json`.

The ground truth is used by the LLM judge during evaluation to score `context_recall` (whether retrieved chunks cover the answer) and `faithfulness` (whether the generated answer matches the ground truth).

Questions are distributed evenly across papers: `count / num_papers`, rounded, minimum 1 per paper. For a 5-paper corpus with `--count 30`, that is 6 questions per paper.

**Manual curation is optional.** `data/questions.json` is plain JSON. You can edit, add, or remove questions before running `evaluate`. Each entry needs: `question_id`, `question`, `ground_truth`, `source_files`, `source_pages`. The `question_id` must be the first 16 hex chars of `sha256(question.strip())`.

The command is idempotent: if `data/questions.json` already contains at least `--count` questions, generation is skipped entirely.

---

## 5. What You Do Not Need to Change

The following components are corpus-agnostic and work unchanged regardless of what documents you ingest:

- **All 4 retrieval pipelines** — Traditional, Contextual, Hybrid, Modern
- **Embedding model** (`BAAI/bge-base-en-v1.5`) — general-purpose, not domain-specific
- **Reranker model** (`BAAI/bge-reranker-v2-m3`) — general-purpose cross-encoder
- **BM25 index** — rebuilt automatically from your chunks on each ingest
- **RRF fusion logic**
- **LLM-as-judge evaluation framework** — the 4 metrics apply to any domain
- **Telemetry and cost gate system**
- **ChromaDB collections layout** — two collections (`rag_traditional_v1`, `rag_contextualized_v1`) serve all pipelines
- **Comparison report generation**

---

## 6. Scaling

### More papers

Drop additional PDFs into `data/papers/` before running `ingest`. No upper limit enforced by the code. Practical limits on a 16 GB machine:

- Embedding: the embedder processes chunks sequentially on MPS. Memory is constant regardless of corpus size.
- Contextualization: each chunk requires one LLM call. Token cost scales linearly with `num_chunks`.
- ChromaDB: handles tens of thousands of vectors without issue. Performance degrades beyond ~500K vectors.

### More questions

`--count` can be set arbitrarily. Evaluation cost scales as:

```
judge calls = num_questions x num_pipelines x num_metrics
             = num_questions x 4 x 4
             = 16 x num_questions
```

At 30 questions: 480 judge calls. At 100 questions: 1,600 judge calls.

Evaluation is resumable. If interrupted, re-run `evaluate --approve` — completed question/pipeline/metric combinations are skipped (results are persisted incrementally to `data/results/`).

### Reducing cost on large corpora

- Use `--provider openai --model gpt-4o-mini` for contextualization (cheaper than Sonnet for context-prefix generation).
- Set `judge_model` to `gpt-4o-mini` or `claude-haiku-4-5-20241022` — judge calls dominate token cost.
- Reduce `num_metrics` in `config/default.yaml` if you only need a subset of scores.
- Use `--count 15` for a faster, cheaper comparison pass before committing to a full 30-question run.
