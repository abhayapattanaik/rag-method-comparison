# RAG Comparison System -- Architecture & Design Document

Status: **Draft -- Pending Author Approval**

---

## 1. System Overview

This system compares four RAG (Retrieval-Augmented Generation) retrieval methods on answer quality, cost, and latency using a corpus of 20 arXiv RAG research papers. The goal is to understand how retrieval technique choices affect outcomes through controlled experimentation where only the retrieval method varies -- all other variables (embedding model, chunk size, LLM, judge) are held constant.

### The 4 Retrieval Methods

| Method | Key Differentiator | Retrieval Steps |
|--------|-------------------|-----------------|
| **Traditional** | Baseline dense retrieval | chunk -> embed -> dense search -> LLM answer |
| **Contextual** | LLM-enriched chunks | chunk -> LLM-contextualize -> embed -> dense search -> LLM answer |
| **Hybrid** | Dense + sparse fusion | chunk -> LLM-contextualize -> embed -> dense + BM25 + RRF fusion -> LLM answer |
| **Modern** | Full pipeline with reranking | chunk -> LLM-contextualize -> embed -> dense + BM25 + RRF + cross-encoder rerank -> LLM answer |

Each method builds on the previous one, adding one technique at a time. This layered design isolates the contribution of each component (contextualization, BM25/RRF fusion, reranking).

### Two-Phase Architecture

- **Ingestion phase** (one-time per configuration): PDF extraction, chunking, optional contextualization, embedding, storage in ChromaDB.
- **Query/Evaluation phase** (repeatable): Question answering, LLM-as-judge scoring, cross-method comparison.

**PENDING AUTHOR APPROVAL**

---

## 2. Directory Structure

```
claude_rc_2/
  config/
    default.yaml                    # All tunable parameters (paths, models, thresholds)
  src/
    __init__.py
    config.py                       # Pydantic config loader -- YAML + CLI overrides
    cost_gate.py                    # Estimate/approve pattern for cost control
    telemetry.py                    # Token usage, latency, cost tracking per LLM call
    llm/
      __init__.py
      base.py                       # Abstract LLM provider interface + LLMResponse dataclass
      anthropic_provider.py         # Anthropic SDK implementation
      openai_provider.py            # OpenAI SDK implementation
      claude_cli_provider.py        # Subprocess to `claude` CLI, parse stdout
    ingestion/
      __init__.py
      pdf_extractor.py              # pymupdf4llm -- PDF to Markdown with page metadata
      chunker.py                    # Document-level chunking (configurable size/overlap)
      contextualizer.py             # LLM chunk contextualization + per-chunk JSON cache
      embedder.py                   # bge-base-en-v1.5 via sentence-transformers + MPS
      store.py                      # ChromaDB collection management (create, upsert, query)
    retrieval/
      __init__.py
      dense.py                      # Dense (vector similarity) retrieval from ChromaDB
      bm25.py                       # BM25 sparse retrieval via rank_bm25
      fusion.py                     # Reciprocal Rank Fusion (RRF) -- merges dense + BM25
      reranker.py                   # bge-reranker-v2-m3 cross-encoder reranking
      pipeline.py                   # Pipeline orchestrator -- selects retrieval path per method
    evaluation/
      __init__.py
      judge.py                      # LLM-as-judge -- 4 metric scoring prompts
      question_gen.py               # Question + ground truth generation from papers
      runner.py                     # Evaluation runner -- incremental persistence per result
      comparison.py                 # Cross-method comparison tables and summaries
    interactive/
      __init__.py
      repl.py                       # Terminal REPL -- query all methods, side-by-side display
    cli/
      __init__.py
      main.py                       # argparse entry point with subcommands
  data/
    papers/                         # Downloaded arXiv PDFs (20 papers)
    chroma_db/                      # ChromaDB persistent storage (2 collections)
    cache/                          # Contextualization cache (per-chunk JSON files)
    results/                        # Evaluation results (per question-pipeline-metric JSON)
    questions.json                  # Generated questions + ground truth
  docs/                             # Output documents (architecture, analysis, glossary, etc.)
  tests/                            # Unit and integration tests
  CLAUDE.md                         # Project-level AI assistant instructions
  project-spec.md                   # Build specification
  project-plan.md                   # Implementation plan with phases
  clarifying-questions.md           # Author Q&A log
```

**PENDING AUTHOR APPROVAL**

---

## 3. Data Flow

### 3.1 Ingestion Flow

#### Traditional Ingestion (raw chunks)

```
                          rag_traditional_v1
                          (ChromaDB collection)
                                  ^
                                  |
                              [upsert]
                                  |
+--------+     +-----------+     +---------+     +----------+
|  PDFs  | --> | Extract   | --> | Chunk   | --> | Embed    |
| (arXiv)|     | (pymupdf  |     | (doc-   |     | (bge-    |
|        |     |  4llm)    |     |  level, |     |  base-en |
+--------+     +-----------+     |  512tok)|     |  -v1.5,  |
                    |            +---------+     |  MPS)    |
                    v                            +----------+
               Markdown +
               page metadata
```

Each PDF is extracted to Markdown preserving page boundaries. The full document text is chunked (not per-page) with configurable size and overlap. Chunks are embedded locally using bge-base-en-v1.5 on MPS and upserted into `rag_traditional_v1`.

#### Contextualized Ingestion (LLM-enriched chunks)

```
                                              rag_contextualized_v1
                                              (ChromaDB collection)
                                                      ^
                                                      |
                                                  [upsert]
                                                      |
+--------+     +-----------+     +---------+     +----------------+     +----------+
|  PDFs  | --> | Extract   | --> | Chunk   | --> | Contextualize  | --> | Embed    |
| (arXiv)|     | (pymupdf  |     | (same   |     | (LLM adds doc |     | (bge-    |
|        |     |  4llm)    |     |  chunks |     |  context to    |     |  base-en |
+--------+     +-----------+     |  as     |     |  each chunk)   |     |  -v1.5,  |
                                 |  above) |     +----------------+     |  MPS)    |
                                 +---------+          |                 +----------+
                                                      v
                                                 JSON cache
                                                 (per chunk_id,
                                                  crash recovery)
```

Same extraction and chunking as Traditional. Each chunk is sent to an LLM which prepends document-level context (what the paper is about, where this chunk fits). The contextualized text is embedded and stored in `rag_contextualized_v1`. A per-chunk JSON cache enables crash recovery.

#### Two-Collection Layout

```
+---------------------------+     +----------------------------------+
| rag_traditional_v1        |     | rag_contextualized_v1            |
|                           |     |                                  |
| Raw chunks + embeddings   |     | Contextualized chunks +          |
|                           |     | embeddings                       |
| Used by: Traditional      |     | Used by: Contextual, Hybrid,     |
|                           |     |          Modern                  |
+---------------------------+     +----------------------------------+
```

Contextual, Hybrid, and Modern share the same collection. Their differences are in post-retrieval processing (BM25 fusion, reranking), not in stored data.

### 3.2 Query/Retrieval Flow

#### Traditional Pipeline

```
+----------+     +----------+     +-----------------+     +----------+     +----------+
|  Query   | --> | Embed    | --> | Dense Search    | --> | Top-K    | --> | LLM      |
|  (text)  |     | (bge-    |     | (rag_traditional|     | chunks   |     | Generate |
|          |     |  base)   |     |  _v1)           |     |          |     | Answer   |
+----------+     +----------+     +-----------------+     +----------+     +----------+
                                                                                |
                                                                                v
                                                                          Answer with
                                                                          source citations
```

#### Contextual Pipeline

```
+----------+     +----------+     +--------------------+     +----------+     +----------+
|  Query   | --> | Embed    | --> | Dense Search       | --> | Top-K    | --> | LLM      |
|  (text)  |     | (bge-    |     | (rag_contextualized|     | chunks   |     | Generate |
|          |     |  base)   |     |  _v1)              |     |          |     | Answer   |
+----------+     +----------+     +--------------------+     +----------+     +----------+
                                                                                   |
                                                                                   v
                                                                             Answer with
                                                                             source citations
```

Same as Traditional but searches the contextualized collection. The richer embeddings should retrieve more relevant chunks.

#### Hybrid Pipeline

```
+----------+     +----------+     +--------------------+
|  Query   | --> | Embed    | --> | Dense Search       | --+
|  (text)  |     | (bge-    |     | (rag_contextualized|   |
|          |     |  base)   |     |  _v1)              |   |
+----------+     +----------+     +--------------------+   |
     |                                                     |     +-----------+     +----------+
     |                                                     +---> | RRF       | --> | LLM      |
     |                                                     |     | Fusion    |     | Generate |
     |           +--------------------+                    |     | (merge +  |     | Answer   |
     +---------> | BM25 Search       | -------------------+     |  re-rank) |     +----------+
                 | (rank_bm25 on     |                           +-----------+          |
                 |  contextualized   |                                                  v
                 |  chunks)          |                                            Answer with
                 +--------------------+                                           source citations
```

Queries both dense (vector similarity) and sparse (BM25 keyword matching) indexes. Results are merged using Reciprocal Rank Fusion (RRF) which combines rankings without requiring score normalization.

#### Modern Pipeline

```
+----------+     +----------+     +--------------------+
|  Query   | --> | Embed    | --> | Dense Search       | --+
|  (text)  |     | (bge-    |     | (rag_contextualized|   |
|          |     |  base)   |     |  _v1)              |   |
+----------+     +----------+     +--------------------+   |
     |                                                     |     +-----------+     +----------+     +----------+
     |                                                     +---> | RRF       | --> | Rerank   | --> | LLM      |
     |                                                     |     | Fusion    |     | (bge-    |     | Generate |
     |           +--------------------+                    |     |           |     |  reranker|     | Answer   |
     +---------> | BM25 Search       | -------------------+     +-----------+     |  -v2-m3, |     +----------+
                 | (rank_bm25 on     |                                            |  MPS)    |          |
                 |  contextualized   |                                            +----------+          v
                 |  chunks)          |                                                           Answer with
                 +--------------------+                                                          source citations
```

Same as Hybrid, plus a cross-encoder reranker that scores each (query, chunk) pair. The reranker uses full attention between query and chunk (not just embedding similarity), providing higher-quality relevance ordering at the cost of additional compute.

### 3.3 Evaluation Flow

```
+-------------------+     +------------------+     +------------------+
| questions.json    | --> | For each         | --> | For each         |
| (20 questions +   |     | pipeline:        |     | metric:          |
|  ground truth)    |     | Traditional,     |     | - context_prec   |
+-------------------+     | Contextual,      |     | - context_recall |
                          | Hybrid,          |     | - faithfulness   |
                          | Modern           |     | - answer_relev   |
                          +------------------+     +------------------+
                                |                        |
                                v                        v
                          +-----------+            +------------+
                          | Pipeline  |            | LLM Judge  |
                          | generates |            | scores     |
                          | answer +  |            | 0.0 - 1.0  |
                          | contexts  |            | + reason   |
                          +-----------+            +------------+
                                                        |
                                                        v
                                                  +------------+
                                                  | Persist    |
                                                  | per Q/P/M  |
                                                  | (JSON,     |
                                                  |  resumable)|
                                                  +------------+
                                                        |
                                                        v
                                                  +------------+
                                                  | Comparison |
                                                  | Table      |
                                                  | (per-metric|
                                                  |  per-pipe) |
                                                  +------------+

Total judge calls: 20 questions x 4 pipelines x 4 metrics = 320
Total answer generation calls: 20 questions x 4 pipelines = 80
```

Results are persisted incrementally (one JSON file per question-pipeline-metric combination) so evaluation can resume after interruption without repeating completed work.

**PENDING AUTHOR APPROVAL**

---

## 4. Module Design

### 4.1 `src/config.py` -- Configuration Loader

**Responsibility:** Load configuration from YAML, merge CLI overrides, validate with Pydantic.

**Interface:**
```python
from pydantic import BaseModel

class PathsConfig(BaseModel):
    data_dir: str
    chroma_dir: str
    cache_dir: str
    results_dir: str
    papers_dir: str

class ModelsConfig(BaseModel):
    embedding_model: str       # e.g. "BAAI/bge-base-en-v1.5"
    reranker_model: str        # e.g. "BAAI/bge-reranker-v2-m3"
    llm_model: str             # e.g. "claude-sonnet-4-20250514"
    judge_model: str           # e.g. "claude-haiku-4-5-20241022"

class ChunkingConfig(BaseModel):
    chunk_size: int            # tokens
    chunk_overlap: int         # tokens

class RetrievalConfig(BaseModel):
    top_k_dense: int
    top_k_bm25: int
    top_k_fusion: int
    top_k_rerank: int
    rrf_k: int                 # RRF constant (typically 60)

class LLMConfig(BaseModel):
    provider: str              # "anthropic" | "openai" | "claude_cli"
    anthropic_api_key_env: str
    openai_api_key_env: str
    temperature: float
    max_tokens: int

class CostConfig(BaseModel):
    pricing: dict[str, dict[str, float]]  # model -> {input_per_1k, output_per_1k}

class EvaluationConfig(BaseModel):
    metrics: list[str]
    judge_temperature: float

class ConcurrencyConfig(BaseModel):
    max_workers: int
    stall_timeout_seconds: int

class AppConfig(BaseModel):
    paths: PathsConfig
    models: ModelsConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    cost: CostConfig
    evaluation: EvaluationConfig
    concurrency: ConcurrencyConfig

def load_config(config_path: str | None = None, cli_overrides: dict | None = None) -> AppConfig:
    """Load config from YAML, apply CLI overrides, validate."""
    ...
```

**Dependencies:** `pydantic`, `pyyaml`

**Design Notes:** Pydantic validates all fields at load time. CLI overrides use dot-notation (e.g., `chunking.chunk_size=400`) and are applied after YAML loading but before validation.

---

### 4.2 `src/cost_gate.py` -- Cost Estimation & Approval

**Responsibility:** Estimate LLM costs before execution, block execution unless `--approve` flag is set.

**Interface:**
```python
from dataclasses import dataclass

@dataclass
class CostEstimate:
    operation: str                           # e.g. "contextualize", "evaluate"
    estimated_calls: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    cost_by_model: dict[str, float]          # model_name -> estimated USD

class CostGate:
    def __init__(self, config: AppConfig, approved: bool = False):
        ...

    def estimate(self, operation: str, num_items: int,
                 avg_input_tokens: int, avg_output_tokens: int) -> CostEstimate:
        """Calculate estimated cost across all configured models."""
        ...

    def display_estimate(self, estimate: CostEstimate) -> None:
        """Print cost table to stdout (haiku/sonnet/opus/gpt-4o/gpt-4o-mini)."""
        ...

    def require_approval(self, estimate: CostEstimate) -> None:
        """Raise SystemExit if not approved."""
        ...
```

**Dependencies:** `src.config`, `src.telemetry`

**Design Notes:** Every CLI subcommand that makes LLM calls instantiates CostGate. Without `--approve`, it prints the estimate table and exits. This is the primary defense against accidental large bills.

---

### 4.3 `src/telemetry.py` -- Usage Tracking

**Responsibility:** Record every LLM call's token usage, latency, and cost. Accumulate per-pipeline.

**Interface:**
```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class LLMCallRecord:
    timestamp: datetime
    provider: str                # "anthropic" | "openai" | "claude_cli"
    model: str                   # e.g. "claude-sonnet-4-20250514"
    operation: str               # "contextualize" | "generate_answer" | "judge" | "question_gen"
    pipeline: str | None         # "traditional" | "contextual" | "hybrid" | "modern" | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    success: bool
    error: str | None = None

class TelemetryTracker:
    def __init__(self, config: AppConfig):
        self._records: list[LLMCallRecord] = []
        ...

    def record(self, call: LLMCallRecord) -> None:
        """Append a call record."""
        ...

    def get_pipeline_summary(self, pipeline: str) -> dict:
        """Return {total_calls, total_input_tokens, total_output_tokens, total_cost, avg_latency}."""
        ...

    def get_operation_summary(self, operation: str) -> dict:
        """Return summary grouped by operation type."""
        ...

    def display_summary(self) -> None:
        """Print formatted summary table to stdout."""
        ...

    def export_json(self, path: str) -> None:
        """Persist all records to JSON."""
        ...
```

**Dependencies:** `src.config`

**Design Notes:** TelemetryTracker is a singleton shared across all modules. Each LLM provider auto-records calls after completion. Results include per-pipeline cost breakdowns so pipelines can be compared on efficiency.

---

### 4.4 `src/llm/base.py` -- Abstract LLM Provider

**Responsibility:** Define the common interface all LLM providers implement.

**Interface:**
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float
    cost_usd: float

class BaseLLMProvider(ABC):
    def __init__(self, config: AppConfig, telemetry: TelemetryTracker):
        self.config = config
        self.telemetry = telemetry

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None,
                 temperature: float | None = None,
                 max_tokens: int | None = None) -> LLMResponse:
        """Send prompt to LLM, return structured response."""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier string."""
        ...

    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Look up per-token pricing from config, compute cost."""
        ...
```

**Dependencies:** `src.config`, `src.telemetry`

---

### 4.5 `src/llm/anthropic_provider.py` -- Anthropic SDK Provider

**Responsibility:** Implement LLM calls via the `anthropic` Python SDK.

**Interface:**
```python
class AnthropicProvider(BaseLLMProvider):
    def __init__(self, config: AppConfig, telemetry: TelemetryTracker):
        super().__init__(config, telemetry)
        self.client = anthropic.Anthropic(api_key=os.environ[config.llm.anthropic_api_key_env])

    def complete(self, prompt: str, system: str | None = None,
                 temperature: float | None = None,
                 max_tokens: int | None = None) -> LLMResponse:
        ...

    def get_model_name(self) -> str:
        return self.config.models.llm_model
```

**Dependencies:** `anthropic` SDK, `src.llm.base`

**Design Notes:** Token counts come directly from the API response (`usage.input_tokens`, `usage.output_tokens`). Cost is calculated using the pricing table in config.

---

### 4.6 `src/llm/openai_provider.py` -- OpenAI SDK Provider

**Responsibility:** Implement LLM calls via the `openai` Python SDK.

**Interface:**
```python
class OpenAIProvider(BaseLLMProvider):
    def __init__(self, config: AppConfig, telemetry: TelemetryTracker):
        super().__init__(config, telemetry)
        self.client = openai.OpenAI(api_key=os.environ[config.llm.openai_api_key_env])

    def complete(self, prompt: str, system: str | None = None,
                 temperature: float | None = None,
                 max_tokens: int | None = None) -> LLMResponse:
        ...

    def get_model_name(self) -> str:
        return self.config.models.llm_model
```

**Dependencies:** `openai` SDK, `src.llm.base`

**Design Notes:** Maps to OpenAI's chat completion API. System message goes in `messages[0]` with `role: "system"`.

---

### 4.7 `src/llm/claude_cli_provider.py` -- Claude CLI Provider

**Responsibility:** Shell out to the `claude` CLI tool, parse stdout for responses.

**Interface:**
```python
class ClaudeCLIProvider(BaseLLMProvider):
    def complete(self, prompt: str, system: str | None = None,
                 temperature: float | None = None,
                 max_tokens: int | None = None) -> LLMResponse:
        """Run `claude` as subprocess, capture stdout, parse response."""
        ...

    def get_model_name(self) -> str:
        return "claude-cli"
```

**Dependencies:** `subprocess`, `src.llm.base`

**Design Notes:** Token tracking is limited -- the CLI does not reliably report token counts. `input_tokens` and `output_tokens` are estimated from character counts (chars / 4). `cost_usd` is approximate. This is an accepted tradeoff per author decision.

---

### 4.8 `src/ingestion/pdf_extractor.py` -- PDF Extraction

**Responsibility:** Extract text from PDFs to Markdown, preserving page boundaries.

**Interface:**
```python
@dataclass
class ExtractedDocument:
    source_file: str
    pages: list[PageContent]       # ordered by page number
    full_text: str                 # concatenated Markdown
    total_pages: int

@dataclass
class PageContent:
    page_number: int
    text: str

def extract_pdf(pdf_path: str) -> ExtractedDocument:
    """Extract PDF to Markdown with page metadata using pymupdf4llm."""
    ...

def extract_all(papers_dir: str) -> list[ExtractedDocument]:
    """Extract all PDFs in directory."""
    ...
```

**Dependencies:** `pymupdf4llm`

**Design Notes:** pymupdf4llm outputs Markdown which preserves structure better than plain text. Page boundaries are tracked so generated answers can cite source page numbers.

---

### 4.9 `src/ingestion/chunker.py` -- Document-Level Chunking

**Responsibility:** Split extracted documents into overlapping chunks at the document level (not per-page).

**Interface:**
```python
@dataclass
class Chunk:
    chunk_id: str                  # deterministic hash (source_file + chunk_index)
    doc_id: str                    # source document identifier
    source_file: str
    chunk_index: int
    text: str
    start_page: int
    end_page: int
    token_count: int

def chunk_document(doc: ExtractedDocument, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split document into overlapping chunks. Track page spans."""
    ...

def chunk_all(docs: list[ExtractedDocument], config: ChunkingConfig) -> list[Chunk]:
    """Chunk all documents."""
    ...
```

**Dependencies:** `src.ingestion.pdf_extractor`, `src.config`

**Design Notes:** Chunks at the document level (concatenated full text), not per-page. This avoids splitting mid-paragraph at page boundaries. Page spans (start_page, end_page) are computed by tracking character offsets against page boundaries. chunk_id is a deterministic hash of `source_file + chunk_index` for idempotency.

---

### 4.10 `src/ingestion/contextualizer.py` -- LLM Chunk Contextualization

**Responsibility:** Enrich each chunk with document-level context using an LLM. Cache results for crash recovery.

**Interface:**
```python
@dataclass
class ContextualizedChunk:
    chunk: Chunk
    contextualized_text: str       # LLM-prepended context + original text
    llm_context: str               # just the context portion

class Contextualizer:
    def __init__(self, provider: BaseLLMProvider, config: AppConfig):
        self.cache_dir = config.paths.cache_dir
        ...

    def contextualize(self, chunk: Chunk, document_summary: str) -> ContextualizedChunk:
        """Add document context to a single chunk. Check cache first."""
        ...

    def contextualize_all(self, chunks: list[Chunk],
                          docs: list[ExtractedDocument]) -> list[ContextualizedChunk]:
        """Contextualize all chunks with progress tracking and stall detection."""
        ...

    def _load_cache(self, chunk_id: str) -> str | None:
        """Load cached contextualization if exists."""
        ...

    def _save_cache(self, chunk_id: str, context: str) -> None:
        """Persist contextualization to JSON cache file."""
        ...
```

**Dependencies:** `src.llm.base`, `src.ingestion.chunker`, `src.config`

**Design Notes:** Cache is stored as one JSON file per chunk_id in `data/cache/`. On restart, completed chunks are skipped. Progress bar and ETA displayed during full run. Stall detection aborts if no chunk completes within `stall_timeout_seconds`.

---

### 4.11 `src/ingestion/embedder.py` -- Embedding

**Responsibility:** Generate vector embeddings for chunks using bge-base-en-v1.5 on MPS.

**Interface:**
```python
class Embedder:
    def __init__(self, config: AppConfig):
        self.model = None          # lazy load for MPS memory management
        ...

    def load_model(self) -> None:
        """Load sentence-transformers model to MPS device."""
        ...

    def unload_model(self) -> None:
        """Free MPS memory."""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of embedding vectors."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query text."""
        ...
```

**Dependencies:** `sentence-transformers`, `torch`, `src.config`

**Design Notes:** Model is loaded on demand and unloaded after use to manage MPS memory (bge-base ~416MB + reranker ~1.1GB would exceed comfortable limits if loaded simultaneously on 16GB M2). Batch embedding for ingestion, single embedding for queries.

---

### 4.12 `src/ingestion/store.py` -- ChromaDB Collection Management

**Responsibility:** Create, populate, and query ChromaDB collections.

**Interface:**
```python
class ChromaStore:
    def __init__(self, config: AppConfig):
        self.client = chromadb.PersistentClient(path=config.paths.chroma_dir)
        ...

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Get existing or create new collection."""
        ...

    def upsert_chunks(self, collection_name: str, chunks: list[Chunk],
                      embeddings: list[list[float]],
                      texts: list[str] | None = None) -> None:
        """Upsert chunks with embeddings and metadata. Idempotent via chunk_id."""
        ...

    def query(self, collection_name: str, query_embedding: list[float],
              top_k: int) -> list[RetrievedChunk]:
        """Dense similarity search. Returns top-K chunks with scores."""
        ...

    def get_all_documents(self, collection_name: str) -> list[dict]:
        """Retrieve all documents (for BM25 index building)."""
        ...

    def count(self, collection_name: str) -> int:
        """Return number of documents in collection."""
        ...

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict                 # doc_id, source_file, start_page, end_page, chunk_index
```

**Dependencies:** `chromadb`, `src.config`

**Design Notes:** `upsert` (not `add`) ensures idempotency -- re-running ingestion with same chunk_ids does not duplicate data. Metadata stored per chunk enables source citations in answers.

---

### 4.13 `src/retrieval/dense.py` -- Dense Retrieval

**Responsibility:** Vector similarity search against ChromaDB.

**Interface:**
```python
class DenseRetriever:
    def __init__(self, store: ChromaStore, embedder: Embedder, config: AppConfig):
        ...

    def retrieve(self, query: str, collection_name: str,
                 top_k: int | None = None) -> list[RetrievedChunk]:
        """Embed query, search collection, return top-K results."""
        ...
```

**Dependencies:** `src.ingestion.store`, `src.ingestion.embedder`, `src.config`

---

### 4.14 `src/retrieval/bm25.py` -- BM25 Sparse Retrieval

**Responsibility:** Keyword-based retrieval using BM25 scoring.

**Interface:**
```python
class BM25Retriever:
    def __init__(self, store: ChromaStore, config: AppConfig):
        self._index: BM25Okapi | None = None
        self._corpus: list[dict] | None = None
        ...

    def build_index(self, collection_name: str) -> None:
        """Load all documents from ChromaDB, tokenize, build BM25 index."""
        ...

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Tokenize query, score against BM25 index, return top-K."""
        ...
```

**Dependencies:** `rank_bm25`, `src.ingestion.store`, `src.config`

**Design Notes:** BM25 index is built in-memory from ChromaDB documents. The index is built once per session and reused across queries. Tokenization uses simple whitespace + lowercasing (can be upgraded to more sophisticated tokenization later).

---

### 4.15 `src/retrieval/fusion.py` -- Reciprocal Rank Fusion

**Responsibility:** Merge ranked lists from dense and BM25 retrieval using RRF.

**Interface:**
```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = 60,
    top_k: int | None = None
) -> list[RetrievedChunk]:
    """
    Merge multiple ranked lists using RRF.

    RRF score for document d = sum over all lists of: 1 / (k + rank(d))

    Args:
        ranked_lists: List of ranked result lists (e.g., [dense_results, bm25_results])
        k: RRF constant (default 60, controls how much rank matters)
        top_k: Number of results to return

    Returns:
        Merged and re-ranked list of chunks with RRF scores.
    """
    ...
```

**Dependencies:** `src.ingestion.store` (RetrievedChunk type)

**Design Notes:** RRF is score-agnostic -- it uses only rank positions, not raw scores. This avoids the score normalization problem when combining dense (cosine similarity) and BM25 (term frequency) scores. The `k` constant (default 60) controls the diminishing-returns curve.

---

### 4.16 `src/retrieval/reranker.py` -- Cross-Encoder Reranking

**Responsibility:** Re-score (query, chunk) pairs using bge-reranker-v2-m3 cross-encoder.

**Interface:**
```python
class Reranker:
    def __init__(self, config: AppConfig):
        self.model = None          # lazy load for MPS memory management
        ...

    def load_model(self) -> None:
        """Load cross-encoder model to MPS device."""
        ...

    def unload_model(self) -> None:
        """Free MPS memory."""
        ...

    def rerank(self, query: str, chunks: list[RetrievedChunk],
               top_k: int | None = None) -> list[RetrievedChunk]:
        """Score each (query, chunk) pair, re-sort by cross-encoder score, return top-K."""
        ...
```

**Dependencies:** `sentence-transformers`, `torch`, `src.config`

**Design Notes:** Cross-encoder performs full attention between query and chunk text (unlike bi-encoder embeddings which encode independently). This is more accurate but slower -- only applied to the already-filtered fusion results (not the full corpus). Model loaded/unloaded on demand to share MPS memory with embedder.

---

### 4.17 `src/retrieval/pipeline.py` -- Pipeline Orchestrator

**Responsibility:** Orchestrate the full retrieval-to-answer flow for each of the 4 methods.

**Interface:**
```python
from enum import Enum

class PipelineType(Enum):
    TRADITIONAL = "traditional"
    CONTEXTUAL = "contextual"
    HYBRID = "hybrid"
    MODERN = "modern"

@dataclass
class PipelineResult:
    pipeline: PipelineType
    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    citations: list[Citation]
    llm_response: LLMResponse
    retrieval_latency_ms: float
    total_latency_ms: float

@dataclass
class Citation:
    source_file: str
    start_page: int
    end_page: int
    chunk_text_preview: str        # first 100 chars

class Pipeline:
    def __init__(self, pipeline_type: PipelineType, config: AppConfig,
                 store: ChromaStore, embedder: Embedder,
                 bm25: BM25Retriever | None, reranker: Reranker | None,
                 llm_provider: BaseLLMProvider, telemetry: TelemetryTracker):
        ...

    def run(self, query: str) -> PipelineResult:
        """Execute full pipeline: retrieve -> (optional fusion/rerank) -> LLM answer."""
        ...

def create_pipeline(pipeline_type: PipelineType, config: AppConfig, ...) -> Pipeline:
    """Factory: create pipeline with correct components for the given type."""
    ...
```

**Dependencies:** All retrieval modules, `src.llm.base`, `src.ingestion.store`, `src.telemetry`

**Design Notes:** Each pipeline type uses only the retrieval components it needs:
- Traditional: dense only (from `rag_traditional_v1`)
- Contextual: dense only (from `rag_contextualized_v1`)
- Hybrid: dense + BM25 + RRF (from `rag_contextualized_v1`)
- Modern: dense + BM25 + RRF + reranker (from `rag_contextualized_v1`)

---

### 4.18 `src/evaluation/judge.py` -- LLM-as-Judge

**Responsibility:** Score answers on 4 metrics using LLM-as-judge prompts.

**Interface:**
```python
class MetricType(Enum):
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"

@dataclass
class JudgeResult:
    metric: MetricType
    score: float                   # 0.0 to 1.0
    justification: str
    llm_response: LLMResponse

class Judge:
    def __init__(self, provider: BaseLLMProvider, config: AppConfig):
        ...

    def score(self, metric: MetricType, question: str, answer: str,
              contexts: list[str], ground_truth: str) -> JudgeResult:
        """Score a single answer on a single metric."""
        ...

    def score_all_metrics(self, question: str, answer: str,
                          contexts: list[str],
                          ground_truth: str) -> list[JudgeResult]:
        """Score all 4 metrics. Returns list of JudgeResults."""
        ...
```

**Dependencies:** `src.llm.base`, `src.config`

**Design Notes:** Each metric has a dedicated prompt template instructing the judge LLM. No RAGAS library -- all prompts are hand-crafted. The judge model can be different from the answer-generation model (configured via `config.models.judge_model`).

---

### 4.19 `src/evaluation/question_gen.py` -- Question Generation

**Responsibility:** Generate evaluation questions with ground-truth answers from the paper corpus.

**Interface:**
```python
@dataclass
class EvalQuestion:
    question_id: str
    question: str
    ground_truth: str
    source_files: list[str]
    source_pages: list[int]

class QuestionGenerator:
    def __init__(self, provider: BaseLLMProvider, config: AppConfig):
        ...

    def generate_candidates(self, documents: list[ExtractedDocument],
                            count: int = 30) -> list[EvalQuestion]:
        """Generate candidate questions from papers. Author curates to 20."""
        ...

    def save(self, questions: list[EvalQuestion], path: str) -> None:
        """Persist to questions.json."""
        ...

    def load(self, path: str) -> list[EvalQuestion]:
        """Load from questions.json."""
        ...
```

**Dependencies:** `src.llm.base`, `src.ingestion.pdf_extractor`, `src.config`

**Design Notes:** Generates ~30 candidates; author reviews and selects 20. Questions should cover diverse RAG topics across the corpus. Each question includes ground truth with page citations for evaluation.

---

### 4.20 `src/evaluation/runner.py` -- Evaluation Runner

**Responsibility:** Run all pipelines against all questions, collect judge scores, persist incrementally.

**Interface:**
```python
@dataclass
class EvalResult:
    question_id: str
    pipeline: PipelineType
    metric: MetricType
    score: float
    justification: str
    answer: str
    retrieved_chunks: list[str]
    cost_usd: float
    latency_ms: float

class EvaluationRunner:
    def __init__(self, pipelines: dict[PipelineType, Pipeline],
                 judge: Judge, config: AppConfig):
        self.results_dir = config.paths.results_dir
        ...

    def run(self, questions: list[EvalQuestion],
            pipeline_filter: list[PipelineType] | None = None,
            question_filter: list[str] | None = None) -> list[EvalResult]:
        """
        Run evaluation. Check cache for completed results.
        Persist each result immediately after completion.
        """
        ...

    def _result_path(self, question_id: str, pipeline: str, metric: str) -> str:
        """Deterministic path: results_dir/{question_id}_{pipeline}_{metric}.json"""
        ...

    def _is_completed(self, question_id: str, pipeline: str, metric: str) -> bool:
        """Check if result already cached on disk."""
        ...
```

**Dependencies:** `src.retrieval.pipeline`, `src.evaluation.judge`, `src.config`

**Design Notes:** Results are persisted as individual JSON files (`{question_id}_{pipeline}_{metric}.json`). On restart, completed results are skipped. Pipeline and question filters allow partial evaluation runs.

---

### 4.21 `src/evaluation/comparison.py` -- Comparison & Reporting

**Responsibility:** Aggregate evaluation results into cross-method comparison tables.

**Interface:**
```python
class ComparisonReport:
    def __init__(self, results: list[EvalResult], telemetry: TelemetryTracker):
        ...

    def per_pipeline_scores(self) -> dict[str, dict[str, float]]:
        """Average scores per pipeline per metric."""
        ...

    def per_question_breakdown(self) -> dict:
        """Detailed per-question, per-pipeline, per-metric breakdown."""
        ...

    def cost_comparison(self) -> dict[str, float]:
        """Total cost per pipeline."""
        ...

    def render_table(self, output_format: str = "table") -> str:
        """Render as terminal table, JSON, or Markdown."""
        ...

    def render_side_by_side(self, question_id: str) -> str:
        """Show all 4 pipeline answers for one question side by side."""
        ...
```

**Dependencies:** `src.evaluation.runner`, `src.telemetry`

---

### 4.22 `src/interactive/repl.py` -- Interactive REPL

**Responsibility:** Terminal-based Q&A interface. Query all methods, display answers side-by-side with citations.

**Interface:**
```python
class REPL:
    def __init__(self, pipelines: dict[PipelineType, Pipeline], config: AppConfig):
        ...

    def start(self) -> None:
        """Enter read-eval-print loop. Commands: query, compare, cost, quit."""
        ...

    def _display_results(self, results: list[PipelineResult]) -> None:
        """Rich terminal table with side-by-side answers and citations."""
        ...
```

**Dependencies:** `rich`, `src.retrieval.pipeline`, `src.config`

---

### 4.23 `src/cli/main.py` -- CLI Entry Point

**Responsibility:** argparse-based CLI with subcommands. Entry point for all operations.

**Interface:**
```python
def main():
    parser = argparse.ArgumentParser(description="RAG Comparison System")
    subparsers = parser.add_subparsers(dest="command")

    # Subcommands: ingest, contextualize, generate-questions, evaluate, interactive, compare
    ...

if __name__ == "__main__":
    main()
```

**Dependencies:** All modules (dispatches to appropriate module per subcommand)

**Design Notes:** Each subcommand instantiates CostGate. Without `--approve`, prints estimate and exits. This is the only user-facing entry point.

**PENDING AUTHOR APPROVAL**

---

## 5. Configuration Schema

### `config/default.yaml`

```yaml
paths:
  data_dir: "data"
  chroma_dir: "data/chroma_db"
  cache_dir: "data/cache"
  results_dir: "data/results"
  papers_dir: "data/papers"

models:
  embedding_model: "BAAI/bge-base-en-v1.5"
  reranker_model: "BAAI/bge-reranker-v2-m3"
  llm_model: "claude-sonnet-4-20250514"
  judge_model: "claude-haiku-4-5-20241022"

chunking:
  chunk_size: 400         # tokens (to be confirmed after research -- max 512 for bge-base)
  chunk_overlap: 50       # tokens

retrieval:
  top_k_dense: 20
  top_k_bm25: 20
  top_k_fusion: 20
  top_k_rerank: 10
  rrf_k: 60               # RRF constant

llm:
  provider: "anthropic"    # "anthropic" | "openai" | "claude_cli"
  anthropic_api_key_env: "ANTHROPIC_API_KEY"
  openai_api_key_env: "OPENAI_API_KEY"
  temperature: 0.0
  max_tokens: 2048

cost:
  pricing:
    claude-haiku-4-5-20241022:
      input_per_1k: 0.001
      output_per_1k: 0.005
    claude-sonnet-4-20250514:
      input_per_1k: 0.003
      output_per_1k: 0.015
    claude-opus-4-20250514:
      input_per_1k: 0.015
      output_per_1k: 0.075
    gpt-4o:
      input_per_1k: 0.005
      output_per_1k: 0.015
    gpt-4o-mini:
      input_per_1k: 0.00015
      output_per_1k: 0.0006

evaluation:
  metrics:
    - context_precision
    - context_recall
    - faithfulness
    - answer_relevancy
  judge_temperature: 0.0

concurrency:
  max_workers: 4
  stall_timeout_seconds: 120
```

### CLI Override Mechanism

CLI arguments override YAML values. Mapping:

| CLI Argument | Config Path | Example |
|-------------|-------------|---------|
| `--config` | (loads alternate YAML) | `--config my_config.yaml` |
| `--chunk-size` | `chunking.chunk_size` | `--chunk-size 400` |
| `--chunk-overlap` | `chunking.chunk_overlap` | `--chunk-overlap 50` |
| `--provider` | `llm.provider` | `--provider openai` |
| `--model` | `models.llm_model` | `--model gpt-4o` |
| `--top-k` | `retrieval.top_k_dense` | `--top-k 15` |
| `--temperature` | `llm.temperature` | `--temperature 0.1` |
| `--max-workers` | `concurrency.max_workers` | `--max-workers 8` |

Precedence: CLI argument > YAML config > Pydantic defaults.

**PENDING AUTHOR APPROVAL**

---

## 6. CLI Interface Design

### Subcommands

#### `ingest` -- Ingest PDFs into traditional ChromaDB collection

```
python -m src.cli.main ingest [OPTIONS]

Options:
  --config PATH          Path to YAML config file (default: config/default.yaml)
  --approve              Execute (without this flag, only prints cost estimate)
  --papers-dir PATH      Override papers directory

Example:
  # Estimate only
  python -m src.cli.main ingest

  # Execute
  python -m src.cli.main ingest --approve --papers-dir data/papers
```

#### `contextualize` -- LLM-contextualize chunks into contextualized collection

```
python -m src.cli.main contextualize [OPTIONS]

Options:
  --config PATH          Path to YAML config file
  --approve              Execute
  --provider TEXT        LLM provider override (anthropic/openai/claude_cli)
  --model TEXT           Model override
  --sample N             Run on N chunks only (for quality review before full run)

Example:
  # Preview cost for full contextualization
  python -m src.cli.main contextualize

  # Run on 5 chunks for quality review
  python -m src.cli.main contextualize --approve --sample 5

  # Full run with OpenAI
  python -m src.cli.main contextualize --approve --provider openai --model gpt-4o-mini
```

#### `generate-questions` -- Generate evaluation questions from papers

```
python -m src.cli.main generate-questions [OPTIONS]

Options:
  --config PATH          Path to YAML config file
  --approve              Execute
  --count N              Number of candidate questions to generate (default: 30)
  --provider TEXT        LLM provider override
  --model TEXT           Model override

Example:
  # Generate 30 candidates for author curation
  python -m src.cli.main generate-questions --approve --count 30
```

#### `evaluate` -- Run evaluation across pipelines

```
python -m src.cli.main evaluate [OPTIONS]

Options:
  --config PATH          Path to YAML config file
  --approve              Execute
  --pipelines TEXT       Comma-separated pipeline list (default: all)
  --questions TEXT       Comma-separated question IDs to evaluate (default: all)
  --provider TEXT        LLM provider override
  --model TEXT           Model override

Example:
  # Cost estimate for full evaluation
  python -m src.cli.main evaluate

  # Evaluate traditional pipeline only
  python -m src.cli.main evaluate --approve --pipelines traditional

  # Evaluate specific questions across all pipelines
  python -m src.cli.main evaluate --approve --questions q1,q5,q10
```

#### `interactive` -- Start interactive Q&A REPL

```
python -m src.cli.main interactive [OPTIONS]

Options:
  --config PATH          Path to YAML config file
  --pipelines TEXT       Comma-separated pipeline list (default: all available)
  --provider TEXT        LLM provider override
  --model TEXT           Model override

Example:
  # Start REPL with all 4 pipelines
  python -m src.cli.main interactive

  # Compare just traditional vs modern
  python -m src.cli.main interactive --pipelines traditional,modern
```

#### `compare` -- Generate comparison report from evaluation results

```
python -m src.cli.main compare [OPTIONS]

Options:
  --config PATH          Path to YAML config file
  --results-dir PATH     Override results directory
  --output-format TEXT   Output format: table, json, md (default: table)

Example:
  # Terminal table
  python -m src.cli.main compare

  # Markdown report
  python -m src.cli.main compare --output-format md > docs/comparison-analysis.md
```

**PENDING AUTHOR APPROVAL**

---

## 7. ChromaDB Collection Design

### Collections

| Collection Name | Contents | Used By |
|----------------|----------|---------|
| `rag_traditional_v1` | Raw chunks with embeddings | Traditional pipeline |
| `rag_contextualized_v1` | LLM-contextualized chunks with embeddings | Contextual, Hybrid, Modern pipelines |

### Document/Chunk Metadata Schema

Each document (chunk) stored in ChromaDB carries the following metadata:

```json
{
  "doc_id": "arxiv_2312.10997",
  "chunk_id": "a3f8b2c1d4e5...",
  "source_file": "2312.10997.pdf",
  "chunk_index": 7,
  "start_page": 3,
  "end_page": 4,
  "is_contextualized": false,
  "token_count": 387
}
```

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | string | Document identifier (derived from filename) |
| `chunk_id` | string | Deterministic hash -- primary key for upsert |
| `source_file` | string | Original PDF filename |
| `chunk_index` | int | Position of chunk within document (0-indexed) |
| `start_page` | int | First page this chunk spans |
| `end_page` | int | Last page this chunk spans |
| `is_contextualized` | bool | Whether chunk has LLM-added context |
| `token_count` | int | Token count of the chunk text |

### Chunk ID Generation Strategy

Chunk IDs are deterministic hashes to ensure idempotency:

```python
import hashlib

def generate_chunk_id(source_file: str, chunk_index: int) -> str:
    """Deterministic chunk ID. Same input always produces same ID."""
    raw = f"{source_file}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

Re-running ingestion with the same PDFs and chunk configuration produces the same chunk_ids. ChromaDB `upsert` (not `add`) means duplicates are overwritten, not created.

### Embedding Function Configuration

Embeddings are generated externally (by `embedder.py`) and passed to ChromaDB at upsert time. ChromaDB is not configured with its own embedding function -- this keeps the embedding model choice centralized in config and avoids ChromaDB's default embedding behavior.

**PENDING AUTHOR APPROVAL**

---

## 8. LLM Provider Interface

### Abstract Base Class

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    text: str                  # Generated text
    input_tokens: int          # Prompt tokens consumed
    output_tokens: int         # Completion tokens generated
    model: str                 # Model identifier (e.g. "claude-sonnet-4-20250514")
    latency_ms: float          # Wall-clock time for the call
    cost_usd: float            # Calculated cost based on pricing table

class BaseLLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: str | None = None,
                 temperature: float | None = None,
                 max_tokens: int | None = None) -> LLMResponse:
        """Send prompt, return structured response with usage metadata."""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        ...
```

### Provider Implementations

| Provider | SDK/Method | Token Tracking | Cost Tracking | Notes |
|----------|-----------|----------------|---------------|-------|
| **Anthropic** | `anthropic` SDK | Exact (from `response.usage`) | Exact (pricing table lookup) | Primary provider. Full structured usage data. |
| **OpenAI** | `openai` SDK | Exact (from `response.usage`) | Exact (pricing table lookup) | Maps to chat completion API. System message in `messages[0]`. |
| **Claude CLI** | `subprocess.run(["claude", ...])` | Estimated (chars/4) | Approximate | Parses stdout. No structured usage data. Slower than API. |

### Provider Selection

Provider is selected via `config.llm.provider`:

```python
def create_provider(config: AppConfig, telemetry: TelemetryTracker) -> BaseLLMProvider:
    match config.llm.provider:
        case "anthropic":
            return AnthropicProvider(config, telemetry)
        case "openai":
            return OpenAIProvider(config, telemetry)
        case "claude_cli":
            return ClaudeCLIProvider(config, telemetry)
        case _:
            raise ValueError(f"Unknown provider: {config.llm.provider}")
```

**PENDING AUTHOR APPROVAL**

---

## 9. Cost Gate Design

### Estimate Phase

When a CLI subcommand runs without `--approve`:

1. Count the number of operations (chunks to contextualize, questions to evaluate, etc.)
2. Estimate average input/output tokens per operation
3. Multiply by per-token pricing for each model tier
4. Display cost table and exit

### Cost Table Format

```
Cost Estimate: contextualize (1,500 chunks)
===========================================================
Model                  Input Tokens  Output Tokens  Est. Cost
-----------------------------------------------------------
claude-haiku-4-5          750,000       150,000      $1.50
claude-sonnet-4           750,000       150,000      $4.50
claude-opus-4             750,000       150,000     $22.50
gpt-4o                   750,000       150,000      $6.00
gpt-4o-mini              750,000       150,000      $0.20
===========================================================

Run with --approve to execute. Use --provider and --model to select.
```

### Approve Phase

When `--approve` is passed:
1. CostGate is constructed with `approved=True`
2. `require_approval()` becomes a no-op
3. Execution proceeds

### Integration with CLI

Every subcommand that makes LLM calls follows this pattern:

```python
def cmd_contextualize(args):
    config = load_config(args.config, cli_overrides_from(args))
    cost_gate = CostGate(config, approved=args.approve)

    # Count work
    chunks = load_chunks(config)
    estimate = cost_gate.estimate("contextualize", len(chunks),
                                  avg_input_tokens=500, avg_output_tokens=100)
    cost_gate.display_estimate(estimate)
    cost_gate.require_approval(estimate)  # exits if not approved

    # Execute
    ...
```

### 9.4 Token Estimation Strategy

Two-phase estimation: **static estimate first, then sample-based refinement** for expensive operations.

**Phase A — Static Estimate (all operations)**

Use configured averages to produce rough estimate before any LLM calls:

| Operation | Formula | Config Params |
|-----------|---------|---------------|
| Contextualize | `num_chunks × (avg_doc_tokens + avg_chunk_tokens)` input, `num_chunks × contextualization_output_tokens` output | `avg_doc_tokens: 8000`, `avg_chunk_tokens: 400`, `contextualization_output_tokens: 100` |
| Generate answer | `num_questions × (avg_chunk_tokens × top_k + query_overhead_tokens)` input, `num_questions × answer_max_tokens` output | `query_overhead_tokens: 200`, `answer_max_tokens: 500` |
| Judge | `num_questions × num_pipelines × num_metrics × (question_tokens + ground_truth_tokens + context_tokens + answer_tokens + judge_prompt_tokens)` input, `num_questions × num_pipelines × num_metrics × judge_output_tokens` output | `judge_prompt_tokens: 300`, `judge_output_tokens: 200` |
| Question gen | `num_papers × avg_paper_tokens` input, `num_questions × question_output_tokens` output | `avg_paper_tokens: 15000`, `question_output_tokens: 300` |

Static estimate displayed on every run without `--approve`. Accuracy: ±50% — sufficient for order-of-magnitude cost awareness.

**Phase B — Sample-Based Refinement (expensive operations only)**

Before full execution of contextualization or evaluation, run on a small sample and measure actual tokens:

```
1. Run operation on N=5 items
2. Record actual input_tokens, output_tokens per item
3. Extrapolate: actual_avg × total_items
4. Display refined estimate alongside static estimate
5. Require --approve again for full run
```

Applies to:
- `contextualize`: sample 5 chunks, measure, extrapolate to all chunks
- `evaluate`: sample 1 question × 1 pipeline × 4 metrics, extrapolate to full matrix
- `generate-questions`: sample 2 papers, extrapolate to all papers

Does NOT apply to (too cheap to bother):
- `interactive` query answering (single LLM call per query)

**Config for estimation defaults** (add to `default.yaml`):

```yaml
cost_estimation:
  # Static estimate defaults (Phase A)
  avg_doc_tokens: 8000
  avg_chunk_tokens: 400
  contextualization_output_tokens: 100
  query_overhead_tokens: 200
  answer_max_tokens: 500
  judge_prompt_tokens: 300
  judge_output_tokens: 200
  avg_paper_tokens: 15000
  question_output_tokens: 300
  # Sample-based refinement (Phase B)
  sample_size_contextualize: 5
  sample_size_evaluate: 1      # 1 question × 1 pipeline × 4 metrics
  sample_size_question_gen: 2
```

**PENDING AUTHOR APPROVAL**

---

## 10. Telemetry Design

### LLMCallRecord Fields

```python
@dataclass
class LLMCallRecord:
    timestamp: datetime            # When the call was made
    provider: str                  # "anthropic" | "openai" | "claude_cli"
    model: str                     # Full model identifier
    operation: str                 # "contextualize" | "generate_answer" | "judge" | "question_gen"
    pipeline: str | None           # Which pipeline this call serves (None for shared ops)
    input_tokens: int              # Prompt tokens
    output_tokens: int             # Completion tokens
    latency_ms: float              # Wall-clock milliseconds
    cost_usd: float                # Calculated cost
    success: bool                  # Whether call succeeded
    error: str | None              # Error message if failed
```

### Per-Pipeline Accumulator

TelemetryTracker maintains an in-memory list of all LLMCallRecords. Summaries are computed on demand:

```
Pipeline Cost Summary
=================================================================
Pipeline        Calls  Input Tokens  Output Tokens  Cost     Avg Latency
-----------------------------------------------------------------
traditional        20       40,000        20,000    $0.90      1,200ms
contextual         20       40,000        20,000    $0.90      1,150ms
hybrid             20       40,000        20,000    $0.90      1,300ms
modern             20       40,000        20,000    $0.90      1,250ms
-----------------------------------------------------------------
contextualize   1,500      750,000       150,000    $4.50        800ms
judge             320      640,000       160,000    $2.40      1,100ms
=================================================================
Total                    1,510,000       390,000    $10.50
```

### Persistence

- All records exported to `data/results/telemetry.json` at end of each CLI run
- Evaluation results include per-question cost data
- Comparison report shows per-pipeline cost breakdown

### Cost Breakdown in Evaluation Results

Each `EvalResult` carries `cost_usd` (the LLM calls for that specific question+pipeline). The comparison report aggregates these to show total cost per pipeline, enabling cost-quality tradeoff analysis.

**PENDING AUTHOR APPROVAL**

---

## 11. Resilience Patterns

### Idempotency

| Operation | Mechanism |
|-----------|-----------|
| Chunk ingestion | Deterministic `chunk_id` (hash of source_file + chunk_index) + ChromaDB `upsert` |
| Contextualization | Per-chunk JSON cache checked before LLM call |
| Evaluation | Per result JSON file checked before judge call |
| Question generation | Output saved to `questions.json`, re-run overwrites |

Re-running any operation with the same configuration produces the same results without duplicating data or repeating completed LLM calls.

### Crash Recovery

| Operation | Cache Location | Recovery Behavior |
|-----------|---------------|-------------------|
| Contextualization | `data/cache/{chunk_id}.json` | On restart, loads existing cache files, skips completed chunks |
| Evaluation | `data/results/{qid}_{pipeline}_{metric}.json` | On restart, checks for existing result files, skips completed |
| Ingestion | ChromaDB persistent storage | ChromaDB handles its own durability; upsert is safe to retry |

Cache files are written atomically (write to temp file, then rename) to prevent corruption from mid-write crashes.

### Stall Detection

- Every LLM call has a configurable timeout (`concurrency.stall_timeout_seconds`, default 120s)
- Long-running batch operations (contextualization, evaluation) track progress
- If no progress within timeout period, operation aborts with clear error message
- Progress bar shows ETA and completion rate

### MPS Memory Management

- bge-base-en-v1.5 (~416MB) and bge-reranker-v2-m3 (~1.1GB) are never loaded simultaneously
- Models loaded on demand via `load_model()`, freed via `unload_model()`
- Pattern: embedder loads for ingestion -> unloads -> reranker loads for Modern pipeline -> unloads
- Total peak MPS usage: ~1.1GB (reranker alone), well within 16GB M2

**PENDING AUTHOR APPROVAL**

---

## 12. Approval Checkpoints

Every item below requires explicit author review and approval before the system proceeds. Artifacts are presented at each checkpoint.

| # | Checkpoint | Phase | Artifacts Presented | What Author Reviews |
|---|-----------|-------|--------------------|--------------------|
| 1 | Directory structure | Phase 0 | Directory tree (section 2 above) | Module layout, file organization |
| 2 | Config schema | Phase 0 | `default.yaml` draft (section 5) | All parameter names, defaults, structure |
| 3 | CLI subcommand design | Phase 0 | Subcommand list with args (section 6) | UX, argument names, workflow |
| 4 | Module interfaces | Phase 0 | Class/function signatures (section 4) | API design, dependencies |
| 5 | Paper list | Phase 1a | Top 20 arXiv RAG papers with titles and URLs | Corpus selection |
| 6 | Chunk sizing | Phase 1c | Research findings + recommended size/overlap | Chunking parameters |
| 7 | Extraction quality | Phase 1b | Sample extracted Markdown from 2-3 papers | PDF extraction fidelity |
| 8 | Contextualization quality | Phase 3 | 5-chunk sample: original vs contextualized | LLM prompt quality, context usefulness |
| 9 | Contextualization cost | Phase 3 | Cost estimate table for full run | Budget approval |
| 10 | Question candidates | Phase 7 | ~30 generated questions + ground truth | Curate to final 20 |
| 11 | Judge prompts | Phase 8 | 4 metric prompt templates | Evaluation methodology |
| 12 | Single-question validation | Phase 8 | 1 question x 1 pipeline x 4 metrics results | Structure and sanity of scores |
| 13 | Full evaluation cost | Phase 8 | Cost estimate for 320 judge + 80 answer calls | Budget approval |
| 14 | Test plan | Phase 9 | Test strategy document | Test coverage, acceptance criteria |

**PENDING AUTHOR APPROVAL**

---

*Document generated for RAG Comparison Project. All sections marked with "PENDING AUTHOR APPROVAL" require explicit sign-off before implementation proceeds.*
