# RAG Concepts Glossary

Key terms encountered during the RAG Comparison Project build. Each entry includes a definition and its specific relevance to this project.

---

## Answer Relevancy

A measure of how directly and completely an answer addresses the question asked, regardless of source grounding. A high-relevancy answer stays on topic, covers the question's full scope, and does not introduce unrelated content. In this project, scored 0.0–1.0 by the LLM judge (claude-haiku-4-5) using a dedicated prompt. All four pipelines scored 0.950 on this metric — indicating the LLM generates relevant answers from any of the retrieval sets provided.

---

## Bi-Encoder

A neural architecture that encodes the query and document independently into a shared vector space, then measures similarity via dot product or cosine distance. Because each is encoded separately, bi-encoders are fast — embeddings can be precomputed and stored offline. The tradeoff is that the model cannot attend to the interaction between query and document during encoding. In this project, `BAAI/bge-base-en-v1.5` is the bi-encoder used for dense retrieval in all four pipelines.

---

## BM25 (Best Match 25)

A probabilistic bag-of-words ranking function derived from TF-IDF. For each document, BM25 scores how well it matches a query based on term frequency within the document (with saturation), inverse document frequency across the corpus, and document length normalization. BM25 is fast, requires no GPU, and excels at exact-match or rare-keyword queries. In this project, BM25 is the sparse retrieval component of the Hybrid and Modern pipelines, implemented via the `rank_bm25` library. It complements dense retrieval by boosting chunks that contain exact query terms (e.g., "RAG-Sequence", "RAG-Token") even if their embedding does not rank them highly.

---

## Chunking

The process of splitting long documents into smaller segments (chunks) before embedding and storing in a vector index. Chunk size controls the granularity of retrieval — smaller chunks are more precise but lose surrounding context; larger chunks carry more context but dilute relevance signals. In this project, documents are chunked at the document level (not per-page) using a 450-token target with 68-token overlap, reflecting a 15% overlap rate. This was chosen after researching academic text characteristics: arXiv paragraphs average 120–200 words (~160–270 tokens), making 450 tokens a natural 1–2 paragraph unit.

---

## ChromaDB

A lightweight, Python-native vector store with local persistent storage and metadata filtering. ChromaDB stores document embeddings alongside metadata (source file, page range, chunk index) and supports cosine similarity search. In this project, two ChromaDB collections are maintained: `rag_traditional_v1` (raw chunks) and `rag_contextualized_v1` (LLM-enriched chunks shared by Contextual, Hybrid, and Modern pipelines). Collections use `upsert` (not `add`) for idempotent re-ingestion.

---

## Context Precision

A retrieval quality metric measuring what fraction of the retrieved chunks are actually relevant to the question. High precision means the retrieval set contains mostly relevant content with little noise. In this project, Modern pipeline achieves 0.700 average precision — the highest of the four — primarily due to cross-encoder reranking filtering out borderline-relevant chunks. Traditional and Contextual score ~0.400–0.450, indicating that roughly half the retrieved chunks are not directly useful for answering the question.

---

## Context Recall

A retrieval quality metric measuring whether the retrieved chunks cover all the information needed to answer the question, relative to the ground truth. High recall means no important details are missed. In this project, Traditional scores 0.750 on context recall (misses relevant chunks on the index hot-swapping question), while Contextual, Hybrid, and Modern all score 1.000. The recall improvement is attributable to LLM contextualization, which anchors specific experimental details to their topic area in the embedding space.

---

## Context Window

The maximum token length an LLM can process in a single call — encompassing both the input (system prompt + retrieved chunks + question) and the output (generated answer). In this project, the answer generation model (claude-sonnet-4-20250514) has a 200K token context window, well above the ~20 chunks × 450 tokens/chunk = ~9,000 tokens of context passed per query. The reranker in Modern pipeline reduces context to 10 chunks (~4,500 tokens), reducing latency and cost per query.

---

## Contextual Chunking

An ingestion technique where each chunk is enriched with document-level context before embedding, using an LLM to prepend a 1–2 sentence summary of where the chunk fits within the document. The goal is to improve retrieval recall by ensuring that specific details (e.g., an experimental result on page 7) carry their document context in the embedding rather than appearing as a decontextualized fragment. In this project, all three advanced pipelines (Contextual, Hybrid, Modern) use contextual chunking. The contextualized text is cached per chunk for crash recovery and cost efficiency.

---

## Cosine Similarity

A measure of similarity between two vectors calculated as the cosine of the angle between them, ranging from -1 (opposite) to 1 (identical direction). In RAG, cosine similarity is used to rank retrieved chunks by how closely their embedding aligns with the query embedding. ChromaDB uses cosine similarity as its distance metric for dense retrieval in this project. Cosine similarity measures semantic direction, not magnitude — two embeddings in the same semantic region score high even if their magnitudes differ.

---

## Cross-Encoder

A neural architecture that takes a (query, document) pair as a single joint input and produces a relevance score using full attention across both. Unlike bi-encoders (which encode query and document independently), cross-encoders can attend to the interaction between query tokens and document tokens, enabling more nuanced relevance judgments. The tradeoff is that cross-encoders cannot precompute document representations — every (query, document) pair must be scored at query time. In this project, `BAAI/bge-reranker-v2-m3` is the cross-encoder used in the Modern pipeline to rerank the RRF-fused results, reducing the final context set from 20 to 10 chunks.

---

## Dense Retrieval

Retrieval based on vector similarity — a query is embedded into a high-dimensional vector, and the most similar document vectors are returned. Dense retrieval captures semantic meaning (e.g., "transformer architecture" matches "attention mechanism" even without shared words) but can miss exact-match keywords. All four pipelines in this project use dense retrieval; they differ in whether BM25 and reranking are added on top. Dense retrieval is performed by ChromaDB using cosine similarity.

---

## Embedding

The process of encoding text into a fixed-size numerical vector that captures its semantic meaning. Semantically similar texts produce vectors that are geometrically close (high cosine similarity). Embeddings are the foundation of dense retrieval — chunks and queries must be embedded with the same model for similarity to be meaningful. In this project, `BAAI/bge-base-en-v1.5` produces 768-dimensional embeddings for all chunks and queries, run locally via `sentence-transformers` on Apple M2 MPS.

---

## Faithfulness

A measure of how well an answer is grounded in the retrieved context — whether the claims in the answer can be attributed to specific retrieved chunks rather than hallucinated by the LLM. A perfectly faithful answer makes no claims beyond what the retrieved chunks support. In this project, scored 0.0–1.0 by the LLM judge. All pipelines score 0.950–1.000, indicating the answer LLM (claude-sonnet-4-20250514) generates faithfully from whatever context it receives.

---

## Few-Shot

A prompting technique where a small number of input-output examples are included in the prompt to guide the LLM's behavior, format, or reasoning style — without any gradient updates to the model. In this project, judge prompts for each metric include implicit few-shot framing (criteria + examples of high/low scores) to calibrate the judge's scoring behavior across different question types.

---

## Ground Truth

A reference answer representing the expected correct response for an evaluation question, used as the gold standard for measuring retrieval and generation quality. In this project, ground truth answers were generated by the question generation module from source paper content and stored in `data/questions.json`. The LLM judge compares pipeline answers against ground truth when scoring context recall and faithfulness.

---

## LLM-as-Judge

An evaluation methodology where a capable LLM is prompted to score another LLM's output on specific quality dimensions, acting as an automated evaluator in place of human raters. In this project, `claude-haiku-4-5-20241022` serves as the judge, scoring each pipeline's answer on four metrics (context precision, context recall, faithfulness, answer relevancy) with a 0.0–1.0 score and written justification. Custom prompt templates are used — the RAGAS library is not used.

---

## MPS (Metal Performance Shaders)

Apple's GPU compute framework for accelerated machine learning inference on Apple Silicon (M1/M2/M3/M4 chips). PyTorch supports MPS as a backend device (`device="mps"`), enabling local model inference without requiring NVIDIA CUDA hardware. In this project, both the embedding model (bge-base-en-v1.5, ~416 MB) and reranker (bge-reranker-v2-m3, ~1.1 GB) run on MPS. The two models are never loaded simultaneously to avoid memory pressure on the 16 GB M2.

---

## RAG (Retrieval-Augmented Generation)

An architecture that augments LLM generation with a retrieval step: given a query, relevant documents are retrieved from an external knowledge base and injected into the LLM prompt as context. RAG allows LLMs to access up-to-date or domain-specific information without retraining, and produces more factual answers by grounding generation in retrieved source text. This project implements and compares four RAG variants — Traditional, Contextual, Hybrid, and Modern — that differ in how they retrieve and process context before generation.

---

## Reciprocal Rank Fusion (RRF)

A rank aggregation algorithm that merges multiple ranked lists without requiring score normalization. Each document's RRF score is the sum of `1 / (k + rank)` across all lists, where `k` is a constant (typically 60) controlling the diminishing-returns curve. Documents appearing near the top of multiple lists receive the highest combined scores. RRF is score-agnostic — it does not matter that cosine similarity scores and BM25 scores are on different scales. In this project, RRF merges dense and BM25 ranked lists in the Hybrid and Modern pipelines using `rrf_k=60`.

---

## Reranking

A post-retrieval step where an initial set of retrieved candidates is re-scored and re-ordered using a more accurate but slower model (typically a cross-encoder). Reranking is applied to a small candidate set (20–100 items) rather than the full corpus, making the computational cost acceptable. In this project, the Modern pipeline applies `bge-reranker-v2-m3` reranking to the top-20 RRF fusion results, returning the top-10 for LLM answer generation. Reranking produces the largest single quality improvement observed across all techniques: +25 percentage points in context precision.

---

## Sparse Retrieval

Retrieval based on lexical term matching, where both documents and queries are represented as high-dimensional sparse vectors (most dimensions are zero, with non-zero values only for terms that appear). BM25 and TF-IDF are the most common sparse retrieval methods. Sparse retrieval excels at exact-match queries and rare technical terms but cannot capture semantic similarity. In this project, BM25 sparse retrieval is used in the Hybrid and Modern pipelines to complement dense (semantic) retrieval.

---

## TF-IDF (Term Frequency-Inverse Document Frequency)

A weighting scheme for text terms that balances how frequently a term appears in a document (TF) against how rare it is across the corpus (IDF). Terms that appear frequently in a document but rarely in the corpus receive high TF-IDF weight, indicating high discriminative value. BM25 is a probabilistic extension of TF-IDF with document length normalization. TF-IDF is the conceptual ancestor of BM25 and is the basis for understanding how sparse retrieval differs from dense retrieval in this project.

---

## Tokenization

The process of splitting text into discrete units (tokens) for processing by a neural model. Tokens are typically subword units — common words may be a single token, while rare words are split into multiple. Token count determines embedding model capacity, LLM context window usage, and API cost. In this project, tokenization matters in two places: (1) chunking uses token count to enforce the 450-token chunk size limit, and (2) LLM API cost is billed per input and output token.

---

## Vector Store

A database optimized for storing and querying high-dimensional vectors (embeddings), supporting operations like approximate nearest neighbor (ANN) search. Unlike relational databases optimized for exact row lookups, vector stores enable "find me the most semantically similar documents to this query" queries via cosine similarity or dot product. In this project, ChromaDB is the vector store. It stores both embeddings and chunk metadata (source file, page range, chunk index) and supports persistent storage on disk.

---

## Zero-Shot

A prompting technique where the LLM is given a task description and input but no worked examples. The model must generalize from its pretraining to perform the task. In this project, answer generation is zero-shot — the retrieval pipeline provides context chunks, and the LLM is instructed to answer based on them without examples of correct answers. Judge scoring is also zero-shot per question (though the metric definition functions as an implicit prompt template).
