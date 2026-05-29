"""Cross-encoder reranker for the Modern RAG pipeline.

Uses bge-reranker-v2-m3 to re-score (query, chunk) pairs with full attention,
providing higher-quality relevance ordering than bi-encoder embeddings alone.

Primary backend: FlagEmbedding (FlagReranker).
Fallback backend: sentence-transformers CrossEncoder (if FlagEmbedding is not installed).

Device selection: CPU always (MPS is detected but not used — CrossEncoder.predict()
hangs indefinitely on MPS; CPU is safe and correct for this model size).
Model is loaded lazily on the first rerank() call and can be freed via unload().
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import AppConfig

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-ranks retrieved chunks using a cross-encoder model.

    Lazy-loads bge-reranker-v2-m3 on the first call to :meth:`rerank`.
    Supports two backends (tried in order):

    1. ``FlagEmbedding.FlagReranker`` — preferred; uses BAAI's optimised inference.
    2. ``sentence_transformers.CrossEncoder`` — fallback if FlagEmbedding is absent.

    Args:
        config: AppConfig — provides ``models.reranker_model`` and
                ``retrieval.top_k_rerank``.
    """

    def __init__(self, config: "AppConfig | None" = None) -> None:
        if config is None:
            from src.config import load_config
            config = load_config()
        self.config = config
        self._model = None          # loaded on first rerank() call
        self._backend: str | None = None   # "flag" or "sentence_transformers"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Score each (query, chunk) pair and return the top-K results.

        Args:
            query:  Natural language query string.
            chunks: Candidate chunks from fusion output. Each dict must contain
                    at minimum a ``text`` key. The full chunk schema (as produced
                    by :func:`src.retrieval.dense.dense_retrieve`) is preserved
                    in the output.
            top_k:  Number of results to return after reranking. Defaults to
                    ``config.retrieval.top_k_rerank``.

        Returns:
            List of chunk dicts (same schema as input plus updated ``score``
            field set to the cross-encoder relevance score), sorted from most
            to least relevant, truncated to *top_k* entries.

        Result dict fields (same as other retrieval modules):
            - ``text``        (str)   chunk text
            - ``doc_id``      (str)   source document identifier
            - ``section``     (str)   section label
            - ``page_start``  (int)   first page this chunk spans
            - ``page_end``    (int)   last page this chunk spans
            - ``source_file`` (str)   original PDF filename
            - ``score``       (float) cross-encoder relevance score (higher = better)
        """
        effective_top_k = top_k if top_k is not None else self.config.retrieval.top_k_rerank

        logger.info(
            "CrossEncoderReranker.rerank: num_candidates=%d top_k=%d",
            len(chunks),
            effective_top_k,
        )

        if not chunks:
            logger.debug("rerank: no candidates, returning empty list")
            return []

        self._load_model()

        t0 = time.perf_counter()

        pairs = [(query, chunk.get("text", "")) for chunk in chunks]
        scores = self._score_pairs(pairs)

        # Attach cross-encoder scores and sort descending.
        scored: list[dict] = []
        for chunk, score in zip(chunks, scores):
            result = dict(chunk)        # shallow copy, preserves all existing keys
            result["score"] = float(score)
            scored.append(result)

        scored.sort(key=lambda x: x["score"], reverse=True)
        top_results = scored[:effective_top_k]

        latency_ms = (time.perf_counter() - t0) * 1000
        top_score = top_results[0]["score"] if top_results else 0.0

        logger.info(
            "CrossEncoderReranker.rerank: returned %d results top_score=%.4f latency_ms=%.1f",
            len(top_results),
            top_score,
            latency_ms,
        )

        return top_results

    def unload(self) -> None:
        """Free the model from memory (MPS or CPU).

        Call this after the Modern pipeline finishes to release MPS memory
        before loading other models (e.g. the embedder).
        """
        if self._model is not None:
            logger.info(
                "CrossEncoderReranker.unload: freeing model backend=%s", self._backend
            )
            self._model = None
            self._backend = None

            # Attempt to flush MPS/CUDA cache if torch is available.
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    logger.debug("unload: MPS cache flushed")
            except Exception:
                pass  # non-fatal

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load the reranker model on first use."""
        if self._model is not None:
            return  # already loaded

        model_name: str = self.config.models.reranker_model
        device: str = self._select_device()

        logger.info(
            "CrossEncoderReranker: loading model=%s device=%s", model_name, device
        )
        t0 = time.perf_counter()

        # Try FlagEmbedding first.
        try:
            from FlagEmbedding import FlagReranker  # type: ignore[import]

            # FlagReranker uses "cuda" / "cpu"; "mps" may not be supported in all
            # versions — use cpu as fallback for Flag if device is mps.
            flag_device = device if device != "mps" else "cpu"
            self._model = FlagReranker(model_name, use_fp16=(flag_device == "cuda"))
            self._backend = "flag"
            logger.info(
                "CrossEncoderReranker: loaded via FlagEmbedding device=%s (flag_device=%s)",
                device,
                flag_device,
            )
        except ImportError:
            logger.info(
                "FlagEmbedding not available — falling back to sentence-transformers CrossEncoder"
            )
            from sentence_transformers import CrossEncoder  # type: ignore[import]

            self._model = CrossEncoder(model_name, device=device)
            self._backend = "sentence_transformers"
            logger.info(
                "CrossEncoderReranker: loaded via sentence-transformers device=%s", device
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        param_count = self._count_params()
        logger.info(
            "CrossEncoderReranker: model ready backend=%s params=%s latency_ms=%.1f",
            self._backend,
            param_count,
            latency_ms,
        )

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score a list of (query, text) pairs using the loaded backend."""
        if self._backend == "flag":
            scores = self._model.compute_score(pairs, normalize=True)
            # compute_score may return a single float when len(pairs)==1.
            if isinstance(scores, float):
                scores = [scores]
            return [float(s) for s in scores]
        else:
            # sentence-transformers CrossEncoder
            # show_progress_bar=False avoids tqdm overhead;
            # num_workers=0 avoids multiprocessing issues on Python 3.14/MPS.
            scores = self._model.predict(
                pairs,
                show_progress_bar=False,
                num_workers=0,
            )
            return [float(s) for s in scores]

    def _select_device(self) -> str:
        """Always return 'cpu' for the cross-encoder reranker.

        MPS is detected and logged, but CrossEncoder.predict() hangs
        indefinitely on MPS (known PyTorch/sentence-transformers issue).
        CPU is the safe, correct device for this model.
        """
        try:
            import torch
            if torch.backends.mps.is_available():
                logger.info(
                    "_select_device: MPS detected but forcing CPU — "
                    "CrossEncoder.predict() hangs on MPS"
                )
        except ImportError:
            pass
        return "cpu"

    def _count_params(self) -> str:
        """Return a human-readable parameter count string, or 'unknown'."""
        try:
            import torch
            if self._backend == "sentence_transformers":
                model_obj = self._model.model
            elif self._backend == "flag":
                model_obj = getattr(self._model, "model", None)
            else:
                return "unknown"

            if model_obj is not None:
                total = sum(p.numel() for p in model_obj.parameters())
                return f"{total / 1_000_000:.1f}M"
        except Exception:
            pass
        return "unknown"
