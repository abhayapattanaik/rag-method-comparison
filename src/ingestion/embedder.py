"""Embedding module for the RAG Comparison System.

Loads BAAI/bge-base-en-v1.5 via sentence-transformers. Uses MPS device when
available (Apple Silicon), falls back to CPU.

Model is loaded lazily on first call and cached at module level. This avoids
paying the load cost at import time and lets other modules manage MPS memory
by unloading/reloading as needed.

BGE models require the prefix "Represent this sentence: " for query embedding
(asymmetric retrieval). Document/passage embedding uses no prefix.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import AppConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------

_model = None          # sentence_transformers.SentenceTransformer instance
_model_name: str = ""  # track which model is loaded so we can detect config changes

_DEFAULT_BATCH_SIZE = 32
_BGE_QUERY_PREFIX = "Represent this sentence: "


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def _get_device() -> str:
    """Return 'mps' if available on Apple Silicon, else 'cpu'."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except (ImportError, AttributeError):
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Lazy loader / unloader
# ---------------------------------------------------------------------------


def _load_model(model_name: str) -> None:
    """Load the sentence-transformers model into the module-level cache."""
    global _model, _model_name

    if _model is not None and _model_name == model_name:
        return  # already loaded, nothing to do

    from sentence_transformers import SentenceTransformer

    device = _get_device()
    logger.info("Loading embedding model %s on device=%s", model_name, device)
    _model = SentenceTransformer(model_name, device=device)
    _model_name = model_name
    logger.info("Embedding model loaded")


def unload_model() -> None:
    """Free the cached model from memory (useful for MPS memory management)."""
    global _model, _model_name

    if _model is None:
        return

    logger.info("Unloading embedding model %s", _model_name)
    del _model
    _model = None
    _model_name = ""

    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_texts(
    texts: list[str],
    config: "AppConfig",
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """Batch-embed a list of document/passage texts.

    Args:
        texts: Texts to embed. No prefix is added (BGE passage encoding).
        config: AppConfig — reads config.models.embedding_model.
        batch_size: Number of texts per encode call. Default 32.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    if not texts:
        return []

    model_name = config.models.embedding_model
    _load_model(model_name)

    logger.info("Embedding %d texts in batches of %d", len(texts), batch_size)
    embeddings = _model.encode(  # type: ignore[union-attr]
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    result = embeddings.tolist()
    logger.debug("Batch embedded: count=%d embedding_dim=%d", len(result), len(result[0]) if result else 0)
    return result


def embed_query(query: str, config: "AppConfig") -> list[float]:
    """Embed a single query string.

    BGE models use the prefix "Represent this sentence: " for queries in
    asymmetric retrieval settings.

    Args:
        query: Raw query text (prefix will be prepended automatically).
        config: AppConfig — reads config.models.embedding_model.

    Returns:
        Embedding vector as a list of floats.
    """
    model_name = config.models.embedding_model
    _load_model(model_name)

    prefixed = _BGE_QUERY_PREFIX + query
    embedding = _model.encode(  # type: ignore[union-attr]
        [prefixed],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding[0].tolist()
