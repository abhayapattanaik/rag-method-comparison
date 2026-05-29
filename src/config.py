"""Configuration loader for the RAG Comparison System.

Loads from YAML (defaults to config/default.yaml), merges CLI overrides
(dot-notation keys such as "chunking.chunk_size"), and validates with Pydantic v2.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section sub-configs
# ---------------------------------------------------------------------------


class PathsConfig(BaseModel):
    data_dir: str
    chroma_dir: str
    cache_dir: str
    results_dir: str
    papers_dir: str


class ModelsConfig(BaseModel):
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    llm_model: str = "claude-sonnet-4-20250514"
    judge_model: str = "claude-haiku-4-5-20241022"


class ChunkingConfig(BaseModel):
    chunk_size: int = 400       # tokens
    chunk_overlap: int = 50     # tokens


class RetrievalConfig(BaseModel):
    top_k_dense: int = 20
    top_k_bm25: int = 20
    top_k_fusion: int = 20
    top_k_rerank: int = 10
    rrf_k: int = 60             # RRF constant (typically 60)


class LLMConfig(BaseModel):
    provider: str = "anthropic"             # "anthropic" | "openai" | "claude_cli"
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"
    openai_api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 2048


class CostEstimationConfig(BaseModel):
    """Static estimate defaults (Phase A) and sample sizes (Phase B)."""

    # Phase A — static estimate defaults
    avg_doc_tokens: int = 8000
    avg_chunk_tokens: int = 400
    contextualization_output_tokens: int = 100
    query_overhead_tokens: int = 200
    answer_max_tokens: int = 500
    judge_prompt_tokens: int = 300
    judge_output_tokens: int = 200
    avg_paper_tokens: int = 15000
    question_output_tokens: int = 300

    # Phase B — sample-based refinement sizes
    sample_size_contextualize: int = 5
    sample_size_evaluate: int = 1       # 1 question x 1 pipeline x 4 metrics
    sample_size_question_gen: int = 2


class CostPricingConfig(BaseModel):
    """Per-model pricing dict: model_name -> {input_per_1k, output_per_1k}."""

    pricing: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "claude-haiku-4-5-20241022": {
                "input_per_1k": 0.001,
                "output_per_1k": 0.005,
            },
            "claude-sonnet-4-20250514": {
                "input_per_1k": 0.003,
                "output_per_1k": 0.015,
            },
            "claude-opus-4-20250514": {
                "input_per_1k": 0.015,
                "output_per_1k": 0.075,
            },
            "gpt-4o": {
                "input_per_1k": 0.005,
                "output_per_1k": 0.015,
            },
            "gpt-4o-mini": {
                "input_per_1k": 0.00015,
                "output_per_1k": 0.0006,
            },
        }
    )


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(
        default_factory=lambda: [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
        ]
    )
    judge_temperature: float = 0.0


class ConcurrencyConfig(BaseModel):
    max_workers: int = 4
    stall_timeout_seconds: int = 120


# ---------------------------------------------------------------------------
# Top-level AppConfig
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    paths: PathsConfig
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    cost_estimation: CostEstimationConfig = Field(default_factory=CostEstimationConfig)
    cost: CostPricingConfig = Field(default_factory=CostPricingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"


def _set_nested(mapping: dict[str, Any], dot_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dot-notation key.

    Example: _set_nested(d, "chunking.chunk_size", 400) sets
    d["chunking"]["chunk_size"] = 400, creating intermediate dicts as needed.
    """
    keys = dot_key.split(".")
    node = mapping
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    leaf = keys[-1]

    # Attempt to coerce to int/float/bool when the current value is a string
    if isinstance(value, str):
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            for cast in (int, float):
                try:
                    value = cast(value)
                    break
                except ValueError:
                    pass

    node[leaf] = value


def load_config(
    config_path: str | None = None,
    cli_overrides: dict | None = None,
) -> AppConfig:
    """Load configuration from a YAML file, apply CLI overrides, and validate.

    Args:
        config_path: Path to YAML config file. Defaults to
            ``config/default.yaml`` relative to the project root.
        cli_overrides: Dict of dot-notation key -> value pairs that override
            YAML values, e.g. ``{"chunking.chunk_size": 400}``. Values are
            coerced to int/float/bool when the string representation allows it.

    Returns:
        Validated ``AppConfig`` instance.

    Raises:
        FileNotFoundError: If the resolved config file does not exist.
        pydantic.ValidationError: If the merged config is invalid.
    """
    resolved = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not resolved.exists():
        raise FileNotFoundError(
            f"Config file not found: {resolved}. "
            "Create config/default.yaml or pass an explicit config_path."
        )

    logger.info("Loading config from %s", resolved)
    with resolved.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    # Apply CLI overrides (dot-notation) on top of YAML data
    if cli_overrides:
        for dot_key, value in cli_overrides.items():
            logger.debug("Config override: %s = %r", dot_key, value)
            _set_nested(raw, dot_key, value)

    cfg = AppConfig.model_validate(raw)
    logger.debug("Config loaded and validated: provider=%s model=%s", cfg.llm.provider, cfg.models.llm_model)
    return cfg
