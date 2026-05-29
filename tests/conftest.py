"""Common fixtures for the RAG comparison test suite."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.config import AppConfig, ChunkingConfig, ConcurrencyConfig, CostEstimationConfig, CostPricingConfig, EvaluationConfig, LLMConfig, ModelsConfig, PathsConfig, RetrievalConfig
from src.llm.base import BaseLLMProvider, LLMResponse
from src.telemetry import TelemetryTracker


# ---------------------------------------------------------------------------
# Minimal AppConfig that doesn't require real filesystem paths
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_config(tmp_path) -> AppConfig:
    """A fully in-memory AppConfig pointing at tmp_path for all dirs."""
    return AppConfig(
        paths=PathsConfig(
            data_dir=str(tmp_path / "data"),
            chroma_dir=str(tmp_path / "chroma_db"),
            cache_dir=str(tmp_path / "cache"),
            results_dir=str(tmp_path / "results"),
            papers_dir=str(tmp_path / "papers"),
        ),
        models=ModelsConfig(
            embedding_model="BAAI/bge-base-en-v1.5",
            reranker_model="BAAI/bge-reranker-v2-m3",
            llm_model="claude-sonnet-4-20250514",
            judge_model="claude-haiku-4-5-20241022",
        ),
        chunking=ChunkingConfig(chunk_size=100, chunk_overlap=10),
        retrieval=RetrievalConfig(
            top_k_dense=5,
            top_k_bm25=5,
            top_k_fusion=5,
            top_k_rerank=3,
            rrf_k=60,
        ),
        llm=LLMConfig(provider="anthropic", temperature=0.0, max_tokens=512),
        cost_estimation=CostEstimationConfig(),
        cost=CostPricingConfig(),
        evaluation=EvaluationConfig(
            metrics=[
                "context_precision",
                "context_recall",
                "faithfulness",
                "answer_relevancy",
            ],
            judge_temperature=0.0,
        ),
        concurrency=ConcurrencyConfig(max_workers=2, stall_timeout_seconds=30),
    )


# ---------------------------------------------------------------------------
# Canned LLMResponse helper
# ---------------------------------------------------------------------------


def make_llm_response(text: str = '{"score": 0.8, "justification": "Good."}') -> LLMResponse:
    return LLMResponse(
        text=text,
        input_tokens=100,
        output_tokens=20,
        model="claude-haiku-4-5-20241022",
        latency_ms=250.0,
        cost_usd=0.000120,
    )


# ---------------------------------------------------------------------------
# Mock LLM provider
# ---------------------------------------------------------------------------


class MockLLMProvider(BaseLLMProvider):
    """A BaseLLMProvider that returns a canned LLMResponse without API calls."""

    def __init__(self, config: AppConfig, telemetry: TelemetryTracker, canned_text: str = '{"score": 0.8, "justification": "Good."}'):
        # Bypass BaseLLMProvider.__init__ logging that expects real attrs
        self.config = config
        self.telemetry = telemetry
        self._canned_text = canned_text
        self._call_count = 0

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self._call_count += 1
        return make_llm_response(self._canned_text)

    def get_model_name(self) -> str:
        return "claude-haiku-4-5-20241022"


@pytest.fixture()
def telemetry(sample_config) -> TelemetryTracker:
    return TelemetryTracker(sample_config)


@pytest.fixture()
def mock_provider(sample_config, telemetry) -> MockLLMProvider:
    return MockLLMProvider(sample_config, telemetry)
