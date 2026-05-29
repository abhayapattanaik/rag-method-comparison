"""LLM provider package for the RAG Comparison System.

Exports:
    LLMProvider   -- abstract base class (alias for BaseLLMProvider)
    LLMResponse   -- response dataclass with token/cost metadata
    get_provider  -- factory function; selects provider from config
"""

from src.llm.base import LLMResponse, BaseLLMProvider as LLMProvider, get_provider

__all__ = ["LLMProvider", "LLMResponse", "get_provider"]
