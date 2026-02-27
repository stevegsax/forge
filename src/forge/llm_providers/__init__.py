"""LLM provider abstraction layer for Forge.

Public API:
- parse_model_id: Split "provider:model" strings
- get_provider: Factory returning cached provider singletons
- ProviderResponse: Normalized response from any provider
- LLMProvider: Protocol for provider implementations
"""

from __future__ import annotations

from forge.llm_providers.models import ProviderResponse
from forge.llm_providers.protocol import LLMProvider
from forge.llm_providers.registry import get_provider, parse_model_id

__all__ = [
    "LLMProvider",
    "ProviderResponse",
    "get_provider",
    "parse_model_id",
]
