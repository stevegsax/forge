"""LLM provider abstraction layer for Forge.

Public API:
- parse_model_id: Split "provider:model" strings
- get_provider: Factory returning cached provider singletons
- ProviderResponse: Normalized response from any provider
- LLMProvider: Protocol for provider implementations
- Message, ContentBlock, TextContent, ImageContent, DocumentContent: Structured messages
- text_messages: Helper to build text-only message pairs
"""

from __future__ import annotations

from forge.llm_providers.models import (
    ContentBlock,
    DocumentContent,
    ImageContent,
    Message,
    ProviderResponse,
    TextContent,
    text_messages,
)
from forge.llm_providers.protocol import LLMProvider
from forge.llm_providers.registry import get_provider, get_provider_by_name, parse_model_id

__all__ = [
    "ContentBlock",
    "DocumentContent",
    "ImageContent",
    "LLMProvider",
    "Message",
    "ProviderResponse",
    "TextContent",
    "get_provider",
    "get_provider_by_name",
    "parse_model_id",
    "text_messages",
]
