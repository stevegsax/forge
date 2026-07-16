"""LLM integration layer for the playbook service.

Re-exports the LLMProvider protocol and ProviderResponse from sax-llm.
Defines pbook-specific structured output models (ExtractionResult, ReviewResult)
and a simple provider registry for runtime injection.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from sax_llm.models import ProviderResponse, text_messages
from sax_llm.protocol import LLMProvider

# Re-export for backward compatibility with existing pbook code
__all__ = [
    "ExtractionEntry",
    "ExtractionResult",
    "LLMProvider",
    "LLMResponse",
    "ProviderResponse",
    "ReviewResult",
    "get_provider",
    "reset_provider",
    "set_provider",
    "text_messages",
]


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

LLMResponse = ProviderResponse


# ---------------------------------------------------------------------------
# Structured output models for extraction and review
# ---------------------------------------------------------------------------


class ExtractionEntry(BaseModel):
    """A single entry extracted by the LLM from push experience data."""

    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    embedding: bytes | None = None


class ExtractionResult(BaseModel):
    """Structured output from the extraction LLM call."""

    entries: list[ExtractionEntry] = Field(default_factory=list)


class ReviewResult(BaseModel):
    """Structured output from the review LLM call."""

    approved: bool = False
    rejection_reason: str = ""
    suggested_title: str = ""
    suggested_content: str = ""
    suggested_tags: list[str] = Field(default_factory=list)


class ConsolidationResult(BaseModel):
    """Structured output from the consolidation LLM call."""

    merged_title: str
    merged_content: str
    merged_tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Provider registry — simple global for runtime injection
# ---------------------------------------------------------------------------

_provider: LLMProvider | None = None


def set_provider(provider: LLMProvider) -> None:
    """Register the LLM provider for pbook activities."""
    global _provider
    _provider = provider


def get_provider() -> LLMProvider:
    """Get the registered LLM provider.

    Raises ``RuntimeError`` if no provider has been registered.
    """
    if _provider is None:
        msg = (
            "No LLM provider registered. Call pbook.llm.set_provider() "
            "before running extraction or review activities."
        )
        raise RuntimeError(msg)
    return _provider


def reset_provider() -> None:
    """Clear the registered provider (for testing)."""
    global _provider
    _provider = None
