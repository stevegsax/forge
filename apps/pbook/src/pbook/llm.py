"""LLM integration layer for the playbook service.

Defines pbook-specific structured output models (ExtractionResult,
ReviewResult, ConsolidationResult) and a minimal provider seam for runtime
injection of a ``sax_platform.llm`` structured-outputs client.

The seam is a module-global registered via :func:`set_provider`; the
generic chat activity calls :func:`get_provider`. Only the one method
``llm_chat`` actually needs — ``complete`` — is captured in the local
:class:`SupportsComplete` Protocol, so tests can inject a stub without
constructing the SDK-backed client (and mypy-strict still checks the shape).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from anthropic.types import MessageParam
    from sax_platform.llm import CacheSpec, Completion, ThinkingPolicy

__all__ = [
    "ConsolidationResult",
    "ExtractionEntry",
    "ExtractionResult",
    "ReviewResult",
    "SupportsComplete",
    "get_provider",
    "reset_provider",
    "set_provider",
]


# ---------------------------------------------------------------------------
# Structured output models for extraction, review, and consolidation
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
# Provider seam — minimal Protocol + module global for runtime injection
# ---------------------------------------------------------------------------


class SupportsComplete(Protocol):
    """The single method ``llm_chat`` needs from a structured-outputs client.

    Matches the signature shape of ``sax_platform.llm.AnthropicLLM.complete``
    (the platform client satisfies this structurally). Keeping it a narrow
    local Protocol lets tests inject a stub client under mypy-strict without
    dragging in the SDK-backed implementation.
    """

    async def complete(
        self,
        messages: Iterable[MessageParam],
        *,
        output_type: type[BaseModel],
        model: str,
        max_tokens: int,
        system: str | list[dict[str, Any]] | None = None,
        cache: CacheSpec | None = None,
        thinking: ThinkingPolicy | None = None,
    ) -> Completion[Any]: ...


# T3.6 deletes this global in favor of explicit dependency passing; until
# then the worker registers a provider at startup and activities read it here.
_provider: SupportsComplete | None = None


def set_provider(provider: SupportsComplete) -> None:
    """Register the LLM provider for pbook activities."""
    global _provider
    _provider = provider


def get_provider() -> SupportsComplete:
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
