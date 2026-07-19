"""LLM integration layer for the playbook service.

Defines pbook-specific structured output models (ExtractionResult,
ReviewResult, ConsolidationResult) and the minimal :class:`SupportsComplete`
Protocol — the one method ``llm_chat`` needs (``complete``) from a
``sax_platform.llm`` structured-outputs client.

As of T3.6 the module-global provider seam is gone: the worker's composition
root builds the client and threads it into
:class:`~pbook.roots.LlmActivities`. The narrow local Protocol lets tests
inject a fake (`sax_platform.testing.FakeLLM`) without constructing the
SDK-backed client, and mypy-strict still checks the shape.
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
# Provider seam — minimal Protocol (injected at the composition root)
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
