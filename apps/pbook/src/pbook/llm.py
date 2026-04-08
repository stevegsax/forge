"""Minimal LLM protocol for the playbook service.

Defines the interface that extraction and review activities depend on.
In Phase 4, this will be replaced by the shared ``sax-llm`` package.
For now, consumers inject a provider implementation at runtime.

Design follows Function Core / Imperative Shell:

- Pure: build_messages, ExtractionResult, ReviewResult models
- Protocol: LLMProvider (async call with structured output)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# LLM protocol — minimal interface for extraction and review
# ---------------------------------------------------------------------------


class LLMResponse(BaseModel):
    """Normalized response from an LLM call."""

    tool_input: dict = Field(default_factory=dict)
    model_name: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal LLM provider protocol for pbook.

    Implementations must support:

    1. Building request parameters from messages + output type
    2. Calling the LLM and returning a normalized response
    """

    def build_request_params(
        self,
        *,
        messages: list[dict],
        output_type: type[BaseModel],
        model: str,
        max_tokens: int = 4096,
    ) -> dict: ...

    async def call(self, params: dict) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def build_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    """Build a simple system + user message list."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------------
# Structured output models for extraction and review
# ---------------------------------------------------------------------------


class ExtractionEntry(BaseModel):
    """A single entry extracted by the LLM from push experience data."""

    title: str
    content: str
    tags: list[str] = Field(default_factory=list)


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
