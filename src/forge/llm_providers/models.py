"""Normalized response types for LLM providers."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ProviderResponse(BaseModel):
    """Normalized response from any LLM provider."""

    tool_input: dict = Field(description="Parsed structured output (tool call arguments).")
    model_name: str = Field(description="Actual model that responded.")
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    raw_response_json: str = Field(description="Serialized original response for message logging.")


class BatchPollStatus(StrEnum):
    """Normalized batch poll statuses across providers."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ENDED = "ended"
    FAILED = "failed"
    CANCELED = "canceled"
    EXPIRED = "expired"


class BatchResultEntry(BaseModel):
    """A single result entry from a batch response."""

    custom_id: str
    succeeded: bool
    raw_response_json: str | None = None
    error: str | None = None


class BatchPollResult(BaseModel):
    """Result of polling a batch job."""

    status: BatchPollStatus
    entries: list[BatchResultEntry] = Field(default_factory=list)
