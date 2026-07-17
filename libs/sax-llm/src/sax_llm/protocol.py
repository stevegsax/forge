"""LLMProvider protocol definition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pydantic import BaseModel

    from sax_llm.models import BatchPollResult, Message, ProviderResponse


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM provider implementations.

    Each provider wraps a specific LLM API (Anthropic, Mistral, etc.)
    behind a common interface for both sync and batch modes.
    """

    # --- Sync mode ---

    def build_request_params(
        self,
        messages: list[Message],
        output_type: type[BaseModel] | None,
        model: str,
        max_tokens: int,
        *,
        cache_instructions: bool = True,
        cache_tool_definitions: bool = True,
        thinking_enabled: bool = False,
        effort: str | None = None,
    ) -> dict[str, Any]:
        """Build provider-specific request parameters."""
        ...

    async def call(self, params: dict[str, Any]) -> ProviderResponse:
        """Execute a synchronous LLM call and return a normalized response."""
        ...

    # --- Batch mode ---

    @property
    def supports_batch(self) -> bool:
        """Whether this provider supports the batch API."""
        ...

    def build_batch_request(self, request_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Wrap request params into a batch request entry."""
        ...

    async def submit_batch(
        self, requests: list[dict[str, Any]], model: str, *, endpoint: str = ""
    ) -> str:
        """Submit a batch of requests. Returns the batch job ID."""
        ...

    async def poll_batch(self, batch_id: str) -> BatchPollResult:
        """Poll for batch results. Returns status and results if done."""
        ...

    def parse_batch_result(
        self,
        raw_json: str,
        output_type_name: str | None,
    ) -> ProviderResponse:
        """Parse a single batch result entry into a normalized response."""
        ...
