"""Anthropic LLM provider — wraps existing llm_client.py functions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from forge.llm_client import (
    build_batch_request,
    build_messages_params,
    extract_usage,
    parse_batch_response_json,
)
from forge.llm_providers.models import (
    BatchPollResult,
    BatchPollStatus,
    BatchResultEntry,
    ProviderResponse,
)

if TYPE_CHECKING:
    from pydantic import BaseModel


class AnthropicProvider:
    """LLM provider backed by the Anthropic Messages API.

    Wraps existing pure functions from ``llm_client.py`` as a thin adapter.
    """

    def __init__(self) -> None:
        from forge.llm_client import get_anthropic_client

        self._get_client = get_anthropic_client

    def build_request_params(
        self,
        system_prompt: str,
        user_prompt: str,
        output_type: type[BaseModel],
        model: str,
        max_tokens: int,
        *,
        cache_instructions: bool = True,
        cache_tool_definitions: bool = True,
        thinking_budget_tokens: int = 0,
    ) -> dict:
        """Build Anthropic messages.create kwargs."""
        return build_messages_params(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_type=output_type,
            model=model,
            max_tokens=max_tokens,
            cache_instructions=cache_instructions,
            cache_tool_definitions=cache_tool_definitions,
            thinking_budget_tokens=thinking_budget_tokens,
        )

    async def call(self, params: dict) -> ProviderResponse:
        """Call the Anthropic API and return a normalized response."""
        client = self._get_client()
        message = await client.messages.create(**params)

        # Extract the output type from the tool definition name
        # We need the output type to validate, but for ProviderResponse
        # we just need the raw tool_input dict
        tool_input: dict = {}
        for block in message.content:
            if block.type == "tool_use":
                tool_input = block.input
                break

        in_tok, out_tok, cache_create, cache_read = extract_usage(message)

        return ProviderResponse(
            tool_input=tool_input,
            model_name=message.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_creation_input_tokens=cache_create,
            cache_read_input_tokens=cache_read,
            raw_response_json=message.model_dump_json(),
        )

    @property
    def supports_batch(self) -> bool:
        """Anthropic supports the Message Batches API."""
        return True

    def build_batch_request(self, request_id: str, params: dict) -> dict:
        """Wrap params into an Anthropic batch request item."""
        return build_batch_request(request_id, params)

    async def submit_batch(self, requests: list[dict], model: str) -> str:
        """Submit a batch to the Anthropic Message Batches API."""
        client = self._get_client()
        batch = await client.messages.batches.create(requests=requests)
        return batch.id

    async def poll_batch(self, batch_id: str) -> BatchPollResult:
        """Poll the Anthropic API for batch results."""
        client = self._get_client()
        batch = await client.messages.batches.retrieve(batch_id)

        if batch.processing_status != "ended":
            return BatchPollResult(status=BatchPollStatus.IN_PROGRESS)

        results_iter = await client.messages.batches.results(batch_id)
        entries: list[BatchResultEntry] = []
        async for entry in results_iter:
            result_type = entry.result.type
            if result_type == "succeeded":
                entries.append(
                    BatchResultEntry(
                        custom_id=entry.custom_id,
                        succeeded=True,
                        raw_response_json=entry.result.message.model_dump_json(),
                    )
                )
            else:
                error_msg = _format_batch_error(entry)
                entries.append(
                    BatchResultEntry(
                        custom_id=entry.custom_id,
                        succeeded=False,
                        error=error_msg,
                    )
                )

        return BatchPollResult(status=BatchPollStatus.ENDED, entries=entries)

    def parse_batch_result(
        self,
        raw_json: str,
        output_type_name: str,
    ) -> ProviderResponse:
        """Parse a raw Anthropic Message JSON from a batch response."""
        parsed, model_name, in_tok, out_tok, cache_create, cache_read = parse_batch_response_json(
            raw_json, output_type_name
        )

        # Extract tool_input dict from the parsed model
        tool_input = json.loads(parsed.model_dump_json())

        return ProviderResponse(
            tool_input=tool_input,
            model_name=model_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_creation_input_tokens=cache_create,
            cache_read_input_tokens=cache_read,
            raw_response_json=raw_json,
        )


def _format_batch_error(entry: object) -> str:
    """Format an error message from a batch result entry."""
    result = getattr(entry, "result", None)
    result_type = getattr(result, "type", "unknown")

    if result_type == "errored":
        return f"Batch error: {getattr(result, 'error', 'unknown')}"
    if result_type == "expired":
        return "Batch request expired (24h limit)"
    if result_type == "canceled":
        return "Batch request was canceled"
    return f"Unknown result type: {result_type}"
