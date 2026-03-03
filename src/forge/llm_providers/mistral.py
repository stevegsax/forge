"""Mistral LLM provider implementation."""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING

from forge.llm_providers.models import (
    BatchPollResult,
    BatchPollStatus,
    BatchResultEntry,
    DocumentContent,
    ImageContent,
    Message,
    ProviderResponse,
    TextContent,
)

if TYPE_CHECKING:
    from pydantic import BaseModel


def _snake_case(name: str) -> str:
    """Convert CamelCase class name to snake_case tool name."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


# ---------------------------------------------------------------------------
# Message translation helpers
# ---------------------------------------------------------------------------


def _content_block_to_mistral(block: TextContent | ImageContent | DocumentContent) -> dict:
    """Convert a ContentBlock to Mistral API format."""
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageContent):
        data_uri = f"data:{block.media_type};base64,{block.data}"
        return {"type": "image_url", "image_url": data_uri}
    if isinstance(block, DocumentContent):
        data_uri = f"data:{block.media_type};base64,{block.data}"
        return {"type": "image_url", "image_url": data_uri}
    msg = f"Unknown content block type: {type(block)}"
    raise TypeError(msg)


def _content_to_mistral(
    content: str | list[TextContent | ImageContent | DocumentContent],
) -> str | list[dict]:
    """Convert message content to Mistral format."""
    if isinstance(content, str):
        return content
    return [_content_block_to_mistral(b) for b in content]


def _messages_to_mistral(messages: list[Message]) -> list[dict]:
    """Convert Message list to Mistral format (system messages stay in array)."""
    return [
        {"role": msg.role, "content": _content_to_mistral(msg.content)}
        for msg in messages
    ]


class MistralProvider:
    """LLM provider backed by the Mistral API.

    Uses the ``mistralai`` SDK for both sync and batch modes.
    Features not supported by Mistral (prompt caching, extended thinking)
    are silently skipped per D63 degradation policy.
    """

    def __init__(self) -> None:
        from mistralai import Mistral

        api_key = os.environ.get("MISTRAL_API_KEY", "")
        self._client = Mistral(api_key=api_key)

    def build_request_params(
        self,
        messages: list[Message],
        output_type: type[BaseModel] | None,
        model: str,
        max_tokens: int,
        *,
        cache_instructions: bool = True,
        cache_tool_definitions: bool = True,
        thinking_budget_tokens: int = 0,
    ) -> dict:
        """Build Mistral chat.complete kwargs.

        Ignores cache_instructions, cache_tool_definitions, and
        thinking_budget_tokens (Mistral has no equivalents).
        """
        params: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": _messages_to_mistral(messages),
        }

        if output_type is not None:
            schema = output_type.model_json_schema()
            tool_name = _snake_case(output_type.__name__)
            description = (output_type.__doc__ or "").strip() or f"Structured output: {tool_name}"

            tool_def = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": description,
                    "parameters": schema,
                },
            }
            params["tools"] = [tool_def]
            params["tool_choice"] = "any"

        return params

    async def call(self, params: dict) -> ProviderResponse:
        """Call the Mistral API and return a normalized response."""
        response = await self._client.chat.complete_async(**params)

        tool_input: dict = {}
        text_content: str | None = None
        has_tools = "tools" in params and params["tools"]

        if has_tools:
            if response.choices and response.choices[0].message.tool_calls:
                args_str = response.choices[0].message.tool_calls[0].function.arguments
                tool_input = json.loads(args_str) if isinstance(args_str, str) else args_str
        else:
            # No tools — extract text content
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                text_content = content if isinstance(content, str) else str(content)

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        return ProviderResponse(
            tool_input=tool_input,
            text_content=text_content,
            model_name=response.model or params.get("model", ""),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            raw_response_json=json.dumps(response.model_dump(), default=str),
        )

    @property
    def supports_batch(self) -> bool:
        """Mistral supports the Batch API."""
        return True

    def build_batch_request(self, request_id: str, params: dict) -> dict:
        """Build a Mistral inline batch entry."""
        return {"custom_id": request_id, "body": params}

    async def submit_batch(self, requests: list[dict], model: str) -> str:
        """Submit a batch to the Mistral Batch API."""
        job = await self._client.batch.jobs.create_async(
            requests=requests,
            model=model,
            endpoint="/v1/chat/completions",
        )
        return job.id

    async def poll_batch(self, batch_id: str) -> BatchPollResult:
        """Poll the Mistral Batch API for results."""
        job = await self._client.batch.jobs.get_async(job_id=batch_id)

        status_map = {
            "QUEUED": BatchPollStatus.PENDING,
            "RUNNING": BatchPollStatus.IN_PROGRESS,
            "SUCCESS": BatchPollStatus.ENDED,
            "FAILED": BatchPollStatus.FAILED,
            "TIMEOUT_EXCEEDED": BatchPollStatus.EXPIRED,
            "CANCELLATION_REQUESTED": BatchPollStatus.CANCELED,
            "CANCELLED": BatchPollStatus.CANCELED,
        }
        poll_status = status_map.get(job.status, BatchPollStatus.IN_PROGRESS)

        if poll_status != BatchPollStatus.ENDED:
            return BatchPollResult(status=poll_status)

        # Download and parse results
        output_file = await self._client.files.download_async(file_id=job.output_file)
        if hasattr(output_file, "read"):
            content = output_file.read().decode("utf-8")
        else:
            content = str(output_file)
        entries: list[BatchResultEntry] = []
        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            entry_data = json.loads(line)
            custom_id = entry_data.get("custom_id", "")
            response_body = entry_data.get("response", {}).get("body", {})
            if response_body.get("choices"):
                entries.append(
                    BatchResultEntry(
                        custom_id=custom_id,
                        succeeded=True,
                        raw_response_json=json.dumps(response_body),
                    )
                )
            else:
                entries.append(
                    BatchResultEntry(
                        custom_id=custom_id,
                        succeeded=False,
                        error=json.dumps(response_body.get("error", "Unknown error")),
                    )
                )

        return BatchPollResult(status=BatchPollStatus.ENDED, entries=entries)

    def parse_batch_result(
        self,
        raw_json: str,
        output_type_name: str | None,
    ) -> ProviderResponse:
        """Parse a Mistral batch result entry into a normalized response."""
        data = json.loads(raw_json)
        usage = data.get("usage", {})
        model_name = data.get("model", "")

        if output_type_name is None:
            # Text-only response (no tool call)
            choices = data.get("choices", [])
            text_content: str | None = None
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                text_content = content if isinstance(content, str) else str(content)

            return ProviderResponse(
                text_content=text_content,
                model_name=model_name,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                raw_response_json=raw_json,
            )

        from forge.llm_client import get_output_type_registry

        registry = get_output_type_registry()
        if output_type_name not in registry:
            msg = f"Unknown output type: {output_type_name!r}"
            raise KeyError(msg)

        choices = data.get("choices", [])
        tool_input: dict = {}
        if choices:
            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                args = tool_calls[0].get("function", {}).get("arguments", "{}")
                tool_input = json.loads(args) if isinstance(args, str) else args

        return ProviderResponse(
            tool_input=tool_input,
            model_name=model_name,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            raw_response_json=raw_json,
        )
