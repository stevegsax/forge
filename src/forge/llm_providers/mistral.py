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
    ProviderResponse,
)

if TYPE_CHECKING:
    from pydantic import BaseModel


def _snake_case(name: str) -> str:
    """Convert CamelCase class name to snake_case tool name."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


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
        """Build Mistral chat.complete kwargs.

        Ignores cache_instructions, cache_tool_definitions, and
        thinking_budget_tokens (Mistral has no equivalents).
        """
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

        return {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "tools": [tool_def],
            "tool_choice": "any",
        }

    async def call(self, params: dict) -> ProviderResponse:
        """Call the Mistral API and return a normalized response."""
        response = await self._client.chat.complete_async(**params)

        tool_input: dict = {}
        if response.choices and response.choices[0].message.tool_calls:
            args_str = response.choices[0].message.tool_calls[0].function.arguments
            tool_input = json.loads(args_str) if isinstance(args_str, str) else args_str

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        return ProviderResponse(
            tool_input=tool_input,
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
            input_data=requests,
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
        output_type_name: str,
    ) -> ProviderResponse:
        """Parse a Mistral batch result entry into a normalized response."""
        from forge.llm_client import get_output_type_registry

        registry = get_output_type_registry()
        if output_type_name not in registry:
            msg = f"Unknown output type: {output_type_name!r}"
            raise KeyError(msg)

        data = json.loads(raw_json)
        choices = data.get("choices", [])
        tool_input: dict = {}
        if choices:
            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                args = tool_calls[0].get("function", {}).get("arguments", "{}")
                tool_input = json.loads(args) if isinstance(args, str) else args

        usage = data.get("usage", {})

        return ProviderResponse(
            tool_input=tool_input,
            model_name=data.get("model", ""),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            raw_response_json=raw_json,
        )
