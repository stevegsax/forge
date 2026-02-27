"""Batch parse activity for Forge.

Deserializes a raw LLM response JSON from a batch response into
a typed ParsedLLMResponse. Routes to the correct provider for parsing.

Design follows Function Core / Imperative Shell:
- Testable function: execute_parse_llm_response
- Imperative shell: parse_llm_response (activity with OTel tracing)
"""

from __future__ import annotations

import logging

from temporalio import activity

from forge.message_log import write_message_log
from forge.models import ParsedLLMResponse, ParseResponseInput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


def execute_parse_llm_response(
    raw_json: str,
    output_type_name: str,
    provider_name: str = "anthropic",
) -> ParsedLLMResponse:
    """Parse raw LLM response JSON into a ParsedLLMResponse.

    Routes to the correct provider for parsing.
    Separated from the imperative shell so tests can call directly.
    """
    from forge.llm_providers import get_provider

    provider = get_provider(provider_name)
    result = provider.parse_batch_result(raw_json, output_type_name)

    import json

    return ParsedLLMResponse(
        parsed_json=json.dumps(result.tool_input),
        model_name=result.model_name,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_creation_input_tokens=result.cache_creation_input_tokens,
        cache_read_input_tokens=result.cache_read_input_tokens,
    )


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


@activity.defn
async def parse_llm_response(input: ParseResponseInput) -> ParsedLLMResponse:
    """Activity wrapper with OTel tracing."""
    from forge.tracing import get_tracer

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.parse_llm_response") as span:
        logger.info(
            "Parse response: task_id=%s output_type=%s", input.task_id, input.output_type_name
        )
        if input.log_messages and input.worktree_path:
            write_message_log(input.worktree_path, "response", input.raw_response_json)

        result = execute_parse_llm_response(
            input.raw_response_json,
            input.output_type_name,
            provider_name=input.provider,
        )

        span.set_attributes(
            {
                "forge.batch.output_type": input.output_type_name,
                "forge.batch.task_id": input.task_id,
                "forge.batch.model_name": result.model_name,
            }
        )

        return result
