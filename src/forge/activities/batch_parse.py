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
from forge.models import ParsedLLMResponse, ParseResponseInput, extract_stop_reason

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


def execute_parse_llm_response(
    raw_json: str,
    output_type_name: str | None,
    provider_name: str = "anthropic",
) -> ParsedLLMResponse:
    """Parse raw LLM response JSON into a ParsedLLMResponse.

    Routes to the correct provider for parsing.
    When output_type_name is None, returns text_content as parsed_json.
    Separated from the imperative shell so tests can call directly.
    """
    import json

    from sax_llm import get_provider_by_name

    provider = get_provider_by_name(provider_name)
    result = provider.parse_batch_result(raw_json, output_type_name)

    if output_type_name is None:
        parsed_json = json.dumps(result.text_content)
    else:
        parsed_json = json.dumps(result.tool_input)

    return ParsedLLMResponse(
        parsed_json=parsed_json,
        model_name=result.model_name,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_creation_input_tokens=result.cache_creation_input_tokens,
        cache_read_input_tokens=result.cache_read_input_tokens,
        # raw_json is the serialized anthropic.types.Message either way (sax_llm
        # echoes it back verbatim on result.raw_response_json too); sax_llm's
        # ProviderResponse has no typed stop_reason field, so pull it from the
        # wire JSON directly rather than depend on a libs/ change (owner note:
        # 2026-07 Phase 3 code review, item 3).
        stop_reason=extract_stop_reason(raw_json),
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
        # The body arrives inline or via an S3 pointer (a result envelope); fetch
        # and unwrap when only s3_key is set. The generic path ignores any images.
        if input.s3_key is not None:
            from sax_platform.contracts.models import BatchResult, resolve_batch_result

            raw_json, _images = resolve_batch_result(
                BatchResult(request_id="", batch_id="", s3_key=input.s3_key, result_type="")
            )
        else:
            raw_json = input.raw_response_json
        if raw_json is None:
            msg = "parse_llm_response: no body resolved (both raw_response_json and s3_key empty)"
            raise ValueError(msg)

        if input.log_messages and input.worktree_path:
            write_message_log(input.worktree_path, "response", raw_json)

        result = execute_parse_llm_response(
            raw_json,
            input.output_type_name,
            provider_name=input.provider,
        )

        if result.stop_reason == "max_tokens":
            logger.warning(
                "Batch LLM call truncated at max_tokens: task_id=%s model=%s "
                "max_tokens=%d output_tokens=%d",
                input.task_id,
                result.model_name,
                input.max_tokens,
                result.output_tokens,
            )

        span.set_attributes(
            {
                "forge.batch.output_type": input.output_type_name or "",
                "forge.batch.task_id": input.task_id,
                "forge.batch.model_name": result.model_name,
            }
        )

        return result
