"""Batch parse activity for Forge.

Deserializes a raw Anthropic Message JSON from a batch response into
a typed ParsedLLMResponse.

Design follows Function Core / Imperative Shell:
- Testable function: execute_parse_llm_response
- Imperative shell: parse_llm_response (activity with OTel tracing)
"""

from __future__ import annotations

from temporalio import activity

from forge.llm_client import parse_batch_response_json
from forge.models import ParsedLLMResponse, ParseResponseInput

# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


def execute_parse_llm_response(
    raw_json: str,
    output_type_name: str,
) -> ParsedLLMResponse:
    """Parse raw Anthropic Message JSON into a ParsedLLMResponse.

    Separated from the imperative shell so tests can call directly.
    """
    parsed, model_name, in_tok, out_tok, cache_create, cache_read = parse_batch_response_json(
        raw_json, output_type_name
    )
    return ParsedLLMResponse(
        parsed_json=parsed.model_dump_json(),
        model_name=model_name,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_creation_input_tokens=cache_create,
        cache_read_input_tokens=cache_read,
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
        result = execute_parse_llm_response(input.raw_response_json, input.output_type_name)

        span.set_attributes(
            {
                "forge.batch.output_type": input.output_type_name,
                "forge.batch.task_id": input.task_id,
                "forge.batch.model_name": result.model_name,
            }
        )

        return result
