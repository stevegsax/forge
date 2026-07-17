"""LLM call activity for Forge.

Sends the assembled context to the LLM provider and extracts the structured response.

Design follows Function Core / Imperative Shell:
- Testable function: execute_llm_call (takes provider as argument)
- Imperative shell: call_llm
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from sax_llm.models import text_messages
from temporalio import activity

from forge.activities._heartbeat import heartbeat_during
from forge.message_log import write_message_log
from forge.models import (
    AssembledContext,
    CapabilityTier,
    LLMCallResult,
    LLMResponse,
    ModelConfig,
    resolve_model,
)

if TYPE_CHECKING:
    from sax_llm.protocol import LLMProvider

# Shadow fallback for a missing context.model_name — the workflow always sets
# it from CapabilityTier.GENERATION via ModelConfig, so this only fires if a
# caller forgets. Resolved through the registry (T3.2) rather than a
# hardcoded literal, so it tracks the tier's pinned default.
DEFAULT_MODEL = resolve_model(CapabilityTier.GENERATION, ModelConfig())
DEFAULT_MAX_TOKENS = 4096

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


async def execute_llm_call(
    context: AssembledContext,
    provider: LLMProvider,
) -> LLMCallResult:
    """Call the LLM provider and extract structured results.

    Separated from the imperative shell so tests can inject a mock provider.
    """
    from sax_llm import parse_model_id

    full_model = context.model_name or DEFAULT_MODEL
    _, model = parse_model_id(full_model)
    start = time.monotonic()

    params = provider.build_request_params(
        messages=text_messages(context.system_prompt, context.user_prompt),
        output_type=LLMResponse,
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    result = await provider.call(params)

    if context.log_messages and context.worktree_path:
        request_json = json.dumps(params, indent=2, default=str)
        write_message_log(context.worktree_path, "request", request_json)
        write_message_log(context.worktree_path, "response", result.raw_response_json)

    elapsed_ms = (time.monotonic() - start) * 1000
    response = LLMResponse.model_validate(result.tool_input)

    return LLMCallResult(
        task_id=context.task_id,
        response=response,
        model_name=result.model_name,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=elapsed_ms,
        cache_creation_input_tokens=result.cache_creation_input_tokens,
        cache_read_input_tokens=result.cache_read_input_tokens,
    )


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


@activity.defn
async def call_llm(context: AssembledContext) -> LLMCallResult:
    """Activity wrapper — creates a provider and delegates to execute_llm_call."""
    from sax_llm import get_provider

    from forge.tracing import get_tracer, llm_call_attributes

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.call_llm") as span:
        logger.info("LLM call start: task_id=%s model=%s", context.task_id, context.model_name)
        provider = get_provider(context.model_name or DEFAULT_MODEL)
        async with heartbeat_during():
            result = await execute_llm_call(context, provider)
        logger.info(
            "LLM call done: task_id=%s tokens=%din/%dout latency=%.0fms",
            context.task_id,
            result.input_tokens,
            result.output_tokens,
            result.latency_ms,
        )

        span.set_attributes(
            llm_call_attributes(
                model_name=result.model_name,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=result.latency_ms,
                task_id=context.task_id,
                cache_creation_input_tokens=result.cache_creation_input_tokens,
                cache_read_input_tokens=result.cache_read_input_tokens,
            )
        )

        return result
