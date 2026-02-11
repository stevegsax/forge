"""LLM call activity for Forge.

Sends the assembled context to the Anthropic API and extracts the structured response.

Design follows Function Core / Imperative Shell:
- Testable function: execute_llm_call (takes client as argument)
- Imperative shell: call_llm, _persist_interaction
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from temporalio import activity

from forge.llm_client import build_messages_params, extract_tool_result, extract_usage
from forge.models import AssembledContext, LLMCallResult, LLMResponse

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_MAX_TOKENS = 4096

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


async def execute_llm_call(
    context: AssembledContext,
    client: AsyncAnthropic,
) -> LLMCallResult:
    """Call the Anthropic API and extract structured results.

    Separated from the imperative shell so tests can inject a mock client.
    """
    model = context.model_name or DEFAULT_MODEL
    start = time.monotonic()

    params = build_messages_params(
        system_prompt=context.system_prompt,
        user_prompt=context.user_prompt,
        output_type=LLMResponse,
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    message = await client.messages.create(**params)

    elapsed_ms = (time.monotonic() - start) * 1000
    response = extract_tool_result(message, LLMResponse)
    in_tok, out_tok, cache_create, cache_read = extract_usage(message)

    return LLMCallResult(
        task_id=context.task_id,
        response=response,
        model_name=model,
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=elapsed_ms,
        cache_creation_input_tokens=cache_create,
        cache_read_input_tokens=cache_read,
    )


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def _persist_interaction(
    context: AssembledContext,
    result: LLMCallResult,
    *,
    role: str = "llm",
) -> None:
    """Best-effort store write. Never raises (D42)."""
    try:
        from forge.store import build_interaction_dict, get_db_path, get_engine, save_interaction

        db_path = get_db_path()
        if db_path is None:
            return

        engine = get_engine(db_path)
        data = build_interaction_dict(
            task_id=context.task_id,
            step_id=context.step_id,
            sub_task_id=context.sub_task_id,
            role=role,
            context=context,
            llm_result=result,
        )
        save_interaction(engine, **data)
    except Exception:
        logger.warning("Failed to persist LLM interaction to store", exc_info=True)


@activity.defn
async def call_llm(context: AssembledContext) -> LLMCallResult:
    """Activity wrapper — creates a client and delegates to execute_llm_call."""
    from forge.llm_client import get_anthropic_client
    from forge.tracing import get_tracer, llm_call_attributes

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.call_llm") as span:
        client = get_anthropic_client()
        result = await execute_llm_call(context, client)

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

        _persist_interaction(context, result)
        return result
