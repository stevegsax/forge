"""LLM call activity for Forge.

Sends the assembled context to a pydantic-ai Agent and extracts the structured response.

Design follows Function Core / Imperative Shell:
- Testable function: execute_llm_call (takes agent as argument)
- Imperative shell: create_agent, call_llm, _persist_interaction
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from temporalio import activity

from forge.models import AssembledContext, LLMCallResult, LLMResponse

if TYPE_CHECKING:
    from pydantic_ai import Agent

DEFAULT_MODEL = "anthropic:claude-sonnet-4-5-20250929"
DEFAULT_MAX_TOKENS = 4096

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


async def execute_llm_call(
    context: AssembledContext,
    agent: Agent[None, LLMResponse],
) -> LLMCallResult:
    """Call the agent and extract structured results.

    Separated from the imperative shell so tests can inject a mock agent.
    """
    start = time.monotonic()

    result = await agent.run(
        context.user_prompt,
        instructions=context.system_prompt,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    usage = result.usage()

    return LLMCallResult(
        task_id=context.task_id,
        response=result.output,
        model_name=str(agent.model),
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        latency_ms=elapsed_ms,
        cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
        cache_read_input_tokens=usage.cache_read_input_tokens or 0,
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


def create_agent(model_name: str = DEFAULT_MODEL) -> Agent[None, LLMResponse]:
    """Create a pydantic-ai Agent configured for code generation."""
    from pydantic_ai import Agent

    return Agent(
        model_name,
        output_type=LLMResponse,
        model_settings={
            "anthropic_cache_instructions": True,
            "anthropic_cache_tool_definitions": True,
        },
    )


@activity.defn
async def call_llm(context: AssembledContext) -> LLMCallResult:
    """Activity wrapper — creates an agent and delegates to execute_llm_call."""
    from forge.tracing import get_tracer, llm_call_attributes

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.call_llm") as span:
        agent = create_agent(context.model_name or DEFAULT_MODEL)
        result = await execute_llm_call(context, agent)

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
