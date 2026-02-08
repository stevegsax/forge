"""LLM call activity for Forge.

Sends the assembled context to a pydantic-ai Agent and extracts the structured response.

Design follows Function Core / Imperative Shell:
- Testable function: execute_llm_call (takes agent as argument)
- Imperative shell: create_agent, call_llm
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from temporalio import activity

from forge.models import AssembledContext, LLMCallResult, LLMResponse

if TYPE_CHECKING:
    from pydantic_ai import Agent

DEFAULT_MODEL = "anthropic:claude-sonnet-4-5-20250929"
DEFAULT_MAX_TOKENS = 4096


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
    )


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def create_agent(model_name: str = DEFAULT_MODEL) -> Agent[None, LLMResponse]:
    """Create a pydantic-ai Agent configured for code generation."""
    from pydantic_ai import Agent

    return Agent(
        model_name,
        output_type=LLMResponse,
    )


@activity.defn
async def call_llm(context: AssembledContext) -> LLMCallResult:
    """Activity wrapper — creates an agent and delegates to execute_llm_call."""
    agent = create_agent()
    return await execute_llm_call(context, agent)
