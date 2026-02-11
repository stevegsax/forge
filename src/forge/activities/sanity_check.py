"""Sanity check activity for Forge.

Periodically evaluates whether the remaining plan is still valid given
completed work. Three verdicts: continue, revise, abort.

Design follows Function Core / Imperative Shell:
- Pure functions: build_sanity_check_system_prompt, build_sanity_check_user_prompt,
  build_step_digest
- Testable function: execute_sanity_check_call (takes agent as argument)
- Imperative shell: assemble_sanity_check_context, call_sanity_check,
  _persist_interaction
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from temporalio import activity

from forge.activities.context import (
    _read_project_instructions,
    build_project_instructions_section,
)
from forge.models import (
    AssembleSanityCheckContextInput,
    Plan,
    PlanStep,
    SanityCheckCallResult,
    SanityCheckInput,
    SanityCheckResponse,
    StepResult,
    TaskDefinition,
)

if TYPE_CHECKING:
    from pydantic_ai import Agent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def build_step_digest(step_result: StepResult) -> str:
    """Build a compact digest string from a StepResult.

    Format: "{step_id}: {status} -- wrote {file_count} files"
    Includes error summary if the step failed.
    """
    file_count = len(step_result.output_files)
    digest = f"{step_result.step_id}: {step_result.status.value} -- wrote {file_count} files"
    if step_result.error:
        digest += f" (error: {step_result.error})"
    return digest


def build_sanity_check_system_prompt(
    task: TaskDefinition,
    plan: Plan,
    completed_steps: list[StepResult],
    remaining_steps: list[PlanStep],
    project_instructions: str = "",
) -> str:
    """Build the system prompt for the sanity check LLM call."""
    parts: list[str] = []

    parts.append("You are a plan evaluation assistant.")
    parts.append(
        "Your job is to evaluate whether the remaining steps of a plan are still "
        "valid given the work completed so far."
    )

    if project_instructions:
        parts.append("")
        parts.append(project_instructions)

    parts.append("")
    parts.append("## Original Task")
    parts.append(task.description)

    parts.append("")
    parts.append("## Full Plan")
    for step in plan.steps:
        parts.append(f"- {step.step_id}: {step.description}")
        if step.target_files:
            parts.append(f"  target_files: {', '.join(step.target_files)}")

    parts.append("")
    parts.append("## Completed Steps")
    if completed_steps:
        for sr in completed_steps:
            parts.append(f"- {build_step_digest(sr)}")
    else:
        parts.append("(none)")

    parts.append("")
    parts.append("## Remaining Steps")
    if remaining_steps:
        for step in remaining_steps:
            parts.append(f"- {step.step_id}: {step.description}")
            if step.target_files:
                parts.append(f"  target_files: {', '.join(step.target_files)}")
    else:
        parts.append("(none)")

    parts.append("")
    parts.append("## Instructions")
    parts.append(
        "Evaluate whether the remaining steps are still valid and appropriate "
        "given the completed work. Consider:"
    )
    parts.append("- Are the remaining steps still necessary?")
    parts.append("- Do they need to be reordered or modified?")
    parts.append("- Has the completed work revealed new requirements or issues?")
    parts.append("- Should the task be aborted due to fundamental problems?")
    parts.append("")
    parts.append("Return one of three verdicts:")
    parts.append("- continue: The remaining plan is still valid. Proceed as planned.")
    parts.append(
        "- revise: The remaining plan needs changes. Provide revised_steps "
        "to replace ALL remaining steps."
    )
    parts.append(
        "- abort: The task should be stopped due to fundamental issues "
        "that cannot be resolved by revising steps."
    )
    parts.append("")
    parts.append(
        "If revising, provide replacement steps in the same schema as the original plan "
        "(step_id, description, target_files, context_files)."
    )

    return "\n".join(parts)


def build_sanity_check_user_prompt(completed_count: int, total_count: int) -> str:
    """Build the user prompt for the sanity check LLM call."""
    return (
        f"{completed_count} of {total_count} steps completed. "
        "Evaluate whether the remaining plan is still valid."
    )


# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


async def execute_sanity_check_call(
    input: SanityCheckInput,
    agent: Agent[None, SanityCheckResponse],
) -> SanityCheckCallResult:
    """Call the sanity check agent and extract structured results.

    Separated from the imperative shell so tests can inject a mock agent.
    """
    start = time.monotonic()

    result = await agent.run(
        input.user_prompt,
        instructions=input.system_prompt,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    usage = result.usage()

    return SanityCheckCallResult(
        task_id=input.task_id,
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
    sanity_input: SanityCheckInput,
    result: SanityCheckCallResult,
) -> None:
    """Best-effort store write. Never raises (D42)."""
    try:
        from forge.models import AssembledContext
        from forge.store import build_interaction_dict, get_db_path, get_engine, save_interaction

        db_path = get_db_path()
        if db_path is None:
            return

        context = AssembledContext(
            task_id=sanity_input.task_id,
            system_prompt=sanity_input.system_prompt,
            user_prompt=sanity_input.user_prompt,
        )

        engine = get_engine(db_path)
        data = build_interaction_dict(
            task_id=sanity_input.task_id,
            step_id=None,
            sub_task_id=None,
            role="sanity_check",
            context=context,
            llm_result=result,
        )
        save_interaction(engine, **data)
    except Exception:
        logger.warning("Failed to persist sanity check interaction to store", exc_info=True)


def create_sanity_check_agent(
    model_name: str | None = None,
    *,
    thinking_budget_tokens: int = 0,
    thinking_effort: str = "high",
) -> Agent[None, SanityCheckResponse]:
    """Create a pydantic-ai Agent configured for plan evaluation."""
    from pydantic_ai import Agent

    from forge.activities.llm import DEFAULT_MODEL
    from forge.activities.planner import build_thinking_settings

    if model_name is None:
        model_name = DEFAULT_MODEL

    settings: dict[str, object] = {
        "anthropic_cache_instructions": True,
        "anthropic_cache_tool_definitions": True,
    }

    if thinking_budget_tokens > 0:
        thinking_settings = build_thinking_settings(
            model_name, thinking_budget_tokens, thinking_effort
        )
        settings.update(thinking_settings)

    return Agent(
        model_name,
        output_type=SanityCheckResponse,
        model_settings=settings,
    )


@activity.defn
async def assemble_sanity_check_context(
    input: AssembleSanityCheckContextInput,
) -> SanityCheckInput:
    """Read project instructions and assemble prompts for the sanity check."""
    repo_root = Path(input.repo_root)
    project_instructions = build_project_instructions_section(
        _read_project_instructions(repo_root)
    )

    system_prompt = build_sanity_check_system_prompt(
        task=input.task,
        plan=input.plan,
        completed_steps=input.completed_steps,
        remaining_steps=input.remaining_steps,
        project_instructions=project_instructions,
    )
    user_prompt = build_sanity_check_user_prompt(
        completed_count=len(input.completed_steps),
        total_count=len(input.plan.steps),
    )

    return SanityCheckInput(
        task_id=input.task.task_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


@activity.defn
async def call_sanity_check(input: SanityCheckInput) -> SanityCheckCallResult:
    """Activity wrapper -- creates an agent and delegates to execute_sanity_check_call."""
    from forge.tracing import get_tracer, llm_call_attributes

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.call_sanity_check") as span:
        agent = create_sanity_check_agent(
            input.model_name or None,
            thinking_budget_tokens=input.thinking_budget_tokens,
            thinking_effort=input.thinking_effort,
        )
        result = await execute_sanity_check_call(input, agent)

        span.set_attributes(
            llm_call_attributes(
                model_name=result.model_name,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=result.latency_ms,
                task_id=input.task_id,
                cache_creation_input_tokens=result.cache_creation_input_tokens,
                cache_read_input_tokens=result.cache_read_input_tokens,
            )
        )

        _persist_interaction(input, result)
        return result
