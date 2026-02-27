"""Sanity check activity for Forge.

Periodically evaluates whether the remaining plan is still valid given
completed work. Three verdicts: continue, revise, abort.

Design follows Function Core / Imperative Shell:
- Pure functions: build_sanity_check_system_prompt, build_sanity_check_user_prompt,
  build_step_digest
- Testable function: execute_sanity_check_call (takes client as argument)
- Imperative shell: assemble_sanity_check_context, call_sanity_check,
  store.persist_interaction
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from temporalio import activity

from forge.activities._heartbeat import heartbeat_during
from forge.activities.context import (
    _read_project_instructions,
    build_project_instructions_section,
)
from forge.llm_client import build_messages_params, extract_tool_result, extract_usage
from forge.message_log import write_message_log
from forge.models import (
    AssembleSanityCheckContextInput,
    Plan,
    PlanStep,
    SanityCheckCallResult,
    SanityCheckInput,
    SanityCheckResponse,
    StepResult,
)

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

DEFAULT_SANITY_CHECK_MAX_TOKENS = 4096


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
    task_id: str,
    task_description: str,
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
    parts.append(task_description)

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
    client: AsyncAnthropic,
) -> SanityCheckCallResult:
    """Call the Anthropic API for sanity check and extract structured results.

    Separated from the imperative shell so tests can inject a mock client.
    """
    model = input.model_name or "claude-sonnet-4-5-20250929"
    start = time.monotonic()

    params = build_messages_params(
        system_prompt=input.system_prompt,
        user_prompt=input.user_prompt,
        output_type=SanityCheckResponse,
        model=model,
        max_tokens=DEFAULT_SANITY_CHECK_MAX_TOKENS,
        thinking_budget_tokens=input.thinking.budget_tokens,
    )
    message = await client.messages.create(**params)

    if input.log_messages and input.worktree_path:
        request_json = json.dumps(params, indent=2, default=str)
        write_message_log(input.worktree_path, "sanity-request", request_json)
        write_message_log(input.worktree_path, "sanity-response", message.model_dump_json(indent=2))

    elapsed_ms = (time.monotonic() - start) * 1000
    response = extract_tool_result(message, SanityCheckResponse)
    in_tok, out_tok, cache_create, cache_read = extract_usage(message)

    return SanityCheckCallResult(
        task_id=input.task_id,
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


@activity.defn
async def assemble_sanity_check_context(
    input: AssembleSanityCheckContextInput,
) -> SanityCheckInput:
    """Read project instructions and assemble prompts for the sanity check."""
    repo_root = Path(input.repo_root)
    project_instructions = build_project_instructions_section(_read_project_instructions(repo_root))

    system_prompt = build_sanity_check_system_prompt(
        task_id=input.task_id,
        task_description=input.task_description,
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
        task_id=input.task_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


@activity.defn
async def call_sanity_check(input: SanityCheckInput) -> SanityCheckCallResult:
    """Activity wrapper -- creates a client and delegates to execute_sanity_check_call."""
    from forge.llm_client import get_anthropic_client
    from forge.tracing import get_tracer, llm_call_attributes

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.call_sanity_check") as span:
        logger.info("Sanity check call: task_id=%s", input.task_id)
        client = get_anthropic_client()
        async with heartbeat_during():
            result = await execute_sanity_check_call(input, client)
        logger.info(
            "Sanity verdict: task_id=%s verdict=%s", input.task_id, result.response.verdict.value
        )

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

        from forge.store import persist_interaction

        persist_interaction(
            task_id=input.task_id,
            role="sanity_check",
            system_prompt=input.system_prompt,
            user_prompt=input.user_prompt,
            llm_result=result,
        )
        return result
