"""Planning activity for Forge.

Decomposes a task into ordered steps using an LLM with structured output.

Design follows Function Core / Imperative Shell:
- Pure functions: build_planner_system_prompt, build_planner_user_prompt
- Testable function: execute_planner_call (takes agent as argument)
- Imperative shell: create_planner_agent, assemble_planner_context, call_planner
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from temporalio import activity

from forge.activities.context import _read_context_files
from forge.models import (
    AssembleContextInput,
    Plan,
    PlanCallResult,
    PlannerInput,
    TaskDefinition,
)

if TYPE_CHECKING:
    from pydantic_ai import Agent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def build_planner_system_prompt(
    task: TaskDefinition,
    context_file_contents: dict[str, str],
) -> str:
    """Build the system prompt for the planning LLM call.

    Includes the task description, any target file hints, context files,
    and instructions for decomposing the task into ordered steps.
    """
    parts: list[str] = []

    parts.append("You are a task decomposition assistant.")
    parts.append("")
    parts.append("## Task")
    parts.append(task.description)

    if task.target_files:
        parts.append("")
        parts.append("## Target Files (hints)")
        for f in task.target_files:
            parts.append(f"- {f}")

    if context_file_contents:
        parts.append("")
        parts.append("## Context Files")
        for file_path, content in context_file_contents.items():
            parts.append("")
            parts.append(f"### {file_path}")
            parts.append("```")
            parts.append(content)
            parts.append("```")

    parts.append("")
    parts.append("## Instructions")
    parts.append(
        "Decompose the task into ordered steps. Each step should be a small, "
        "independent unit of work that produces one or more files. Steps are "
        "executed sequentially — later steps can read files created by earlier "
        "steps. Each step will be committed separately."
    )
    parts.append("")
    parts.append("For each step, specify:")
    parts.append("- step_id: A short identifier (e.g. 'step-1', 'create-models')")
    parts.append("- description: What the step should accomplish")
    parts.append("- target_files: Files to create or modify in this step")
    parts.append(
        "- context_files: Files from the repo (or created by prior steps) to include as context"
    )

    return "\n".join(parts)


def build_planner_user_prompt(task: TaskDefinition) -> str:
    """Build the user prompt for the planning LLM call."""
    return (
        f"Decompose the following task into ordered steps: {task.description}\n\n"
        "Produce a plan with at least one step."
    )


# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


async def execute_planner_call(
    input: PlannerInput,
    agent: Agent[None, Plan],
) -> PlanCallResult:
    """Call the planning agent and extract structured results.

    Separated from the imperative shell so tests can inject a mock agent.
    """
    start = time.monotonic()

    result = await agent.run(
        input.user_prompt,
        instructions=input.system_prompt,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    usage = result.usage()

    return PlanCallResult(
        task_id=input.task_id,
        plan=result.output,
        model_name=str(agent.model),
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        latency_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def create_planner_agent(model_name: str | None = None) -> Agent[None, Plan]:
    """Create a pydantic-ai Agent configured for task decomposition."""
    from pydantic_ai import Agent

    from forge.activities.llm import DEFAULT_MODEL

    if model_name is None:
        model_name = DEFAULT_MODEL

    return Agent(
        model_name,
        output_type=Plan,
    )


@activity.defn
async def assemble_planner_context(input: AssembleContextInput) -> PlannerInput:
    """Read context files and assemble the prompts for the planning call."""
    repo_root = Path(input.repo_root)
    context_contents = _read_context_files(repo_root, input.task.context_files)

    system_prompt = build_planner_system_prompt(input.task, context_contents)
    user_prompt = build_planner_user_prompt(input.task)

    return PlannerInput(
        task_id=input.task.task_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


@activity.defn
async def call_planner(input: PlannerInput) -> PlanCallResult:
    """Activity wrapper — creates an agent and delegates to execute_planner_call."""
    agent = create_planner_agent()
    return await execute_planner_call(input, agent)
