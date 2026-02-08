"""Context assembly activity for Forge.

Builds the system and user prompts from a task definition and context files.

Design follows Function Core / Imperative Shell:
- Pure functions: build_system_prompt, build_user_prompt,
  build_step_system_prompt, build_step_user_prompt
- Imperative shell: _read_context_files, assemble_context, assemble_step_context
"""

from __future__ import annotations

import logging
from pathlib import Path

from temporalio import activity

from forge.models import (
    AssembleContextInput,
    AssembledContext,
    AssembleStepContextInput,
    AssembleSubTaskContextInput,
    PlanStep,
    StepResult,
    SubTask,
    TaskDefinition,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def build_system_prompt(task: TaskDefinition, context_file_contents: dict[str, str]) -> str:
    """Build the system prompt from a task definition and context file contents.

    Includes the task description, target files list, and any context files
    with delimiters.
    """
    parts: list[str] = []

    parts.append("You are a code generation assistant.")
    parts.append("")
    parts.append("## Task")
    parts.append(task.description)
    parts.append("")
    parts.append("## Target Files")
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

    return "\n".join(parts)


def build_user_prompt() -> str:
    """Build the user prompt.

    Short instruction directing the LLM to produce all target files.
    """
    return "Generate the code described above. Produce all target files with complete content."


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def _read_context_files(base_path: Path, file_paths: list[str]) -> dict[str, str]:
    """Read context files from disk.

    Skips missing files with a warning log. Context files are read from
    the repo root (not the worktree) so the LLM sees the current state
    of reference files.
    """
    contents: dict[str, str] = {}
    for file_path in file_paths:
        full_path = base_path / file_path
        if not full_path.is_file():
            logger.warning("Context file not found, skipping: %s", full_path)
            continue
        contents[file_path] = full_path.read_text()
    return contents


@activity.defn
async def assemble_context(input: AssembleContextInput) -> AssembledContext:
    """Read context files and assemble the prompts for the LLM call."""
    repo_root = Path(input.repo_root)
    context_contents = _read_context_files(repo_root, input.task.context_files)

    system_prompt = build_system_prompt(input.task, context_contents)
    user_prompt = build_user_prompt()

    return AssembledContext(
        task_id=input.task.task_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


# ---------------------------------------------------------------------------
# Step-level context (Phase 2)
# ---------------------------------------------------------------------------


def build_step_system_prompt(
    task: TaskDefinition,
    step: PlanStep,
    step_index: int,
    total_steps: int,
    completed_steps: list[StepResult],
    context_file_contents: dict[str, str],
) -> str:
    """Build the system prompt for a single step execution.

    Includes the overall task summary, plan progress, current step details,
    and context files read from the worktree.
    """
    parts: list[str] = []

    parts.append("You are a code generation assistant.")
    parts.append("")
    parts.append("## Overall Task")
    parts.append(task.description)
    parts.append("")
    parts.append(f"## Plan Progress ({step_index + 1}/{total_steps})")

    if completed_steps:
        parts.append("")
        parts.append("Completed steps:")
        for cs in completed_steps:
            parts.append(f"- {cs.step_id}: {cs.status.value}")
    else:
        parts.append("No steps completed yet.")

    parts.append("")
    parts.append("## Current Step")
    parts.append(f"**Step ID:** {step.step_id}")
    parts.append(f"**Description:** {step.description}")
    parts.append("")
    parts.append("### Target Files")
    for f in step.target_files:
        parts.append(f"- {f}")

    if context_file_contents:
        parts.append("")
        parts.append("### Context Files")
        for file_path, content in context_file_contents.items():
            parts.append("")
            parts.append(f"#### {file_path}")
            parts.append("```")
            parts.append(content)
            parts.append("```")

    return "\n".join(parts)


def build_step_user_prompt(step: PlanStep) -> str:
    """Build the user prompt for a single step execution."""
    return (
        f"Execute step '{step.step_id}': {step.description}\n\n"
        "Produce all target files with complete content."
    )


@activity.defn
async def assemble_step_context(input: AssembleStepContextInput) -> AssembledContext:
    """Read context files from the worktree and assemble step-level prompts.

    Context files are read from the **worktree** (not repo root) so that
    later steps can see files created by earlier steps.
    """
    worktree = Path(input.worktree_path)
    context_contents = _read_context_files(worktree, input.step.context_files)

    system_prompt = build_step_system_prompt(
        task=input.task,
        step=input.step,
        step_index=input.step_index,
        total_steps=input.total_steps,
        completed_steps=input.completed_steps,
        context_file_contents=context_contents,
    )
    user_prompt = build_step_user_prompt(input.step)

    return AssembledContext(
        task_id=input.task.task_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


# ---------------------------------------------------------------------------
# Sub-task context (Phase 3)
# ---------------------------------------------------------------------------


def build_sub_task_system_prompt(
    parent_task_id: str,
    parent_description: str,
    sub_task: SubTask,
    context_file_contents: dict[str, str],
) -> str:
    """Build the system prompt for a sub-task execution.

    Includes parent task context, sub-task description, target files,
    and context files read from the parent worktree.
    """
    parts: list[str] = []

    parts.append("You are a code generation assistant.")
    parts.append("")
    parts.append("## Parent Task")
    parts.append(f"**Task ID:** {parent_task_id}")
    parts.append(f"**Description:** {parent_description}")
    parts.append("")
    parts.append("## Sub-Task")
    parts.append(f"**Sub-Task ID:** {sub_task.sub_task_id}")
    parts.append(f"**Description:** {sub_task.description}")
    parts.append("")
    parts.append("### Target Files")
    for f in sub_task.target_files:
        parts.append(f"- {f}")

    if context_file_contents:
        parts.append("")
        parts.append("### Context Files")
        for file_path, content in context_file_contents.items():
            parts.append("")
            parts.append(f"#### {file_path}")
            parts.append("```")
            parts.append(content)
            parts.append("```")

    return "\n".join(parts)


def build_sub_task_user_prompt(sub_task: SubTask) -> str:
    """Build the user prompt for a sub-task execution."""
    return (
        f"Execute sub-task '{sub_task.sub_task_id}': {sub_task.description}\n\n"
        "Produce all target files with complete content."
    )


@activity.defn
async def assemble_sub_task_context(
    input: AssembleSubTaskContextInput,
) -> AssembledContext:
    """Read context files from the parent worktree and assemble sub-task prompts.

    Context files are read from the **parent worktree** because the sub-task
    worktree starts empty (branched from parent branch).
    """
    parent_worktree = Path(input.worktree_path)
    context_contents = _read_context_files(parent_worktree, input.sub_task.context_files)

    system_prompt = build_sub_task_system_prompt(
        parent_task_id=input.parent_task_id,
        parent_description=input.parent_description,
        sub_task=input.sub_task,
        context_file_contents=context_contents,
    )
    user_prompt = build_sub_task_user_prompt(input.sub_task)

    return AssembledContext(
        task_id=input.parent_task_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
