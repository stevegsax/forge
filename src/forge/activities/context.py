"""Context assembly activity for Forge.

Builds the system and user prompts from a task definition and context files.

Design follows Function Core / Imperative Shell:
- Pure functions: build_system_prompt, build_user_prompt,
  build_step_system_prompt, build_step_user_prompt,
  build_system_prompt_with_context
- Imperative shell: _read_context_files, assemble_context, assemble_step_context
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from temporalio import activity

from forge.models import (
    AssembleContextInput,
    AssembledContext,
    AssembleStepContextInput,
    AssembleSubTaskContextInput,
    ContextStats,
    PlanStep,
    StepResult,
    SubTask,
    TaskDefinition,
)

if TYPE_CHECKING:
    from forge.code_intel.budget import PackedContext

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


def build_system_prompt_with_context(task: TaskDefinition, packed: PackedContext) -> str:
    """Build the system prompt using auto-discovered packed context.

    Organizes context items by priority tier with labeled sections.
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

    # Group items by representation type
    _append_context_section(
        parts,
        "Target File Contents",
        packed,
        priority=2,
    )
    _append_context_section(
        parts,
        "Direct Dependencies (full content)",
        packed,
        priority=3,
    )
    _append_context_section(
        parts,
        "Interface Context (signatures)",
        packed,
        priority=4,
    )

    # Repo map
    repo_map_items = [i for i in packed.items if i.priority == 5]
    if repo_map_items:
        parts.append("")
        parts.append("## Repository Structure")
        parts.append("```")
        parts.append(repo_map_items[0].content)
        parts.append("```")

    _append_context_section(
        parts,
        "Additional Context",
        packed,
        priority=6,
    )

    return "\n".join(parts)


def _append_context_section(
    parts: list[str],
    section_title: str,
    packed: PackedContext,
    priority: int,
) -> None:
    """Append a context section for items at the given priority level."""
    from forge.code_intel.budget import Representation

    items = [i for i in packed.items if i.priority == priority]
    if not items:
        return

    parts.append("")
    parts.append(f"## {section_title}")
    for item in items:
        parts.append("")
        if item.representation == Representation.SIGNATURES:
            parts.append(f"### {item.file_path} (signatures)")
        else:
            parts.append(f"### {item.file_path}")
        parts.append("```")
        parts.append(item.content)
        parts.append("```")


def _build_context_stats(packed: PackedContext) -> ContextStats:
    """Build ContextStats from a PackedContext."""
    from forge.code_intel.budget import Representation

    full_count = sum(1 for i in packed.items if i.representation == Representation.FULL)
    sig_count = sum(1 for i in packed.items if i.representation == Representation.SIGNATURES)
    repo_map_tokens = sum(
        i.estimated_tokens for i in packed.items if i.representation == Representation.REPO_MAP
    )

    return ContextStats(
        files_discovered=packed.items_included + packed.items_truncated,
        files_included_full=full_count,
        files_included_signatures=sig_count,
        files_truncated=packed.items_truncated,
        total_estimated_tokens=packed.total_estimated_tokens,
        budget_utilization=packed.budget_utilization,
        repo_map_tokens=repo_map_tokens,
    )


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
    """Read context files and assemble the prompts for the LLM call.

    When auto_discover is enabled and target_files are specified, uses
    automatic context discovery via import graph analysis. Otherwise
    falls back to manual context_files.
    """
    task = input.task
    repo_root = Path(input.repo_root)

    if task.context.auto_discover and task.target_files:
        from forge.code_intel import discover_context

        manual_contents = _read_context_files(repo_root, task.context_files)

        packed = discover_context(
            target_files=task.target_files,
            project_root=input.repo_root,
            package_name=task.context.package_name or _detect_package_name(input.repo_root),
            src_root="src",
            manual_context=manual_contents,
            token_budget=task.context.token_budget,
            max_import_depth=task.context.max_import_depth,
            include_repo_map=task.context.include_repo_map,
            repo_map_tokens=task.context.repo_map_tokens,
        )

        system_prompt = build_system_prompt_with_context(task, packed)
        context_stats = _build_context_stats(packed)
    else:
        context_contents = _read_context_files(repo_root, task.context_files)
        system_prompt = build_system_prompt(task, context_contents)
        context_stats = None

    user_prompt = build_user_prompt()

    return AssembledContext(
        task_id=task.task_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_stats=context_stats,
    )


def _detect_package_name(repo_root: str) -> str:
    """Detect the Python package name from the src/ directory.

    Looks for the first directory in src/ that contains __init__.py.
    Falls back to the repo directory name.
    """
    src_dir = Path(repo_root) / "src"
    if src_dir.is_dir():
        for child in sorted(src_dir.iterdir()):
            if child.is_dir() and (child / "__init__.py").exists():
                return child.name
    return Path(repo_root).name


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
