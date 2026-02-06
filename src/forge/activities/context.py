"""Context assembly activity for Forge.

Builds the system and user prompts from a task definition and context files.

Design follows Function Core / Imperative Shell:
- Pure functions: build_system_prompt, build_user_prompt
- Imperative shell: _read_context_files, assemble_context
"""

from __future__ import annotations

import logging
from pathlib import Path

from temporalio import activity

from forge.models import AssembleContextInput, AssembledContext, TaskDefinition

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
