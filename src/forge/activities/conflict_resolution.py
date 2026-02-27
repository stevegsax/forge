"""Conflict resolution activity for Forge.

Resolves file conflicts when multiple sub-tasks produce the same file during
fan-out execution. Uses REASONING-tier LLM to merge competing versions.

Design follows Function Core / Imperative Shell:
- Pure functions: classify_file_conflicts, build_conflict_resolution_system_prompt,
  build_conflict_resolution_user_prompt
- I/O wrapper: detect_file_conflicts (calls classify + reads originals from disk)
- Temporal activities: detect_file_conflicts_activity, assemble_conflict_resolution_context,
  call_conflict_resolution
- Testable function: execute_conflict_resolution_call (takes client as argument)
- Imperative shell: store.persist_interaction
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
    ConflictResolutionCallInput,
    ConflictResolutionCallResult,
    ConflictResolutionInput,
    ConflictResolutionResponse,
    DetectFileConflictsInput,
    DetectFileConflictsOutput,
    FileConflict,
    FileConflictVersion,
    SubTaskResult,
    TransitionSignal,
)

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

DEFAULT_CONFLICT_RESOLUTION_MAX_TOKENS = 8192


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def classify_file_conflicts(
    sub_task_results: list[SubTaskResult],
) -> tuple[dict[str, str], list[FileConflict]]:
    """Separate sub-task outputs into non-conflicting files and conflicts.

    Pure function — no filesystem I/O. Conflicts have ``original_content=None``.

    Returns:
        A tuple of (non_conflicting_files, conflicts) where:
        - non_conflicting_files: dict mapping file_path -> content for unique files
        - conflicts: list of FileConflict for files produced by multiple sub-tasks
    """
    file_sources: dict[str, list[FileConflictVersion]] = {}
    for result in sub_task_results:
        if result.status != TransitionSignal.SUCCESS:
            continue
        for file_path, content in result.output_files.items():
            if file_path not in file_sources:
                file_sources[file_path] = []
            file_sources[file_path].append(
                FileConflictVersion(source_id=result.sub_task_id, content=content)
            )

    non_conflicting: dict[str, str] = {}
    conflicts: list[FileConflict] = []

    for file_path, versions in file_sources.items():
        if len(versions) == 1:
            non_conflicting[file_path] = versions[0].content
        else:
            conflicts.append(
                FileConflict(
                    file_path=file_path,
                    versions=versions,
                    original_content=None,
                )
            )

    return non_conflicting, conflicts


def detect_file_conflicts(
    sub_task_results: list[SubTaskResult],
    worktree_path: str | None = None,
) -> tuple[dict[str, str], list[FileConflict]]:
    """Classify conflicts and read original content from the worktree.

    Thin wrapper around :func:`classify_file_conflicts` that adds filesystem
    reads for ``original_content``.  Preserves backward compatibility for
    existing callers and tests.
    """
    non_conflicting, conflicts = classify_file_conflicts(sub_task_results)

    if worktree_path:
        for conflict in conflicts:
            original_path = Path(worktree_path) / conflict.file_path
            try:
                if original_path.is_file():
                    conflict.original_content = original_path.read_text()
            except Exception:
                logger.debug("Could not read original content for %s", conflict.file_path)

    return non_conflicting, conflicts


@activity.defn
async def detect_file_conflicts_activity(
    input: DetectFileConflictsInput,
) -> DetectFileConflictsOutput:
    """Temporal activity wrapper for detect_file_conflicts."""
    non_conflicting, conflicts = detect_file_conflicts(
        input.sub_task_results, input.worktree_path
    )
    return DetectFileConflictsOutput(
        non_conflicting_files=non_conflicting,
        conflicts=conflicts,
    )


def build_conflict_resolution_system_prompt(
    task_description: str,
    step_description: str,
    conflicts: list[FileConflict],
    non_conflicting_file_paths: list[str],
    project_instructions: str = "",
) -> str:
    """Build the system prompt for conflict resolution."""
    parts: list[str] = []

    parts.append("You are a code conflict resolution assistant.")
    parts.append(
        "Multiple sub-tasks have produced different versions of the same file(s). "
        "Your job is to merge these versions into a single coherent result that "
        "preserves the intent of all sub-tasks."
    )

    if project_instructions:
        parts.append("")
        parts.append(project_instructions)

    parts.append("")
    parts.append("## Task Context")
    parts.append(f"Task: {task_description}")
    parts.append(f"Step: {step_description}")

    if non_conflicting_file_paths:
        parts.append("")
        parts.append("## Non-Conflicting Files (already merged)")
        for fp in non_conflicting_file_paths:
            parts.append(f"- {fp}")

    parts.append("")
    parts.append("## Conflicting Files")
    for conflict in conflicts:
        parts.append("")
        parts.append(f"### {conflict.file_path}")
        if conflict.original_content is not None:
            parts.append("")
            parts.append("**Original content (before sub-tasks):**")
            parts.append("```")
            parts.append(conflict.original_content)
            parts.append("```")
        else:
            parts.append("")
            parts.append("**Original: (new file)**")

        for version in conflict.versions:
            parts.append("")
            parts.append(f"**Version from sub-task `{version.source_id}`:**")
            parts.append("```")
            parts.append(version.content)
            parts.append("```")

    parts.append("")
    parts.append("## Instructions")
    parts.append(
        "For each conflicting file, produce a single merged version that combines "
        "the contributions from all sub-tasks. Preserve all functionality from each "
        "version. If the versions add different functions/classes/sections, include all "
        "of them. If they modify the same section differently, use your judgment to "
        "create the best combined result."
    )
    parts.append("")
    parts.append(
        "Return a `resolved_files` list with one entry per conflicting file. "
        "Each entry must have the exact `file_path` and the merged `content`."
    )

    return "\n".join(parts)


def build_conflict_resolution_user_prompt(conflict_count: int) -> str:
    """Build the user prompt for conflict resolution."""
    return (
        f"Resolve {conflict_count} file conflict(s). "
        "Merge all competing versions into single coherent files."
    )


# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


async def execute_conflict_resolution_call(
    input: ConflictResolutionCallInput,
    client: AsyncAnthropic,
) -> ConflictResolutionCallResult:
    """Call the Anthropic API for conflict resolution and extract structured results.

    Separated from the imperative shell so tests can inject a mock client.
    """
    model = input.model_name or "claude-sonnet-4-5-20250929"
    start = time.monotonic()

    params = build_messages_params(
        system_prompt=input.system_prompt,
        user_prompt=input.user_prompt,
        output_type=ConflictResolutionResponse,
        model=model,
        max_tokens=DEFAULT_CONFLICT_RESOLUTION_MAX_TOKENS,
        thinking_budget_tokens=input.thinking.budget_tokens,
    )
    message = await client.messages.create(**params)

    if input.log_messages and input.worktree_path:
        request_json = json.dumps(params, indent=2, default=str)
        write_message_log(input.worktree_path, "conflict-request", request_json)
        write_message_log(
            input.worktree_path,
            "conflict-response",
            message.model_dump_json(indent=2),
        )

    elapsed_ms = (time.monotonic() - start) * 1000
    response = extract_tool_result(message, ConflictResolutionResponse)
    in_tok, out_tok, cache_create, cache_read = extract_usage(message)

    return ConflictResolutionCallResult(
        task_id=input.task_id,
        resolved_files={f.file_path: f.content for f in response.resolved_files},
        explanation=response.explanation,
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
async def assemble_conflict_resolution_context(
    input: ConflictResolutionInput,
) -> ConflictResolutionCallInput:
    """Read project instructions and assemble prompts for conflict resolution."""
    repo_root = Path(input.repo_root)
    project_instructions = build_project_instructions_section(_read_project_instructions(repo_root))

    system_prompt = build_conflict_resolution_system_prompt(
        task_description=input.task_description,
        step_description=input.step_description,
        conflicts=input.conflicts,
        non_conflicting_file_paths=list(input.non_conflicting_files.keys()),
        project_instructions=project_instructions,
    )
    user_prompt = build_conflict_resolution_user_prompt(len(input.conflicts))

    return ConflictResolutionCallInput(
        task_id=input.task_id,
        step_id=input.step_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=input.model_name,
        thinking=input.thinking,
    )


@activity.defn
async def call_conflict_resolution(
    input: ConflictResolutionCallInput,
) -> ConflictResolutionCallResult:
    """Activity wrapper -- creates a client and delegates to execute_conflict_resolution_call."""
    from forge.llm_client import get_anthropic_client
    from forge.tracing import get_tracer, llm_call_attributes

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.call_conflict_resolution") as span:
        logger.info("Conflict resolution call: task_id=%s", input.task_id)
        client = get_anthropic_client()
        async with heartbeat_during():
            result = await execute_conflict_resolution_call(input, client)

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
            role="conflict_resolution",
            system_prompt=input.system_prompt,
            user_prompt=input.user_prompt,
            llm_result=result,
            step_id=input.step_id,
        )
        return result
