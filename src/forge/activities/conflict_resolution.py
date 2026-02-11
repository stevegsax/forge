"""Conflict resolution activity for Forge.

Resolves file conflicts when multiple sub-tasks produce the same file during
fan-out execution. Uses REASONING-tier LLM to merge competing versions.

Design follows Function Core / Imperative Shell:
- Pure functions: detect_file_conflicts, build_conflict_resolution_system_prompt,
  build_conflict_resolution_user_prompt
- Testable function: execute_conflict_resolution_call (takes agent as argument)
- Imperative shell: assemble_conflict_resolution_context, call_conflict_resolution,
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
    ConflictResolutionCallInput,
    ConflictResolutionCallResult,
    ConflictResolutionInput,
    ConflictResolutionResponse,
    FileConflict,
    FileConflictVersion,
    SubTaskResult,
    TaskDomain,
    TransitionSignal,
)

if TYPE_CHECKING:
    from pydantic_ai import Agent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def detect_file_conflicts(
    sub_task_results: list[SubTaskResult],
    worktree_path: str | None = None,
) -> tuple[dict[str, str], list[FileConflict]]:
    """Separate sub-task outputs into non-conflicting files and conflicts.

    Returns:
        A tuple of (non_conflicting_files, conflicts) where:
        - non_conflicting_files: dict mapping file_path -> content for unique files
        - conflicts: list of FileConflict for files produced by multiple sub-tasks
    """
    # Track which sub-tasks produced each file path
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
            # Read original content from worktree if available
            original_content: str | None = None
            if worktree_path:
                original_path = Path(worktree_path) / file_path
                try:
                    if original_path.is_file():
                        original_content = original_path.read_text()
                except Exception:
                    logger.debug("Could not read original content for %s", file_path)

            conflicts.append(
                FileConflict(
                    file_path=file_path,
                    versions=versions,
                    original_content=original_content,
                )
            )

    return non_conflicting, conflicts


def build_conflict_resolution_system_prompt(
    task_description: str,
    step_description: str,
    conflicts: list[FileConflict],
    non_conflicting_file_paths: list[str],
    project_instructions: str = "",
    domain: TaskDomain = TaskDomain.CODE_GENERATION,
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
    agent: Agent[None, ConflictResolutionResponse],
) -> ConflictResolutionCallResult:
    """Call the conflict resolution agent and extract structured results.

    Separated from the imperative shell so tests can inject a mock agent.
    """
    start = time.monotonic()

    result = await agent.run(
        input.user_prompt,
        instructions=input.system_prompt,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    usage = result.usage()

    return ConflictResolutionCallResult(
        task_id=input.task_id,
        resolved_files={f.file_path: f.content for f in result.output.resolved_files},
        explanation=result.output.explanation,
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
    call_input: ConflictResolutionCallInput,
    result: ConflictResolutionCallResult,
) -> None:
    """Best-effort store write. Never raises (D42)."""
    try:
        from forge.models import AssembledContext
        from forge.store import build_interaction_dict, get_db_path, get_engine, save_interaction

        db_path = get_db_path()
        if db_path is None:
            return

        context = AssembledContext(
            task_id=call_input.task_id,
            system_prompt=call_input.system_prompt,
            user_prompt=call_input.user_prompt,
            step_id=call_input.step_id,
        )

        engine = get_engine(db_path)
        data = build_interaction_dict(
            task_id=call_input.task_id,
            step_id=call_input.step_id,
            sub_task_id=None,
            role="conflict_resolution",
            context=context,
            llm_result=result,
        )
        save_interaction(engine, **data)
    except Exception:
        logger.warning("Failed to persist conflict resolution interaction to store", exc_info=True)


def create_conflict_resolution_agent(
    model_name: str | None = None,
    *,
    thinking_budget_tokens: int = 0,
    thinking_effort: str = "high",
) -> Agent[None, ConflictResolutionResponse]:
    """Create a pydantic-ai Agent configured for conflict resolution."""
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
        output_type=ConflictResolutionResponse,
        model_settings=settings,
    )


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
        domain=input.domain,
    )
    user_prompt = build_conflict_resolution_user_prompt(len(input.conflicts))

    return ConflictResolutionCallInput(
        task_id=input.task_id,
        step_id=input.step_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=input.model_name,
        thinking_budget_tokens=input.thinking_budget_tokens,
        thinking_effort=input.thinking_effort,
    )


@activity.defn
async def call_conflict_resolution(
    input: ConflictResolutionCallInput,
) -> ConflictResolutionCallResult:
    """Activity wrapper -- creates an agent and delegates to execute_conflict_resolution_call."""
    from forge.tracing import get_tracer, llm_call_attributes

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.call_conflict_resolution") as span:
        agent = create_conflict_resolution_agent(
            input.model_name or None,
            thinking_budget_tokens=input.thinking_budget_tokens,
            thinking_effort=input.thinking_effort,
        )
        result = await execute_conflict_resolution_call(input, agent)

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
