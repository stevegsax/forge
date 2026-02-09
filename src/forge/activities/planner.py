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
    *,
    repo_map: str | None = None,
) -> str:
    """Build the system prompt for the planning LLM call.

    Includes the task description, any target file hints, context files,
    optional repo map, and instructions for decomposing the task into ordered steps.
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

    if repo_map:
        parts.append("")
        parts.append("## Repository Structure")
        parts.append(
            "This is a compressed overview of the codebase, showing file paths "
            "and key signatures ranked by importance."
        )
        parts.append("```")
        parts.append(repo_map)
        parts.append("```")

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

    if repo_map:
        parts.append("")
        parts.append(
            "Note: Context is automatically discovered from import graphs for each step's "
            "target files. Use context_files for files not reachable via imports "
            "(e.g., config files, documentation, test fixtures)."
        )

    parts.append("")
    parts.append("## Fan-Out Sub-Tasks")
    parts.append(
        "Steps can optionally include `sub_tasks` for independent parallel work. "
        "When `sub_tasks` is set, the step's own `target_files` is ignored — "
        "each sub-task specifies its own target files."
    )
    parts.append("")
    parts.append("Rules for sub-tasks:")
    parts.append("- Sub-tasks run simultaneously and cannot see each other's outputs")
    parts.append("- Two sub-tasks must not write to the same file")
    parts.append(
        "- Use fan-out only when work items are genuinely independent "
        "and can be validated separately"
    )
    parts.append("")
    parts.append("Each sub-task specifies:")
    parts.append("- sub_task_id: A short identifier (e.g. 'analyze-schema', 'write-tests')")
    parts.append("- description: What the sub-task should produce")
    parts.append("- target_files: Files to create or modify")
    parts.append("- context_files: Files from the parent worktree to include as context")

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
    """Read context files and assemble the prompts for the planning call.

    When auto_discover is enabled, builds a repo map to include in the
    planner prompt, giving it a structural overview of the codebase.
    """
    repo_root = Path(input.repo_root)
    context_contents = _read_context_files(repo_root, input.task.context_files)

    repo_map_text: str | None = None

    if input.task.context.auto_discover:
        try:
            from forge.code_intel import (
                build_import_graph,
                extract_symbols,
                generate_repo_map,
                rank_files,
            )

            package_name = input.task.context.package_name or _detect_package_name(input.repo_root)
            graph = build_import_graph(package_name)

            # For the planner, rank all files (no specific targets)
            all_modules = list(graph.modules)
            ranked_set = rank_files(graph, all_modules[:5], "src", max_depth=3)

            # Build summaries for ranked files
            all_summaries = {}
            for rf in ranked_set.ranked_files:
                full = repo_root / rf.file_path
                if full.is_file():
                    source = full.read_text()
                    summary = extract_symbols(source, rf.file_path, rf.module_name)
                    all_summaries[rf.file_path] = summary

            repo_map_result = generate_repo_map(
                ranked_set.ranked_files,
                all_summaries,
                token_budget=input.task.context.repo_map_tokens,
            )
            if repo_map_result.content:
                repo_map_text = repo_map_result.content
        except Exception:
            logger.warning("Failed to build repo map for planner", exc_info=True)

    system_prompt = build_planner_system_prompt(
        input.task, context_contents, repo_map=repo_map_text
    )
    user_prompt = build_planner_user_prompt(input.task)

    return PlannerInput(
        task_id=input.task.task_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _detect_package_name(repo_root: str) -> str:
    """Detect the Python package name from the src/ directory."""
    src_dir = Path(repo_root) / "src"
    if src_dir.is_dir():
        for child in sorted(src_dir.iterdir()):
            if child.is_dir() and (child / "__init__.py").exists():
                return child.name
    return Path(repo_root).name


@activity.defn
async def call_planner(input: PlannerInput) -> PlanCallResult:
    """Activity wrapper — creates an agent and delegates to execute_planner_call."""
    agent = create_planner_agent()
    return await execute_planner_call(input, agent)
