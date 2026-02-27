"""Planning activity for Forge.

Decomposes a task into ordered steps using an LLM with structured output.

Design follows Function Core / Imperative Shell:
- Pure functions: build_planner_system_prompt, build_planner_user_prompt
- Testable function: execute_planner_call (takes client as argument)
- Imperative shell: assemble_planner_context, call_planner, store.persist_interaction
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
    _read_context_files,
    _read_project_instructions,
    build_project_instructions_section,
)
from forge.domains import get_domain_config
from forge.llm_client import build_messages_params, extract_tool_result, extract_usage
from forge.message_log import write_message_log
from forge.models import (
    AssembleContextInput,
    Plan,
    PlanCallResult,
    PlannerInput,
    TaskDefinition,
)

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

DEFAULT_PLANNER_MAX_TOKENS = 8192


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def build_planner_system_prompt(
    task: TaskDefinition,
    context_file_contents: dict[str, str],
    *,
    repo_map: str | None = None,
    project_instructions: str = "",
) -> str:
    """Build the system prompt for the planning LLM call.

    Includes the task description, any target file hints, context files,
    optional repo map, and instructions for decomposing the task into ordered steps.
    """
    parts: list[str] = []

    parts.append("You are a task decomposition assistant.")

    if project_instructions:
        parts.append("")
        parts.append(project_instructions)

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
    parts.append("## Capability Tier")
    parts.append(
        "Each step can optionally specify a `capability_tier` to control which "
        "model class handles it. When omitted, the step uses the GENERATION tier."
    )
    parts.append("")
    parts.append("Available tiers:")
    parts.append("- REASONING — for complex analysis, planning, or architectural decisions")
    parts.append("- GENERATION — for standard code generation (default)")
    parts.append("- SUMMARIZATION — for summarization and extraction tasks")
    parts.append("- CLASSIFICATION — for lightweight classification or triage tasks")

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
    parts.append("")
    parts.append("## Recursive Sub-Tasks")
    parts.append(
        "Sub-tasks can themselves contain nested `sub_tasks` for recursive fan-out. "
        "The maximum nesting depth is configurable (default: 1, meaning flat fan-out only). "
        "Prefer flat fan-out unless the decomposition clearly requires hierarchy — "
        "e.g. a sub-task that itself contains genuinely independent work items."
    )

    domain_config = get_domain_config(task.domain)
    parts.append("")
    parts.append("## Task Domain")
    parts.append(domain_config.planner_domain_instruction)

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
    client: AsyncAnthropic,
) -> PlanCallResult:
    """Call the Anthropic API for planning and extract structured results.

    Separated from the imperative shell so tests can inject a mock client.
    """
    model = input.model_name or "claude-sonnet-4-5-20250929"
    start = time.monotonic()

    params = build_messages_params(
        system_prompt=input.system_prompt,
        user_prompt=input.user_prompt,
        output_type=Plan,
        model=model,
        max_tokens=DEFAULT_PLANNER_MAX_TOKENS,
        thinking_budget_tokens=input.thinking.budget_tokens,
        thinking_effort=input.thinking.effort,
    )
    message = await client.messages.create(**params)

    if input.log_messages and input.worktree_path:
        request_json = json.dumps(params, indent=2, default=str)
        write_message_log(input.worktree_path, "planner-request", request_json)
        write_message_log(
            input.worktree_path, "planner-response", message.model_dump_json(indent=2)
        )

    elapsed_ms = (time.monotonic() - start) * 1000
    plan = extract_tool_result(message, Plan)
    in_tok, out_tok, cache_create, cache_read = extract_usage(message)

    return PlanCallResult(
        task_id=input.task_id,
        plan=plan,
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
async def assemble_planner_context(input: AssembleContextInput) -> PlannerInput:
    """Read context files and assemble the prompts for the planning call.

    When auto_discover is enabled, builds a repo map to include in the
    planner prompt, giving it a structural overview of the codebase.
    """
    repo_root = Path(input.repo_root)
    context_contents = _read_context_files(repo_root, input.context_files)

    repo_map_text: str | None = None

    if input.context_config.auto_discover:
        try:
            from forge.code_intel import (
                build_import_graph,
                extract_symbols,
                generate_repo_map,
                rank_files,
            )

            package_name = input.context_config.package_name or _detect_package_name(
                input.repo_root
            )
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
                token_budget=input.context_config.repo_map_tokens,
            )
            if repo_map_result.content:
                repo_map_text = repo_map_result.content
        except Exception:
            logger.warning("Failed to build repo map for planner", exc_info=True)

    project_instructions = build_project_instructions_section(_read_project_instructions(repo_root))

    task_mock = TaskDefinition(
        task_id=input.task_id,
        description=input.description,
        target_files=input.target_files,
        context_files=input.context_files,
        context=input.context_config,
    )

    system_prompt = build_planner_system_prompt(
        task_mock,
        context_contents,
        repo_map=repo_map_text,
        project_instructions=project_instructions,
    )
    user_prompt = build_planner_user_prompt(task_mock)

    return PlannerInput(
        task_id=input.task_id,
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
    """Activity wrapper — creates a client and delegates to execute_planner_call."""
    from forge.llm_client import get_anthropic_client
    from forge.tracing import get_tracer, llm_call_attributes

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.call_planner") as span:
        logger.info("Planner call: task_id=%s", input.task_id)
        client = get_anthropic_client()
        async with heartbeat_during():
            result = await execute_planner_call(input, client)
        logger.info("Plan produced: task_id=%s steps=%d", input.task_id, len(result.plan.steps))

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
            role="planner",
            system_prompt=input.system_prompt,
            user_prompt=input.user_prompt,
            llm_result=result,
        )
        return result
