"""Knowledge extraction activities for Forge.

Extracts structured lessons from completed task results via an LLM call.

Design follows Function Core / Imperative Shell:
- Pure functions: build_extraction_system_prompt, build_extraction_user_prompt,
  infer_tags_from_task
- Testable function: execute_extraction_call (takes agent as argument)
- Imperative shell: create_extraction_agent, fetch_extraction_input,
  call_extraction_llm, save_extraction_results
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from temporalio import activity

from forge.models import (
    ExtractionCallResult,
    ExtractionInput,
    ExtractionResult,
    FetchExtractionInput,
    SaveExtractionInput,
)

if TYPE_CHECKING:
    from pydantic_ai import Agent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def build_extraction_system_prompt(run_data: list[dict]) -> str:
    """Build the system prompt for knowledge extraction.

    Each item in run_data is a dict from the runs table with keys:
    task_id, workflow_id, status, result_json, created_at.
    The result_json is already parsed into a 'result' dict key.
    """
    parts: list[str] = []

    parts.append("You are a knowledge extraction assistant.")
    parts.append("")
    parts.append("## Instructions")
    parts.append(
        "Analyze the following completed task results and extract actionable "
        "lessons, patterns, and anti-patterns. Each entry should be a specific, "
        "reusable insight that would help future tasks succeed."
    )
    parts.append("")
    parts.append("For each entry, provide:")
    parts.append(
        "- title: A short descriptive title (e.g., 'Include type stubs for Pydantic models')"
    )
    parts.append("- content: The actionable lesson (2-4 sentences)")
    parts.append("- tags: Index tags from these categories:")
    parts.append(
        "  - task_type: code-generation, refactoring, test-writing, bug-fix, documentation"
    )
    parts.append("  - domain: python, api, database, cli, testing, validation")
    parts.append("  - pattern: success-pattern, failure-pattern, retry-pattern, context-pattern")
    parts.append(
        "  - error: import-error, type-error, lint-failure, test-failure, validation-error"
    )
    parts.append("")
    parts.append("Focus on:")
    parts.append("- What context was needed for success (or missing for failure)")
    parts.append("- Common validation failures and how to avoid them")
    parts.append("- Patterns that required retries and why")
    parts.append("- File organization patterns that worked well")
    parts.append("")
    parts.append("Do NOT extract:")
    parts.append("- Generic advice ('write clean code')")
    parts.append("- Task-specific details that won't generalize")
    parts.append("- Entries without at least 2 tags")

    parts.append("")
    parts.append("## Completed Task Results")

    for run in run_data:
        parts.append("")
        parts.append(f"### Task: {run['task_id']} (workflow: {run['workflow_id']})")
        parts.append(f"Status: {run['status']}")

        result = run.get("result", {})
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                result = {}

        if result.get("error"):
            parts.append(f"Error: {result['error']}")

        step_results = result.get("step_results", [])
        if step_results:
            parts.append(f"Steps: {len(step_results)}")
            for sr in step_results:
                step_status = sr.get("status", "unknown")
                step_id = sr.get("step_id", "unknown")
                parts.append(f"  - {step_id}: {step_status}")
                if sr.get("error"):
                    parts.append(f"    Error: {sr['error']}")
                for vr in sr.get("validation_results", []):
                    if not vr.get("passed"):
                        parts.append(f"    [{vr['check_name']}] FAIL: {vr.get('summary', '')}")

        for vr in result.get("validation_results", []):
            tag = "PASS" if vr.get("passed") else "FAIL"
            parts.append(f"  [{tag}] {vr.get('check_name', '?')}: {vr.get('summary', '')}")

        output_files = result.get("output_files", {})
        if output_files:
            parts.append(f"Output files: {', '.join(output_files.keys())}")

    return "\n".join(parts)


def build_extraction_user_prompt() -> str:
    """Build the user prompt for knowledge extraction."""
    return (
        "Extract actionable lessons from the completed task results above. "
        "Produce entries that would help future tasks of similar types succeed. "
        "Include the source_task_id and source_workflow_id for each entry."
    )


def infer_tags_from_task(
    task_id: str,
    description: str,
    target_files: list[str],
) -> list[str]:
    """Infer search tags from task metadata.

    Deterministic: extracts tags from file extensions and description keywords.
    Used both during extraction (to tag entries) and during retrieval (to query).
    """
    tags: list[str] = []

    for f in target_files:
        if f.endswith(".py"):
            tags.append("python")
        elif f.endswith(".ts") or f.endswith(".tsx"):
            tags.append("typescript")
        elif f.endswith(".js") or f.endswith(".jsx"):
            tags.append("javascript")

    desc_lower = description.lower()
    keyword_map = {
        "test": "test-writing",
        "refactor": "refactoring",
        "api": "api",
        "database": "database",
        "migration": "migration",
        "cli": "cli",
        "validate": "validation",
        "bug": "bug-fix",
        "fix": "bug-fix",
    }
    for keyword, tag in keyword_map.items():
        if keyword in desc_lower:
            tags.append(tag)

    if not tags:
        tags.append("code-generation")

    return sorted(set(tags))


# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


async def execute_extraction_call(
    input: ExtractionInput,
    agent: Agent[None, ExtractionResult],
) -> ExtractionCallResult:
    """Call the extraction agent and return structured results.

    Separated from the imperative shell so tests can inject a mock agent.
    Pattern: identical to execute_llm_call in activities/llm.py.
    """
    start = time.monotonic()

    result = await agent.run(
        input.user_prompt,
        instructions=input.system_prompt,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    usage = result.usage()

    for entry in result.output.entries:
        if not entry.source_workflow_id:
            entry.source_workflow_id = (
                input.source_workflow_ids[0] if input.source_workflow_ids else ""
            )

    return ExtractionCallResult(
        result=result.output,
        source_workflow_ids=input.source_workflow_ids,
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


def create_extraction_agent(
    model_name: str | None = None,
) -> Agent[None, ExtractionResult]:
    """Create a pydantic-ai Agent configured for knowledge extraction."""
    from pydantic_ai import Agent

    from forge.activities.llm import DEFAULT_MODEL

    if model_name is None:
        model_name = DEFAULT_MODEL

    return Agent(
        model_name,
        output_type=ExtractionResult,
        model_settings={
            "anthropic_cache_instructions": True,
            "anthropic_cache_tool_definitions": True,
        },
    )


@activity.defn
async def fetch_extraction_input(input: FetchExtractionInput) -> ExtractionInput:
    """Read unextracted runs from the store and build the extraction prompt.

    Returns an ExtractionInput with empty source_workflow_ids if no runs found.
    """
    from forge.store import get_db_path, get_engine, get_unextracted_runs

    db_path = get_db_path()
    if db_path is None or not db_path.exists():
        return ExtractionInput(
            system_prompt="",
            user_prompt="",
            source_workflow_ids=[],
        )

    engine = get_engine(db_path)
    runs = get_unextracted_runs(engine, limit=input.limit)

    if not runs:
        return ExtractionInput(
            system_prompt="",
            user_prompt="",
            source_workflow_ids=[],
        )

    for run in runs:
        if "result_json" in run:
            try:
                run["result"] = json.loads(run["result_json"])
            except (json.JSONDecodeError, TypeError):
                run["result"] = {}

    system_prompt = build_extraction_system_prompt(runs)
    user_prompt = build_extraction_user_prompt()
    source_ids = [r["workflow_id"] for r in runs]

    return ExtractionInput(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        source_workflow_ids=source_ids,
    )


def _persist_extraction_interaction(
    input: ExtractionInput,
    result: ExtractionCallResult,
) -> None:
    """Best-effort store write for extraction interactions. Never raises (D42)."""
    try:
        from forge.models import AssembledContext
        from forge.store import build_interaction_dict, get_db_path, get_engine, save_interaction

        db_path = get_db_path()
        if db_path is None:
            return

        context = AssembledContext(
            task_id="__extraction__",
            system_prompt=input.system_prompt,
            user_prompt=input.user_prompt,
        )

        engine = get_engine(db_path)
        data = build_interaction_dict(
            task_id="__extraction__",
            step_id=None,
            sub_task_id=None,
            role="extraction",
            context=context,
            llm_result=result,
        )
        save_interaction(engine, **data)
    except Exception:
        logger.warning("Failed to persist extraction interaction to store", exc_info=True)


@activity.defn
async def call_extraction_llm(input: ExtractionInput) -> ExtractionCallResult:
    """Activity wrapper — creates agent and delegates to execute_extraction_call.

    Pattern: identical to call_llm in activities/llm.py.
    """
    from forge.tracing import get_tracer, llm_call_attributes

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.call_extraction_llm") as span:
        agent = create_extraction_agent(input.model_name or None)
        result = await execute_extraction_call(input, agent)

        span.set_attributes(
            llm_call_attributes(
                model_name=result.model_name,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=result.latency_ms,
                task_id="__extraction__",
                cache_creation_input_tokens=result.cache_creation_input_tokens,
                cache_read_input_tokens=result.cache_read_input_tokens,
            )
        )

        _persist_extraction_interaction(input, result)
        return result


@activity.defn
async def save_extraction_results(input: SaveExtractionInput) -> None:
    """Write extracted playbook entries to the store."""
    from forge.store import build_playbook_dict, get_db_path, get_engine, save_playbooks

    db_path = get_db_path()
    if db_path is None:
        return

    engine = get_engine(db_path)
    dicts = [build_playbook_dict(entry, input.extraction_workflow_id) for entry in input.entries]
    save_playbooks(engine, dicts)
