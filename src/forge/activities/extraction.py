"""Knowledge extraction activities for Forge.

Extracts structured lessons from completed task results via an LLM call.

Design follows Function Core / Imperative Shell:
- Pure functions: build_extraction_system_prompt, build_extraction_user_prompt,
  infer_tags_from_task
- Testable function: execute_extraction_call (takes the LLM client as an argument)
- Imperative shells: the ``call_extraction_llm`` (LlmActivities),
  ``fetch_extraction_input`` and ``save_extraction_results`` (StoreActivities)
  bound methods on the composition-root classes (forge.activities.roots)
  delegate to these pure functions and the store helpers.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from forge.models import (
    ExtractionCallResult,
    ExtractionInput,
    ExtractionResult,
)

if TYPE_CHECKING:
    from sax_platform.llm import AnthropicLLM

logger = logging.getLogger(__name__)

DEFAULT_EXTRACTION_MAX_TOKENS = 4096


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def build_extraction_system_prompt(run_data: list[dict[str, Any]]) -> str:
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
    llm: AnthropicLLM,
) -> ExtractionCallResult:
    """Call the LLM for extraction and return structured results.

    Separated from the imperative shell so tests can inject a stub client.
    """
    from sax_platform.llm.tiers import split_provider

    from forge.activities.llm import DEFAULT_MODEL

    full_model = input.model_name or DEFAULT_MODEL
    _, model = split_provider(full_model)
    start = time.monotonic()

    completion = await llm.complete(
        [{"role": "user", "content": input.user_prompt}],
        output_type=ExtractionResult,
        model=model,
        max_tokens=DEFAULT_EXTRACTION_MAX_TOKENS,
        system=input.system_prompt,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    extraction_result = completion.output

    for entry in extraction_result.entries:
        if not entry.source_workflow_id:
            entry.source_workflow_id = (
                input.source_workflow_ids[0] if input.source_workflow_ids else ""
            )

    return ExtractionCallResult(
        result=extraction_result,
        source_workflow_ids=input.source_workflow_ids,
        model_name=completion.model,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        latency_ms=elapsed_ms,
        cache_creation_input_tokens=completion.cache_creation_input_tokens,
        cache_read_input_tokens=completion.cache_read_input_tokens,
        stop_reason=completion.stop_reason,
    )
