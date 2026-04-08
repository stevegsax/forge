"""Extraction activities for the playbook service.

Receives pushed experience data, calls the LLM to extract lessons
targeting unexpected + actionable situations, and saves entries
with needs-review=True.

Design follows Function Core / Imperative Shell:

- Pure functions: build_extraction_system_prompt, build_extraction_user_prompt
- Testable function: execute_extraction_call (takes provider as argument)
- Temporal activities: extract_from_experience, save_extracted_entries
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from temporalio import activity

from pbook.llm import ExtractionResult, build_messages
from pbook.models import PushExperienceInput

if TYPE_CHECKING:
    from pbook.llm import LLMProvider


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def build_extraction_system_prompt(experiences: list[PushExperienceInput]) -> str:
    """Build the system prompt for knowledge extraction from pushed experience.

    The prompt instructs the LLM to find situations that are both:
    1. Unexpected — the obvious or default approach did not work
    2. Actionable — there is specific advice that would help next time

    Generic advice, standard rules, and expected outcomes are excluded.
    """
    parts: list[str] = []

    parts.append("You are a knowledge extraction assistant.")
    parts.append("")
    parts.append("## Instructions")
    parts.append("")
    parts.append(
        "Analyze the following experience reports and extract lessons that are both "
        "UNEXPECTED and ACTIONABLE."
    )
    parts.append("")
    parts.append("An entry is worth extracting ONLY if:")
    parts.append(
        "- The default or obvious approach did NOT work — "
        "the LLM's first instinct would have been wrong"
    )
    parts.append(
        "- There is specific, concrete advice that would help "
        "someone encountering this situation for the first time"
    )
    parts.append("")
    parts.append("Signals that something is worth extracting:")
    parts.append("- Multiple attempts were needed before finding the right approach")
    parts.append("- An API behaved differently than its documentation suggests")
    parts.append("- A standard pattern fails in a specific context")
    parts.append("- A workaround was needed for a library or framework quirk")
    parts.append("")
    parts.append("Do NOT extract:")
    parts.append("- Generic advice ('use proper error handling', 'write tests')")
    parts.append("- Standard rules that any experienced developer knows")
    parts.append("- Entries about expected or normal behavior")
    parts.append("- Vague or over-prescriptive advice that constrains future decisions")
    parts.append("- Entries without specific, actionable guidance")
    parts.append("")
    parts.append(
        "It is better to extract NOTHING than to extract a misleading "
        "or overly generic entry. Quality over quantity."
    )
    parts.append("")
    parts.append("For each entry, provide:")
    parts.append("- title: A short, specific descriptive title")
    parts.append("- content: The actionable lesson (2-4 sentences, minimal)")
    parts.append("- tags: Relevant tags (e.g., python, sqlalchemy, testing)")
    parts.append("")
    parts.append("## Experience Reports")

    for exp in experiences:
        parts.append("")
        parts.append(f"### Project: {exp.project}")
        parts.append(f"**Problem:** {exp.problem}")
        parts.append(f"**Resolution:** {exp.resolution}")
        if exp.context:
            parts.append(f"**Context:** {exp.context}")
        if exp.metadata:
            parts.append(f"**Metadata:** {json.dumps(exp.metadata)}")

    return "\n".join(parts)


def build_extraction_user_prompt() -> str:
    """Build the user prompt for knowledge extraction."""
    return (
        "Extract only the unexpected and actionable lessons from the experience "
        "reports above. If nothing meets the quality bar, return an empty list. "
        "Remember: it is better to extract nothing than to extract something "
        "misleading or generic."
    )


# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


async def execute_extraction_call(
    system_prompt: str,
    user_prompt: str,
    provider: LLMProvider,
    model: str = "",
) -> tuple[ExtractionResult, int, int, float]:
    """Call the LLM provider for extraction and return structured results.

    Returns ``(result, input_tokens, output_tokens, latency_ms)``.
    Separated from the imperative shell so tests can inject a mock provider.
    """
    messages = build_messages(system_prompt, user_prompt)
    start = time.monotonic()

    params = provider.build_request_params(
        messages=messages,
        output_type=ExtractionResult,
        model=model,
        max_tokens=4096,
    )
    response = await provider.call(params)
    latency_ms = (time.monotonic() - start) * 1000

    result = ExtractionResult.model_validate(response.tool_input)
    return result, response.input_tokens, response.output_tokens, latency_ms


# ---------------------------------------------------------------------------
# Temporal activities
# ---------------------------------------------------------------------------


@activity.defn
async def extract_from_experience(input_json: str) -> str:
    """Extract lessons from pushed experience data.

    Accepts JSON-serialized list of PushExperienceInput dicts.
    Returns JSON-serialized ExtractionResult.
    """
    from pbook.llm import get_provider

    experiences = [
        PushExperienceInput.model_validate(exp)
        for exp in json.loads(input_json)
    ]

    if not experiences:
        return ExtractionResult().model_dump_json()

    provider = get_provider()
    system_prompt = build_extraction_system_prompt(experiences)
    user_prompt = build_extraction_user_prompt()

    result, _in_tok, _out_tok, _latency = await execute_extraction_call(
        system_prompt, user_prompt, provider,
    )

    return result.model_dump_json()


@activity.defn
async def save_extracted_entries(input_json: str) -> int:
    """Save extracted entries to the store with needs_review=True.

    Accepts JSON with keys: entries (list), project (str).
    Returns the number of entries saved.
    """
    from pbook.models import EntryType, PlaybookEntry
    from pbook.store import build_entry_dict, get_db_path, get_engine, run_migrations, save_entries

    data = json.loads(input_json)
    entries_raw = data.get("entries", [])
    project = data.get("project", "")

    if not entries_raw:
        return 0

    db_path = get_db_path()
    if db_path is None:
        return 0

    run_migrations(db_path)
    engine = get_engine(db_path)

    entry_dicts = []
    for raw in entries_raw:
        entry = PlaybookEntry(
            title=raw["title"],
            content=raw["content"],
            tags=raw.get("tags", []),
            entry_type=EntryType.PITFALL,
            source_project=project,
            needs_review=True,
        )
        entry_dicts.append(build_entry_dict(entry))

    save_entries(engine, entry_dicts)
    return len(entry_dicts)
