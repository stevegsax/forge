"""Review activities for the playbook service.

LLM-based review of manually submitted playbook entries.
Checks clarity, correctness, completeness, and duplication.

Design follows Function Core / Imperative Shell:

- Pure functions: build_review_system_prompt, build_review_user_prompt,
  apply_suggestions
- Testable function: execute_review_call (takes provider as argument)
- Temporal activities: validate_entry, fetch_existing_entries,
  review_entry
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from temporalio import activity

from pbook.llm import ReviewResult, build_messages
from pbook.models import PlaybookEntry

if TYPE_CHECKING:
    from pbook.llm import LLMProvider


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def build_review_system_prompt(existing_entries: list[dict]) -> str:
    """Build the system prompt for reviewing a proposed playbook entry.

    Instructs the LLM to check clarity, correctness, completeness, and
    duplication against existing entries.  Encodes the quality bar:
    better to reject than to accept a misleading or over-prescriptive entry.
    """
    lines = [
        "You are a playbook entry reviewer for a knowledge base.",
        "Your job is to evaluate a proposed entry and decide whether it should be stored.",
        "",
        "## Quality Bar",
        "",
        "It is BETTER TO REJECT than to accept an entry that is:",
        "- Misleading or inaccurate",
        "- Over-prescriptive (constrains future decisions unnecessarily)",
        "- Too generic ('always use X' without explaining when and why)",
        "- Vague or not actionable",
        "",
        "Good entries are MINIMAL and ACCURATE:",
        "- State specific, concrete advice",
        "- Include when the advice applies (not just what to do)",
        "- Avoid over-constraining — leave room for judgment",
        "",
        "## Evaluation Criteria",
        "",
        "1. **Accuracy** — Is the advice technically correct?",
        "2. **Specificity** — Is it specific enough to be actionable?",
        "3. **Minimality** — Is it concise without unnecessary prescriptions?",
        "4. **Duplication** — Is this substantially covered by an existing entry?",
        "",
        "If the entry is acceptable, set approved=true.",
        "If not, set approved=false and explain why.",
        "You may suggest improvements to title, content, or tags.",
    ]

    if existing_entries:
        lines.append("")
        lines.append("## Existing entries (check for duplication)")
        lines.append("")
        for entry in existing_entries:
            tags = entry.get("tags_json", "[]")
            if isinstance(tags, str):
                tags = json.loads(tags)
            lines.append(f"- **{entry['title']}** (tags: {', '.join(tags)})")

    return "\n".join(lines)


def build_review_user_prompt(entry: PlaybookEntry) -> str:
    """Format the proposed entry as the user message for review."""
    lines = [
        "## Proposed entry",
        "",
        f"**Title:** {entry.title}",
        "",
        f"**Content:** {entry.content}",
        "",
        f"**Tags:** {', '.join(entry.tags)}",
    ]
    if entry.source_project:
        lines.append(f"**Project:** {entry.source_project}")
    return "\n".join(lines)


def apply_suggestions(entry: PlaybookEntry, review: ReviewResult) -> PlaybookEntry:
    """Return a new entry with suggested improvements applied.

    Uses suggested values where non-empty; keeps originals otherwise.
    Tags are merged (union of original and suggested).
    """
    merged_tags = list(dict.fromkeys(entry.tags + review.suggested_tags))

    return entry.model_copy(
        update={
            "title": review.suggested_title if review.suggested_title else entry.title,
            "content": review.suggested_content if review.suggested_content else entry.content,
            "tags": merged_tags if review.suggested_tags else entry.tags,
        }
    )


# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


async def execute_review_call(
    system_prompt: str,
    user_prompt: str,
    provider: LLMProvider,
    model: str = "",
) -> ReviewResult:
    """Call the LLM provider for review and return structured results.

    Separated from the imperative shell so tests can inject a mock provider.
    """
    messages = build_messages(system_prompt, user_prompt)

    params = provider.build_request_params(
        messages=messages,
        output_type=ReviewResult,
        model=model,
        max_tokens=1024,
    )
    response = await provider.call(params)

    return ReviewResult.model_validate(response.tool_input)


# ---------------------------------------------------------------------------
# Temporal activities
# ---------------------------------------------------------------------------


@activity.defn
async def validate_entry(raw_json: str) -> str:
    """Parse and validate raw JSON against the PlaybookEntry schema.

    Returns JSON with keys: valid (bool), entry (dict|null), error (str|null).
    """
    from pydantic import ValidationError

    try:
        entry = PlaybookEntry.model_validate_json(raw_json)
        return json.dumps({
            "valid": True,
            "entry": entry.model_dump(),
            "error": None,
        })
    except (ValidationError, ValueError) as exc:
        return json.dumps({
            "valid": False,
            "entry": None,
            "error": str(exc),
        })


@activity.defn
async def fetch_existing_entries(limit: int = 50) -> list[dict]:
    """Query recent entries for duplication context."""
    from pbook.store import get_db_path, get_engine, list_recent_entries, run_migrations

    db_path = get_db_path()
    if db_path is None or not db_path.exists():
        return []

    run_migrations(db_path)
    engine = get_engine(db_path)
    return list_recent_entries(engine, limit=limit)


@activity.defn
async def review_entry(input_json: str) -> str:
    """Review a proposed entry via LLM and apply suggestions.

    Accepts JSON with keys: entry (dict), existing_entries (list[dict]),
    model_name (str, optional).
    Returns JSON with keys: approved (bool), rejection_reason (str),
    final_entry (dict).
    """
    from pbook.llm import get_provider

    data = json.loads(input_json)
    entry = PlaybookEntry.model_validate(data["entry"])
    existing = data.get("existing_entries", [])
    model_name = data.get("model_name", "")

    provider = get_provider()
    system_prompt = build_review_system_prompt(existing)
    user_prompt = build_review_user_prompt(entry)

    review = await execute_review_call(
        system_prompt, user_prompt, provider, model=model_name,
    )

    if not review.approved:
        return json.dumps({
            "approved": False,
            "rejection_reason": review.rejection_reason,
            "final_entry": entry.model_dump(),
        })

    final_entry = apply_suggestions(entry, review)
    return json.dumps({
        "approved": True,
        "rejection_reason": "",
        "final_entry": final_entry.model_dump(),
    })
