"""LLM-based review of manually submitted playbook entries.

Follows Function Core / Imperative Shell:
- Pure functions: build_review_system_prompt, build_review_user_prompt, apply_suggestions
- Async shell: review_playbook_entry
- Temporal activities: validate_playbook_entry, fetch_existing_playbooks, review_manual_playbook
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from temporalio import activity

from forge.llm_client import get_llm
from forge.models import (  # noqa: TC001 — Temporal needs these at runtime for activity deserialization
    FetchExistingPlaybooksInput,
    ReviewManualPlaybookInput,
    ReviewManualPlaybookResult,
    ValidatePlaybookInput,
    ValidatePlaybookResult,
)

if TYPE_CHECKING:
    from sax_platform.llm import AnthropicLLM

    from forge.models import PlaybookEntry, PlaybookReviewResult


# ---------------------------------------------------------------------------
# Function core
# ---------------------------------------------------------------------------


def build_review_system_prompt(existing_playbooks: list[dict[str, Any]]) -> str:
    """Build the system prompt for reviewing a proposed playbook entry.

    Instructs the LLM to check clarity, correctness, completeness, and
    duplication against the provided existing entries.
    """
    lines = [
        "You are a playbook entry reviewer for a software engineering knowledge base.",
        "Your job is to evaluate a proposed playbook entry and decide whether it should be stored.",
        "",
        "Evaluate the entry on these criteria:",
        "1. **Clarity** — Is the content clear and actionable?",
        "2. **Correctness** — Does the advice appear technically sound?",
        "3. **Completeness** — Is enough context provided to be useful?",
        "4. **Duplication** — Is this substantially duplicated by an existing entry?",
        "",
        "If the entry is acceptable, set approved=true.",
        "If not, set approved=false and explain why.",
        "You may suggest improvements to the title, content, or tags",
        "even if you approve the entry.",
        "Leave suggested_title, suggested_content, and suggested_tags",
        "empty if no changes are needed.",
    ]

    if existing_playbooks:
        lines.append("")
        lines.append("## Existing playbook entries (check for duplication)")
        lines.append("")
        for entry in existing_playbooks:
            tags = entry.get("tags_json", "[]")
            if isinstance(tags, str):
                tags = json.loads(tags)
            lines.append(f"- **{entry['title']}** (tags: {', '.join(tags)})")

    return "\n".join(lines)


def build_review_user_prompt(entry: PlaybookEntry) -> str:
    """Format the proposed entry as the user message for review."""
    lines = [
        "## Proposed playbook entry",
        "",
        f"**Title:** {entry.title}",
        "",
        f"**Content:** {entry.content}",
        "",
        f"**Tags:** {', '.join(entry.tags)}",
        "",
        f"**Source task:** {entry.source_task_id}",
    ]
    return "\n".join(lines)


def apply_suggestions(entry: PlaybookEntry, review: PlaybookReviewResult) -> PlaybookEntry:
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
# Imperative shell
# ---------------------------------------------------------------------------


async def review_playbook_entry(
    entry: PlaybookEntry,
    existing_playbooks: list[dict[str, Any]],
    llm: AnthropicLLM,
    model_name: str = "",
) -> PlaybookReviewResult:
    """Send a proposed playbook entry to the LLM for review.

    Uses the CLASSIFICATION tier model (Haiku) since this is a
    straightforward accept/reject with light suggestions.
    """
    from sax_platform.llm.tiers import split_provider

    from forge.models import CapabilityTier, ModelConfig, PlaybookReviewResult, resolve_model

    if not model_name:
        model_name = resolve_model(CapabilityTier.CLASSIFICATION, ModelConfig())
    _, model = split_provider(model_name)

    system_prompt = build_review_system_prompt(existing_playbooks)
    user_prompt = build_review_user_prompt(entry)

    completion = await llm.complete(
        [{"role": "user", "content": user_prompt}],
        output_type=PlaybookReviewResult,
        model=model,
        max_tokens=1024,
        system=system_prompt,
    )

    return completion.output


# ---------------------------------------------------------------------------
# Temporal activities
# ---------------------------------------------------------------------------


@activity.defn
async def validate_playbook_entry(input: ValidatePlaybookInput) -> ValidatePlaybookResult:
    """Parse and validate raw JSON against the PlaybookEntry schema."""
    from pydantic import ValidationError

    from forge.models import PlaybookEntry, ValidatePlaybookResult

    try:
        entry = PlaybookEntry.model_validate_json(input.raw_json)
        return ValidatePlaybookResult(valid=True, entry=entry)
    except (ValidationError, ValueError) as exc:
        return ValidatePlaybookResult(valid=False, error=str(exc))


@activity.defn
async def fetch_existing_playbooks(input: FetchExistingPlaybooksInput) -> list[dict[str, Any]]:
    """Query recent playbooks for duplication context."""
    from forge.store import get_store_engine, list_recent_playbooks

    engine = get_store_engine()
    return list_recent_playbooks(engine, limit=input.limit)


@activity.defn
async def review_manual_playbook(input: ReviewManualPlaybookInput) -> ReviewManualPlaybookResult:
    """Review a proposed playbook entry via LLM and apply suggestions."""
    from forge.models import ReviewManualPlaybookResult

    review = await review_playbook_entry(
        input.entry, input.existing_playbooks, get_llm(), model_name=input.model_name
    )
    if not review.approved:
        return ReviewManualPlaybookResult(
            approved=False,
            rejection_reason=review.rejection_reason,
            final_entry=input.entry,
        )
    final_entry = apply_suggestions(input.entry, review)
    return ReviewManualPlaybookResult(approved=True, final_entry=final_entry)
