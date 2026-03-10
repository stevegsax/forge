"""LLM-based review of manually submitted playbook entries.

Follows Function Core / Imperative Shell:
- Pure functions: build_review_system_prompt, build_review_user_prompt, apply_suggestions
- Async shell: review_playbook_entry
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.models import PlaybookEntry, PlaybookReviewResult


# ---------------------------------------------------------------------------
# Function core
# ---------------------------------------------------------------------------


def build_review_system_prompt(existing_playbooks: list[dict]) -> str:
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
    existing_playbooks: list[dict],
) -> PlaybookReviewResult:
    """Send a proposed playbook entry to the LLM for review.

    Uses the CLASSIFICATION tier model (Haiku) since this is a
    straightforward accept/reject with light suggestions.
    """
    from forge.llm_providers.models import text_messages
    from forge.llm_providers.registry import get_provider, parse_model_id
    from forge.models import CapabilityTier, ModelConfig, PlaybookReviewResult, resolve_model

    model_id = resolve_model(CapabilityTier.CLASSIFICATION, ModelConfig())
    _, model_name = parse_model_id(model_id)
    provider = get_provider(model_id)

    system_prompt = build_review_system_prompt(existing_playbooks)
    user_prompt = build_review_user_prompt(entry)
    messages = text_messages(system_prompt, user_prompt)

    params = provider.build_request_params(
        messages=messages,
        output_type=PlaybookReviewResult,
        model=model_name,
        max_tokens=1024,
    )
    response = await provider.call(params)

    return PlaybookReviewResult.model_validate(response.tool_input)
