"""Prompt construction and post-LLM helpers for the review path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pbook.llm import ReviewResult
    from pbook.models import PlaybookEntry


def build_review_system_prompt(existing_entries: list[dict[str, Any]]) -> str:
    """Build the system prompt for reviewing a proposed playbook entry.

    Instructs the LLM to check clarity, correctness, completeness, and
    duplication against existing entries. Encodes the quality bar:
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
            tags = entry.get("tags", [])
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


def apply_suggestions(
    entry: PlaybookEntry,
    review: ReviewResult,
) -> PlaybookEntry:
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
