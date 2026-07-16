"""Prompt construction for the experience-extraction LLM call."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pbook.models import PushExperienceInput


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
    parts.append("## Verify each candidate against the Resolution")
    parts.append("")
    parts.append(
        "Each candidate lesson must be supported by the **Resolution** text, "
        "not just the Problem or symptom narrative. Symptoms and dismissed "
        "hypotheses (red herrings) often appear in the Problem text but are "
        "explicitly debunked by the Resolution."
    )
    parts.append("")
    parts.append(
        "If the Resolution explicitly says something was NOT the cause "
        "('root cause was not X', 'turned out X wasn't the issue', "
        "'X was a red herring'), do NOT extract a lesson naming X as a pitfall."
    )
    parts.append("")
    parts.append("## One root cause, one entry")
    parts.append("")
    parts.append(
        "Most experiences yield 0 or 1 entries. Multiple entries from one "
        "experience are rare and should be reserved for cases where the "
        "Resolution genuinely teaches independent lessons with distinct "
        "root causes."
    )
    parts.append("")
    parts.append(
        "If you find yourself producing multiple entries from a single "
        "experience, ask whether they are facets of the same root cause. "
        "If so, pick the sharpest framing and drop the others — multiple "
        "near-duplicate framings of one lesson degrade the playbook."
    )
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
