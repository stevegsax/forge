"""Prompt construction for the consolidation LLM call."""

from __future__ import annotations

import json


def build_consolidation_system_prompt() -> str:
    """System prompt instructing the LLM how to merge similar entries."""
    return (
        "You are a knowledge curation assistant. You will be given a set of "
        "semantically similar playbook entries (lessons/pitfalls). Your task is "
        "to merge them into a single, comprehensive, and clear entry that "
        "captures all unique insights from the source entries without redundancy."
        "\n\n"
        "Rules:\n"
        "- The merged entry must be accurate and actionable.\n"
        "- Avoid generic advice; keep the specific insights from the sources.\n"
        "- Combine tags into a deduplicated list.\n"
        "- Quality over quantity: be concise but thorough."
    )


def build_consolidation_user_prompt(entries: list[dict]) -> str:
    """User prompt containing the cluster of entries to merge."""
    parts = ["## Source Entries to Merge\n"]
    for e in entries:
        tags = json.loads(e.get("tags_json", "[]"))
        parts.append(
            f"### {e['title']}\n"
            f"**Content:** {e['content']}\n"
            f"**Tags:** {', '.join(tags)}\n",
        )
    return "".join(parts)
