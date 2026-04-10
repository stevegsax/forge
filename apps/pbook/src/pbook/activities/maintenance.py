"""Maintenance activities for the playbook service.

Design follows Function Core / Imperative Shell:

- Pure functions: identify_prune_candidates, group_similar_entries
- Temporal activities: fetch_all_entries_for_maintenance, prune_entries,
  consolidate_entries_llm
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from temporalio import activity

logger = logging.getLogger(__name__)

from pbook.llm import ConsolidationResult, text_messages
from pbook.models import CapabilityTier, ModelConfig, resolve_model

if TYPE_CHECKING:
    from pbook.llm import LLMProvider


def _resolve_default_model() -> tuple[str, str]:
    """Resolve the default model for consolidation."""
    from sax_llm.registry import parse_model_id

    model_id = resolve_model(CapabilityTier.REASONING, ModelConfig())
    return parse_model_id(model_id)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def identify_prune_candidates(
    entries: list[dict],
    *,
    now: datetime | None = None,
    min_retrievals: int = 5,
    max_harmful_ratio: float = 0.5,
    max_stale_days: int = 180,
) -> list[dict]:
    """Identify entries that should be flagged for review.

    An entry is a prune candidate if:

    1. It has been retrieved >= ``min_retrievals`` times AND its harmful
       ratio (harmful_count / retrieval_count) exceeds ``max_harmful_ratio``.
    2. It has never been retrieved AND is older than ``max_stale_days``.

    Returns a copy of each matching entry with a ``prune_reason`` key added.
    Entries that do not match either criterion are excluded.
    """
    if now is None:
        now = datetime.now(UTC)

    candidates: list[dict] = []

    for entry in entries:
        retrieval_count = entry.get("retrieval_count", 0)
        harmful_count = entry.get("harmful_count", 0)

        # Rule 1: consistently harmful
        if retrieval_count >= min_retrievals:
            ratio = harmful_count / retrieval_count
            if ratio > max_harmful_ratio:
                candidates.append({
                    **entry,
                    "prune_reason": f"harmful ratio {ratio:.0%} exceeds {max_harmful_ratio:.0%} "
                                    f"({harmful_count}/{retrieval_count} retrievals)",
                })
                continue

        # Rule 2: never retrieved and stale
        if retrieval_count == 0:
            created_at = entry.get("created_at")
            if created_at is not None:
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=UTC)
                age_days = (now - created_at).days
                if age_days > max_stale_days:
                    candidates.append({
                        **entry,
                        "prune_reason": f"never retrieved and {age_days} days old "
                                        f"(threshold: {max_stale_days} days)",
                    })

    return candidates


def group_similar_entries(
    entries: list[dict],
    *,
    threshold: float = 0.85,
) -> list[list[dict]]:
    """Group entries into clusters based on semantic similarity.

    Returns a list of clusters, where each cluster is a list of entries.
    Only clusters with more than one entry are returned.
    """
    from pbook.embeddings import cosine_similarity

    if not entries:
        return []

    # Filter out entries without embeddings
    candidates = [e for e in entries if e.get("embedding")]
    if not candidates:
        return []

    clusters: list[list[dict]] = []
    processed_ids = set()

    for i, entry in enumerate(candidates):
        if entry["id"] in processed_ids:
            continue

        cluster = [entry]
        processed_ids.add(entry["id"])

        for other in candidates[i + 1 :]:
            if other["id"] in processed_ids:
                continue

            sim = cosine_similarity(entry["embedding"], other["embedding"])
            if sim >= threshold:
                cluster.append(other)
                processed_ids.add(other["id"])

        if len(cluster) > 1:
            clusters.append(cluster)

    return clusters


# ---------------------------------------------------------------------------
# Temporal activities
# ---------------------------------------------------------------------------


@activity.defn
async def fetch_all_entries_for_maintenance() -> list[dict]:
    """Fetch all entries from the store for maintenance analysis."""
    from pbook.store import get_db_path, get_engine, list_all_entries, run_migrations

    db_path = get_db_path()
    if db_path is None or not db_path.exists():
        return []

    run_migrations(db_path)
    engine = get_engine(db_path)
    return list_all_entries(engine)


@activity.defn
async def prune_entries(entry_ids: list[int]) -> int:
    """Delete the given entries from the store."""
    from pbook.store import delete_entry, get_db_path, get_engine, run_migrations

    if not entry_ids:
        return 0

    db_path = get_db_path()
    if db_path is None:
        return 0

    run_migrations(db_path)
    engine = get_engine(db_path)

    for entry_id in entry_ids:
        delete_entry(engine, entry_id)

    logger.info("Pruned %d entries", len(entry_ids))
    return len(entry_ids)


@activity.defn
async def consolidate_entries_llm(entries_json: str) -> str:
    """Merge semantically similar entries using an LLM.

    Accepts JSON-serialized list of entries.
    Returns JSON-serialized ConsolidationResult.
    """
    from pbook.llm import get_provider

    entries = json.loads(entries_json)
    if not entries:
        return ConsolidationResult(merged_title="", merged_content="").model_dump_json()

    logger.info("Consolidating %d entries via LLM", len(entries))
    provider = get_provider()
    _, model = _resolve_default_model()

    system_prompt = (
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

    user_parts = ["## Source Entries to Merge\n"]
    for e in entries:
        tags = json.loads(e.get("tags_json", "[]"))
        user_parts.append(f"### {e['title']}\n**Content:** {e['content']}\n**Tags:** {', '.join(tags)}\n")

    messages = text_messages(system_prompt, "".join(user_parts))

    params = provider.build_request_params(
        messages=messages,
        output_type=ConsolidationResult,
        model=model,
        max_tokens=2048,
    )
    response = await provider.call(params)
    result = ConsolidationResult.model_validate(response.tool_input)

    return result.model_dump_json()
