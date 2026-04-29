"""Maintenance activities for the playbook service.

Design follows Function Core / Imperative Shell:

- Pure functions: identify_prune_candidates, group_similar_entries
- Temporal activities: fetch_all_entries_for_maintenance, prune_entries,
  save_consolidated_entry

The LLM consolidation call now goes through the generic ``llm_chat``
step in :mod:`pbook.workflow_steps.llm`; this module owns only the
database-side maintenance work.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from temporalio import activity

logger = logging.getLogger(__name__)


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
async def save_consolidated_entry(input_json: str) -> int:
    """Save a consolidated entry and re-parent the cluster's source rows.

    Accepts JSON with keys:
      - merged_entry: dict with title, content, tags, embedding (base64).
      - cluster_ids: list[int] of the entry ids that produced the merge.

    Inserts the merged entry directly (NO match-or-attach — the merged
    entry is meant to replace the cluster, not match against it),
    re-parents every `entry_sources` row from `cluster_ids` to the new
    entry, and returns the new entry's id. The maintenance workflow is
    responsible for pruning the original cluster ids after this call.
    """
    import base64

    from pbook.models import EntryType, PlaybookEntry
    from pbook.store import (
        build_entry_dict,
        get_db_path,
        get_engine,
        reparent_entry_sources,
        run_migrations,
        save_entry_returning_id,
    )

    data = json.loads(input_json)
    merged = data["merged_entry"]
    cluster_ids = list(data.get("cluster_ids", []))

    db_path = get_db_path()
    if db_path is None:
        return 0

    run_migrations(db_path)
    engine = get_engine(db_path)

    raw_embedding = merged.get("embedding") or ""
    embedding = base64.b64decode(raw_embedding) if raw_embedding else None

    entry = PlaybookEntry(
        title=merged["title"],
        content=merged["content"],
        tags=merged.get("tags", []),
        entry_type=EntryType.CURATED,
        needs_review=False,
        embedding=embedding,
    )
    new_id = save_entry_returning_id(engine, build_entry_dict(entry))

    if cluster_ids:
        reparent_entry_sources(
            engine,
            from_entry_ids=cluster_ids,
            to_entry_id=new_id,
        )

    return new_id


