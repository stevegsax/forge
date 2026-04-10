"""Maintenance activities for the playbook service.

Design follows Function Core / Imperative Shell:

- Pure functions: identify_prune_candidates
"""

from __future__ import annotations

from datetime import UTC, datetime


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
