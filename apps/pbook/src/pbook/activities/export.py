"""Export activities for the playbook service.

Design follows Function Core / Imperative Shell:

- Pure function: db_row_to_entry_dict
- Temporal activities: fetch_entry_ids, export_single_entry
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pbook.models import PlaybookEntry

if TYPE_CHECKING:
    from sqlalchemy import Engine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def db_row_to_entry_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a DB row dict to a PlaybookEntry-compatible dict.

    Reads the ``tags`` list attached by the store read helpers and drops
    DB-only fields (id, created_at, updated_at).
    """
    return PlaybookEntry(
        title=row["title"],
        content=row["content"],
        tags=row.get("tags", []),
        entry_type=row.get("entry_type", "curated"),
        source_project=row.get("source_project", ""),
        source_task_id=row.get("source_task_id", ""),
        needs_review=row.get("needs_review", False),
        helpful_count=row.get("helpful_count", 0),
        harmful_count=row.get("harmful_count", 0),
        retrieval_count=row.get("retrieval_count", 0),
    ).model_dump()


# ---------------------------------------------------------------------------
# Temporal activities
# ---------------------------------------------------------------------------


async def fetch_entry_ids(engine: Engine | None, input_json: str) -> list[int]:
    """Query store for matching entry IDs."""
    import json as json_mod

    from pbook.store import get_entries_by_tags

    params = json_mod.loads(input_json)
    tags = params.get("tags", [])
    limit = params.get("limit", 50)

    if engine is None:
        return []

    entries = get_entries_by_tags(engine, tags, limit=limit)
    logger.info("Fetched %d entry IDs for export (tags=%s)", len(entries), tags)
    return [e["id"] for e in entries]


async def export_single_entry(engine: Engine | None, entry_id: int) -> dict[str, Any]:
    """Fetch one entry by ID and convert to PlaybookEntry dict."""
    from pbook.store import get_entry_by_id

    if engine is None:
        msg = "No store available"
        raise RuntimeError(msg)

    row = get_entry_by_id(engine, entry_id)
    if row is None:
        msg = f"Entry {entry_id} not found"
        raise RuntimeError(msg)

    logger.debug("Exported entry %d: %s", entry_id, row["title"])
    return db_row_to_entry_dict(row)
