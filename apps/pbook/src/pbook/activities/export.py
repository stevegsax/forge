"""Export activities for the playbook service.

Design follows Function Core / Imperative Shell:

- Pure function: db_row_to_entry_dict
- Temporal activities: fetch_entry_ids, export_single_entry
"""

from __future__ import annotations

import json
import logging

from temporalio import activity

logger = logging.getLogger(__name__)

from pbook.models import PlaybookEntry

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def db_row_to_entry_dict(row: dict) -> dict:
    """Convert a DB row dict to a PlaybookEntry-compatible dict.

    Maps ``tags_json`` (JSON string) to ``tags`` (list[str]) and drops
    DB-only fields (id, created_at, updated_at).
    """
    tags_raw = row.get("tags_json", "[]")
    tags = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw

    return PlaybookEntry(
        title=row["title"],
        content=row["content"],
        tags=tags,
        entry_type=row.get("entry_type", "curated"),
        source_project=row.get("source_project", ""),
        source_task_id=row.get("source_task_id", ""),
        needs_review=row.get("needs_review", False),
    ).model_dump()


# ---------------------------------------------------------------------------
# Temporal activities
# ---------------------------------------------------------------------------


@activity.defn
async def fetch_entry_ids(input_json: str) -> list[int]:
    """Query store for matching entry IDs."""
    import json as json_mod

    from pbook.store import get_db_path, get_engine, run_migrations

    params = json_mod.loads(input_json)
    tags = params.get("tags", [])
    limit = params.get("limit", 50)

    db_path = get_db_path()
    if db_path is None or not db_path.exists():
        return []

    run_migrations(db_path)
    engine = get_engine(db_path)

    from pbook.store import get_entries_by_tags

    entries = get_entries_by_tags(engine, tags, limit=limit)
    logger.info("Fetched %d entry IDs for export (tags=%s)", len(entries), tags)
    return [e["id"] for e in entries]


@activity.defn
async def export_single_entry(entry_id: int) -> dict:
    """Fetch one entry by ID and convert to PlaybookEntry dict."""
    from pbook.store import get_db_path, get_engine, get_entry_by_id, run_migrations

    db_path = get_db_path()
    if db_path is None or not db_path.exists():
        msg = "No store available"
        raise RuntimeError(msg)

    run_migrations(db_path)
    engine = get_engine(db_path)
    row = get_entry_by_id(engine, entry_id)
    if row is None:
        msg = f"Entry {entry_id} not found"
        raise RuntimeError(msg)

    logger.debug("Exported entry %d: %s", entry_id, row["title"])
    return db_row_to_entry_dict(row)
