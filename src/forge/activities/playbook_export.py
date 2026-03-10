"""Activities for playbook export workflow.

Follows Function Core / Imperative Shell:
- Pure function: db_row_to_playbook_entry
- Temporal activities: fetch_playbook_ids, export_single_playbook
"""

from __future__ import annotations

import json

from temporalio import activity

from forge.models import (  # noqa: TC001 — Temporal needs these at runtime for activity deserialization
    ExportSinglePlaybookInput,
    FetchPlaybookIdsInput,
    PlaybookEntry,
)

# ---------------------------------------------------------------------------
# Function core
# ---------------------------------------------------------------------------


def db_row_to_playbook_entry(row: dict) -> PlaybookEntry:
    """Convert a DB row dict to a PlaybookEntry.

    Maps tags_json (JSON string) to tags (list[str]) and drops DB-only
    fields (id, created_at, extraction_workflow_id).
    """
    from forge.models import PlaybookEntry

    tags_raw = row.get("tags_json", "[]")
    tags = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw

    return PlaybookEntry(
        title=row["title"],
        content=row["content"],
        tags=tags,
        source_task_id=row.get("source_task_id", ""),
        source_workflow_id=row.get("source_workflow_id", ""),
    )


# ---------------------------------------------------------------------------
# Temporal activities
# ---------------------------------------------------------------------------


@activity.defn
async def fetch_playbook_ids(input: FetchPlaybookIdsInput) -> list[int]:
    """Query store for matching playbook IDs."""
    from forge.store import get_db_path, get_engine, get_playbook_ids

    db_path = get_db_path()
    if db_path is None or not db_path.exists():
        return []

    engine = get_engine(db_path)
    return get_playbook_ids(
        engine,
        tags=input.tags if input.tags else None,
        source_task_id=input.source_task_id,
        limit=input.limit,
    )


@activity.defn
async def export_single_playbook(input: ExportSinglePlaybookInput) -> PlaybookEntry:
    """Fetch one playbook row by ID and convert to PlaybookEntry."""
    from forge.store import get_db_path, get_engine, get_playbook_by_id

    db_path = get_db_path()
    if db_path is None or not db_path.exists():
        msg = "No store available"
        raise RuntimeError(msg)

    engine = get_engine(db_path)
    row = get_playbook_by_id(engine, input.playbook_id)
    if row is None:
        msg = f"Playbook {input.playbook_id} not found"
        raise RuntimeError(msg)

    return db_row_to_playbook_entry(row)
