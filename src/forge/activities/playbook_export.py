"""Activities for playbook export workflow.

Follows Function Core / Imperative Shell:
- Pure function: db_row_to_playbook_entry. The ``fetch_playbook_ids`` and
  ``export_single_playbook`` bound methods on ``StoreActivities``
  (forge.activities.roots) delegate to it and the store helpers.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forge.models import PlaybookEntry

# ---------------------------------------------------------------------------
# Function core
# ---------------------------------------------------------------------------


def db_row_to_playbook_entry(row: dict[str, Any]) -> PlaybookEntry:
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
