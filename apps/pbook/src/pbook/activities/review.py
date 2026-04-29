"""Persistence-side review activities.

The LLM review call now goes through the generic ``llm_chat`` step.
This module owns the database side: validating raw JSON against the
``PlaybookEntry`` schema, fetching existing entries for context, and
finding semantic duplicates.
"""

from __future__ import annotations

import json
import logging

from temporalio import activity

from pbook.models import PlaybookEntry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temporal activities
# ---------------------------------------------------------------------------


@activity.defn
async def validate_entry(raw_json: str) -> str:
    """Parse and validate raw JSON against the PlaybookEntry schema.

    Returns JSON with keys: valid (bool), entry (dict|null), error (str|null).
    """
    from pydantic import ValidationError

    try:
        entry = PlaybookEntry.model_validate_json(raw_json)
        logger.debug("Entry validated: %s", entry.title)
        return json.dumps({
            "valid": True,
            "entry": entry.model_dump(),
            "error": None,
        })
    except (ValidationError, ValueError) as exc:
        logger.warning("Entry validation failed: %s", exc)
        return json.dumps({
            "valid": False,
            "entry": None,
            "error": str(exc),
        })


@activity.defn
async def fetch_existing_entries(limit: int = 50) -> list[dict]:
    """Query recent entries for duplication context."""
    from pbook.store import get_db_path, get_engine, list_recent_entries, run_migrations

    db_path = get_db_path()
    if db_path is None or not db_path.exists():
        return []

    run_migrations(db_path)
    engine = get_engine(db_path)
    return list_recent_entries(engine, limit=limit)


@activity.defn
async def find_duplicates(input_json: str) -> list[dict]:
    """Find semantic duplicates for a proposed entry.

    Accepts JSON with keys: embedding (bytes), threshold (float).
    """
    import base64

    from pbook.store import find_semantic_duplicates, get_db_path, get_engine, run_migrations

    data = json.loads(input_json)
    embedding = base64.b64decode(data["embedding"])
    threshold = data.get("threshold", 0.85)

    db_path = get_db_path()
    if db_path is None or not db_path.exists():
        return []

    run_migrations(db_path)
    engine = get_engine(db_path)
    return find_semantic_duplicates(engine, embedding, threshold=threshold)


