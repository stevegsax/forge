"""Activities backing the worker-only CLI surface.

Every direct-DB CLI command (except ``pbook migrate``) routes through
one of these activities. The worker process is the only place that
opens the DB file — its ``PBOOK_DB_PATH`` is the single source of
truth for which DB the operation hits. CLI processes are thin clients
that submit workflows and render results.

The grouping helper ``group_review_by_experience`` is pure and
reusable; tests import it directly. Everything else is an
activity-level wrapper around an existing ``pbook.store`` helper.
"""

from __future__ import annotations

import json
import logging

from temporalio import activity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_BINARY_FIELDS = ("embedding", "source_context_embedding")


def _strip_binary(row: dict) -> dict:
    """Drop binary BLOB columns from a row dict before returning across
    the activity wire. Pydantic's to_json (used by Temporal's data
    converter) doesn't auto-base64 raw bytes inside arbitrary dicts —
    random float32 byte sequences fail UTF-8 validation in serde_json
    and the activity result fails to serialize.
    """
    return {k: v for k, v in row.items() if k not in _BINARY_FIELDS}


def _engine():
    """Open a SQLAlchemy engine against the worker's configured DB.

    The worker resolves ``PBOOK_DB_PATH`` once per activity call. If the
    DB hasn't been configured (env unset and no XDG fallback file),
    raise — callers should see a clear error rather than a confusing
    silent fallback.
    """
    from pbook.store import get_db_path, get_engine, run_migrations

    db_path = get_db_path()
    if db_path is None:
        msg = (
            "Worker has no DB configured. Set PBOOK_DB_PATH on the worker "
            "process and restart."
        )
        raise RuntimeError(msg)
    run_migrations(db_path)
    return get_engine(db_path)


def group_review_by_experience(
    entries_with_sources: list[dict],
) -> tuple[list[tuple[str, list[dict]]], list[dict]]:
    """Pure: cluster review-queue entries by their primary experience_hash.

    Each entry's first source row's ``experience_hash`` is the cluster
    key. Hashes mapped to ≥ 2 entries become clusters; everything else
    is a singleton. Used by ``review_queue_activity`` (when
    ``by_experience=True``).
    """
    by_hash: dict[str, list[dict]] = {}
    no_hash: list[dict] = []
    for entry in entries_with_sources:
        sources = entry.get("sources", [])
        primary = sources[0].get("experience_hash") if sources else None
        if primary is None:
            no_hash.append(entry)
        else:
            by_hash.setdefault(primary, []).append(entry)

    clusters: list[tuple[str, list[dict]]] = []
    singletons: list[dict] = list(no_hash)
    for h, ents in sorted(by_hash.items()):
        if len(ents) >= 2:
            clusters.append((h, ents))
        else:
            singletons.extend(ents)
    return clusters, singletons


# ---------------------------------------------------------------------------
# Read activities
# ---------------------------------------------------------------------------


@activity.defn
async def get_entry_activity(input: dict) -> dict | None:
    """Fetch a single entry by id; return None if missing."""
    from pbook.store import get_entry_by_id

    engine = _engine()
    row = get_entry_by_id(engine, int(input["entry_id"]))
    return _strip_binary(row) if row else None


@activity.defn
async def list_entries_activity(input: dict) -> list[dict]:
    """List entries with optional tag/type/project/needs-review filters."""
    from pbook.store import get_entries_by_tags, list_recent_entries

    engine = _engine()
    tags = input.get("tags") or []
    if tags:
        entries = get_entries_by_tags(
            engine, tags,
            limit=input.get("limit", 20),
            include_rejected=input.get("include_rejected", False),
        )
    else:
        entries = list_recent_entries(
            engine,
            limit=input.get("limit", 20),
            include_rejected=input.get("include_rejected", False),
        )

    entry_type = input.get("entry_type")
    if entry_type:
        entries = [e for e in entries if e.get("entry_type") == entry_type]

    project = input.get("project")
    if project:
        entries = [e for e in entries if e.get("source_project") == project]

    if input.get("needs_review_only"):
        entries = [e for e in entries if e.get("needs_review")]

    return [_strip_binary(e) for e in entries]


@activity.defn
async def list_sources_activity(input: dict) -> dict:
    """List entry_sources rows for an entry; return ``{"found": False}``
    when the entry id is not present so the workflow can surface a
    not_found error."""
    from pbook.store import get_entry_by_id, list_entry_sources_for_entry

    engine = _engine()
    entry_id = int(input["entry_id"])
    if get_entry_by_id(engine, entry_id) is None:
        return {"found": False, "rows": []}
    rows = list_entry_sources_for_entry(engine, entry_id)
    return {"found": True, "rows": [_strip_binary(r) for r in rows]}


@activity.defn
async def list_tags_activity(_input: dict) -> dict:
    """Return canonical tag namespaces plus the values currently in use."""
    from pbook.store import list_tag_values_in_use
    from pbook.tags import EXTRACTED_NAMESPACES, GENERAL_NAMESPACES

    engine = _engine()
    return {
        "namespaces": {
            "general": sorted(GENERAL_NAMESPACES),
            "extracted": sorted(EXTRACTED_NAMESPACES),
        },
        "values_in_use": list_tag_values_in_use(engine),
    }


@activity.defn
async def review_queue_activity(input: dict) -> dict:
    """Return either a flat list (default) or a clustered view (when
    ``by_experience=True``) of the review queue."""
    from pbook.store import list_recent_entries, list_review_queue_with_sources

    def _strip_entry_for_wire(e: dict) -> dict:
        # Drop both the sibling sources list (added by
        # list_review_queue_with_sources) and the binary embedding columns.
        return _strip_binary({k: v for k, v in e.items() if k != "sources"})

    engine = _engine()
    limit = input.get("limit", 20)
    if input.get("by_experience"):
        entries = list_review_queue_with_sources(engine)[:limit]
        clusters, singletons = group_review_by_experience(entries)
        return {
            "entries": [],
            "clusters": [
                {
                    "experience_hash": h,
                    "project_name": (
                        ents[0].get("sources", [{}])[0].get("project_name", "")
                    ),
                    "entries": [_strip_entry_for_wire(e) for e in ents],
                }
                for h, ents in clusters
            ],
            "singletons": [_strip_entry_for_wire(e) for e in singletons],
        }

    entries = list_recent_entries(engine, limit=limit)
    needs_review = [e for e in entries if e.get("needs_review")]
    return {
        "entries": [_strip_binary(e) for e in needs_review],
        "clusters": [],
        "singletons": [],
    }


@activity.defn
async def list_sessions_activity(input: dict) -> list[dict]:
    """List ingested_sessions rows, optionally filtered by project."""
    from pbook.store import list_ingested_sessions

    engine = _engine()
    return list_ingested_sessions(
        engine,
        project=input.get("project") or None,
        limit=input.get("limit", 20),
    )


@activity.defn
async def get_session_text_activity(input: dict) -> dict:
    """Render a session transcript by id (or explicit path); return text
    plus the resolved project name when discoverable."""
    from pathlib import Path

    from pbook.transcript import discover_sessions, parse_jsonl_file, render_transcript

    session_id = input["session_id"]
    explicit_path = input.get("path")
    raw = bool(input.get("raw", False))

    project_name = ""
    jsonl_path: Path | None = None

    if explicit_path:
        candidate = Path(explicit_path)
        if not candidate.exists():
            return {"text": "", "project_name": "", "error": "session_file_missing"}
        jsonl_path = candidate
    else:
        for s in discover_sessions(min_size=0, exclude_subagents=False):
            if s.session_id == session_id:
                jsonl_path = Path(s.path)
                project_name = s.project_name
                break

    if jsonl_path is None or not jsonl_path.exists():
        return {"text": "", "project_name": project_name, "error": "session_file_missing"}

    if raw:
        return {"text": jsonl_path.read_text(), "project_name": project_name}

    transcript = parse_jsonl_file(jsonl_path)
    return {
        "text": render_transcript(transcript),
        "project_name": project_name,
    }


@activity.defn
async def check_duplicate_activity(input: dict) -> list[dict]:
    """Return entries whose title matches the input string (and tags, if given)."""
    from pbook.store import check_duplicate

    engine = _engine()
    matches = check_duplicate(engine, input["title"], tags=input.get("tags"))
    return [_strip_binary(m) for m in matches]


# ---------------------------------------------------------------------------
# Write activities
# ---------------------------------------------------------------------------


@activity.defn
async def add_entry_activity(input: dict) -> dict:
    """Insert a new playbook entry. Returns ``{id, title, needs_review,
    rejected}`` on success, or ``{error: 'tag_invalid', messages: [...]}``
    on tag validation failure.

    Note: CLI-added entries are not auto-embedded — the existing
    ``build_entry_dict`` carries whatever the input provided. Closing
    that gap is a separate change.
    """
    from pbook.models import PlaybookEntry
    from pbook.store import build_entry_dict, save_entry_returning_id
    from pbook.tags import validate_tags

    entry_payload = input["entry"]
    needs_review = bool(input.get("needs_review", False))

    entry_model = PlaybookEntry.model_validate(entry_payload)
    tag_errors = validate_tags(entry_model.tags)
    if tag_errors:
        return {"error": "tag_invalid", "messages": tag_errors}

    if needs_review:
        entry_model = entry_model.model_copy(update={"needs_review": True})

    entry_dict = build_entry_dict(entry_model)

    engine = _engine()
    new_id = save_entry_returning_id(engine, entry_dict)
    return {
        "id": new_id,
        "title": entry_model.title,
        "needs_review": entry_model.needs_review,
        "rejected": False,
    }


@activity.defn
async def approve_entry_activity(input: dict) -> dict:
    """Clear ``needs_review`` on an entry; return the resulting status dict."""
    from pbook.store import get_entry_by_id, update_entry

    engine = _engine()
    entry_id = int(input["entry_id"])
    if get_entry_by_id(engine, entry_id) is None:
        return {"error": "not_found", "id": entry_id}

    update_entry(engine, entry_id, {"needs_review": False})
    row = get_entry_by_id(engine, entry_id)
    assert row is not None
    return {
        "id": row["id"],
        "title": row["title"],
        "approved": not (row.get("needs_review", False) or row.get("rejected", False)),
        "needs_review": row.get("needs_review", False),
        "rejected": row.get("rejected", False),
        "rejection_reason": row.get("rejection_reason"),
    }


@activity.defn
async def reject_entry_activity(input: dict) -> dict:
    """Soft-mark an entry as rejected with an optional reason."""
    from pbook.store import get_entry_by_id, mark_rejected

    engine = _engine()
    entry_id = int(input["entry_id"])
    if get_entry_by_id(engine, entry_id) is None:
        return {"error": "not_found", "id": entry_id}

    mark_rejected(engine, entry_id, reason=input.get("reason"))
    row = get_entry_by_id(engine, entry_id)
    assert row is not None
    return {
        "id": row["id"],
        "title": row["title"],
        "approved": not (row.get("needs_review", False) or row.get("rejected", False)),
        "needs_review": row.get("needs_review", False),
        "rejected": row.get("rejected", False),
        "rejection_reason": row.get("rejection_reason"),
    }


@activity.defn
async def update_entry_activity(input: dict) -> dict:
    """Update arbitrary entry columns (validated against the entries table)."""
    from pbook.store import get_entry_by_id, update_entry

    engine = _engine()
    entry_id = int(input["entry_id"])
    updates = input.get("updates") or {}
    if get_entry_by_id(engine, entry_id) is None:
        return {"error": "not_found", "id": entry_id}

    update_entry(engine, entry_id, updates)
    row = get_entry_by_id(engine, entry_id)
    assert row is not None
    return {
        "id": row["id"],
        "title": row["title"],
        "approved": not (row.get("needs_review", False) or row.get("rejected", False)),
        "needs_review": row.get("needs_review", False),
        "rejected": row.get("rejected", False),
        "rejection_reason": row.get("rejection_reason"),
    }


@activity.defn
async def record_feedback_activity(input: dict) -> dict:
    """Record helpful/harmful feedback. Returns the updated counters and
    a guidance flag noting whether the 3-retrieval threshold has been
    met (so the CLI can warn the user when feedback won't immediately
    move ranking)."""
    from pbook.store import get_entry_by_id, record_feedback

    engine = _engine()
    entry_id = int(input["entry_id"])
    if get_entry_by_id(engine, entry_id) is None:
        return {"error": "not_found", "id": entry_id}

    # Note: ``context`` is accepted on the CLI surface but the current
    # ``record_feedback`` schema doesn't have a column for it. Carrying
    # the value here lets a future schema migration plumb it through
    # without changing the workflow API.
    record_feedback(engine, entry_id, helpful=bool(input["helpful"]))
    row = get_entry_by_id(engine, entry_id)
    assert row is not None
    return {
        "id": row["id"],
        "title": row["title"],
        "helpful_count": row.get("helpful_count", 0),
        "harmful_count": row.get("harmful_count", 0),
        "retrieval_count": row.get("retrieval_count", 0),
        "below_threshold": row.get("retrieval_count", 0) < 3,
    }


@activity.defn
async def filter_already_ingested_activity(input: dict) -> dict:
    """Given a list of session_ids, return the subset that has not been
    ingested yet. Used by ``pbook ingest`` to skip duplicates.
    """
    from pbook.store import get_ingested_session_ids

    engine = _engine()
    ingested = get_ingested_session_ids(engine)
    candidate_ids = input.get("session_ids", [])
    return {
        "fresh_ids": [sid for sid in candidate_ids if sid not in ingested],
        "skipped_count": sum(1 for sid in candidate_ids if sid in ingested),
    }


@activity.defn
async def record_started_sessions_activity(input: dict) -> dict:
    """Seed ``ingested_sessions`` rows in 'running' state for the given
    sessions. Called after the BatchIngestionWorkflow has been submitted
    so ``pbook sessions`` shows them mid-flight."""
    from pbook.store import record_ingested_session_started

    engine = _engine()
    for s in input["sessions"]:
        record_ingested_session_started(
            engine,
            session_id=s["session_id"],
            project_name=s["project_name"],
            workflow_id=input["workflow_id"],
            run_id=input["run_id"],
        )
    return {"recorded_count": len(input["sessions"])}


@activity.defn
async def prune_activity(input: dict) -> dict:
    """Identify (and optionally apply) prune candidates."""
    from pbook.activities.maintenance import identify_prune_candidates
    from pbook.store import list_all_entries, update_entry

    engine = _engine()
    all_entries = list_all_entries(engine)
    candidates = identify_prune_candidates(
        all_entries,
        min_retrievals=input.get("min_retrievals", 5),
        max_harmful_ratio=input.get("max_harmful_ratio", 0.5),
        max_stale_days=input.get("max_stale_days", 90),
    )

    applied_count = 0
    if input.get("apply"):
        for entry in candidates:
            entry_id = entry["id"]
            existing_tags = json.loads(entry.get("tags_json", "[]"))
            if "pattern:prune-candidate" not in existing_tags:
                existing_tags.append("pattern:prune-candidate")
            update_entry(engine, entry_id, {
                "needs_review": True,
                "tags_json": json.dumps(existing_tags),
            })
            applied_count += 1

    return {
        "candidates": [_strip_binary(c) for c in candidates],
        "applied_count": applied_count,
    }
