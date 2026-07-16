"""Persistence-side extraction activities.

The LLM call itself runs through the generic ``llm_chat`` step — see
:mod:`pbook.workflow_steps.llm`. This module owns the database side:
saving extracted entries via match-or-attach and recording session
lifecycle events for forge's IngestionWorkflow callbacks.
"""

from __future__ import annotations

import json
import logging

from temporalio import activity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temporal activities
# ---------------------------------------------------------------------------


@activity.defn
async def save_extracted_entries(input_json: str) -> int:
    """Save extracted entries via match-or-attach, then write source rows.

    Accepts JSON with keys:
      - entries: list of extracted-entry dicts (title, content, tags,
        embedding base64).
      - project: source project name for any newly-inserted entries.
      - source: dict carrying the originating experience's identifying
        info (session_id, project_name, experience_hash, source_context,
        source_context_embedding base64). Threaded through by
        ExtractionWorkflow on a per-experience basis.

    For each candidate entry:
      1. Look up semantic duplicates among existing entries
         (threshold ENTRY_MATCH_THRESHOLD = 0.85). If a match is found,
         the candidate is NOT inserted — the existing entry simply
         gains a new entry_sources row.
      2. If no match, the candidate becomes a new entry (needs_review=True)
         and gets its first entry_sources row.
      3. Source-row writes are deduped against existing rows on the
         target entry (threshold SOURCE_DEDUP_THRESHOLD = 0.92): if the
         same justification is already recorded, the new row is skipped.

    Returns the number of NEW entries created. Source rows attached to
    pre-existing entries do not count toward this number — they're
    incremental enrichment.
    """
    from pbook.embeddings import decode_embedding
    from pbook.models import EntryType, PlaybookEntry
    from pbook.store import (
        ENTRY_MATCH_THRESHOLD,
        add_entry_source,
        build_entry_dict,
        find_semantic_duplicates,
        find_similar_source_contexts,
        get_store_engine,
        insert_entry,
    )

    data = json.loads(input_json)
    entries_raw = data.get("entries", [])
    project = data.get("project", "")
    source = data.get("source") or {}

    if not entries_raw:
        logger.debug("No extracted entries to save")
        return 0

    engine = get_store_engine()
    if engine is None:
        return 0

    src_session_id = source.get("session_id", "")
    src_project_name = source.get("project_name", project)
    src_experience_hash = source.get("experience_hash")
    src_context = source.get("source_context", "")
    raw_src_embedding = source.get("source_context_embedding") or ""
    src_context_embedding = decode_embedding(raw_src_embedding) if raw_src_embedding else None

    new_entry_count = 0

    for raw in entries_raw:
        raw_embedding = raw.get("embedding")
        embedding = decode_embedding(raw_embedding) if raw_embedding else None

        target_entry_id: int | None = None

        # Step 1: try to attach to an existing semantically-similar entry.
        if embedding is not None:
            matches = find_semantic_duplicates(
                engine,
                embedding,
                threshold=ENTRY_MATCH_THRESHOLD,
                limit=1,
            )
            if matches:
                target_entry_id = matches[0]["id"]
                logger.info(
                    "Match-or-attach: candidate '%s' matches existing entry %d "
                    "(similarity=%.3f). Attaching source instead of inserting.",
                    raw.get("title", ""),
                    target_entry_id,
                    matches[0]["similarity"],
                )

        # Step 2: no match — insert a new entry.
        if target_entry_id is None:
            entry = PlaybookEntry(
                title=raw["title"],
                content=raw["content"],
                tags=raw.get("tags", []),
                entry_type=EntryType.PITFALL,
                source_project=project,
                needs_review=True,
                embedding=embedding,
            )
            target_entry_id = insert_entry(engine, build_entry_dict(entry), entry.tags)
            new_entry_count += 1

        # Step 3: source-row dedup, then write.
        if src_context_embedding is not None:
            similar = find_similar_source_contexts(
                engine,
                target_entry_id,
                src_context_embedding,
            )
            if similar:
                logger.info(
                    "Source dedup: justification for entry %d already recorded "
                    "(similarity=%.3f). Skipping source row.",
                    target_entry_id,
                    similar[0]["similarity"],
                )
                continue

        add_entry_source(
            engine,
            entry_id=target_entry_id,
            session_id=src_session_id,
            project_name=src_project_name,
            experience_hash=src_experience_hash,
            source_context=src_context,
            source_context_embedding=src_context_embedding,
        )

    return new_entry_count


@activity.defn
async def record_ingested_session(input_json: str) -> None:
    """Record that a Claude Code session has been ingested.

    Accepts JSON with keys: session_id, project_name, experiences_found, entries_created.
    Called cross-queue from forge's IngestionWorkflow.
    """
    from pbook.store import get_store_engine
    from pbook.store import record_ingested_session as _record

    data = json.loads(input_json)
    session_id = data["session_id"]

    engine = get_store_engine()
    if engine is None:
        return

    _record(
        engine,
        session_id=session_id,
        project_name=data.get("project_name", ""),
        experiences_found=data.get("experiences_found", 0),
        entries_created=data.get("entries_created", 0),
    )


@activity.defn
async def record_ingested_session_error(input_json: str) -> None:
    """Mark a Claude Code session as failed.

    Accepts JSON with keys: session_id, error_message, project_name (optional).
    Called cross-queue from forge's IngestionWorkflow on failure paths.
    """
    from pbook.store import get_store_engine
    from pbook.store import (
        record_ingested_session_error as _record_error,
    )

    data = json.loads(input_json)
    session_id = data["session_id"]

    engine = get_store_engine()
    if engine is None:
        return

    _record_error(
        engine,
        session_id=session_id,
        error_message=data.get("error_message", ""),
        project_name=data.get("project_name", ""),
    )
