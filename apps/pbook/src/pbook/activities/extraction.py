"""Extraction activities for the playbook service.

Receives pushed experience data, calls the LLM to extract lessons
targeting unexpected + actionable situations, and saves entries
with needs-review=True.

Design follows Function Core / Imperative Shell:

- Pure functions: build_extraction_system_prompt, build_extraction_user_prompt
- Testable function: execute_extraction_call (takes provider as argument)
- Temporal activities: extract_from_experience, save_extracted_entries
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from temporalio import activity

logger = logging.getLogger(__name__)

from pbook.llm import ExtractionResult, text_messages
from pbook.models import CapabilityTier, ModelConfig, PushExperienceInput, resolve_model
from pbook.prompts.extraction import (
    build_extraction_system_prompt,
    build_extraction_user_prompt,
)

if TYPE_CHECKING:
    from pbook.llm import LLMProvider


def _resolve_default_model() -> tuple[str, str]:
    """Resolve the default model for extraction, returning (provider, model)."""
    from sax_llm.registry import parse_model_id

    model_id = resolve_model(CapabilityTier.CLASSIFICATION, ModelConfig())
    return parse_model_id(model_id)


# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


async def execute_extraction_call(
    system_prompt: str,
    user_prompt: str,
    provider: LLMProvider,
    model: str = "",
) -> tuple[ExtractionResult, int, int, float]:
    """Call the LLM provider for extraction and return structured results.

    Returns ``(result, input_tokens, output_tokens, latency_ms)``.
    Separated from the imperative shell so tests can inject a mock provider.
    """
    messages = text_messages(system_prompt, user_prompt)
    start = time.monotonic()

    params = provider.build_request_params(
        messages=messages,
        output_type=ExtractionResult,
        model=model,
        max_tokens=4096,
    )
    response = await provider.call(params)
    latency_ms = (time.monotonic() - start) * 1000

    result = ExtractionResult.model_validate(response.tool_input)
    logger.info(
        "Extraction LLM call: %d entries, %d input tokens, %d output tokens, %.0fms",
        len(result.entries), response.input_tokens, response.output_tokens, latency_ms,
    )
    return result, response.input_tokens, response.output_tokens, latency_ms


# ---------------------------------------------------------------------------
# Temporal activities
# ---------------------------------------------------------------------------


@activity.defn
async def extract_from_experience(input_json: str) -> str:
    """Extract lessons from pushed experience data.

    Accepts JSON-serialized list of PushExperienceInput dicts.
    Returns JSON-serialized ExtractionResult.
    """
    from pbook.llm import get_provider

    experiences = [
        PushExperienceInput.model_validate(exp)
        for exp in json.loads(input_json)
    ]

    if not experiences:
        logger.debug("No experiences to extract from")
        return ExtractionResult().model_dump_json()

    logger.info("Extracting from %d experience(s)", len(experiences))
    provider = get_provider()
    system_prompt = build_extraction_system_prompt(experiences)
    user_prompt = build_extraction_user_prompt()
    _, model = _resolve_default_model()

    result, _in_tok, _out_tok, _latency = await execute_extraction_call(
        system_prompt, user_prompt, provider, model=model,
    )

    logger.info("Extracted %d entries", len(result.entries))
    return result.model_dump_json()


@activity.defn
async def compute_embedding(text: str) -> str:
    """Generate an embedding for the given text using OpenAI.

    Returns a base64-encoded string (JSON-safe) rather than raw bytes.
    Decode with ``base64.b64decode()`` at the DB boundary.
    """
    import base64

    from pbook.embeddings import get_embedding

    raw = await get_embedding(text)
    return base64.b64encode(raw).decode("ascii")


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
    import base64

    from pbook.models import EntryType, PlaybookEntry
    from pbook.store import (
        ENTRY_MATCH_THRESHOLD,
        add_entry_source,
        build_entry_dict,
        find_semantic_duplicates,
        find_similar_source_contexts,
        get_db_path,
        get_engine,
        run_migrations,
        save_entry_returning_id,
    )

    data = json.loads(input_json)
    entries_raw = data.get("entries", [])
    project = data.get("project", "")
    source = data.get("source") or {}

    if not entries_raw:
        logger.debug("No extracted entries to save")
        return 0

    db_path = get_db_path()
    if db_path is None:
        return 0

    run_migrations(db_path)
    engine = get_engine(db_path)

    src_session_id = source.get("session_id", "")
    src_project_name = source.get("project_name", project)
    src_experience_hash = source.get("experience_hash")
    src_context = source.get("source_context", "")
    raw_src_embedding = source.get("source_context_embedding") or ""
    src_context_embedding = (
        base64.b64decode(raw_src_embedding) if raw_src_embedding else None
    )

    new_entry_count = 0

    for raw in entries_raw:
        raw_embedding = raw.get("embedding")
        embedding = base64.b64decode(raw_embedding) if raw_embedding else None

        target_entry_id: int | None = None

        # Step 1: try to attach to an existing semantically-similar entry.
        if embedding is not None:
            matches = find_semantic_duplicates(
                engine, embedding,
                threshold=ENTRY_MATCH_THRESHOLD,
                limit=1,
            )
            if matches:
                target_entry_id = matches[0]["id"]
                logger.info(
                    "Match-or-attach: candidate '%s' matches existing entry %d "
                    "(similarity=%.3f). Attaching source instead of inserting.",
                    raw.get("title", ""), target_entry_id, matches[0]["similarity"],
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
            target_entry_id = save_entry_returning_id(engine, build_entry_dict(entry))
            new_entry_count += 1

        # Step 3: source-row dedup, then write.
        if src_context_embedding is not None:
            similar = find_similar_source_contexts(
                engine, target_entry_id, src_context_embedding,
            )
            if similar:
                logger.info(
                    "Source dedup: justification for entry %d already recorded "
                    "(similarity=%.3f). Skipping source row.",
                    target_entry_id, similar[0]["similarity"],
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
    from pbook.store import get_db_path, get_engine, record_ingested_session as _record, run_migrations

    data = json.loads(input_json)
    session_id = data["session_id"]

    db_path = get_db_path()
    if db_path is None:
        return

    run_migrations(db_path)
    engine = get_engine(db_path)
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
    from pbook.store import (
        get_db_path,
        get_engine,
        run_migrations,
    )
    from pbook.store import (
        record_ingested_session_error as _record_error,
    )

    data = json.loads(input_json)
    session_id = data["session_id"]

    db_path = get_db_path()
    if db_path is None:
        return

    run_migrations(db_path)
    engine = get_engine(db_path)
    _record_error(
        engine,
        session_id=session_id,
        error_message=data.get("error_message", ""),
        project_name=data.get("project_name", ""),
    )
