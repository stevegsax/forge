"""Review activities for the playbook service.

LLM-based review of manually submitted playbook entries.
Checks clarity, correctness, completeness, and duplication.

Design follows Function Core / Imperative Shell:

- Pure functions: build_review_system_prompt, build_review_user_prompt,
  apply_suggestions
- Testable function: execute_review_call (takes provider as argument)
- Temporal activities: validate_entry, fetch_existing_entries,
  review_entry
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from temporalio import activity

logger = logging.getLogger(__name__)

from pbook.llm import ReviewResult, text_messages
from pbook.models import CapabilityTier, ModelConfig, PlaybookEntry, resolve_model
from pbook.prompts.review import (
    apply_suggestions,
    build_review_system_prompt,
    build_review_user_prompt,
)

if TYPE_CHECKING:
    from pbook.llm import LLMProvider


def _resolve_default_model() -> tuple[str, str]:
    """Resolve the default model for review, returning (provider, model)."""
    from sax_llm.registry import parse_model_id

    model_id = resolve_model(CapabilityTier.CLASSIFICATION, ModelConfig())
    return parse_model_id(model_id)


# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


async def execute_review_call(
    system_prompt: str,
    user_prompt: str,
    provider: LLMProvider,
    model: str = "",
) -> ReviewResult:
    """Call the LLM provider for review and return structured results.

    Separated from the imperative shell so tests can inject a mock provider.
    """
    messages = text_messages(system_prompt, user_prompt)

    params = provider.build_request_params(
        messages=messages,
        output_type=ReviewResult,
        model=model,
        max_tokens=1024,
    )
    response = await provider.call(params)

    return ReviewResult.model_validate(response.tool_input)


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
    from pbook.store import find_semantic_duplicates, get_db_path, get_engine, run_migrations

    import base64

    data = json.loads(input_json)
    embedding = base64.b64decode(data["embedding"])
    threshold = data.get("threshold", 0.85)

    db_path = get_db_path()
    if db_path is None or not db_path.exists():
        return []

    run_migrations(db_path)
    engine = get_engine(db_path)
    return find_semantic_duplicates(engine, embedding, threshold=threshold)


@activity.defn
async def review_entry(input_json: str) -> str:
    """Review a proposed entry via LLM and apply suggestions.

    Accepts JSON with keys: entry (dict), existing_entries (list[dict]),
    model_name (str, optional).
    Returns JSON with keys: approved (bool), rejection_reason (str),
    final_entry (dict).
    """
    from pbook.llm import get_provider

    data = json.loads(input_json)
    entry = PlaybookEntry.model_validate(data["entry"])
    existing = data.get("existing_entries", [])
    _, default_model = _resolve_default_model()
    model_name = data.get("model_name", "") or default_model

    logger.info("Reviewing entry: %s", entry.title)
    provider = get_provider()
    system_prompt = build_review_system_prompt(existing)
    user_prompt = build_review_user_prompt(entry)

    review = await execute_review_call(
        system_prompt, user_prompt, provider, model=model_name,
    )

    import base64

    def _serialize_entry(e: PlaybookEntry) -> dict:
        d = e.model_dump()
        if d.get("embedding") is not None:
            d["embedding"] = base64.b64encode(d["embedding"]).decode("ascii")
        return d

    if not review.approved:
        logger.info("Entry rejected: %s — %s", entry.title, review.rejection_reason)
        return json.dumps({
            "approved": False,
            "rejection_reason": review.rejection_reason,
            "final_entry": _serialize_entry(entry),
        })

    final_entry = apply_suggestions(entry, review)
    logger.info("Entry approved: %s", final_entry.title)

    # Preserve the embedding if it was already computed
    final_entry.embedding = entry.embedding

    return json.dumps({
        "approved": True,
        "rejection_reason": "",
        "final_entry": _serialize_entry(final_entry),
    })
