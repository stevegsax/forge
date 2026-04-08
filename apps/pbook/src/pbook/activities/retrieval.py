"""Retrieval activities for the playbook service.

Design follows Function Core / Imperative Shell:

- Pure functions: score_entry, rank_and_pack
- Temporal activities: fetch_candidates
"""

from __future__ import annotations

import json

from temporalio import activity

from pbook.models import EntryType, RetrievalMode
from pbook.tags import EXTRACTED_NAMESPACES, GENERAL_NAMESPACES, parse_tag

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return len(text) // 4


def score_entry(
    entry: dict,
    query_tags: set[str],
    mode: RetrievalMode,
) -> float:
    """Score an entry for ranking.

    Combines tag overlap count with mode-based weighting:

    - ``create`` mode boosts general tags (lang, lib, domain) and API docs
    - ``fix`` mode boosts extracted tags (project, pattern) and pitfalls
    """
    entry_tags = set(json.loads(entry.get("tags_json", "[]")))
    overlap = len(entry_tags & query_tags)

    if overlap == 0:
        return 0.0

    # Base score is tag overlap count
    score = float(overlap)

    # Mode-based boosting
    entry_type = entry.get("entry_type", "curated")

    if mode == RetrievalMode.CREATE:
        # Boost general knowledge and API docs
        general_overlap = sum(
            1 for t in entry_tags & query_tags
            if _tag_namespace(t) in GENERAL_NAMESPACES
        )
        score += general_overlap * 0.5

        if entry_type == EntryType.API_DOC:
            score += 1.0

    elif mode == RetrievalMode.FIX:
        # Boost project-specific and pitfalls
        extracted_overlap = sum(
            1 for t in entry_tags & query_tags
            if _tag_namespace(t) in EXTRACTED_NAMESPACES
        )
        score += extracted_overlap * 0.5

        if entry_type == EntryType.PITFALL:
            score += 1.0

    return score


def _tag_namespace(tag: str) -> str:
    """Extract namespace from a tag, returning empty string on parse failure."""
    try:
        ns, _ = parse_tag(tag)
    except ValueError:
        return ""
    return ns


def rank_and_pack(
    candidates: list[dict],
    query_tags: list[str],
    mode: RetrievalMode,
    token_budget: int,
) -> tuple[list[dict], int]:
    """Rank candidates by score and pack within the token budget.

    Returns ``(packed_entries, total_tokens)``.
    """
    tag_set = set(query_tags)

    scored = [
        (score_entry(entry, tag_set, mode), entry)
        for entry in candidates
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    packed: list[dict] = []
    total_tokens = 0

    for _score, entry in scored:
        entry_text = f"{entry['title']}\n{entry['content']}"
        entry_tokens = _estimate_tokens(entry_text)

        if total_tokens + entry_tokens > token_budget:
            continue

        packed.append(entry)
        total_tokens += entry_tokens

    return packed, total_tokens


# ---------------------------------------------------------------------------
# Temporal activities
# ---------------------------------------------------------------------------


@activity.defn
async def fetch_candidates(input_json: str) -> list[dict]:
    """Fetch candidate entries matching the query tags.

    Accepts JSON-serialized RetrievalInput, returns matching entries from store.
    """
    from pbook.models import RetrievalInput
    from pbook.store import get_db_path, get_engine, get_entries_by_tags, run_migrations

    inp = RetrievalInput.model_validate_json(input_json)

    db_path = get_db_path()
    if db_path is None:
        return []

    run_migrations(db_path)
    engine = get_engine(db_path)

    return get_entries_by_tags(
        engine,
        inp.tags,
        limit=100,  # Over-fetch for ranking
        approved_only=inp.approved_only,
    )
