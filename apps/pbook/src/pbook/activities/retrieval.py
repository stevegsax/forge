"""Retrieval activities for the playbook service.

Design follows Function Core / Imperative Shell:

- Pure functions: score_entry, rank_and_pack
- Temporal activities: fetch_candidates
"""

from __future__ import annotations

import json
import logging

from temporalio import activity

from pbook.models import EntryType, RetrievalMode
from pbook.tags import EXTRACTED_NAMESPACES, GENERAL_NAMESPACES, parse_tag

logger = logging.getLogger(__name__)

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
        # Boost general knowledge
        general_overlap = sum(
            1 for t in entry_tags & query_tags
            if _tag_namespace(t) in GENERAL_NAMESPACES
        )
        score += general_overlap * 0.5

    elif mode == RetrievalMode.FIX:
        # Boost project-specific and pitfalls
        extracted_overlap = sum(
            1 for t in entry_tags & query_tags
            if _tag_namespace(t) in EXTRACTED_NAMESPACES
        )
        score += extracted_overlap * 0.5

        if entry_type == EntryType.PITFALL:
            score += 1.0

    # Helpfulness signal from feedback counters
    score += _helpfulness_adjustment(entry)

    return score


_MIN_RETRIEVALS_FOR_SIGNAL = 3
_HELPFULNESS_WEIGHT = 2.0


def _helpfulness_adjustment(entry: dict) -> float:
    """Compute a score adjustment based on feedback counters.

    Requires at least ``_MIN_RETRIEVALS_FOR_SIGNAL`` retrievals before
    applying any adjustment.  Returns a value in the range
    ``[-_HELPFULNESS_WEIGHT, +_HELPFULNESS_WEIGHT]``.  Entries without
    feedback data receive no adjustment (backward compatible).
    """
    retrievals = entry.get("retrieval_count", 0)
    if retrievals < _MIN_RETRIEVALS_FOR_SIGNAL:
        return 0.0

    helpful = entry.get("helpful_count", 0)
    harmful = entry.get("harmful_count", 0)
    ratio = (helpful - harmful) / retrievals
    return ratio * _HELPFULNESS_WEIGHT


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
    *,
    similarities: dict[int, float] | None = None,
    threshold: float = 0.0,
) -> tuple[list[dict], int]:
    """Rank candidates by score and pack within the token budget.

    When ``similarities`` is provided (i.e. the workflow ran a
    free-text query), ranking is **semantic-primary**: entries are
    ordered by cosine similarity, with the existing tag-overlap score
    used as a tiebreaker. Candidates below ``threshold`` are dropped.

    When ``similarities`` is ``None``, the legacy tag-overlap +
    mode-boost score drives ordering (forge consumers keep working).

    Returns ``(packed_entries, total_tokens)``. Each packed entry
    carries a ``similarity`` key when it was scored against a query.
    """
    tag_set = set(query_tags)

    scored: list[tuple[float, float, dict]] = []
    for entry in candidates:
        tag_score = score_entry(entry, tag_set, mode)
        if similarities is not None:
            sim = similarities.get(entry.get("id", -1), 0.0)
            if sim < threshold:
                continue
            scored.append((sim, tag_score, entry))
        else:
            scored.append((tag_score, 0.0, entry))

    # Sort: primary descending, tag_score descending as tiebreaker.
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    packed: list[dict] = []
    total_tokens = 0

    for primary, _, entry in scored:
        entry_text = f"{entry['title']}\n{entry['content']}"
        entry_tokens = _estimate_tokens(entry_text)

        if total_tokens + entry_tokens > token_budget:
            continue

        if similarities is not None:
            entry = {**entry, "similarity": primary}
        packed.append(entry)
        total_tokens += entry_tokens

    logger.debug(
        "Ranked %d candidates, packed %d within %d token budget (%d tokens used)",
        len(candidates), len(packed), token_budget, total_tokens,
    )
    return packed, total_tokens


# ---------------------------------------------------------------------------
# Temporal activities
# ---------------------------------------------------------------------------


@activity.defn
async def compute_similarities(input_json: str) -> dict[str, float]:
    """Compute cosine similarity between a query embedding and a list of candidates.

    Accepts JSON with keys:
      - query_embedding_b64: base64-encoded query embedding bytes.
      - candidates: list of {id, embedding_b64} dicts.
    Returns a dict mapping str(id) → similarity. (JSON serialization
    requires string keys; the workflow re-keys to int.)

    Lives in an activity because numpy operations inside a workflow
    body trip Temporal's determinism sandbox.
    """
    import base64

    from pbook.embeddings import cosine_similarity

    data = json.loads(input_json)
    query_embedding = base64.b64decode(data["query_embedding_b64"])
    out: dict[str, float] = {}
    for candidate in data["candidates"]:
        emb_b64 = candidate.get("embedding_b64")
        if not emb_b64:
            continue
        emb = base64.b64decode(emb_b64)
        out[str(candidate["id"])] = cosine_similarity(query_embedding, emb)
    return out


@activity.defn
async def record_retrieval_event(entry_ids_json: str) -> None:
    """Record that entries were served in a retrieval result.

    Accepts JSON-serialized list of entry IDs.  Increments retrieval_count
    for each entry.  Failures are logged but do not propagate.
    """
    from pbook.store import get_db_path, get_engine, record_retrieval, run_migrations

    entry_ids = json.loads(entry_ids_json)
    if not entry_ids:
        return

    db_path = get_db_path()
    if db_path is None:
        return

    run_migrations(db_path)
    engine = get_engine(db_path)
    record_retrieval(engine, entry_ids)
    logger.info("Recorded retrieval for %d entries", len(entry_ids))


_MAX_QUERY_ONLY_CANDIDATES = 200


@activity.defn
async def fetch_candidates(input_json: str) -> list[dict]:
    """Fetch candidate entries matching the query tags or query string.

    Accepts JSON-serialized RetrievalInput.

    - Tags + (optional query) → existing tag fetch.
    - Query-only (no tags) → broad pool of entries with embeddings,
      capped at ``_MAX_QUERY_ONLY_CANDIDATES``. The semantic ranking
      step that follows narrows it.
    - Neither → empty (preserves prior contract for forge callers).
    """
    from pbook.models import RetrievalInput
    from pbook.store import (
        Entry,
        get_db_path,
        get_engine,
        get_entries_by_tags,
        run_migrations,
    )

    inp = RetrievalInput.model_validate_json(input_json)
    logger.info(
        "Fetching candidates: tags=%s mode=%s query=%r",
        inp.tags, inp.mode, inp.query,
    )

    db_path = get_db_path()
    if db_path is None:
        return []

    run_migrations(db_path)
    engine = get_engine(db_path)

    if inp.tags:
        candidates = get_entries_by_tags(
            engine,
            inp.tags,
            limit=100,  # Over-fetch for ranking
            approved_only=inp.approved_only,
            include_rejected=inp.include_rejected,
        )
    elif inp.query:
        # Query-only: pull a broad pool of entries that have embeddings
        # so the semantic step can rank them. Filtered by approval and
        # rejection per the input.
        t = Entry.__table__
        stmt = t.select().where(t.c.embedding.is_not(None))
        if inp.approved_only:
            stmt = stmt.where(t.c.needs_review == False)  # noqa: E712
        if not inp.include_rejected:
            stmt = stmt.where(t.c.rejected == False)  # noqa: E712
        stmt = stmt.order_by(t.c.created_at.desc()).limit(_MAX_QUERY_ONLY_CANDIDATES)
        with engine.connect() as conn:
            candidates = [dict(r) for r in conn.execute(stmt).mappings().all()]
    else:
        candidates = []

    logger.info("Found %d candidates", len(candidates))
    return candidates
