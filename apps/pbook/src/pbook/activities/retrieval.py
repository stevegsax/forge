"""Retrieval activities for the playbook service.

Design follows Function Core / Imperative Shell:

- Pure functions: score_entry, rank_meta, pack_within_budget
- Temporal activities: fetch_candidates, compute_similarities_by_id,
  score_and_pack, record_retrieval_event

The activity layout exists to keep heavy entry data (content, embeddings)
off the workflow boundary. ``fetch_candidates`` returns minimal ranking
metadata; full entry content is loaded inside ``score_and_pack`` for only
the top-N entries that fit in the token budget. Embedding bytes are
loaded inside ``compute_similarities_by_id`` and never cross the wire.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pbook.models import EntryType, RetrievalMode
from pbook.tags import EXTRACTED_NAMESPACES, GENERAL_NAMESPACES, parse_tag

if TYPE_CHECKING:
    from sqlalchemy import Engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return len(text) // 4


def score_entry(
    entry: dict[str, Any],
    query_tags: set[str],
    mode: RetrievalMode,
) -> float:
    """Score an entry by tag overlap, mode boost, and helpfulness signal.

    Operates on minimal fields: ``tags``, ``entry_type``,
    ``helpful_count``, ``harmful_count``, ``retrieval_count``. Title and
    content are not consulted, so this works on the lightweight metadata
    dicts returned by ``fetch_candidates``.
    """
    entry_tags = set(entry.get("tags", []))
    overlap = len(entry_tags & query_tags)

    if overlap == 0:
        return 0.0

    score = float(overlap)
    entry_type = entry.get("entry_type", "curated")

    if mode == RetrievalMode.CREATE:
        general_overlap = sum(
            1 for t in entry_tags & query_tags if _tag_namespace(t) in GENERAL_NAMESPACES
        )
        score += general_overlap * 0.5
    elif mode == RetrievalMode.FIX:
        extracted_overlap = sum(
            1 for t in entry_tags & query_tags if _tag_namespace(t) in EXTRACTED_NAMESPACES
        )
        score += extracted_overlap * 0.5
        if entry_type == EntryType.PITFALL:
            score += 1.0

    score += _helpfulness_adjustment(entry)
    return score


_MIN_RETRIEVALS_FOR_SIGNAL = 3
_HELPFULNESS_WEIGHT = 2.0


def _helpfulness_adjustment(entry: dict[str, Any]) -> float:
    """Adjustment based on feedback counters; gated by 3-retrieval threshold,
    bounded ±_HELPFULNESS_WEIGHT.
    """
    retrievals: int = entry.get("retrieval_count", 0)
    if retrievals < _MIN_RETRIEVALS_FOR_SIGNAL:
        return 0.0
    helpful: int = entry.get("helpful_count", 0)
    harmful: int = entry.get("harmful_count", 0)
    ratio = (helpful - harmful) / retrievals
    return ratio * _HELPFULNESS_WEIGHT


def _tag_namespace(tag: str) -> str:
    """Extract namespace from a tag, returning empty string on parse failure."""
    try:
        ns, _ = parse_tag(tag)
    except ValueError:
        return ""
    return ns


def rank_meta(
    meta_list: list[dict[str, Any]],
    query_tags: list[str],
    mode: RetrievalMode,
    *,
    similarities: dict[int, float] | None = None,
    threshold: float = 0.0,
) -> list[tuple[float, float, int]]:
    """Pure: rank minimal candidates and return sorted ``(primary,
    secondary, id)`` tuples descending.

    With ``similarities`` provided, primary = cosine similarity and
    secondary = tag-overlap score (semantic-primary, tag tiebreaker).
    Candidates with similarity below ``threshold`` are dropped.

    Without ``similarities``, primary = tag-overlap score and secondary
    = 0. This is the legacy path forge has called for a long time.
    """
    tag_set = set(query_tags)
    scored: list[tuple[float, float, int]] = []
    for meta in meta_list:
        tag_score = score_entry(meta, tag_set, mode)
        entry_id = meta.get("id", -1)
        if similarities is not None:
            sim = similarities.get(entry_id, 0.0)
            if sim < threshold:
                continue
            scored.append((sim, tag_score, entry_id))
        else:
            scored.append((tag_score, 0.0, entry_id))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored


def pack_within_budget(
    scored: list[tuple[float, float, int]],
    full_entries: dict[int, dict[str, Any]],
    token_budget: int,
    *,
    annotate_similarity: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Pure: pack entries within a token budget in score order.

    ``full_entries`` maps id → full entry dict (with title, content,
    tags, etc.). IDs in ``scored`` but absent from ``full_entries`` are
    skipped — useful when the caller only loaded the top-K by score.

    ``annotate_similarity=True`` adds the primary score as ``similarity``
    to each packed entry (only meaningful when the workflow ran a query).

    The ``embedding`` field is stripped from packed entries — consumers
    don't need it and including it would re-introduce the bytes-on-the-wire
    problem this layout was designed to avoid.
    """
    packed: list[dict[str, Any]] = []
    total_tokens = 0
    for primary, _secondary, entry_id in scored:
        if entry_id not in full_entries:
            continue
        entry = {k: v for k, v in full_entries[entry_id].items() if k != "embedding"}
        entry_text = f"{entry.get('title', '')}\n{entry.get('content', '')}"
        entry_tokens = _estimate_tokens(entry_text)
        if total_tokens + entry_tokens > token_budget:
            continue
        if annotate_similarity:
            entry = {**entry, "similarity": primary}
        packed.append(entry)
        total_tokens += entry_tokens
    return packed, total_tokens


# ---------------------------------------------------------------------------
# Temporal activities
# ---------------------------------------------------------------------------


_MAX_QUERY_ONLY_CANDIDATES = 200
_TOP_K_FULL_LOAD = 50  # Cap on how many full entries score_and_pack loads
_MINIMAL_FIELDS = (
    "id",
    "tags",
    "entry_type",
    "helpful_count",
    "harmful_count",
    "retrieval_count",
)


def _minimize(row: dict[str, Any]) -> dict[str, Any]:
    """Project a candidate row to just the fields needed for ranking.

    Title, content, embedding, timestamps, and other heavy fields are
    loaded later for only the top-N entries that survive ranking.
    """
    return {k: row.get(k) for k in _MINIMAL_FIELDS}


async def fetch_candidates(engine: Engine | None, input_json: str) -> list[dict[str, Any]]:
    """Fetch candidate entries matching the query tags or query string.

    Returns **minimal** dicts (id + ranking fields only). Heavy fields
    are not shipped over the wire — ``score_and_pack`` loads full
    content for only the top-N entries that fit the token budget.
    """
    from pbook.models import RetrievalInput
    from pbook.store import get_entries_by_tags, list_embedded_entries

    inp = RetrievalInput.model_validate_json(input_json)
    logger.info(
        "Fetching candidates: tags=%s mode=%s query=%r",
        inp.tags,
        inp.mode,
        inp.query,
    )

    if engine is None:
        return []

    if inp.tags:
        candidates = get_entries_by_tags(
            engine,
            inp.tags,
            limit=100,
            approved_only=inp.approved_only,
            include_rejected=inp.include_rejected,
        )
    elif inp.query:
        # Query-only: pull a broad pool of entries with embeddings so
        # the semantic step has signal to rank them.
        candidates = list_embedded_entries(
            engine,
            approved_only=inp.approved_only,
            include_rejected=inp.include_rejected,
            limit=_MAX_QUERY_ONLY_CANDIDATES,
        )
    else:
        candidates = []

    minimized = [_minimize(c) for c in candidates]
    logger.info("Found %d candidates", len(minimized))
    return minimized


async def compute_similarities_by_id(engine: Engine | None, input_json: str) -> dict[str, float]:
    """Compute cosine similarity between a query embedding and the
    embeddings of the named entries.

    Loads embeddings from the DB inside the activity so embedding bytes
    never cross the workflow boundary (they're random float32 sequences
    that fail JSON UTF-8 validation otherwise).

    Input JSON:
      - ``query_embedding_b64``: base64-encoded query embedding bytes
      - ``ids``: list of entry IDs to score

    Returns dict mapping ``str(id)`` → similarity. (JSON requires str
    keys; the workflow re-keys to int as needed.) Entries with NULL
    embeddings or IDs not found in the DB are silently absent.
    """
    from pbook.embeddings import decode_embedding
    from pbook.store import cosine_similarities_for_ids

    data = json.loads(input_json)
    query_embedding = decode_embedding(data["query_embedding_b64"])
    ids: list[int] = data.get("ids", [])
    if not ids:
        return {}

    if engine is None:
        return {}

    sims = cosine_similarities_for_ids(engine, query_embedding, ids)
    return {str(entry_id): sim for entry_id, sim in sims.items()}


async def score_and_pack(engine: Engine | None, input_json: str) -> dict[str, Any]:
    """Orchestrate ranking and packing on the activity side.

    1. Score the minimal candidate list with pure ``rank_meta``.
    2. Load full content for the top ``_TOP_K_FULL_LOAD`` entries.
    3. Pack within the token budget in score order with pure
       ``pack_within_budget``.

    Lives in an activity so heavy entry data (title, content, etc.)
    stays out of the workflow boundary; the workflow ferries only
    ranking metadata.

    Input JSON:
      - ``meta``: list of minimal dicts (from ``fetch_candidates``)
      - ``similarities``: dict ``str(id) → float`` or null
      - ``tags``: list[str]
      - ``mode``: ``"create"`` or ``"fix"``
      - ``token_budget``: int
      - ``threshold``: float (cosine floor; 0.0 = no filter)

    Output:
      - ``packed``: list of full entry dicts (no embedding, with
        ``similarity`` annotation when applicable)
      - ``token_count``: total tokens packed
    """
    from pbook.store import get_entries_by_ids

    data = json.loads(input_json)
    meta_list: list[dict[str, Any]] = data["meta"]
    similarities_raw = data.get("similarities")
    similarities: dict[int, float] | None = (
        {int(k): float(v) for k, v in similarities_raw.items()} if similarities_raw else None
    )
    tags: list[str] = data.get("tags", [])
    mode = RetrievalMode(data["mode"])
    token_budget: int = data["token_budget"]
    threshold: float = data.get("threshold", 0.0)

    scored = rank_meta(
        meta_list,
        tags,
        mode,
        similarities=similarities,
        threshold=threshold,
    )
    if not scored:
        return {"packed": [], "token_count": 0}

    if engine is None:
        return {"packed": [], "token_count": 0}

    top_ids = [entry_id for _, _, entry_id in scored[:_TOP_K_FULL_LOAD]]
    full_by_id = {e["id"]: e for e in get_entries_by_ids(engine, top_ids)}

    packed, total_tokens = pack_within_budget(
        scored,
        full_by_id,
        token_budget,
        annotate_similarity=similarities is not None,
    )
    logger.debug(
        "Ranked %d candidates; packed %d within %d-token budget (%d tokens used)",
        len(meta_list),
        len(packed),
        token_budget,
        total_tokens,
    )
    return {"packed": packed, "token_count": total_tokens}


async def record_retrieval_event(engine: Engine | None, entry_ids_json: str) -> None:
    """Record that entries were served in a retrieval result.

    Accepts JSON-serialized list of entry IDs. Increments
    retrieval_count for each entry. Failures are logged but do not
    propagate.
    """
    from pbook.store import record_retrieval

    entry_ids = json.loads(entry_ids_json)
    if not entry_ids:
        return

    if engine is None:
        return

    record_retrieval(engine, entry_ids)
    logger.info("Recording retrieval for %d entries", len(entry_ids))
