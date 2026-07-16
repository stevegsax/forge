"""Tests for retrieval activities and workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from pbook.activities.retrieval import (
    compute_similarities_by_id,
    fetch_candidates,
    pack_within_budget,
    rank_meta,
    record_retrieval_event,
    score_and_pack,
    score_entry,
)
from pbook.embeddings import encode_embedding
from pbook.models import PlaybookEntry, RetrievalInput, RetrievalMode
from pbook.store import (
    build_entry_dict,
    get_entry_by_id,
    save_entries,
)
from pbook.worker import PBOOK_TASK_QUEUE
from pbook.workflows.retrieval import RetrievalWorkflow
from tests.conftest import make_embedding, setup_db

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
async def env():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        yield env


def _setup_db(_tmp_path: Path | None = None):
    """Return the session test engine (migrations already applied)."""
    return setup_db()[0]


def _make_entry(title: str, tags: list[str], entry_type: str = "curated") -> dict:
    return build_entry_dict(
        PlaybookEntry(
            title=title,
            content=f"Content for {title}",
            tags=tags,
            entry_type=entry_type,
        )
    )


# ---------------------------------------------------------------------------
# score_entry
# ---------------------------------------------------------------------------


class TestScoreEntry:
    def test_no_overlap_returns_zero(self):
        entry = {"tags": ["lang:python"], "entry_type": "curated"}
        assert score_entry(entry, {"lang:go"}, RetrievalMode.CREATE) == 0.0

    def test_single_overlap(self):
        entry = {"tags": ["lang:python"], "entry_type": "curated"}
        score = score_entry(entry, {"lang:python"}, RetrievalMode.CREATE)
        assert score > 0

    def test_create_mode_boosts_general(self):
        general = {"tags": ["lang:python"], "entry_type": "curated"}
        extracted = {"tags": ["project:forge"], "entry_type": "pitfall"}

        general_score = score_entry(
            general,
            {"lang:python", "project:forge"},
            RetrievalMode.CREATE,
        )
        extracted_score = score_entry(
            extracted,
            {"lang:python", "project:forge"},
            RetrievalMode.CREATE,
        )
        assert general_score > extracted_score

    def test_fix_mode_boosts_extracted(self):
        general = {"tags": ["lang:python"], "entry_type": "curated"}
        extracted = {"tags": ["project:forge"], "entry_type": "pitfall"}

        general_score = score_entry(
            general,
            {"lang:python", "project:forge"},
            RetrievalMode.FIX,
        )
        extracted_score = score_entry(
            extracted,
            {"lang:python", "project:forge"},
            RetrievalMode.FIX,
        )
        assert extracted_score > general_score

    def test_pitfall_boost_in_fix(self):
        curated = {"tags": ["project:forge"], "entry_type": "curated"}
        pitfall = {"tags": ["project:forge"], "entry_type": "pitfall"}

        curated_score = score_entry(curated, {"project:forge"}, RetrievalMode.FIX)
        pitfall_score = score_entry(pitfall, {"project:forge"}, RetrievalMode.FIX)
        assert pitfall_score > curated_score

    def test_helpful_entry_boosted(self):
        base = {"tags": ["lang:python"], "entry_type": "curated"}
        no_feedback = {**base}
        helpful = {**base, "helpful_count": 5, "harmful_count": 0, "retrieval_count": 5}

        base_score = score_entry(no_feedback, {"lang:python"}, RetrievalMode.CREATE)
        boosted_score = score_entry(helpful, {"lang:python"}, RetrievalMode.CREATE)
        assert boosted_score > base_score

    def test_harmful_entry_penalized(self):
        base = {"tags": ["lang:python"], "entry_type": "curated"}
        no_feedback = {**base}
        harmful = {**base, "helpful_count": 0, "harmful_count": 5, "retrieval_count": 5}

        base_score = score_entry(no_feedback, {"lang:python"}, RetrievalMode.CREATE)
        penalized_score = score_entry(harmful, {"lang:python"}, RetrievalMode.CREATE)
        assert penalized_score < base_score

    def test_insufficient_retrievals_ignored(self):
        base = {"tags": ["lang:python"], "entry_type": "curated"}
        no_feedback = {**base}
        # Only 2 retrievals — below threshold of 3
        few = {**base, "helpful_count": 2, "harmful_count": 0, "retrieval_count": 2}

        base_score = score_entry(no_feedback, {"lang:python"}, RetrievalMode.CREATE)
        few_score = score_entry(few, {"lang:python"}, RetrievalMode.CREATE)
        assert base_score == few_score

    def test_mixed_feedback(self):
        base = {"tags": ["lang:python"], "entry_type": "curated"}
        no_feedback = {**base}
        mixed = {**base, "helpful_count": 3, "harmful_count": 1, "retrieval_count": 5}

        base_score = score_entry(no_feedback, {"lang:python"}, RetrievalMode.CREATE)
        mixed_score = score_entry(mixed, {"lang:python"}, RetrievalMode.CREATE)
        # Net positive (3-1)/5 = 0.4 ratio → should boost
        assert mixed_score > base_score

    def test_zero_counters_unchanged(self):
        base = {"tags": ["lang:python"], "entry_type": "curated"}
        no_counters = {**base}
        zero_counters = {**base, "helpful_count": 0, "harmful_count": 0, "retrieval_count": 0}

        score_without = score_entry(no_counters, {"lang:python"}, RetrievalMode.CREATE)
        score_with = score_entry(zero_counters, {"lang:python"}, RetrievalMode.CREATE)
        assert score_without == score_with


# ---------------------------------------------------------------------------
# rank_meta (pure)
# ---------------------------------------------------------------------------


class TestRankMeta:
    """rank_meta operates on minimal candidate dicts (id + ranking fields)
    and returns a sorted list of (primary, secondary, id) tuples."""

    def test_ranks_by_tag_score(self):
        meta = [
            {"id": 1, "tags": ["lang:python"], "entry_type": "curated"},
            {"id": 2, "tags": ["lang:python", "lib:sqlalchemy"], "entry_type": "curated"},
        ]
        scored = rank_meta(
            meta,
            ["lang:python", "lib:sqlalchemy"],
            RetrievalMode.CREATE,
        )
        assert scored[0][2] == 2  # higher overlap ranks first

    def test_empty_meta(self):
        assert rank_meta([], ["lang:python"], RetrievalMode.CREATE) == []

    def test_semantic_primary_when_similarities_provided(self):
        """With similarities, ordering is similarity-first; tag overlap is
        the tiebreaker."""
        base = {"tags": ["lang:python"], "entry_type": "curated"}
        meta = [
            {"id": 1, **base},
            {"id": 2, **base},
        ]
        scored = rank_meta(
            meta,
            ["lang:python"],
            RetrievalMode.CREATE,
            similarities={1: 0.5, 2: 0.95},
        )
        assert scored[0][2] == 2
        assert scored[0][0] == 0.95

    def test_threshold_filters_low_similarity(self):
        base = {"tags": ["lang:python"], "entry_type": "curated"}
        meta = [{"id": 1, **base}, {"id": 2, **base}]
        scored = rank_meta(
            meta,
            [],
            RetrievalMode.CREATE,
            similarities={1: 0.4, 2: 0.9},
            threshold=0.6,
        )
        assert len(scored) == 1
        assert scored[0][2] == 2

    def test_no_similarities_returns_secondary_zero(self):
        """When no similarities are provided, the secondary (tiebreaker)
        score is 0; the primary carries the tag-overlap score."""
        meta = [
            {"id": 1, "tags": ["lang:python"], "entry_type": "curated"},
        ]
        scored = rank_meta(meta, ["lang:python"], RetrievalMode.CREATE)
        assert scored[0][1] == 0.0
        assert scored[0][0] > 0


# ---------------------------------------------------------------------------
# pack_within_budget (pure)
# ---------------------------------------------------------------------------


class TestPackWithinBudget:
    """pack_within_budget walks scored entries in order, fetching content
    from full_entries (id-keyed dict) and packing until the token budget
    is exhausted."""

    def test_packs_within_budget(self):
        scored = [(1.0, 0.0, 1), (0.9, 0.0, 2), (0.8, 0.0, 3)]
        full = {
            1: {"id": 1, "title": "A", "content": "x" * 400},
            2: {"id": 2, "title": "B", "content": "y" * 400},
            3: {"id": 3, "title": "C", "content": "z" * 400},
        }
        # Budget for ~2 entries (each ~100 tokens after title)
        packed, tokens = pack_within_budget(scored, full, 250, annotate_similarity=False)
        assert len(packed) == 2
        assert tokens <= 250

    def test_skips_ids_not_in_full_entries(self):
        """Caller may load only top-K full entries; lower-ranked IDs in
        scored should be skipped silently."""
        scored = [(0.9, 0.0, 1), (0.5, 0.0, 999)]
        full = {1: {"id": 1, "title": "A", "content": "c"}}
        packed, _ = pack_within_budget(scored, full, 5000, annotate_similarity=False)
        assert len(packed) == 1
        assert packed[0]["id"] == 1

    def test_strips_embedding_from_packed_output(self):
        """Embedding bytes don't belong in retrieval results — consumers
        don't use them and shipping bytes re-introduces the JSON-encoder
        problem this layout was built to avoid."""
        scored = [(1.0, 0.0, 1)]
        full = {1: {"id": 1, "title": "A", "content": "c", "embedding": b"\x01\x02"}}
        packed, _ = pack_within_budget(scored, full, 5000, annotate_similarity=False)
        assert "embedding" not in packed[0]

    def test_annotates_similarity_when_requested(self):
        scored = [(0.85, 1.5, 1)]
        full = {1: {"id": 1, "title": "A", "content": "c"}}
        packed, _ = pack_within_budget(scored, full, 5000, annotate_similarity=True)
        assert packed[0]["similarity"] == 0.85

    def test_no_similarity_key_when_not_annotated(self):
        scored = [(2.0, 0.0, 1)]
        full = {1: {"id": 1, "title": "A", "content": "c"}}
        packed, _ = pack_within_budget(scored, full, 5000, annotate_similarity=False)
        assert "similarity" not in packed[0]

    def test_empty_scored_returns_empty(self):
        packed, tokens = pack_within_budget([], {}, 5000, annotate_similarity=False)
        assert packed == []
        assert tokens == 0


# ---------------------------------------------------------------------------
# compute_similarities_by_id (activity, DB-backed)
# ---------------------------------------------------------------------------


class TestComputeSimilaritiesByID:
    """The activity loads embeddings from the DB itself so the workflow
    body never ferries embedding bytes."""

    @pytest.mark.asyncio
    async def test_aligned_query_scores_high(self, tmp_path: Path):
        import json as _json

        from pbook.models import EntryType, PlaybookEntry

        engine = _setup_db(tmp_path)

        emb_a = make_embedding(1.0, 0.0, 0.0, 0.0)
        emb_b = make_embedding(0.0, 1.0, 0.0, 0.0)
        save_entries(
            engine,
            [
                build_entry_dict(
                    PlaybookEntry(
                        title="A",
                        content="A",
                        tags=["lang:python"],
                        entry_type=EntryType.CURATED,
                        embedding=emb_a,
                    )
                ),
                build_entry_dict(
                    PlaybookEntry(
                        title="B",
                        content="B",
                        tags=["lang:python"],
                        entry_type=EntryType.CURATED,
                        embedding=emb_b,
                    )
                ),
            ],
        )

        # Query embedding aligned with A.
        query = make_embedding(1.0, 0.0, 0.0, 0.0)
        result = await compute_similarities_by_id(
            _json.dumps(
                {
                    "query_embedding_b64": encode_embedding(query),
                    "ids": [1, 2],
                }
            )
        )
        assert result["1"] > 0.99  # aligned
        assert result["2"] < 0.01  # orthogonal

    @pytest.mark.asyncio
    async def test_skips_entries_without_embeddings(self, tmp_path: Path):
        import json as _json

        from pbook.models import EntryType, PlaybookEntry

        engine = _setup_db(tmp_path)
        save_entries(
            engine,
            [
                build_entry_dict(
                    PlaybookEntry(
                        title="No emb",
                        content="x",
                        tags=["lang:python"],
                        entry_type=EntryType.CURATED,
                    )
                ),
            ],
        )

        query = make_embedding(1.0, 0.0, 0.0, 0.0)
        result = await compute_similarities_by_id(
            _json.dumps(
                {
                    "query_embedding_b64": encode_embedding(query),
                    "ids": [1],
                }
            )
        )
        assert "1" not in result
        assert result == {}

    @pytest.mark.asyncio
    async def test_empty_ids_returns_empty_dict(self, tmp_path: Path):
        import json as _json

        _setup_db(tmp_path)

        query = make_embedding(1.0, 0.0, 0.0, 0.0)
        result = await compute_similarities_by_id(
            _json.dumps(
                {
                    "query_embedding_b64": encode_embedding(query),
                    "ids": [],
                }
            )
        )
        assert result == {}


# ---------------------------------------------------------------------------
# fetch_candidates wire format
# ---------------------------------------------------------------------------


class TestFetchCandidatesWireFormat:
    """fetch_candidates returns minimal dicts (id + ranking fields) to
    keep the activity-result payload small. Heavy fields (title, content,
    embedding, timestamps) are not on the wire — they are loaded later
    by score_and_pack for the top-N entries that fit the token budget.

    This also avoids the pydantic-to-json bytes issue entirely: there is
    no embedding in the result, so there are no raw float32 bytes for
    the JSON encoder to choke on."""

    _MINIMAL_KEYS: ClassVar[set[str]] = {
        "id",
        "tags",
        "entry_type",
        "helpful_count",
        "harmful_count",
        "retrieval_count",
    }

    @pytest.mark.asyncio
    async def test_query_only_branch_returns_minimal_dicts(
        self,
        tmp_path: Path,
    ) -> None:
        from pbook.models import EntryType, PlaybookEntry

        engine = _setup_db(tmp_path)
        emb = make_embedding(1.0, 0.0, 0.0, 0.0)
        save_entries(
            engine,
            [
                build_entry_dict(
                    PlaybookEntry(
                        title="A",
                        content="content with stuff",
                        tags=["lang:python"],
                        entry_type=EntryType.CURATED,
                        embedding=emb,
                    )
                ),
            ],
        )

        result = await fetch_candidates(
            RetrievalInput(query="anything", token_budget=5000).model_dump_json(),
        )

        assert len(result) == 1
        assert set(result[0].keys()) == self._MINIMAL_KEYS, (
            f"fetch_candidates must return only ranking fields; got {set(result[0])}"
        )
        # Heavy fields explicitly absent.
        assert "embedding" not in result[0]
        assert "title" not in result[0]
        assert "content" not in result[0]

    @pytest.mark.asyncio
    async def test_tag_branch_returns_minimal_dicts(
        self,
        tmp_path: Path,
    ) -> None:
        """Same minimal contract for the tag branch — forge consumers
        get the same wire shape; full content is loaded by score_and_pack."""
        from pbook.models import EntryType, PlaybookEntry

        engine = _setup_db(tmp_path)
        emb = make_embedding(0.0, 1.0, 0.0, 0.0)
        save_entries(
            engine,
            [
                build_entry_dict(
                    PlaybookEntry(
                        title="B",
                        content="content",
                        tags=["lang:python"],
                        entry_type=EntryType.CURATED,
                        embedding=emb,
                    )
                ),
            ],
        )

        result = await fetch_candidates(
            RetrievalInput(
                tags=["lang:python"],
                token_budget=5000,
            ).model_dump_json(),
        )

        assert len(result) == 1
        assert set(result[0].keys()) == self._MINIMAL_KEYS
        assert "embedding" not in result[0]


# ---------------------------------------------------------------------------
# RetrievalWorkflow
# ---------------------------------------------------------------------------


_WORKFLOW_ACTIVITIES = [
    fetch_candidates,
    compute_similarities_by_id,
    score_and_pack,
    record_retrieval_event,
]


class TestRetrievalWorkflow:
    @pytest.mark.asyncio
    async def test_retrieval_returns_entries(
        self,
        env: WorkflowEnvironment,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        engine = _setup_db(tmp_path)
        save_entries(
            engine,
            [
                _make_entry("Python advice", ["lang:python"]),
                _make_entry("Go advice", ["lang:go"]),
            ],
        )

        async with Worker(
            env.client,
            task_queue=PBOOK_TASK_QUEUE,
            workflows=[RetrievalWorkflow],
            activities=_WORKFLOW_ACTIVITIES,
        ):
            result = await env.client.execute_workflow(
                RetrievalWorkflow.run,
                RetrievalInput(tags=["lang:python"], token_budget=5000),
                id="test-retrieval-1",
                task_queue=PBOOK_TASK_QUEUE,
            )

        assert result.total_candidates == 1
        assert len(result.entries) == 1
        assert result.entries[0]["title"] == "Python advice"

    @pytest.mark.asyncio
    async def test_retrieval_respects_token_budget(
        self,
        env: WorkflowEnvironment,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        engine = _setup_db(tmp_path)
        # Create entries that total more than a tiny budget
        for i in range(10):
            save_entries(
                engine,
                [
                    _make_entry(f"Entry {i}", ["lang:python"]),
                ],
            )

        async with Worker(
            env.client,
            task_queue=PBOOK_TASK_QUEUE,
            workflows=[RetrievalWorkflow],
            activities=_WORKFLOW_ACTIVITIES,
        ):
            result = await env.client.execute_workflow(
                RetrievalWorkflow.run,
                RetrievalInput(tags=["lang:python"], token_budget=50),
                id="test-retrieval-budget",
                task_queue=PBOOK_TASK_QUEUE,
            )

        assert result.total_candidates == 10
        assert len(result.entries) < 10
        assert result.token_count <= 50

    @pytest.mark.asyncio
    async def test_retrieval_empty_store(
        self,
        env: WorkflowEnvironment,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        _setup_db(tmp_path)

        async with Worker(
            env.client,
            task_queue=PBOOK_TASK_QUEUE,
            workflows=[RetrievalWorkflow],
            activities=_WORKFLOW_ACTIVITIES,
        ):
            result = await env.client.execute_workflow(
                RetrievalWorkflow.run,
                RetrievalInput(tags=["lang:python"]),
                id="test-retrieval-empty",
                task_queue=PBOOK_TASK_QUEUE,
            )

        assert result.entries == []
        assert result.total_candidates == 0

    @pytest.mark.asyncio
    async def test_retrieval_records_served_entries(
        self,
        env: WorkflowEnvironment,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        engine = _setup_db(tmp_path)
        save_entries(
            engine,
            [
                _make_entry("Python advice", ["lang:python"]),
            ],
        )

        async with Worker(
            env.client,
            task_queue=PBOOK_TASK_QUEUE,
            workflows=[RetrievalWorkflow],
            activities=_WORKFLOW_ACTIVITIES,
        ):
            result = await env.client.execute_workflow(
                RetrievalWorkflow.run,
                RetrievalInput(tags=["lang:python"], token_budget=5000),
                id="test-retrieval-record",
                task_queue=PBOOK_TASK_QUEUE,
            )

        assert len(result.entries) == 1
        entry_id = result.entries[0]["id"]
        entry = get_entry_by_id(engine, entry_id)
        assert entry["retrieval_count"] == 1

    @pytest.mark.asyncio
    async def test_retrieval_with_query_uses_semantic_ranking(
        self,
        env: WorkflowEnvironment,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """When `query` is non-empty, the workflow embeds it via llm_embed
        and ranks candidates by cosine similarity. Mocking llm_embed lets
        us test the wiring without needing OpenAI."""
        from temporalio import activity

        engine = _setup_db(tmp_path)

        # Two entries with deliberately different stored embeddings.
        # We'll mock the query embedding so it aligns with the second.
        emb_a = make_embedding(1.0, 0.0, 0.0, 0.0)
        emb_b = make_embedding(0.0, 1.0, 0.0, 0.0)
        from pbook.models import EntryType, PlaybookEntry

        save_entries(
            engine,
            [
                build_entry_dict(
                    PlaybookEntry(
                        title="A entry",
                        content="content A",
                        tags=["lang:python"],
                        entry_type=EntryType.CURATED,
                        embedding=emb_a,
                    )
                ),
                build_entry_dict(
                    PlaybookEntry(
                        title="B entry",
                        content="content B",
                        tags=["lang:python"],
                        entry_type=EntryType.CURATED,
                        embedding=emb_b,
                    )
                ),
            ],
        )

        # Query embedding aligned with B
        query_b64 = encode_embedding(emb_b)

        @activity.defn(name="llm_embed")
        async def mock_embed(_text: str) -> str:
            return query_b64

        async with Worker(
            env.client,
            task_queue=PBOOK_TASK_QUEUE,
            workflows=[RetrievalWorkflow],
            activities=[*_WORKFLOW_ACTIVITIES, mock_embed],
        ):
            result = await env.client.execute_workflow(
                RetrievalWorkflow.run,
                RetrievalInput(
                    tags=["lang:python"],
                    query="anything",
                    token_budget=5000,
                ),
                id="test-retrieval-query",
                task_queue=PBOOK_TASK_QUEUE,
            )

        assert len(result.entries) == 2
        # Semantic-primary: B (similarity 1.0) ranks above A (0.0).
        assert result.entries[0]["title"] == "B entry"
        assert result.entries[0]["similarity"] > 0.99
        assert result.entries[1]["title"] == "A entry"
