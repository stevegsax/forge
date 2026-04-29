"""Tests for retrieval activities and workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from pbook.activities.retrieval import (
    compute_similarities,
    fetch_candidates,
    rank_and_pack,
    record_retrieval_event,
    score_entry,
)
from pbook.models import PlaybookEntry, RetrievalInput, RetrievalMode
from pbook.store import (
    build_entry_dict,
    get_engine,
    get_entry_by_id,
    run_migrations,
    save_entries,
)
from pbook.worker import PBOOK_TASK_QUEUE
from pbook.workflows.retrieval import RetrievalWorkflow

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
async def env():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        yield env


def _setup_db(tmp_path: Path):
    db_path = tmp_path / "test.db"
    run_migrations(db_path)
    return get_engine(db_path)


def _make_entry(title: str, tags: list[str], entry_type: str = "curated") -> dict:
    return build_entry_dict(PlaybookEntry(
        title=title,
        content=f"Content for {title}",
        tags=tags,
        entry_type=entry_type,
    ))


# ---------------------------------------------------------------------------
# score_entry
# ---------------------------------------------------------------------------


class TestScoreEntry:
    def test_no_overlap_returns_zero(self):
        entry = {"tags_json": '["lang:python"]', "entry_type": "curated"}
        assert score_entry(entry, {"lang:go"}, RetrievalMode.CREATE) == 0.0

    def test_single_overlap(self):
        entry = {"tags_json": '["lang:python"]', "entry_type": "curated"}
        score = score_entry(entry, {"lang:python"}, RetrievalMode.CREATE)
        assert score > 0

    def test_create_mode_boosts_general(self):
        general = {"tags_json": '["lang:python"]', "entry_type": "curated"}
        extracted = {"tags_json": '["project:forge"]', "entry_type": "pitfall"}

        general_score = score_entry(
            general, {"lang:python", "project:forge"}, RetrievalMode.CREATE,
        )
        extracted_score = score_entry(
            extracted, {"lang:python", "project:forge"}, RetrievalMode.CREATE,
        )
        assert general_score > extracted_score

    def test_fix_mode_boosts_extracted(self):
        general = {"tags_json": '["lang:python"]', "entry_type": "curated"}
        extracted = {"tags_json": '["project:forge"]', "entry_type": "pitfall"}

        general_score = score_entry(
            general, {"lang:python", "project:forge"}, RetrievalMode.FIX,
        )
        extracted_score = score_entry(
            extracted, {"lang:python", "project:forge"}, RetrievalMode.FIX,
        )
        assert extracted_score > general_score

    def test_pitfall_boost_in_fix(self):
        curated = {"tags_json": '["project:forge"]', "entry_type": "curated"}
        pitfall = {"tags_json": '["project:forge"]', "entry_type": "pitfall"}

        curated_score = score_entry(curated, {"project:forge"}, RetrievalMode.FIX)
        pitfall_score = score_entry(pitfall, {"project:forge"}, RetrievalMode.FIX)
        assert pitfall_score > curated_score

    def test_helpful_entry_boosted(self):
        base = {"tags_json": '["lang:python"]', "entry_type": "curated"}
        no_feedback = {**base}
        helpful = {**base, "helpful_count": 5, "harmful_count": 0, "retrieval_count": 5}

        base_score = score_entry(no_feedback, {"lang:python"}, RetrievalMode.CREATE)
        boosted_score = score_entry(helpful, {"lang:python"}, RetrievalMode.CREATE)
        assert boosted_score > base_score

    def test_harmful_entry_penalized(self):
        base = {"tags_json": '["lang:python"]', "entry_type": "curated"}
        no_feedback = {**base}
        harmful = {**base, "helpful_count": 0, "harmful_count": 5, "retrieval_count": 5}

        base_score = score_entry(no_feedback, {"lang:python"}, RetrievalMode.CREATE)
        penalized_score = score_entry(harmful, {"lang:python"}, RetrievalMode.CREATE)
        assert penalized_score < base_score

    def test_insufficient_retrievals_ignored(self):
        base = {"tags_json": '["lang:python"]', "entry_type": "curated"}
        no_feedback = {**base}
        # Only 2 retrievals — below threshold of 3
        few = {**base, "helpful_count": 2, "harmful_count": 0, "retrieval_count": 2}

        base_score = score_entry(no_feedback, {"lang:python"}, RetrievalMode.CREATE)
        few_score = score_entry(few, {"lang:python"}, RetrievalMode.CREATE)
        assert base_score == few_score

    def test_mixed_feedback(self):
        base = {"tags_json": '["lang:python"]', "entry_type": "curated"}
        no_feedback = {**base}
        mixed = {**base, "helpful_count": 3, "harmful_count": 1, "retrieval_count": 5}

        base_score = score_entry(no_feedback, {"lang:python"}, RetrievalMode.CREATE)
        mixed_score = score_entry(mixed, {"lang:python"}, RetrievalMode.CREATE)
        # Net positive (3-1)/5 = 0.4 ratio → should boost
        assert mixed_score > base_score

    def test_zero_counters_unchanged(self):
        base = {"tags_json": '["lang:python"]', "entry_type": "curated"}
        no_counters = {**base}
        zero_counters = {**base, "helpful_count": 0, "harmful_count": 0, "retrieval_count": 0}

        score_without = score_entry(no_counters, {"lang:python"}, RetrievalMode.CREATE)
        score_with = score_entry(zero_counters, {"lang:python"}, RetrievalMode.CREATE)
        assert score_without == score_with


# ---------------------------------------------------------------------------
# rank_and_pack
# ---------------------------------------------------------------------------


class TestRankAndPack:
    def test_packs_within_budget(self):
        base = {"tags_json": '["lang:python"]', "entry_type": "curated"}
        candidates = [
            {"title": "A", "content": "x" * 400, **base},
            {"title": "B", "content": "y" * 400, **base},
            {"title": "C", "content": "z" * 400, **base},
        ]
        # Budget for ~2 entries (each ~100 tokens)
        packed, tokens = rank_and_pack(candidates, ["lang:python"], RetrievalMode.CREATE, 250)
        assert len(packed) == 2
        assert tokens <= 250

    def test_ranks_by_score(self):
        candidates = [
            {
                "title": "Low", "content": "c",
                "tags_json": '["lang:python"]', "entry_type": "curated",
            },
            {
                "title": "High", "content": "c",
                "tags_json": '["lang:python", "lib:sqlalchemy"]',
                "entry_type": "curated",
            },
        ]
        packed, _ = rank_and_pack(
            candidates, ["lang:python", "lib:sqlalchemy"], RetrievalMode.CREATE, 5000,
        )
        assert packed[0]["title"] == "High"

    def test_empty_candidates(self):
        packed, tokens = rank_and_pack([], ["lang:python"], RetrievalMode.CREATE, 5000)
        assert packed == []
        assert tokens == 0

    def test_semantic_primary_when_similarities_provided(self):
        """When similarities are passed in, ordering is similarity-first."""
        base = {"tags_json": '["lang:python"]', "entry_type": "curated"}
        candidates = [
            {"id": 1, "title": "low-sim", "content": "c", **base},
            {"id": 2, "title": "high-sim", "content": "c", **base},
        ]
        # Even though both have identical tag overlap, the one with
        # higher similarity must come first.
        packed, _ = rank_and_pack(
            candidates, ["lang:python"], RetrievalMode.CREATE, 5000,
            similarities={1: 0.5, 2: 0.95},
        )
        assert packed[0]["id"] == 2
        assert packed[0]["similarity"] == 0.95

    def test_threshold_filters_low_similarity(self):
        base = {"tags_json": '["lang:python"]', "entry_type": "curated"}
        candidates = [
            {"id": 1, "title": "below", "content": "c", **base},
            {"id": 2, "title": "above", "content": "c", **base},
        ]
        packed, _ = rank_and_pack(
            candidates, [], RetrievalMode.CREATE, 5000,
            similarities={1: 0.4, 2: 0.9},
            threshold=0.6,
        )
        assert len(packed) == 1
        assert packed[0]["id"] == 2

    def test_encode_candidate_embedding_handles_bytes_list_str_none(self):
        from pbook.workflows.retrieval import _encode_candidate_embedding

        assert _encode_candidate_embedding(None) == ""
        assert _encode_candidate_embedding("") == ""
        # bytes → base64
        assert _encode_candidate_embedding(b"\x01\x02") == "AQI="
        # list of ints (Temporal-serialized bytes) → base64
        assert _encode_candidate_embedding([1, 2]) == "AQI="
        # str input passes through (assumed already base64)
        assert _encode_candidate_embedding("AQI=") == "AQI="
        # unknown type falls back to empty
        assert _encode_candidate_embedding({"unexpected": True}) == ""

    @pytest.mark.asyncio
    async def test_compute_similarities_activity(self):
        """Direct unit test of the activity function (not via Temporal)."""
        import base64
        import json
        import struct

        from pbook.activities.retrieval import compute_similarities

        emb_a = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        emb_b = struct.pack("4f", 0.0, 1.0, 0.0, 0.0)
        query = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)

        result = await compute_similarities(json.dumps({
            "query_embedding_b64": base64.b64encode(query).decode("ascii"),
            "candidates": [
                {"id": 1, "embedding_b64": base64.b64encode(emb_a).decode("ascii")},
                {"id": 2, "embedding_b64": base64.b64encode(emb_b).decode("ascii")},
                {"id": 3, "embedding_b64": ""},  # missing — skipped
            ],
        }))
        assert result["1"] > 0.99   # aligned with query
        assert result["2"] < 0.01   # orthogonal
        assert "3" not in result

    def test_tag_only_unchanged_when_no_similarities(self):
        """Regression: existing forge consumers pass no similarities and
        must see the same scoring behavior as before this change."""
        candidates = [
            {
                "id": 1, "title": "Low", "content": "c",
                "tags_json": '["lang:python"]', "entry_type": "curated",
            },
            {
                "id": 2, "title": "High", "content": "c",
                "tags_json": '["lang:python", "lib:sqlalchemy"]',
                "entry_type": "curated",
            },
        ]
        packed, _ = rank_and_pack(
            candidates, ["lang:python", "lib:sqlalchemy"],
            RetrievalMode.CREATE, 5000,
        )
        assert packed[0]["id"] == 2
        # No similarity attached when ranking was tag-only.
        assert "similarity" not in packed[0]


# ---------------------------------------------------------------------------
# RetrievalWorkflow
# ---------------------------------------------------------------------------


_WORKFLOW_ACTIVITIES = [fetch_candidates, compute_similarities, record_retrieval_event]


class TestRetrievalWorkflow:
    @pytest.mark.asyncio
    async def test_retrieval_returns_entries(
        self, env: WorkflowEnvironment, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        save_entries(engine, [
            _make_entry("Python advice", ["lang:python"]),
            _make_entry("Go advice", ["lang:go"]),
        ])

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
        self, env: WorkflowEnvironment, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        # Create entries that total more than a tiny budget
        for i in range(10):
            save_entries(engine, [
                _make_entry(f"Entry {i}", ["lang:python"]),
            ])

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
        self, env: WorkflowEnvironment, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
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
        self, env: WorkflowEnvironment, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        save_entries(engine, [
            _make_entry("Python advice", ["lang:python"]),
        ])

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
        self, env: WorkflowEnvironment, tmp_path: Path, monkeypatch,
    ) -> None:
        """When `query` is non-empty, the workflow embeds it via llm_embed
        and ranks candidates by cosine similarity. Mocking llm_embed lets
        us test the wiring without needing OpenAI."""
        import base64
        import struct

        from temporalio import activity

        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)

        # Two entries with deliberately different stored embeddings.
        # We'll mock the query embedding so it aligns with the second.
        emb_a = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        emb_b = struct.pack("4f", 0.0, 1.0, 0.0, 0.0)
        from pbook.models import EntryType, PlaybookEntry

        save_entries(engine, [
            build_entry_dict(PlaybookEntry(
                title="A entry", content="content A",
                tags=["lang:python"], entry_type=EntryType.CURATED,
                embedding=emb_a,
            )),
            build_entry_dict(PlaybookEntry(
                title="B entry", content="content B",
                tags=["lang:python"], entry_type=EntryType.CURATED,
                embedding=emb_b,
            )),
        ])

        # Query embedding aligned with B
        query_b64 = base64.b64encode(emb_b).decode("ascii")

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
