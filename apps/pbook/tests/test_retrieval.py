"""Tests for retrieval activities and workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from pbook.activities.retrieval import (
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

    def test_api_doc_boost_in_create(self):
        curated = {"tags_json": '["lib:sqlalchemy"]', "entry_type": "curated"}
        api_doc = {"tags_json": '["lib:sqlalchemy"]', "entry_type": "api_doc"}

        curated_score = score_entry(curated, {"lib:sqlalchemy"}, RetrievalMode.CREATE)
        api_doc_score = score_entry(api_doc, {"lib:sqlalchemy"}, RetrievalMode.CREATE)
        assert api_doc_score > curated_score

    def test_pitfall_boost_in_fix(self):
        curated = {"tags_json": '["project:forge"]', "entry_type": "curated"}
        pitfall = {"tags_json": '["project:forge"]', "entry_type": "pitfall"}

        curated_score = score_entry(curated, {"project:forge"}, RetrievalMode.FIX)
        pitfall_score = score_entry(pitfall, {"project:forge"}, RetrievalMode.FIX)
        assert pitfall_score > curated_score


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


# ---------------------------------------------------------------------------
# RetrievalWorkflow
# ---------------------------------------------------------------------------


_WORKFLOW_ACTIVITIES = [fetch_candidates, record_retrieval_event]


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
