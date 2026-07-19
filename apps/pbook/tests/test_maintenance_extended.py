"""Tests for maintenance activities and group_similar_entries.

Extends test_maintenance.py which covers identify_prune_candidates.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from pbook.activities.maintenance import group_similar_entries
from pbook.models import PlaybookEntry
from pbook.roots import StoreActivities
from pbook.store import build_entry_dict, save_entries
from tests.conftest import make_embedding, setup_db

if TYPE_CHECKING:
    from pathlib import Path


def _embedding(values: list[float]) -> list[float]:
    """A plain float vector. group_similar_entries scores in Python, so
    these short vectors need not match the DB's fixed pgvector width."""
    return list(values)


def _setup_db(_tmp_path: Path | None = None):
    """Return the session test engine (migrations already applied)."""
    return setup_db()[0]


# ---------------------------------------------------------------------------
# group_similar_entries (pure function)
# ---------------------------------------------------------------------------


class TestGroupSimilarEntries:
    def test_empty_list(self):
        assert group_similar_entries([]) == []

    def test_no_embeddings(self):
        entries = [
            {"id": 1, "title": "A", "embedding": None},
            {"id": 2, "title": "B", "embedding": None},
        ]
        assert group_similar_entries(entries) == []

    def test_identical_embeddings_grouped(self):
        emb = _embedding([1.0, 0.0, 0.0])
        entries = [
            {"id": 1, "title": "A", "embedding": emb},
            {"id": 2, "title": "B", "embedding": emb},
        ]
        clusters = group_similar_entries(entries, threshold=0.9)
        assert len(clusters) == 1
        assert {e["id"] for e in clusters[0]} == {1, 2}

    def test_orthogonal_embeddings_not_grouped(self):
        entries = [
            {"id": 1, "title": "A", "embedding": _embedding([1.0, 0.0, 0.0])},
            {"id": 2, "title": "B", "embedding": _embedding([0.0, 1.0, 0.0])},
        ]
        clusters = group_similar_entries(entries, threshold=0.5)
        assert clusters == []

    def test_mixed_entries(self):
        """Two similar entries cluster; one dissimilar entry stays out."""
        similar_a = _embedding([1.0, 0.1, 0.0])
        similar_b = _embedding([1.0, 0.0, 0.0])
        different = _embedding([0.0, 0.0, 1.0])
        entries = [
            {"id": 1, "title": "A", "embedding": similar_a},
            {"id": 2, "title": "B", "embedding": similar_b},
            {"id": 3, "title": "C", "embedding": different},
        ]
        clusters = group_similar_entries(entries, threshold=0.9)
        assert len(clusters) == 1
        assert {e["id"] for e in clusters[0]} == {1, 2}

    def test_entries_without_embedding_skipped(self):
        emb = _embedding([1.0, 0.0])
        entries = [
            {"id": 1, "title": "A", "embedding": emb},
            {"id": 2, "title": "B", "embedding": None},
            {"id": 3, "title": "C", "embedding": emb},
        ]
        clusters = group_similar_entries(entries, threshold=0.9)
        assert len(clusters) == 1
        assert {e["id"] for e in clusters[0]} == {1, 3}

    def test_single_entry_no_cluster(self):
        entries = [{"id": 1, "title": "A", "embedding": _embedding([1.0])}]
        assert group_similar_entries(entries) == []

    def test_custom_threshold(self):
        """Lowering threshold allows less-similar entries to cluster."""
        a = _embedding([1.0, 0.5, 0.0])
        b = _embedding([1.0, 0.0, 0.5])
        entries = [
            {"id": 1, "title": "A", "embedding": a},
            {"id": 2, "title": "B", "embedding": b},
        ]
        # High threshold — not clustered
        assert group_similar_entries(entries, threshold=0.95) == []
        # Low threshold — clustered
        clusters = group_similar_entries(entries, threshold=0.5)
        assert len(clusters) == 1


# ---------------------------------------------------------------------------
# fetch_all_entries_for_maintenance activity
# ---------------------------------------------------------------------------


class TestFetchAllEntriesForMaintenance:
    @pytest.mark.asyncio
    async def test_returns_entries(self, tmp_path: Path):
        engine = _setup_db(tmp_path)
        save_entries(
            engine,
            [
                {
                    "title": "Test",
                    "content": "Content",
                    "tags": [],
                    "entry_type": "curated",
                    "source_project": "",
                    "source_task_id": "",
                    "needs_review": False,
                    "helpful_count": 0,
                    "harmful_count": 0,
                    "retrieval_count": 0,
                    "embedding": None,
                }
            ],
        )

        result = await StoreActivities(engine).fetch_all_entries_for_maintenance()
        assert len(result) == 1
        assert result[0]["title"] == "Test"
        # Embeddings are stripped before crossing the wire.
        assert "embedding" not in result[0]

    @pytest.mark.asyncio
    async def test_empty_when_store_empty(self):
        result = await StoreActivities(_setup_db()).fetch_all_entries_for_maintenance()
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_when_disabled(self):
        result = await StoreActivities(None).fetch_all_entries_for_maintenance()
        assert result == []


# ---------------------------------------------------------------------------
# prune_entries activity
# ---------------------------------------------------------------------------


class TestPruneEntries:
    @pytest.mark.asyncio
    async def test_deletes_entries(self, tmp_path: Path):
        engine = _setup_db(tmp_path)
        save_entries(
            engine,
            [
                {
                    "title": f"Entry {i}",
                    "content": "Content",
                    "tags": [],
                    "entry_type": "curated",
                    "source_project": "",
                    "source_task_id": "",
                    "needs_review": False,
                    "helpful_count": 0,
                    "harmful_count": 0,
                    "retrieval_count": 0,
                    "embedding": None,
                }
                for i in range(3)
            ],
        )

        count = await StoreActivities(engine).prune_entries([1, 2])
        assert count == 2

        # Only entry 3 remains
        from pbook.store import list_all_entries

        remaining = list_all_entries(engine)
        assert len(remaining) == 1
        assert remaining[0]["id"] == 3

    @pytest.mark.asyncio
    async def test_empty_list(self):
        count = await StoreActivities(None).prune_entries([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_disabled_store(self):
        count = await StoreActivities(None).prune_entries([1, 2])
        assert count == 0


# ---------------------------------------------------------------------------
# cluster_similar_entries activity (server-side clustering over pgvector)
# ---------------------------------------------------------------------------


class TestClusterSimilarEntries:
    def _seed(self, engine, *vectors):
        save_entries(
            engine,
            [
                build_entry_dict(
                    PlaybookEntry(
                        title=f"E{i}",
                        content="c",
                        tags=[],
                        embedding=vec,
                    )
                )
                for i, vec in enumerate(vectors)
            ],
        )

    @pytest.mark.asyncio
    async def test_groups_similar_ids(self, tmp_path):
        engine = _setup_db(tmp_path)
        # Two near-identical vectors plus one orthogonal.
        self._seed(
            engine,
            make_embedding(1.0, 0.0),
            make_embedding(1.0, 0.0001),
            make_embedding(0.0, 1.0),
        )
        clusters = await StoreActivities(engine).cluster_similar_entries(
            json.dumps({"threshold": 0.9, "exclude_ids": []}),
        )
        assert len(clusters) == 1
        assert set(clusters[0]) == {1, 2}

    @pytest.mark.asyncio
    async def test_excluded_ids_break_the_cluster(self, tmp_path):
        engine = _setup_db(tmp_path)
        self._seed(engine, make_embedding(1.0, 0.0), make_embedding(1.0, 0.0001))
        clusters = await StoreActivities(engine).cluster_similar_entries(
            json.dumps({"threshold": 0.9, "exclude_ids": [2]}),
        )
        assert clusters == []

    @pytest.mark.asyncio
    async def test_disabled_store(self):
        assert (
            await StoreActivities(None).cluster_similar_entries(json.dumps({"threshold": 0.9}))
            == []
        )


# consolidate_entries_llm activity is gone — consolidation now goes
# through the generic llm_chat step. Coverage of the merge prompt
# lives in tests/test_workflow_steps_llm.py and the workflow-level
# tests below (TestMaintenanceWorkflow, when added).
