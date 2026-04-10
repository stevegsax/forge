"""Tests for maintenance activities and group_similar_entries.

Extends test_maintenance.py which covers identify_prune_candidates.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from sax_llm.models import ProviderResponse

from pbook.activities.maintenance import (
    consolidate_entries_llm,
    fetch_all_entries_for_maintenance,
    group_similar_entries,
    prune_entries,
)
from pbook.llm import ConsolidationResult, set_provider
from pbook.store import get_engine, run_migrations, save_entries

if TYPE_CHECKING:
    from pathlib import Path


def _embedding(values: list[float]) -> bytes:
    return np.array(values, dtype=np.float32).tobytes()


def _setup_db(tmp_path: Path):
    db_path = tmp_path / "test.db"
    run_migrations(db_path)
    return get_engine(db_path)


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
    async def test_returns_entries(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        save_entries(engine, [{
            "title": "Test",
            "content": "Content",
            "tags_json": "[]",
            "entry_type": "curated",
            "source_project": "",
            "source_task_id": "",
            "needs_review": False,
            "helpful_count": 0,
            "harmful_count": 0,
            "retrieval_count": 0,
            "embedding": None,
        }])

        result = await fetch_all_entries_for_maintenance()
        assert len(result) == 1
        assert result[0]["title"] == "Test"

    @pytest.mark.asyncio
    async def test_empty_when_no_db(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "nonexistent.db"))
        result = await fetch_all_entries_for_maintenance()
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_when_disabled(self, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", "")
        result = await fetch_all_entries_for_maintenance()
        assert result == []


# ---------------------------------------------------------------------------
# prune_entries activity
# ---------------------------------------------------------------------------


class TestPruneEntries:
    @pytest.mark.asyncio
    async def test_deletes_entries(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        save_entries(engine, [
            {
                "title": f"Entry {i}",
                "content": "Content",
                "tags_json": "[]",
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
        ])

        count = await prune_entries([1, 2])
        assert count == 2

        # Only entry 3 remains
        from pbook.store import list_all_entries
        remaining = list_all_entries(engine)
        assert len(remaining) == 1
        assert remaining[0]["id"] == 3

    @pytest.mark.asyncio
    async def test_empty_list(self):
        count = await prune_entries([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_disabled_store(self, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", "")
        count = await prune_entries([1, 2])
        assert count == 0


# ---------------------------------------------------------------------------
# consolidate_entries_llm activity
# ---------------------------------------------------------------------------


class TestConsolidateEntriesLlm:
    @pytest.fixture(autouse=True)
    def _cleanup_provider(self):
        yield
        from pbook.llm import reset_provider
        reset_provider()

    @pytest.mark.asyncio
    async def test_merges_entries(self):
        mock_response = ProviderResponse(
            tool_input={
                "merged_title": "Combined Lesson",
                "merged_content": "Unified content",
                "merged_tags": ["lang:python", "tool:pytest"],
            },
            model_name="test",
            input_tokens=0,
            output_tokens=0,
            raw_response_json="{}",
        )
        provider = MagicMock()
        provider.build_request_params.return_value = {}
        provider.call = AsyncMock(return_value=mock_response)
        set_provider(provider)

        entries = [
            {"title": "Lesson A", "content": "Content A", "tags_json": '["lang:python"]'},
            {"title": "Lesson B", "content": "Content B", "tags_json": '["tool:pytest"]'},
        ]
        result_json = await consolidate_entries_llm(json.dumps(entries))
        result = ConsolidationResult.model_validate_json(result_json)

        assert result.merged_title == "Combined Lesson"
        assert result.merged_content == "Unified content"
        assert "lang:python" in result.merged_tags

    @pytest.mark.asyncio
    async def test_empty_entries(self):
        result_json = await consolidate_entries_llm("[]")
        result = ConsolidationResult.model_validate_json(result_json)
        assert result.merged_title == ""
        assert result.merged_content == ""
