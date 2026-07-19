"""Tests for export activities and workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from temporalio.worker import Worker

from pbook.activities.export import db_row_to_entry_dict
from pbook.models import PlaybookEntry
from pbook.roots import StoreActivities
from pbook.store import build_entry_dict, save_entries
from pbook.worker import PBOOK_TASK_QUEUE
from pbook.workflows.export import ExportWorkflow
from tests.conftest import setup_db

if TYPE_CHECKING:
    from pathlib import Path

    from temporalio.testing import WorkflowEnvironment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _setup_db(_tmp_path: Path | None = None):
    """Return the session test engine (migrations already applied)."""
    return setup_db()[0]


def _export_activities(engine):
    """The two ExportWorkflow store activities, bound to one engine."""
    store = StoreActivities(engine)
    return [store.fetch_entry_ids, store.export_single_entry]


# ---------------------------------------------------------------------------
# db_row_to_entry_dict
# ---------------------------------------------------------------------------


class TestDbRowToEntryDict:
    def test_basic(self):
        row = {
            "id": 1,
            "title": "Test",
            "content": "Content",
            "tags": ["lang:python"],
            "entry_type": "curated",
            "source_project": "forge",
            "source_task_id": "task-1",
            "needs_review": False,
            "created_at": "2026-04-08",
            "updated_at": "2026-04-08",
        }
        result = db_row_to_entry_dict(row)
        assert result["title"] == "Test"
        assert result["tags"] == ["lang:python"]
        assert "id" not in result
        assert "created_at" not in result


# ---------------------------------------------------------------------------
# export_single_entry (error branches)
# ---------------------------------------------------------------------------


class TestExportSingleEntry:
    @pytest.mark.asyncio
    async def test_no_store_configured_raises(self) -> None:
        """With the store disabled (engine None), export_single_entry fails
        loudly rather than no-op-ing."""
        with pytest.raises(RuntimeError, match="No store available"):
            await StoreActivities(None).export_single_entry(1)

    @pytest.mark.asyncio
    async def test_missing_entry_raises(self) -> None:
        """A well-formed but nonexistent entry id is reported by id, not swallowed."""
        store = StoreActivities(_setup_db())
        with pytest.raises(RuntimeError, match="Entry 999 not found"):
            await store.export_single_entry(999)


# ---------------------------------------------------------------------------
# ExportWorkflow
# ---------------------------------------------------------------------------


class TestExportWorkflow:
    @pytest.mark.asyncio
    async def test_export_entries(
        self,
        env: WorkflowEnvironment,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        engine = _setup_db(tmp_path)

        entry = PlaybookEntry(
            title="Export me",
            content="Content to export",
            tags=["lang:python"],
        )
        save_entries(engine, [build_entry_dict(entry)])

        import json

        async with Worker(
            env.client,
            task_queue=PBOOK_TASK_QUEUE,
            workflows=[ExportWorkflow],
            activities=_export_activities(engine),
        ):
            result = await env.client.execute_workflow(
                ExportWorkflow.run,
                json.dumps({"tags": ["lang:python"], "limit": 50}),
                id="test-export-1",
                task_queue=PBOOK_TASK_QUEUE,
            )

        assert result["count"] == 1
        assert result["entries"][0]["title"] == "Export me"

    @pytest.mark.asyncio
    async def test_export_empty(
        self,
        env: WorkflowEnvironment,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        engine = _setup_db(tmp_path)

        import json

        async with Worker(
            env.client,
            task_queue=PBOOK_TASK_QUEUE,
            workflows=[ExportWorkflow],
            activities=_export_activities(engine),
        ):
            result = await env.client.execute_workflow(
                ExportWorkflow.run,
                json.dumps({"tags": ["lang:python"], "limit": 50}),
                id="test-export-empty",
                task_queue=PBOOK_TASK_QUEUE,
            )

        assert result["count"] == 0
        assert result["entries"] == []
