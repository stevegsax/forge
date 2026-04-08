"""Tests for export activities and workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from pbook.activities.export import (
    db_row_to_entry_dict,
    export_single_entry,
    fetch_entry_ids,
)
from pbook.models import PlaybookEntry
from pbook.store import build_entry_dict, get_engine, run_migrations, save_entries
from pbook.worker import PBOOK_TASK_QUEUE
from pbook.workflows.export import ExportWorkflow

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


# ---------------------------------------------------------------------------
# db_row_to_entry_dict
# ---------------------------------------------------------------------------


class TestDbRowToEntryDict:
    def test_basic(self):
        row = {
            "id": 1,
            "title": "Test",
            "content": "Content",
            "tags_json": '["lang:python"]',
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
# ExportWorkflow
# ---------------------------------------------------------------------------


class TestExportWorkflow:
    @pytest.mark.asyncio
    async def test_export_entries(
        self, env: WorkflowEnvironment, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
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
            activities=[fetch_entry_ids, export_single_entry],
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
        self, env: WorkflowEnvironment, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        import json

        async with Worker(
            env.client,
            task_queue=PBOOK_TASK_QUEUE,
            workflows=[ExportWorkflow],
            activities=[fetch_entry_ids, export_single_entry],
        ):
            result = await env.client.execute_workflow(
                ExportWorkflow.run,
                json.dumps({"tags": ["lang:python"], "limit": 50}),
                id="test-export-empty",
                task_queue=PBOOK_TASK_QUEUE,
            )

        assert result["count"] == 0
        assert result["entries"] == []
