"""Tests for ExportPlaybookWorkflow using Temporal test server."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from temporalio import activity
from temporalio.worker import Worker

from forge.activities.playbook_export import db_row_to_playbook_entry
from forge.export_playbook_workflow import ExportPlaybookWorkflow
from forge.models import (
    ExportPlaybookInput,
    ExportSinglePlaybookInput,
    FetchPlaybookIdsInput,
    PlaybookEntry,
)
from forge.workflows import FORGE_TASK_QUEUE

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestDbRowToPlaybookEntry:
    def test_converts_full_row(self) -> None:
        row = {
            "id": 42,
            "title": "Use type stubs",
            "content": "Always include type stubs for Pydantic models.",
            "tags_json": '["python", "pydantic"]',
            "source_task_id": "task-1",
            "source_workflow_id": "wf-abc",
            "extraction_workflow_id": "wf-extract-1",
            "created_at": "2025-01-01 00:00:00",
        }
        entry = db_row_to_playbook_entry(row)

        assert entry.title == "Use type stubs"
        assert entry.content == "Always include type stubs for Pydantic models."
        assert entry.tags == ["python", "pydantic"]
        assert entry.source_task_id == "task-1"
        assert entry.source_workflow_id == "wf-abc"

    def test_drops_db_only_fields(self) -> None:
        row = {
            "id": 99,
            "title": "Test",
            "content": "Content",
            "tags_json": "[]",
            "source_task_id": "t-1",
            "source_workflow_id": "",
            "extraction_workflow_id": "manual",
            "created_at": "2025-06-01 12:00:00",
        }
        entry = db_row_to_playbook_entry(row)
        dumped = entry.model_dump()

        assert "id" not in dumped
        assert "created_at" not in dumped
        assert "extraction_workflow_id" not in dumped

    def test_empty_tags(self) -> None:
        row = {
            "title": "Minimal",
            "content": "Content",
            "tags_json": "[]",
            "source_task_id": "t-2",
        }
        entry = db_row_to_playbook_entry(row)
        assert entry.tags == []

    def test_tags_already_list(self) -> None:
        """Handles case where tags_json is already parsed."""
        row = {
            "title": "Parsed",
            "content": "Content",
            "tags_json": ["a", "b"],
            "source_task_id": "t-3",
        }
        entry = db_row_to_playbook_entry(row)
        assert entry.tags == ["a", "b"]


# ---------------------------------------------------------------------------
# Stub activities for workflow tests
# ---------------------------------------------------------------------------

_SAMPLE_ENTRIES = [
    PlaybookEntry(
        title="Entry one",
        content="First lesson.",
        tags=["python"],
        source_task_id="task-1",
        source_workflow_id="wf-1",
    ),
    PlaybookEntry(
        title="Entry two",
        content="Second lesson.",
        tags=["api"],
        source_task_id="task-2",
        source_workflow_id="wf-2",
    ),
]


@activity.defn(name="fetch_playbook_ids")
async def stub_fetch_ids(input: FetchPlaybookIdsInput) -> list[int]:
    return [1, 2]


@activity.defn(name="fetch_playbook_ids")
async def stub_fetch_ids_empty(input: FetchPlaybookIdsInput) -> list[int]:
    return []


@activity.defn(name="fetch_playbook_ids")
async def stub_fetch_ids_filtered(input: FetchPlaybookIdsInput) -> list[int]:
    if input.tags == ["python"]:
        return [1]
    if input.source_task_id == "task-2":
        return [2]
    return [1, 2]


@activity.defn(name="export_single_playbook")
async def stub_export_single(input: ExportSinglePlaybookInput) -> PlaybookEntry:
    idx = input.playbook_id - 1
    return _SAMPLE_ENTRIES[idx]


# ---------------------------------------------------------------------------
# Workflow tests
# ---------------------------------------------------------------------------


class TestExportPlaybookWorkflow:
    @pytest.mark.asyncio
    async def test_export_all(self, env: WorkflowEnvironment) -> None:
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ExportPlaybookWorkflow],
            activities=[stub_fetch_ids, stub_export_single],
        ):
            result = await env.client.execute_workflow(
                ExportPlaybookWorkflow.run,
                ExportPlaybookInput(),
                id="test-export-all",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.count == 2
        assert len(result.entries) == 2
        assert result.entries[0].title == "Entry one"
        assert result.entries[1].title == "Entry two"

    @pytest.mark.asyncio
    async def test_export_empty(self, env: WorkflowEnvironment) -> None:
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ExportPlaybookWorkflow],
            activities=[stub_fetch_ids_empty, stub_export_single],
        ):
            result = await env.client.execute_workflow(
                ExportPlaybookWorkflow.run,
                ExportPlaybookInput(),
                id="test-export-empty",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.count == 0
        assert result.entries == []

    @pytest.mark.asyncio
    async def test_export_with_tag_filter(self, env: WorkflowEnvironment) -> None:
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ExportPlaybookWorkflow],
            activities=[stub_fetch_ids_filtered, stub_export_single],
        ):
            result = await env.client.execute_workflow(
                ExportPlaybookWorkflow.run,
                ExportPlaybookInput(tags=["python"]),
                id="test-export-tag-filter",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.count == 1
        assert result.entries[0].title == "Entry one"

    @pytest.mark.asyncio
    async def test_export_with_task_id_filter(self, env: WorkflowEnvironment) -> None:
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ExportPlaybookWorkflow],
            activities=[stub_fetch_ids_filtered, stub_export_single],
        ):
            result = await env.client.execute_workflow(
                ExportPlaybookWorkflow.run,
                ExportPlaybookInput(source_task_id="task-2"),
                id="test-export-task-filter",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.count == 1
        assert result.entries[0].title == "Entry two"

    @pytest.mark.asyncio
    async def test_entries_are_playbook_entry_compatible(
        self, env: WorkflowEnvironment
    ) -> None:
        """Exported entries can round-trip through PlaybookEntry."""
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ExportPlaybookWorkflow],
            activities=[stub_fetch_ids, stub_export_single],
        ):
            result = await env.client.execute_workflow(
                ExportPlaybookWorkflow.run,
                ExportPlaybookInput(),
                id="test-export-roundtrip",
                task_queue=FORGE_TASK_QUEUE,
            )

        for entry in result.entries:
            roundtripped = PlaybookEntry.model_validate(entry.model_dump())
            assert roundtripped.title == entry.title
            assert roundtripped.content == entry.content
            assert roundtripped.tags == entry.tags
