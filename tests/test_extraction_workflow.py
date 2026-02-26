"""Tests for forge.extraction_workflow — Temporal extraction workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from temporalio.worker import Worker

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment

from forge.extraction_workflow import ForgeExtractionWorkflow
from forge.models import (
    ExtractionCallResult,
    ExtractionInput,
    ExtractionResult,
    ExtractionWorkflowInput,
    FetchExtractionInput,
    PlaybookEntry,
    SaveExtractionInput,
)

TASK_QUEUE = "test-extraction-queue"


# ---------------------------------------------------------------------------
# Mock activities
# ---------------------------------------------------------------------------


def _make_fetch_no_runs():
    """Return a fetch activity that returns no runs."""
    from temporalio import activity

    @activity.defn(name="fetch_extraction_input")
    async def fetch_extraction_input(input: FetchExtractionInput) -> ExtractionInput:
        return ExtractionInput(
            system_prompt="",
            user_prompt="",
            source_workflow_ids=[],
        )

    return fetch_extraction_input


def _make_fetch_with_runs():
    """Return a fetch activity that returns runs."""
    from temporalio import activity

    @activity.defn(name="fetch_extraction_input")
    async def fetch_extraction_input(input: FetchExtractionInput) -> ExtractionInput:
        return ExtractionInput(
            system_prompt="Extract lessons.",
            user_prompt="Do it.",
            source_workflow_ids=["wf-1", "wf-2"],
        )

    return fetch_extraction_input


def _make_call_llm(entries: list[PlaybookEntry] | None = None):
    """Return a call_extraction_llm activity."""
    from temporalio import activity

    if entries is None:
        entries = [
            PlaybookEntry(
                title="Test lesson",
                content="Always include type stubs.",
                tags=["python"],
                source_task_id="t1",
                source_workflow_id="wf-1",
            ),
        ]

    @activity.defn(name="call_extraction_llm")
    async def call_extraction_llm(input: ExtractionInput) -> ExtractionCallResult:
        return ExtractionCallResult(
            result=ExtractionResult(
                entries=entries,
                summary="Extracted lessons.",
            ),
            source_workflow_ids=input.source_workflow_ids,
            model_name="test-model",
            input_tokens=500,
            output_tokens=200,
            latency_ms=100.0,
        )

    return call_extraction_llm


def _make_save():
    """Return a save activity that records calls."""
    from temporalio import activity

    saved: list[SaveExtractionInput] = []

    @activity.defn(name="save_extraction_results")
    async def save_extraction_results(input: SaveExtractionInput) -> None:
        saved.append(input)

    return save_extraction_results, saved


# ---------------------------------------------------------------------------
# Workflow tests
# ---------------------------------------------------------------------------


class TestForgeExtractionWorkflow:
    @pytest.mark.asyncio
    async def test_no_runs_returns_zero(self, env: WorkflowEnvironment) -> None:
        save_fn, saved = _make_save()
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[ForgeExtractionWorkflow],
            activities=[
                _make_fetch_no_runs(),
                _make_call_llm(),
                save_fn,
            ],
        ):
            result = await env.client.execute_workflow(
                ForgeExtractionWorkflow.run,
                ExtractionWorkflowInput(limit=10, since_hours=24),
                id="test-extraction-empty",
                task_queue=TASK_QUEUE,
            )
        assert result.entries_created == 0
        assert result.source_workflow_ids == []
        assert len(saved) == 0

    @pytest.mark.asyncio
    async def test_runs_found_produces_entries(self, env: WorkflowEnvironment) -> None:
        save_fn, saved = _make_save()
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[ForgeExtractionWorkflow],
            activities=[
                _make_fetch_with_runs(),
                _make_call_llm(),
                save_fn,
            ],
        ):
            result = await env.client.execute_workflow(
                ForgeExtractionWorkflow.run,
                ExtractionWorkflowInput(limit=10, since_hours=24),
                id="test-extraction-with-runs",
                task_queue=TASK_QUEUE,
            )
        assert result.entries_created == 1
        assert result.source_workflow_ids == ["wf-1", "wf-2"]
        assert len(saved) == 1
        assert saved[0].entries[0].title == "Test lesson"

    @pytest.mark.asyncio
    async def test_empty_entries_skips_save(self, env: WorkflowEnvironment) -> None:
        save_fn, saved = _make_save()
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[ForgeExtractionWorkflow],
            activities=[
                _make_fetch_with_runs(),
                _make_call_llm(entries=[]),
                save_fn,
            ],
        ):
            result = await env.client.execute_workflow(
                ForgeExtractionWorkflow.run,
                ExtractionWorkflowInput(limit=10, since_hours=24),
                id="test-extraction-no-entries",
                task_queue=TASK_QUEUE,
            )
        assert result.entries_created == 0
        assert len(saved) == 0
