"""Tests for ManualPlaybookWorkflow using Temporal test server."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sax_platform.contracts.constants import FORGE_TASK_QUEUE
from temporalio import activity
from temporalio.worker import Worker

from forge.manual_playbook_workflow import ManualPlaybookWorkflow
from forge.models import (
    FetchExistingPlaybooksInput,
    ManualPlaybookInput,
    PlaybookEntry,
    ReviewManualPlaybookInput,
    ReviewManualPlaybookResult,
    SaveExtractionInput,
    ValidatePlaybookInput,
    ValidatePlaybookResult,
)

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment


# ---------------------------------------------------------------------------
# Stub activities
# ---------------------------------------------------------------------------

_VALID_ENTRY = PlaybookEntry(
    title="Test entry",
    content="Do the thing.",
    tags=["test"],
    source_task_id="manual-1",
)

_VALID_JSON = _VALID_ENTRY.model_dump_json()


@activity.defn(name="validate_playbook_entry")
async def stub_validate_valid(input: ValidatePlaybookInput) -> ValidatePlaybookResult:
    entry = PlaybookEntry.model_validate_json(input.raw_json)
    return ValidatePlaybookResult(valid=True, entry=entry)


@activity.defn(name="validate_playbook_entry")
async def stub_validate_invalid(input: ValidatePlaybookInput) -> ValidatePlaybookResult:
    return ValidatePlaybookResult(valid=False, error="Missing required field: title")


@activity.defn(name="fetch_existing_playbooks")
async def stub_fetch_empty(input: FetchExistingPlaybooksInput) -> list[dict]:
    return []


@activity.defn(name="fetch_existing_playbooks")
async def stub_fetch_with_entries(input: FetchExistingPlaybooksInput) -> list[dict]:
    return [{"title": "Existing entry", "tags_json": '["python"]'}]


@activity.defn(name="review_manual_playbook")
async def stub_review_approved(input: ReviewManualPlaybookInput) -> ReviewManualPlaybookResult:
    return ReviewManualPlaybookResult(approved=True, final_entry=input.entry)


@activity.defn(name="review_manual_playbook")
async def stub_review_approved_with_suggestions(
    input: ReviewManualPlaybookInput,
) -> ReviewManualPlaybookResult:
    improved = input.entry.model_copy(update={"title": "Improved title"})
    return ReviewManualPlaybookResult(approved=True, final_entry=improved)


@activity.defn(name="review_manual_playbook")
async def stub_review_rejected(input: ReviewManualPlaybookInput) -> ReviewManualPlaybookResult:
    return ReviewManualPlaybookResult(
        approved=False,
        rejection_reason="Too vague and not actionable.",
        final_entry=input.entry,
    )


_save_calls: list[SaveExtractionInput] = []


@activity.defn(name="save_extraction_results")
async def stub_save(input: SaveExtractionInput) -> None:
    _save_calls.append(input)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestManualPlaybookWorkflow:
    @pytest.mark.asyncio
    async def test_valid_entry_approved_and_saved(self, env: WorkflowEnvironment) -> None:
        _save_calls.clear()
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ManualPlaybookWorkflow],
            activities=[
                stub_validate_valid,
                stub_fetch_empty,
                stub_review_approved,
                stub_save,
            ],
        ):
            result = await env.client.execute_workflow(
                ManualPlaybookWorkflow.run,
                ManualPlaybookInput(raw_json=_VALID_JSON),
                id="test-manual-playbook-happy",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.approved is True
        assert result.entry is not None
        assert result.entry.title == "Test entry"
        assert result.validation_error == ""
        assert len(_save_calls) == 1
        assert _save_calls[0].entries[0].title == "Test entry"

    @pytest.mark.asyncio
    async def test_invalid_json_returns_validation_error(self, env: WorkflowEnvironment) -> None:
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ManualPlaybookWorkflow],
            activities=[
                stub_validate_invalid,
                stub_fetch_empty,
                stub_review_approved,
                stub_save,
            ],
        ):
            result = await env.client.execute_workflow(
                ManualPlaybookWorkflow.run,
                ManualPlaybookInput(raw_json="{not valid json}"),
                id="test-manual-playbook-invalid",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.approved is False
        assert "title" in result.validation_error.lower()
        assert result.entry is None

    @pytest.mark.asyncio
    async def test_rejected_entry_not_saved(self, env: WorkflowEnvironment) -> None:
        _save_calls.clear()
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ManualPlaybookWorkflow],
            activities=[
                stub_validate_valid,
                stub_fetch_empty,
                stub_review_rejected,
                stub_save,
            ],
        ):
            result = await env.client.execute_workflow(
                ManualPlaybookWorkflow.run,
                ManualPlaybookInput(raw_json=_VALID_JSON),
                id="test-manual-playbook-rejected",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.approved is False
        assert "Too vague" in result.rejection_reason
        assert len(_save_calls) == 0

    @pytest.mark.asyncio
    async def test_suggestions_applied_to_saved_entry(self, env: WorkflowEnvironment) -> None:
        _save_calls.clear()
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ManualPlaybookWorkflow],
            activities=[
                stub_validate_valid,
                stub_fetch_empty,
                stub_review_approved_with_suggestions,
                stub_save,
            ],
        ):
            result = await env.client.execute_workflow(
                ManualPlaybookWorkflow.run,
                ManualPlaybookInput(raw_json=_VALID_JSON),
                id="test-manual-playbook-suggestions",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.approved is True
        assert result.entry is not None
        assert result.entry.title == "Improved title"
        assert len(_save_calls) == 1
        assert _save_calls[0].entries[0].title == "Improved title"

    @pytest.mark.asyncio
    async def test_empty_store_still_reviews(self, env: WorkflowEnvironment) -> None:
        """Fetch returns empty list, but review still proceeds."""
        _save_calls.clear()
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ManualPlaybookWorkflow],
            activities=[
                stub_validate_valid,
                stub_fetch_empty,
                stub_review_approved,
                stub_save,
            ],
        ):
            result = await env.client.execute_workflow(
                ManualPlaybookWorkflow.run,
                ManualPlaybookInput(raw_json=_VALID_JSON),
                id="test-manual-playbook-empty-store",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.approved is True
        assert len(_save_calls) == 1
