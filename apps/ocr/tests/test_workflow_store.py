"""Tests for OcrStoreWorkflow — timer-loop poll, fetch+store, terminal outcomes.

The signal path is gone (T4.2): the workflow polls ``ocr_batch_status`` on a
backoff timer (the time-skipping ``env`` fast-forwards the sleeps), fetches and
stores via ``fetch_and_store_ocr_result`` on "ended", and records a
``PersistBatchOutcome`` on the platform ``batch_jobs`` ledger cross-queue
(``persist_to_store`` on ``forge-task-queue``). Both an OCR worker and a stand-in
platform worker run so the cross-queue persist resolves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sax_platform.contracts.constants import FORGE_TASK_QUEUE, OCR_TASK_QUEUE
from sax_platform.contracts.persist import PersistBatchOutcome, PersistResult
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.exceptions import ApplicationError
from temporalio.worker import Worker

from ocr.models import (
    OcrBatchStatusInput,
    OcrFetchStoreInput,
    OcrProcessingStatus,
    OcrStatusUpsertInput,
    OcrStoreInput,
    OcrStoreResult,
)
from ocr.workflow_store import OcrStoreWorkflow

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment


def _store_input() -> OcrStoreInput:
    return OcrStoreInput(batch_id="b1", request_id="r1", document_id="doc-1", file_path="/a.pdf")


def _ocr_activities(
    calls: dict[str, list], *, states: list[str], fetch_raises: bool = False
) -> list:
    """OCR-queue mocks. ``states`` is the status sequence the poll loop observes."""
    state_iter = iter(states)

    @activity.defn(name="ocr_batch_status")
    async def ocr_batch_status(input: OcrBatchStatusInput) -> str:
        calls["status"].append(input.batch_id)
        try:
            return next(state_iter)
        except StopIteration:
            return states[-1]  # steady-state after the sequence is exhausted

    @activity.defn(name="fetch_and_store_ocr_result")
    async def fetch_and_store_ocr_result(input: OcrFetchStoreInput) -> OcrStoreResult:
        calls["fetch"].append(input.request_id)
        if fetch_raises:
            raise ApplicationError("fetch/store boom", non_retryable=True)
        return OcrStoreResult(document_id=input.document_id, text_length=5, page_count=1)

    @activity.defn(name="upsert_ocr_status")
    async def upsert_ocr_status(input: OcrStatusUpsertInput) -> None:
        calls["status_upsert"].append(input.status)

    return [ocr_batch_status, fetch_and_store_ocr_result, upsert_ocr_status]


def _platform_activities(calls: dict[str, list]) -> list:
    @activity.defn(name="persist_to_store")
    async def persist_to_store(req: PersistBatchOutcome) -> PersistResult:
        calls["persist"].append(req)
        return PersistResult(kind=req.kind, applied=True)

    return [persist_to_store]


async def _run_store(
    env: WorkflowEnvironment,
    calls: dict[str, list],
    *,
    states: list[str],
    fetch_raises: bool = False,
    wf_id: str,
) -> None:
    ocr_worker = Worker(
        env.client,
        task_queue=OCR_TASK_QUEUE,
        workflows=[OcrStoreWorkflow],
        activities=_ocr_activities(calls, states=states, fetch_raises=fetch_raises),
    )
    platform_worker = Worker(
        env.client,
        task_queue=FORGE_TASK_QUEUE,
        activities=_platform_activities(calls),
    )
    async with ocr_worker, platform_worker:
        await env.client.execute_workflow(
            OcrStoreWorkflow.run,
            _store_input(),
            id=wf_id,
            task_queue=OCR_TASK_QUEUE,
        )


def _new_calls() -> dict[str, list]:
    return {"status": [], "fetch": [], "status_upsert": [], "persist": []}


class TestOcrStoreWorkflow:
    @pytest.mark.asyncio
    async def test_ended_fetches_stores_and_persists_ended(self, env: WorkflowEnvironment) -> None:
        calls = _new_calls()
        await _run_store(env, calls, states=["in_progress", "ended"], wf_id="store-ended")

        assert calls["fetch"] == ["r1"]
        # Terminal ENDED recorded on batch_jobs cross-queue; no failure status.
        assert [p.status for p in calls["persist"]] == ["ended"]
        assert calls["status_upsert"] == []

    @pytest.mark.asyncio
    async def test_terminal_failed_marks_failed_and_persists(
        self, env: WorkflowEnvironment
    ) -> None:
        calls = _new_calls()
        with pytest.raises(WorkflowFailureError):
            await _run_store(env, calls, states=["failed"], wf_id="store-failed")

        assert calls["fetch"] == []  # never fetched — terminal before ended
        assert len(calls["status_upsert"]) == 1
        assert calls["status_upsert"][0] == OcrProcessingStatus.FAILED
        assert [p.status for p in calls["persist"]] == ["failed"]

    @pytest.mark.asyncio
    async def test_expired_persists_expired(self, env: WorkflowEnvironment) -> None:
        calls = _new_calls()
        with pytest.raises(WorkflowFailureError):
            await _run_store(env, calls, states=["expired"], wf_id="store-expired")

        assert [p.status for p in calls["persist"]] == ["expired"]

    @pytest.mark.asyncio
    async def test_gave_up_persists_missing(self, env: WorkflowEnvironment) -> None:
        """A batch that never ends gives up at the ceiling and records MISSING."""
        calls = _new_calls()
        with pytest.raises(WorkflowFailureError):
            await _run_store(env, calls, states=["in_progress"], wf_id="store-giveup")

        assert calls["fetch"] == []
        assert [p.status for p in calls["persist"]] == ["missing"]
        assert len(calls["status_upsert"]) == 1
        assert calls["status_upsert"][0] == OcrProcessingStatus.FAILED

    @pytest.mark.asyncio
    async def test_fetch_store_error_marks_failed(self, env: WorkflowEnvironment) -> None:
        calls = _new_calls()
        with pytest.raises(WorkflowFailureError):
            await _run_store(
                env, calls, states=["ended"], fetch_raises=True, wf_id="store-fetch-err"
            )

        assert calls["fetch"] == ["r1"]
        assert len(calls["status_upsert"]) == 1
        assert calls["status_upsert"][0] == OcrProcessingStatus.FAILED
        assert [p.status for p in calls["persist"]] == ["failed"]
