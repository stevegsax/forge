"""Tests for OcrStoreWorkflow — signal handling, store, terminal status."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from forge_contracts.constants import OCR_TASK_QUEUE
from forge_contracts.models import BatchResult
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.worker import Worker

from ocr.models import OcrStoreInput, OcrStoreResult
from ocr.workflow_store import OcrStoreWorkflow

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment


def _store_input() -> OcrStoreInput:
    return OcrStoreInput(
        batch_id="b1", request_id="r1", document_id="doc-1", file_path="/a.pdf"
    )


def _mock_activities(calls: dict[str, list]) -> list:
    @activity.defn(name="store_ocr_result")
    async def store_ocr_result(input_json: str) -> OcrStoreResult:
        calls["store"].append(input_json)
        return OcrStoreResult(document_id="doc-1", text_length=5, page_count=1)

    @activity.defn(name="upsert_ocr_status")
    async def upsert_ocr_status(input_json: str) -> None:
        calls["status"].append(input_json)

    return [store_ocr_result, upsert_ocr_status]


class TestOcrStoreWorkflow:
    @pytest.mark.asyncio
    async def test_success_stores_result(self, env: WorkflowEnvironment) -> None:
        calls: dict[str, list] = {"store": [], "status": []}
        async with Worker(
            env.client,
            task_queue=OCR_TASK_QUEUE,
            workflows=[OcrStoreWorkflow],
            activities=_mock_activities(calls),
        ):
            handle = await env.client.start_workflow(
                OcrStoreWorkflow.run,
                _store_input(),
                id="test-store-ok",
                task_queue=OCR_TASK_QUEUE,
            )
            await handle.signal(
                OcrStoreWorkflow.batch_result_received,
                BatchResult(
                    request_id="r1",
                    batch_id="b1",
                    raw_response_json='{"pages": []}',
                    result_type="succeeded",
                ),
            )
            result = await handle.result()

        assert result.document_id == "doc-1"
        assert len(calls["store"]) == 1

    @pytest.mark.asyncio
    async def test_error_signal_marks_failed(self, env: WorkflowEnvironment) -> None:
        calls: dict[str, list] = {"store": [], "status": []}
        async with Worker(
            env.client,
            task_queue=OCR_TASK_QUEUE,
            workflows=[OcrStoreWorkflow],
            activities=_mock_activities(calls),
        ):
            handle = await env.client.start_workflow(
                OcrStoreWorkflow.run,
                _store_input(),
                id="test-store-err",
                task_queue=OCR_TASK_QUEUE,
            )
            await handle.signal(
                OcrStoreWorkflow.batch_result_received,
                BatchResult(
                    request_id="r1", batch_id="b1", error="provider failed", result_type="errored"
                ),
            )
            with pytest.raises(WorkflowFailureError):
                await handle.result()

        # No store; a terminal failed status was written.
        assert calls["store"] == []
        assert len(calls["status"]) == 1
        assert "failed" in calls["status"][0]

    @pytest.mark.asyncio
    async def test_wait_timeout_marks_failed(self, env: WorkflowEnvironment) -> None:
        calls: dict[str, list] = {"store": [], "status": []}
        async with Worker(
            env.client,
            task_queue=OCR_TASK_QUEUE,
            workflows=[OcrStoreWorkflow],
            activities=_mock_activities(calls),
        ):
            handle = await env.client.start_workflow(
                OcrStoreWorkflow.run,
                _store_input(),
                id="test-store-timeout",
                task_queue=OCR_TASK_QUEUE,
            )
            # Never signal — the 25h wait_condition times out (time-skipped).
            with pytest.raises(WorkflowFailureError):
                await handle.result()

        assert calls["store"] == []
        assert len(calls["status"]) == 1
        assert "failed" in calls["status"][0]
