"""Cross-queue test for OcrSubmitWorkflow.

Runs two workers — the OCR worker (ocr-task-queue) and a stand-in platform worker
(forge-task-queue) — to prove the submit path crosses the queue boundary purely via
forge-contracts: OCR builds the request, the platform submit SPI is invoked
cross-queue, and the batch_jobs record is written cross-queue.

The store child is stubbed (returns immediately) so the test isn't coupled to its
25h result wait — this test asserts the submission, not the store path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sax_platform.contracts.constants import FORGE_TASK_QUEUE, OCR_TASK_QUEUE
from sax_platform.contracts.models import BatchSubmitResult, BatchSubmitSpiInput
from sax_platform.contracts.persist import PersistResult
from temporalio import activity, workflow
from temporalio.worker import Worker

from ocr.models import (
    ChunkRef,
    FileContentRef,
    OcrBatchRequestRef,
    OcrDuplicateCheckResult,
    OcrStoreInput,
    OcrStoreResult,
    OcrSubmitInput,
    SplitResult,
)
from ocr.workflow_submit import OcrSubmitWorkflow

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment


@workflow.defn(name="OcrStoreWorkflow")
class _StubStoreWorkflow:
    """Stand-in store child that completes immediately (no 25h result wait)."""

    @workflow.run
    async def run(self, input: OcrStoreInput) -> OcrStoreResult:
        return OcrStoreResult(document_id=input.document_id, text_length=0, page_count=0)


def _ocr_activities(calls: dict[str, list]) -> list:
    @activity.defn(name="check_ocr_duplicate")
    async def check_ocr_duplicate(file_path: str) -> OcrDuplicateCheckResult:
        return OcrDuplicateCheckResult(is_duplicate=False)

    @activity.defn(name="read_and_store_file_content")
    async def read_and_store_file_content(file_path: str) -> FileContentRef:
        return FileContentRef(content_id="c1", mime_type="application/pdf", file_size_bytes=10)

    @activity.defn(name="split_file_into_chunks")
    async def split_file_into_chunks(input_json: str) -> SplitResult:
        return SplitResult(
            chunks=[
                ChunkRef(
                    content_id="c1",
                    mime_type="application/pdf",
                    file_size_bytes=10,
                    chunk_index=0,
                    page_start=1,
                    page_end=1,
                )
            ],
            total_pages=1,
            original_content_id="c1",
        )

    @activity.defn(name="build_ocr_request_blob")
    async def build_ocr_request_blob(input_json: str) -> OcrBatchRequestRef:
        return OcrBatchRequestRef(request_id="rid-1", s3_key="ocr-request-rid-1", model="m")

    @activity.defn(name="upsert_ocr_status")
    async def upsert_ocr_status(input_json: str) -> None:
        calls["status"].append(input_json)

    @activity.defn(name="delete_file_content_blob")
    async def delete_file_content_blob(content_id: str) -> None:
        calls["deleted"].append(content_id)

    return [
        check_ocr_duplicate,
        read_and_store_file_content,
        split_file_into_chunks,
        build_ocr_request_blob,
        upsert_ocr_status,
        delete_file_content_blob,
    ]


def _platform_activities(calls: dict[str, list]) -> list:
    @activity.defn(name="submit_batch_blob")
    async def submit_batch_blob(spi: BatchSubmitSpiInput) -> BatchSubmitResult:
        calls["submit"].append(spi)
        return BatchSubmitResult(request_id=spi.custom_id, batch_id="bid-1", provider=spi.provider)

    @activity.defn(name="persist_to_store")
    async def persist_to_store(req: object) -> PersistResult:
        calls["persist"].append(req)
        return PersistResult(kind="batch_submission", applied=True)

    return [submit_batch_blob, persist_to_store]


class TestOcrSubmitCrossQueue:
    @pytest.mark.asyncio
    async def test_submit_crosses_queue_boundary(self, env: WorkflowEnvironment) -> None:
        calls: dict[str, list] = {"status": [], "deleted": [], "submit": [], "persist": []}

        ocr_worker = Worker(
            env.client,
            task_queue=OCR_TASK_QUEUE,
            workflows=[OcrSubmitWorkflow, _StubStoreWorkflow],
            activities=_ocr_activities(calls),
        )
        platform_worker = Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            activities=_platform_activities(calls),
        )

        async with ocr_worker, platform_worker:
            result = await env.client.execute_workflow(
                OcrSubmitWorkflow.run,
                OcrSubmitInput(file_path="/tmp/x.pdf", document_id="doc-fixed"),
                id="test-submit-xq",
                task_queue=OCR_TASK_QUEUE,
            )

        assert result.document_id == "doc-fixed"
        assert result.chunk_count == 1
        assert result.batch_refs[0].batch_id == "bid-1"

        # The platform submit SPI was invoked cross-queue with the OCR endpoint.
        assert len(calls["submit"]) == 1
        assert calls["submit"][0].endpoint == "/v1/ocr"
        assert calls["submit"][0].custom_id == "rid-1"
        # batch_jobs was recorded cross-queue; OCR wrote its own submitted status.
        assert len(calls["persist"]) == 1
        assert any("submitted" in s for s in calls["status"])
