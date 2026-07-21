"""Tests for OcrSubmitWorkflow — submit ordering + parent-awaited children (T4.2).

The workflow now submits each chunk's Mistral batch through ocr's own
``submit_ocr_batch`` activity, records the ``batch_jobs`` row cross-queue, and
starts a parent-awaited OcrStoreWorkflow child with the REAL batch_id (no
ABANDON). Two workers run: the OCR worker (``ocr-task-queue``) and a stand-in
platform worker (``forge-task-queue``) for the cross-queue ``persist_to_store``.

The headline test is the failed-chunk AC: with the real self-polling
OcrStoreWorkflow as the child, a chunk whose provider status is terminal fails
PROMPTLY (recording ``batch_outcome`` FAILED, never MISSING — the 25h-ceiling
outcome), its sibling still completes, and the document fails.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from sax_platform.contracts.constants import FORGE_TASK_QUEUE, OCR_TASK_QUEUE
from sax_platform.contracts.persist import (
    PersistBatchOutcome,
    PersistBatchSubmission,
    PersistResult,
)
from temporalio import activity, workflow
from temporalio.client import WorkflowFailureError
from temporalio.worker import Worker

from ocr.models import (
    ChunkRef,
    FileContentRef,
    OcrBatchRequestRef,
    OcrBatchStatusInput,
    OcrBuildRequestInput,
    OcrDuplicateCheckResult,
    OcrFetchStoreInput,
    OcrProcessingStatus,
    OcrReassembleInput,
    OcrSplitInput,
    OcrStatusUpsertInput,
    OcrStoreInput,
    OcrStoreResult,
    OcrSubmitBatchInput,
    OcrSubmitInput,
    SplitResult,
)
from ocr.workflow_store import OcrStoreWorkflow
from ocr.workflow_submit import OcrSubmitWorkflow

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment


# ---------------------------------------------------------------------------
# Test 1 — single-chunk submit ordering (submit before child; real batch_id)
# ---------------------------------------------------------------------------


@workflow.defn(name="OcrStoreWorkflow")
class _StubStoreWorkflow:
    """Stand-in store child that records its received batch_id and returns."""

    @workflow.run
    async def run(self, input: OcrStoreInput) -> OcrStoreResult:
        await workflow.execute_activity(
            "record_child_start", input.batch_id, start_to_close_timeout=timedelta(seconds=5)
        )
        return OcrStoreResult(document_id=input.document_id, text_length=0, page_count=0)


def _ocr_activities_single(calls: dict[str, list]) -> list:
    @activity.defn(name="check_ocr_duplicate")
    async def check_ocr_duplicate(file_path: str) -> OcrDuplicateCheckResult:
        return OcrDuplicateCheckResult(is_duplicate=False)

    @activity.defn(name="read_and_store_file_content")
    async def read_and_store_file_content(file_path: str) -> FileContentRef:
        return FileContentRef(content_id="c1", mime_type="application/pdf", file_size_bytes=10)

    @activity.defn(name="split_file_into_chunks")
    async def split_file_into_chunks(input: OcrSplitInput) -> SplitResult:
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
    async def build_ocr_request_blob(input: OcrBuildRequestInput) -> OcrBatchRequestRef:
        return OcrBatchRequestRef(request_id="rid-1", s3_key="key-1", model="m")

    @activity.defn(name="submit_ocr_batch")
    async def submit_ocr_batch(input: OcrSubmitBatchInput) -> str:
        calls["submit"].append(input)
        return "bid-1"

    @activity.defn(name="record_child_start")
    async def record_child_start(batch_id: str) -> None:
        calls["child_batch_id"].append(batch_id)

    @activity.defn(name="upsert_ocr_status")
    async def upsert_ocr_status(input: OcrStatusUpsertInput) -> None:
        calls["status"].append(input.status)

    @activity.defn(name="delete_file_content_blob")
    async def delete_file_content_blob(content_id: str) -> None:
        calls["deleted"].append(content_id)

    return [
        check_ocr_duplicate,
        read_and_store_file_content,
        split_file_into_chunks,
        build_ocr_request_blob,
        submit_ocr_batch,
        record_child_start,
        upsert_ocr_status,
        delete_file_content_blob,
    ]


def _persist_activity(calls: dict[str, list]) -> list:
    @activity.defn(name="persist_to_store")
    async def persist_to_store(req: PersistBatchSubmission | PersistBatchOutcome) -> PersistResult:
        calls["persist"].append(req)
        return PersistResult(kind=req.kind, applied=True)

    return [persist_to_store]


class TestOcrSubmitOrdering:
    @pytest.mark.asyncio
    async def test_submit_before_child_and_real_batch_id(self, env: WorkflowEnvironment) -> None:
        calls: dict[str, list] = {
            "submit": [],
            "child_batch_id": [],
            "status": [],
            "deleted": [],
            "persist": [],
        }
        ocr_worker = Worker(
            env.client,
            task_queue=OCR_TASK_QUEUE,
            workflows=[OcrSubmitWorkflow, _StubStoreWorkflow],
            activities=_ocr_activities_single(calls),
        )
        platform_worker = Worker(
            env.client, task_queue=FORGE_TASK_QUEUE, activities=_persist_activity(calls)
        )

        async with ocr_worker, platform_worker:
            result = await env.client.execute_workflow(
                OcrSubmitWorkflow.run,
                OcrSubmitInput(file_path="/tmp/x.pdf", document_id="doc-fixed"),
                id="test-submit-order",
                task_queue=OCR_TASK_QUEUE,
            )

        assert result.document_id == "doc-fixed"
        assert result.chunk_count == 1
        assert result.batch_refs[0].batch_id == "bid-1"

        # ocr's own submit activity ran with the request blob key.
        assert len(calls["submit"]) == 1
        assert calls["submit"][0].s3_key == "key-1"
        # The child received the REAL batch_id (only known post-submit).
        assert calls["child_batch_id"] == ["bid-1"]

        # batch_jobs submission recorded cross-queue, keyed to the store child id.
        submissions = [p for p in calls["persist"] if isinstance(p, PersistBatchSubmission)]
        assert len(submissions) == 1
        assert submissions[0].batch_id == "bid-1"
        assert submissions[0].workflow_id == "ocr-store-doc-fixed"
        assert OcrProcessingStatus.SUBMITTED in calls["status"]


# ---------------------------------------------------------------------------
# Test 2 — failed-chunk AC: fails promptly, sibling completes, no 26h wait
# ---------------------------------------------------------------------------


def _ocr_activities_multi(calls: dict[str, list], *, fail_c0: bool = True) -> list:
    """Chunk-aware mocks: with ``fail_c0`` set, batch-c0 fails; else both end."""

    @activity.defn(name="check_ocr_duplicate")
    async def check_ocr_duplicate(file_path: str) -> OcrDuplicateCheckResult:
        return OcrDuplicateCheckResult(is_duplicate=False)

    @activity.defn(name="read_and_store_file_content")
    async def read_and_store_file_content(file_path: str) -> FileContentRef:
        return FileContentRef(content_id="orig", mime_type="application/pdf", file_size_bytes=10)

    @activity.defn(name="split_file_into_chunks")
    async def split_file_into_chunks(input: OcrSplitInput) -> SplitResult:
        return SplitResult(
            chunks=[
                ChunkRef(
                    content_id=f"c{i}",
                    mime_type="application/pdf",
                    file_size_bytes=10,
                    chunk_index=i,
                    page_start=i * 10 + 1,
                    page_end=i * 10 + 10,
                )
                for i in range(2)
            ],
            total_pages=20,
            original_content_id="orig",
        )

    @activity.defn(name="build_ocr_request_blob")
    async def build_ocr_request_blob(input: OcrBuildRequestInput) -> OcrBatchRequestRef:
        cid = input.content_id
        return OcrBatchRequestRef(request_id=f"rid-{cid}", s3_key=f"key-{cid}", model="m")

    @activity.defn(name="submit_ocr_batch")
    async def submit_ocr_batch(input: OcrSubmitBatchInput) -> str:
        # key-c0 -> batch-c0, key-c1 -> batch-c1
        return input.s3_key.replace("key-", "batch-")

    @activity.defn(name="ocr_batch_status")
    async def ocr_batch_status(input: OcrBatchStatusInput) -> str:
        if fail_c0 and input.batch_id == "batch-c0":
            return "failed"
        return "ended"

    @activity.defn(name="fetch_and_store_ocr_result")
    async def fetch_and_store_ocr_result(input: OcrFetchStoreInput) -> OcrStoreResult:
        calls["fetch"].append(input.request_id)
        return OcrStoreResult(document_id=input.document_id, text_length=3, page_count=1)

    @activity.defn(name="upsert_ocr_status")
    async def upsert_ocr_status(input: OcrStatusUpsertInput) -> None:
        calls["status"].append(input.status)

    @activity.defn(name="delete_file_content_blob")
    async def delete_file_content_blob(content_id: str) -> None:
        calls["deleted"].append(content_id)

    @activity.defn(name="reassemble_ocr_chunks")
    async def reassemble_ocr_chunks(input: OcrReassembleInput) -> OcrStoreResult:
        calls["reassembled"].append(input)
        return OcrStoreResult(document_id="doc-x", text_length=6, page_count=2)

    return [
        check_ocr_duplicate,
        read_and_store_file_content,
        split_file_into_chunks,
        build_ocr_request_blob,
        submit_ocr_batch,
        ocr_batch_status,
        fetch_and_store_ocr_result,
        upsert_ocr_status,
        delete_file_content_blob,
        reassemble_ocr_chunks,
    ]


class TestFailedChunkPropagatesPromptly:
    @pytest.mark.asyncio
    async def test_failed_chunk_fails_document_without_hanging(
        self, env: WorkflowEnvironment
    ) -> None:
        """AC1: a store child that fails surfaces promptly; its sibling completes.

        chunk 0's provider status is terminal (``failed``), so the real
        OcrStoreWorkflow child records a ``batch_outcome`` FAILED and raises after
        one poll — NOT the 25h-ceiling MISSING outcome. chunk 1 ends normally and
        stores. The parent gathers both (``return_exceptions=True``) and fails the
        document. If a failed chunk hung, the time-skipping env would show a
        ceiling MISSING instead of a fast FAILED.
        """
        calls: dict[str, list] = {
            "fetch": [],
            "status": [],
            "deleted": [],
            "reassembled": [],
            "persist": [],
        }
        ocr_worker = Worker(
            env.client,
            task_queue=OCR_TASK_QUEUE,
            # Real self-polling store child — this is what the AC exercises.
            workflows=[OcrSubmitWorkflow, OcrStoreWorkflow],
            activities=_ocr_activities_multi(calls),
        )
        platform_worker = Worker(
            env.client, task_queue=FORGE_TASK_QUEUE, activities=_persist_activity(calls)
        )

        async with ocr_worker, platform_worker:
            with pytest.raises(WorkflowFailureError):
                await env.client.execute_workflow(
                    OcrSubmitWorkflow.run,
                    OcrSubmitInput(file_path="/tmp/x.pdf", document_id="doc-x"),
                    id="test-submit-failed-chunk",
                    task_queue=OCR_TASK_QUEUE,
                )

        outcomes = [p for p in calls["persist"] if isinstance(p, PersistBatchOutcome)]
        by_req = {p.request_id: p.status for p in outcomes}
        # Promptness: the failed chunk took the terminal path (FAILED), not the
        # 25h-ceiling give-up path (MISSING).
        assert by_req["rid-c0"] == "failed"
        assert "missing" not in by_req.values()
        # The sibling still completed: it ended, fetched, and recorded ENDED.
        assert by_req["rid-c1"] == "ended"
        assert calls["fetch"] == ["rid-c1"]
        # A failed document is never reassembled.
        assert calls["reassembled"] == []

    @pytest.mark.asyncio
    async def test_multichunk_success_reassembles_inline(self, env: WorkflowEnvironment) -> None:
        """Both chunks store, then the parent reassembles inline and returns."""
        calls: dict[str, list] = {
            "fetch": [],
            "status": [],
            "deleted": [],
            "reassembled": [],
            "persist": [],
        }
        ocr_worker = Worker(
            env.client,
            task_queue=OCR_TASK_QUEUE,
            workflows=[OcrSubmitWorkflow, OcrStoreWorkflow],
            activities=_ocr_activities_multi(calls, fail_c0=False),
        )
        platform_worker = Worker(
            env.client, task_queue=FORGE_TASK_QUEUE, activities=_persist_activity(calls)
        )

        async with ocr_worker, platform_worker:
            result = await env.client.execute_workflow(
                OcrSubmitWorkflow.run,
                OcrSubmitInput(file_path="/tmp/x.pdf", document_id="doc-y"),
                id="test-submit-multichunk-ok",
                task_queue=OCR_TASK_QUEUE,
            )

        assert result.chunk_count == 2
        # Both chunks fetched + stored, then reassembled once, inline in the parent.
        assert sorted(calls["fetch"]) == ["rid-c0", "rid-c1"]
        assert len(calls["reassembled"]) == 1
        outcomes = [p for p in calls["persist"] if isinstance(p, PersistBatchOutcome)]
        assert sorted(p.status for p in outcomes) == ["ended", "ended"]
