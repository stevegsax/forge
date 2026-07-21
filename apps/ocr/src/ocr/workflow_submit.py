"""OcrSubmitWorkflow — OCR a document via parent-awaited, self-polling children.

Per chunk it builds the /v1/ocr request blob, submits the batch through ocr's own
Mistral submit activity, records the platform ``batch_jobs`` row cross-queue (an
activity call, not a signal), and starts a parent-awaited OcrStoreWorkflow child
(no ABANDON) carrying the REAL provider ``batch_id``. It then awaits every child
concurrently; a failed chunk fails the document without abandoning its siblings
(``asyncio.gather(return_exceptions=True)``), and a split document is reassembled
inline in this workflow.

This is no longer fire-and-forget at the chunk level — the workflow now runs for
up to the batch-wait ceiling (each child polls its own batch). The CLI starts it
without waiting for that full run.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from sax_platform.contracts.constants import FORGE_TASK_QUEUE
    from sax_platform.contracts.persist import PersistBatchSubmission, persist_block
    from sax_platform.temporal.polling import BATCH_WAIT_CEILING

    from ocr.models import (
        FileContentRef,
        OcrBatchRef,
        OcrBatchRequestRef,
        OcrBuildRequestInput,
        OcrDuplicateCheckResult,
        OcrProcessingStatus,
        OcrReassembleInput,
        OcrSplitInput,
        OcrStatusUpsertInput,
        OcrStoreInput,
        OcrStoreResult,
        OcrSubmitBatchInput,
        OcrSubmitInput,
        OcrSubmitResult,
        SplitResult,
    )

_IO_TIMEOUT = timedelta(seconds=30)
_SPLIT_TIMEOUT = timedelta(seconds=120)
_BLOB_TIMEOUT = timedelta(seconds=60)
_SUBMIT_TIMEOUT = timedelta(seconds=60)
_STATUS_TIMEOUT = timedelta(seconds=15)
_REASSEMBLE_TIMEOUT = timedelta(seconds=60)
_IO_RETRY = RetryPolicy(maximum_attempts=2)
_SUBMIT_RETRY = RetryPolicy(maximum_attempts=2)
# Each store child polls its own batch up to the 25h ceiling; give it a margin
# for the final fetch-and-store plus the terminal ledger/status writes.
_CHILD_EXECUTION_TIMEOUT = BATCH_WAIT_CEILING + timedelta(hours=1)


def _ocr_provider(model_name: str) -> str:
    """Extract the provider prefix from a 'provider:model' id (e.g. 'mistral')."""
    return model_name.split(":", 1)[0]


@workflow.defn
class OcrSubmitWorkflow:
    """Submit a document for OCR and await its self-polling store children."""

    @workflow.run
    async def run(self, input: OcrSubmitInput) -> OcrSubmitResult:
        document_id = input.document_id or str(workflow.uuid4())
        workflow.logger.info(
            "OcrSubmit started: file=%s document_id=%s", input.file_path, document_id
        )

        if not input.skip_duplicate_detection:
            dup_check = await workflow.execute_activity(
                "check_ocr_duplicate",
                input.file_path,
                start_to_close_timeout=_IO_TIMEOUT,
                retry_policy=_IO_RETRY,
                result_type=OcrDuplicateCheckResult,
            )
            if dup_check.is_duplicate:
                return OcrSubmitResult(
                    document_id=dup_check.existing_document_id,
                    skipped=True,
                    skip_reason="Duplicate document",
                )

        file_content_ref = await workflow.execute_activity(
            "read_and_store_file_content",
            input.file_path,
            start_to_close_timeout=_IO_TIMEOUT,
            retry_policy=_IO_RETRY,
            result_type=FileContentRef,
        )

        split_result = await workflow.execute_activity(
            "split_file_into_chunks",
            OcrSplitInput(
                content_id=file_content_ref.content_id,
                mime_type=file_content_ref.mime_type,
                file_size_bytes=file_content_ref.file_size_bytes,
            ),
            start_to_close_timeout=_SPLIT_TIMEOUT,
            retry_policy=_IO_RETRY,
            result_type=SplitResult,
        )
        chunk_count = len(split_result.chunks)

        chunk_document_ids = [
            document_id if chunk_count == 1 else f"{document_id}__chunk_{c.chunk_index}"
            for c in split_result.chunks
        ]

        provider = _ocr_provider(input.model_name)
        handles: list[workflow.ChildWorkflowHandle[Any, Any]] = []
        submitted: list[tuple[str, str]] = []  # (request_id, chunk_doc_id), aligned to handles
        batch_refs: list[OcrBatchRef] = []

        for i, chunk in enumerate(split_result.chunks):
            chunk_doc_id = chunk_document_ids[i]

            # Build the request body + mint the single correlation id, stash to S3.
            request_ref = await workflow.execute_activity(
                "build_ocr_request_blob",
                OcrBuildRequestInput(
                    content_id=chunk.content_id,
                    mime_type=chunk.mime_type,
                    model_name=input.model_name,
                ),
                start_to_close_timeout=_BLOB_TIMEOUT,
                retry_policy=_IO_RETRY,
                result_type=OcrBatchRequestRef,
            )

            # Submit the Mistral batch (ocr's own activity, own queue). On a genuine
            # submit failure, write a terminal failed status and fail the workflow.
            try:
                batch_id: str = await workflow.execute_activity(
                    "submit_ocr_batch",
                    OcrSubmitBatchInput(s3_key=request_ref.s3_key, model=request_ref.model),
                    start_to_close_timeout=_SUBMIT_TIMEOUT,
                    retry_policy=_SUBMIT_RETRY,
                    result_type=str,
                )
            except Exception as exc:
                await self._upsert_status(
                    request_ref.request_id,
                    chunk_doc_id,
                    input.file_path,
                    OcrProcessingStatus.FAILED,
                    error_message=str(exc),
                )
                raise

            await self._upsert_status(
                request_ref.request_id,
                chunk_doc_id,
                input.file_path,
                OcrProcessingStatus.SUBMITTED,
            )

            # Record the platform batch_jobs row cross-queue before starting the
            # child; workflow_id is the store child that will poll this batch.
            store_child_id = f"ocr-store-{chunk_doc_id}"
            await persist_block(
                PersistBatchSubmission(
                    request_id=request_ref.request_id,
                    batch_id=batch_id,
                    workflow_id=store_child_id,
                    provider=provider,
                ),
                task_queue=FORGE_TASK_QUEUE,
            )

            # Start the parent-awaited store child with the REAL batch_id (no ABANDON).
            store_handle = await workflow.start_child_workflow(
                "OcrStoreWorkflow",
                OcrStoreInput(
                    batch_id=batch_id,
                    request_id=request_ref.request_id,
                    document_id=chunk_doc_id,
                    file_path=input.file_path,
                ),
                id=store_child_id,
                execution_timeout=_CHILD_EXECUTION_TIMEOUT,
                result_type=OcrStoreResult,
            )
            handles.append(store_handle)
            submitted.append((request_ref.request_id, chunk_doc_id))
            batch_refs.append(OcrBatchRef(batch_id=batch_id, request_id=request_ref.request_id))

            # The input blob is now redundant (the request blob is in S3); reaped by
            # bucket TTL, but delete eagerly to keep the store tidy.
            try:
                await workflow.execute_activity(
                    "delete_file_content_blob",
                    chunk.content_id,
                    start_to_close_timeout=_IO_TIMEOUT,
                    retry_policy=_IO_RETRY,
                )
            except Exception:
                workflow.logger.warning("Failed to delete blob %s after submit", chunk.content_id)

        # Await every child concurrently. return_exceptions=True lets one child's
        # failure surface without abandoning siblings mid-flight — each settles on
        # its own (a failed store child raises promptly; it never hangs the wait).
        results = await asyncio.gather(
            *(self._child_result(handle) for handle in handles), return_exceptions=True
        )

        failures = [
            (submitted[i], result)
            for i, result in enumerate(results)
            if isinstance(result, BaseException)
        ]
        if failures:
            for (request_id, chunk_doc_id), child_exc in failures:
                await self._upsert_status(
                    request_id,
                    chunk_doc_id,
                    input.file_path,
                    OcrProcessingStatus.FAILED,
                    error_message=str(child_exc),
                )
            error_detail = "; ".join(f"{doc_id}: {err}" for (_req, doc_id), err in failures)
            raise ApplicationError(
                f"OCR document {document_id} failed ({len(failures)}/{chunk_count} chunks): "
                f"{error_detail}",
                non_retryable=True,
            )

        if chunk_count > 1:
            # All chunks stored: combine them into the single document inline.
            await workflow.execute_activity(
                "reassemble_ocr_chunks",
                OcrReassembleInput(
                    document_id=document_id,
                    chunk_document_ids=chunk_document_ids,
                    file_path=input.file_path,
                    total_pages=split_result.total_pages,
                ),
                start_to_close_timeout=_REASSEMBLE_TIMEOUT,
                retry_policy=_IO_RETRY,
                result_type=OcrStoreResult,
            )

        workflow.logger.info("OcrSubmit done: document_id=%s chunks=%d", document_id, chunk_count)
        return OcrSubmitResult(
            document_id=document_id, batch_refs=batch_refs, chunk_count=chunk_count
        )

    async def _child_result(self, handle: workflow.ChildWorkflowHandle[Any, Any]) -> OcrStoreResult:
        """Await one child handle (a coroutine so ``asyncio.gather`` can wrap it)."""
        result: OcrStoreResult = await handle
        return result

    async def _upsert_status(
        self,
        request_id: str,
        document_id: str,
        file_path: str,
        status: OcrProcessingStatus,
        *,
        error_message: str | None = None,
    ) -> None:
        await workflow.execute_activity(
            "upsert_ocr_status",
            OcrStatusUpsertInput(
                request_id=request_id,
                document_id=document_id,
                file_path=file_path,
                status=status,
                error_message=error_message,
            ),
            start_to_close_timeout=_STATUS_TIMEOUT,
            retry_policy=_IO_RETRY,
        )
