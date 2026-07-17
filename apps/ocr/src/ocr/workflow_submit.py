"""OcrSubmitWorkflow — submit a document for OCR via the platform batch service.

Fire-and-forget: returns once batches are submitted. Per chunk it builds the
/v1/ocr request, stashes it to S3, and calls the platform's opaque-blob submit SPI
cross-queue (the platform makes the provider call). The single correlation id
(request_id == provider custom_id == batch_jobs PK) is minted once, in
``build_ocr_request_blob``. Child OcrStoreWorkflows wait for the poller's signal and
store the results.
"""

from __future__ import annotations

import json
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from sax_platform.contracts.constants import FORGE_TASK_QUEUE
    from sax_platform.contracts.models import BatchSubmitResult, BatchSubmitSpiInput
    from sax_platform.contracts.persist import PersistBatchSubmission, persist_block

    from ocr.models import (
        FileContentRef,
        OcrBatchRef,
        OcrBatchRequestRef,
        OcrDuplicateCheckResult,
        OcrGatherInput,
        OcrStoreInput,
        OcrStoreResult,
        OcrSubmitInput,
        OcrSubmitResult,
        SplitResult,
    )

_IO_TIMEOUT = timedelta(seconds=30)
_SPLIT_TIMEOUT = timedelta(seconds=120)
_BLOB_TIMEOUT = timedelta(seconds=60)
_SUBMIT_TIMEOUT = timedelta(seconds=60)
_STATUS_TIMEOUT = timedelta(seconds=15)
_IO_RETRY = RetryPolicy(maximum_attempts=2)
_SUBMIT_RETRY = RetryPolicy(maximum_attempts=2)


def _ocr_provider(model_name: str) -> str:
    """Extract the provider prefix from a 'provider:model' id (e.g. 'mistral')."""
    return model_name.split(":", 1)[0]


@workflow.defn
class OcrSubmitWorkflow:
    """Submit a document for OCR; child workflows store results asynchronously."""

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
            json.dumps(
                {
                    "content_id": file_content_ref.content_id,
                    "mime_type": file_content_ref.mime_type,
                    "file_size_bytes": file_content_ref.file_size_bytes,
                }
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

        gather_workflow_id = ""
        if chunk_count > 1:
            gather_handle = await workflow.start_child_workflow(
                "OcrGatherWorkflow",
                OcrGatherInput(
                    document_id=document_id,
                    chunk_document_ids=chunk_document_ids,
                    store_workflow_ids=[],
                    file_path=input.file_path,
                    total_pages=split_result.total_pages,
                ),
                id=f"ocr-gather-{document_id}",
                parent_close_policy=workflow.ParentClosePolicy.ABANDON,
                result_type=OcrStoreResult,
            )
            gather_workflow_id = gather_handle.id

        provider = _ocr_provider(input.model_name)
        batch_refs: list[OcrBatchRef] = []

        for i, chunk in enumerate(split_result.chunks):
            chunk_doc_id = chunk_document_ids[i]

            # Build the request body + mint the single correlation id, stash to S3.
            request_ref = await workflow.execute_activity(
                "build_ocr_request_blob",
                json.dumps(
                    {
                        "content_id": chunk.content_id,
                        "mime_type": chunk.mime_type,
                        "model_name": input.model_name,
                    }
                ),
                start_to_close_timeout=_BLOB_TIMEOUT,
                retry_policy=_IO_RETRY,
                result_type=OcrBatchRequestRef,
            )

            # Start the store child (its id is batch_jobs.workflow_id, so the poller
            # signals it). request_id is already minted, so the child knows it.
            store_handle = await workflow.start_child_workflow(
                "OcrStoreWorkflow",
                OcrStoreInput(
                    batch_id="",
                    request_id=request_ref.request_id,
                    document_id=chunk_doc_id,
                    file_path=input.file_path,
                    gather_workflow_id=gather_workflow_id,
                ),
                id=f"ocr-store-{chunk_doc_id}",
                parent_close_policy=workflow.ParentClosePolicy.ABANDON,
                result_type=OcrStoreResult,
            )

            # Record OCR-side submitted status (single-writer, own queue).
            await self._upsert_status(
                request_ref.request_id, chunk_doc_id, input.file_path, "submitted"
            )

            try:
                submit_result: BatchSubmitResult = await workflow.execute_activity(
                    "submit_batch_blob",
                    BatchSubmitSpiInput(
                        s3_key=request_ref.s3_key,
                        model=request_ref.model,
                        endpoint="/v1/ocr",
                        provider=provider,
                        custom_id=request_ref.request_id,
                    ),
                    task_queue=FORGE_TASK_QUEUE,
                    start_to_close_timeout=_SUBMIT_TIMEOUT,
                    retry_policy=_SUBMIT_RETRY,
                    result_type=BatchSubmitResult,
                )
            except Exception as exc:
                # Provider submit failed for good: write terminal failed status so
                # the store child (which would otherwise wait 25h) and the status
                # view both reflect it, then fail the workflow.
                await self._upsert_status(
                    request_ref.request_id,
                    chunk_doc_id,
                    input.file_path,
                    "failed",
                    error_message=str(exc),
                )
                raise

            # Record the platform batch_jobs row cross-queue: workflow_id is the
            # store child, so the poller signals it. Writes nothing on submit.
            await persist_block(
                PersistBatchSubmission(
                    request_id=submit_result.request_id,
                    batch_id=submit_result.batch_id,
                    workflow_id=store_handle.id,
                    provider=provider,
                ),
                task_queue=FORGE_TASK_QUEUE,
            )

            # The input blob is now redundant (the request blob is in S3); reaped
            # by bucket TTL, but delete eagerly to keep the store tidy.
            try:
                await workflow.execute_activity(
                    "delete_file_content_blob",
                    chunk.content_id,
                    start_to_close_timeout=_IO_TIMEOUT,
                    retry_policy=_IO_RETRY,
                )
            except Exception:
                workflow.logger.warning("Failed to delete blob %s after submit", chunk.content_id)

            batch_refs.append(
                OcrBatchRef(batch_id=submit_result.batch_id, request_id=submit_result.request_id)
            )

        workflow.logger.info(
            "OcrSubmit done (submitted): document_id=%s chunks=%d", document_id, chunk_count
        )
        return OcrSubmitResult(
            document_id=document_id, batch_refs=batch_refs, chunk_count=chunk_count
        )

    async def _upsert_status(
        self,
        request_id: str,
        document_id: str,
        file_path: str,
        status: str,
        *,
        error_message: str | None = None,
    ) -> None:
        await workflow.execute_activity(
            "upsert_ocr_status",
            json.dumps(
                {
                    "request_id": request_id,
                    "document_id": document_id,
                    "file_path": file_path,
                    "status": status,
                    "error_message": error_message,
                }
            ),
            start_to_close_timeout=_STATUS_TIMEOUT,
            retry_policy=_IO_RETRY,
        )
