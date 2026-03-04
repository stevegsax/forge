"""OcrSubmitWorkflow — submit a document for OCR via batch API.

Steps:
1. Read file and store in database (returns lightweight ref)
2. Split into chunks (1 chunk for small files, N for large PDFs)
3. For each chunk: start child OcrStoreWorkflow + submit batch request
4. Await all OcrStoreWorkflow handles (any failure → workflow fails)
5. If multi-chunk: reassemble into a single OCR result
6. Return OcrSubmitResult
"""

from __future__ import annotations

import json
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from forge.ocr.models import (
        FileContentRef,
        OcrStoreInput,
        OcrStoreResult,
        OcrSubmitInput,
        OcrSubmitResult,
        SplitResult,
    )

_IO_TIMEOUT = timedelta(seconds=30)
_SPLIT_TIMEOUT = timedelta(seconds=120)
_SUBMIT_TIMEOUT = timedelta(seconds=60)
_REASSEMBLE_TIMEOUT = timedelta(seconds=60)
_IO_RETRY = RetryPolicy(maximum_attempts=2)


@workflow.defn
class OcrSubmitWorkflow:
    """Submit a document for OCR processing via batch API."""

    @workflow.run
    async def run(self, input: OcrSubmitInput) -> OcrSubmitResult:
        document_id = input.document_id or str(workflow.uuid4())
        workflow.logger.info(
            "OcrSubmit started: file=%s document_id=%s",
            input.file_path,
            document_id,
        )

        # Step 1: Read file and store in database (returns lightweight ref)
        file_content_ref = await workflow.execute_activity(
            "read_and_store_file_content",
            input.file_path,
            start_to_close_timeout=_IO_TIMEOUT,
            retry_policy=_IO_RETRY,
            result_type=FileContentRef,
        )

        # Step 2: Split into chunks
        split_data = json.dumps({
            "content_id": file_content_ref.content_id,
            "mime_type": file_content_ref.mime_type,
            "file_size_bytes": file_content_ref.file_size_bytes,
        })
        split_result = await workflow.execute_activity(
            "split_file_into_chunks",
            split_data,
            start_to_close_timeout=_SPLIT_TIMEOUT,
            retry_policy=_IO_RETRY,
            result_type=SplitResult,
        )

        chunk_count = len(split_result.chunks)
        workflow.logger.info(
            "Split result: %d chunk(s), %d total pages",
            chunk_count,
            split_result.total_pages,
        )

        # Step 3: For each chunk, start child OcrStoreWorkflow + submit batch
        store_handles = []
        chunk_document_ids: list[str] = []
        first_batch_id = ""
        first_request_id = ""

        for chunk in split_result.chunks:
            # Single chunk: use real document_id. Multi-chunk: use suffix.
            if chunk_count == 1:
                chunk_doc_id = document_id
            else:
                chunk_doc_id = f"{document_id}__chunk_{chunk.chunk_index}"

            chunk_document_ids.append(chunk_doc_id)

            # Start child OcrStoreWorkflow for this chunk
            store_input = OcrStoreInput(
                batch_id="",  # resolved from BatchResult signal
                request_id="",
                document_id=chunk_doc_id,
                file_path=input.file_path,
            )
            store_handle = await workflow.start_child_workflow(
                "OcrStoreWorkflow",
                store_input,
                id=f"ocr-store-{chunk_doc_id}",
                parent_close_policy=workflow.ParentClosePolicy.ABANDON,
            )
            store_handles.append(store_handle)

            # Submit batch request for this chunk
            chunk_ref_dict = {
                "content_id": chunk.content_id,
                "mime_type": chunk.mime_type,
                "file_size_bytes": chunk.file_size_bytes,
            }
            submit_data = json.dumps({
                "submit_input": input.model_copy(
                    update={"document_id": chunk_doc_id}
                ).model_dump(),
                "file_content_ref": chunk_ref_dict,
                "store_workflow_id": store_handle.id,
            })
            result = await workflow.execute_activity(
                "submit_ocr_batch",
                submit_data,
                start_to_close_timeout=_SUBMIT_TIMEOUT,
                retry_policy=_IO_RETRY,
                result_type=OcrSubmitResult,
            )

            if not first_batch_id:
                first_batch_id = result.batch_id
                first_request_id = result.request_id

        # Step 4: Await all OcrStoreWorkflow children
        for handle in store_handles:
            await handle.result()

        # Step 5: If multi-chunk, reassemble into single result
        if chunk_count > 1:
            reassemble_data = json.dumps({
                "document_id": document_id,
                "chunk_document_ids": chunk_document_ids,
                "file_path": input.file_path,
                "total_pages": split_result.total_pages,
            })
            await workflow.execute_activity(
                "reassemble_ocr_chunks",
                reassemble_data,
                start_to_close_timeout=_REASSEMBLE_TIMEOUT,
                retry_policy=_IO_RETRY,
                result_type=OcrStoreResult,
            )

        workflow.logger.info(
            "OcrSubmit done: batch_id=%s document_id=%s chunks=%d",
            first_batch_id,
            document_id,
            chunk_count,
        )

        return OcrSubmitResult(
            batch_id=first_batch_id,
            request_id=first_request_id,
            document_id=document_id,
            workflow_id=store_handles[0].id if store_handles else "",
            chunk_count=chunk_count,
            total_pages=split_result.total_pages,
        )
