"""OcrSubmitWorkflow — submit a document for OCR via batch API.

Steps:
1. Read file and store in database (returns lightweight ref)
2. Split into chunks (1 chunk for small files, N for large PDFs)
3. If multi-chunk: start OcrGatherWorkflow (receives completion signals)
4. For each chunk: start child OcrStoreWorkflow + submit batch request
5. Return OcrSubmitResult immediately (children run independently)
"""

from __future__ import annotations

import json
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from forge.ocr.models import (
        FileContentRef,
        OcrGatherInput,
        OcrStoreInput,
        OcrSubmitInput,
        OcrSubmitResult,
        SplitResult,
    )

_IO_TIMEOUT = timedelta(seconds=30)
_SPLIT_TIMEOUT = timedelta(seconds=120)
_SUBMIT_TIMEOUT = timedelta(seconds=60)
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

        # Step 3: If multi-chunk, start gather workflow first (so store
        # workflows can signal it upon completion)
        gather_workflow_id = ""
        chunk_document_ids: list[str] = []
        for chunk in split_result.chunks:
            if chunk_count == 1:
                chunk_document_ids.append(document_id)
            else:
                chunk_document_ids.append(
                    f"{document_id}__chunk_{chunk.chunk_index}"
                )

        if chunk_count > 1:
            gather_input = OcrGatherInput(
                document_id=document_id,
                chunk_document_ids=chunk_document_ids,
                store_workflow_ids=[],  # not needed — uses signals
                file_path=input.file_path,
                total_pages=split_result.total_pages,
            )
            gather_handle = await workflow.start_child_workflow(
                "OcrGatherWorkflow",
                gather_input,
                id=f"ocr-gather-{document_id}",
                parent_close_policy=workflow.ParentClosePolicy.ABANDON,
            )
            gather_workflow_id = gather_handle.id

        # Step 4: For each chunk, start child OcrStoreWorkflow + submit batch
        first_store_workflow_id = ""
        first_batch_id = ""
        first_request_id = ""

        for i, chunk in enumerate(split_result.chunks):
            chunk_doc_id = chunk_document_ids[i]

            # Start child OcrStoreWorkflow for this chunk
            store_input = OcrStoreInput(
                batch_id="",  # resolved from BatchResult signal
                request_id="",
                document_id=chunk_doc_id,
                file_path=input.file_path,
                gather_workflow_id=gather_workflow_id,
            )
            store_handle = await workflow.start_child_workflow(
                "OcrStoreWorkflow",
                store_input,
                id=f"ocr-store-{chunk_doc_id}",
                parent_close_policy=workflow.ParentClosePolicy.ABANDON,
            )

            if not first_store_workflow_id:
                first_store_workflow_id = store_handle.id

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
            workflow_id=first_store_workflow_id,
            chunk_count=chunk_count,
            total_pages=split_result.total_pages,
        )
