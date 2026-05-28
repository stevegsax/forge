"""OcrSyncWorkflow — synchronous OCR without batch API.

Calls the OCR endpoint directly and waits for the response, bypassing
the batch-submit → poll → signal pipeline.  Best suited for small
documents where batch overhead is not justified.

Steps:
1. Duplicate detection (optional)
2. Read file and store in database (returns lightweight ref)
3. Split into chunks (1 chunk for small files, N for large PDFs)
4. For each chunk: call OCR synchronously and store result
5. If multi-chunk: reassemble chunks
6. Return OcrStoreResult
"""

from __future__ import annotations

import json
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from forge.ocr.models import (
        FileContentRef,
        OcrDuplicateCheckResult,
        OcrStoreResult,
        OcrSyncCallResult,
        OcrSyncInput,
        SplitResult,
    )
    from forge.persist_models import PersistOcrResult
    from forge.workflow_blocks import (
        _PERSIST_RETRY,
        _PERSIST_SCHEDULE_TO_CLOSE,
    )
    from forge.workflow_blocks import persist_block as _persist_block

_IO_TIMEOUT = timedelta(seconds=30)
_SPLIT_TIMEOUT = timedelta(seconds=120)
_OCR_CALL_TIMEOUT = timedelta(minutes=10)
_REASSEMBLE_TIMEOUT = timedelta(seconds=60)
_IO_RETRY = RetryPolicy(maximum_attempts=2)
_OCR_RETRY = RetryPolicy(maximum_attempts=3)


@workflow.defn
class OcrSyncWorkflow:
    """Submit a document for synchronous OCR processing."""

    @workflow.run
    async def run(self, input: OcrSyncInput) -> OcrStoreResult:
        document_id = input.document_id or str(workflow.uuid4())
        workflow.logger.info(
            "OcrSync started: file=%s document_id=%s",
            input.file_path,
            document_id,
        )

        # Step 0: Duplicate detection (hash-based, path format irrelevant)
        if not input.skip_duplicate_detection:
            dup_check = await workflow.execute_activity(
                "check_ocr_duplicate",
                input.file_path,
                start_to_close_timeout=_IO_TIMEOUT,
                retry_policy=_IO_RETRY,
                result_type=OcrDuplicateCheckResult,
            )
            if dup_check.is_duplicate:
                workflow.logger.info(
                    "Duplicate document: file=%s existing_document_id=%s",
                    input.file_path,
                    dup_check.existing_document_id,
                )
                return OcrStoreResult(
                    document_id=dup_check.existing_document_id,
                    text_length=0,
                    page_count=0,
                    stored=False,
                    skipped=True,
                    skip_reason="Duplicate document",
                )

        # Step 1: Read file and store in database
        file_content_ref = await workflow.execute_activity(
            "read_and_store_file_content",
            input.file_path,
            start_to_close_timeout=_IO_TIMEOUT,
            retry_policy=_IO_RETRY,
            result_type=FileContentRef,
        )

        # Step 2: Split into chunks
        split_data = json.dumps(
            {
                "content_id": file_content_ref.content_id,
                "mime_type": file_content_ref.mime_type,
                "file_size_bytes": file_content_ref.file_size_bytes,
            }
        )
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

        # Step 3: Call OCR synchronously for each chunk
        chunk_document_ids: list[str] = []
        result = OcrStoreResult(document_id=document_id, text_length=0, page_count=0, stored=False)
        for chunk in split_result.chunks:
            if chunk_count == 1:
                chunk_doc_id = document_id
            else:
                chunk_doc_id = f"{document_id}__chunk_{chunk.chunk_index}"
            chunk_document_ids.append(chunk_doc_id)

            ocr_data = json.dumps(
                {
                    "content_id": chunk.content_id,
                    "mime_type": chunk.mime_type,
                    "model_name": input.model_name,
                    "document_id": chunk_doc_id,
                    "file_path": input.file_path,
                    "workflow_id": workflow.info().workflow_id,
                }
            )
            call_result: OcrSyncCallResult = await workflow.execute_activity(
                "call_ocr_sync",
                ocr_data,
                start_to_close_timeout=_OCR_CALL_TIMEOUT,
                retry_policy=_OCR_RETRY,
                result_type=OcrSyncCallResult,
            )
            # Survivably persist the ocr_results row (idempotent on document_id);
            # the expensive OCR call already returned and is never re-run on a blip.
            await _persist_block(
                PersistOcrResult(
                    document_id=call_result.document_id,
                    file_path=call_result.file_path,
                    text=call_result.text,
                    model_name=call_result.model_name,
                    input_tokens=call_result.input_tokens,
                    output_tokens=call_result.output_tokens,
                    page_count=call_result.page_count,
                    batch_id="",
                    workflow_id=workflow.info().workflow_id,
                    file_hash=call_result.file_hash,
                )
            )
            result = OcrStoreResult(
                document_id=call_result.document_id,
                text_length=len(call_result.text),
                page_count=call_result.page_count,
                stored=True,
            )

        # Step 4: If multi-chunk, reassemble
        if chunk_count > 1:
            reassemble_data = json.dumps(
                {
                    "document_id": document_id,
                    "chunk_document_ids": chunk_document_ids,
                    "file_path": input.file_path,
                    "total_pages": split_result.total_pages,
                }
            )
            result = await workflow.execute_activity(
                "reassemble_ocr_chunks",
                reassemble_data,
                start_to_close_timeout=_REASSEMBLE_TIMEOUT,
                schedule_to_close_timeout=_PERSIST_SCHEDULE_TO_CLOSE,
                retry_policy=_PERSIST_RETRY,
                result_type=OcrStoreResult,
            )

        workflow.logger.info(
            "OcrSync done: document_id=%s text_length=%d page_count=%d",
            result.document_id,
            result.text_length,
            result.page_count,
        )

        return result
