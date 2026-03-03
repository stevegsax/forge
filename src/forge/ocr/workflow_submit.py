"""OcrSubmitWorkflow — submit a document for OCR via batch API.

Steps:
1. Read file and store in database (returns lightweight ref)
2. Start child OcrStoreWorkflow (abandoned on parent close)
3. Submit batch request to Mistral (loads file from DB, cleans up BLOB)
4. Return OcrSubmitResult
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
        OcrSubmitInput,
        OcrSubmitResult,
    )

_IO_TIMEOUT = timedelta(seconds=30)
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

        # Step 2: Start child OcrStoreWorkflow
        store_input = OcrStoreInput(
            batch_id="",  # resolved from BatchResult signal in OcrStoreWorkflow
            request_id="",
            document_id=document_id,
            file_path=input.file_path,
        )
        store_handle = await workflow.start_child_workflow(
            "OcrStoreWorkflow",
            store_input,
            id=f"ocr-store-{document_id}",
            parent_close_policy=workflow.ParentClosePolicy.ABANDON,
        )

        # Step 3: Submit batch request
        submit_data = json.dumps({
            "submit_input": input.model_copy(update={"document_id": document_id}).model_dump(),
            "file_content_ref": file_content_ref.model_dump(),
            "store_workflow_id": store_handle.id,
        })
        result = await workflow.execute_activity(
            "submit_ocr_batch",
            submit_data,
            start_to_close_timeout=_SUBMIT_TIMEOUT,
            retry_policy=_IO_RETRY,
            result_type=OcrSubmitResult,
        )

        workflow.logger.info(
            "OcrSubmit done: batch_id=%s document_id=%s",
            result.batch_id,
            document_id,
        )

        return OcrSubmitResult(
            batch_id=result.batch_id,
            request_id=result.request_id,
            document_id=document_id,
            workflow_id=store_handle.id,
        )
