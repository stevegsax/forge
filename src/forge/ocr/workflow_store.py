"""OcrStoreWorkflow — wait for OCR batch result and store it.

Steps:
1. Wait for batch_result_received signal (up to 25h)
2. Parse the raw OCR result
3. Store the extracted text in the database
4. Return OcrStoreResult

The batch poller automatically signals this workflow when the batch
completes, using the workflow_id stored in the batch_jobs table.
"""

from __future__ import annotations

import json
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from forge.models import BatchResult  # noqa: TC001
    from forge.ocr.models import (
        OcrParseResult,
        OcrStoreInput,
        OcrStoreResult,
    )

_PARSE_TIMEOUT = timedelta(seconds=30)
_STORE_TIMEOUT = timedelta(seconds=30)
_BATCH_WAIT_TIMEOUT = timedelta(hours=25)
_LOCAL_RETRY = RetryPolicy(maximum_attempts=2)


@workflow.defn
class OcrStoreWorkflow:
    """Wait for OCR batch result, parse, and store.

    Canonical signal pattern: __init__ sets up state, @workflow.signal
    appends to list, workflow.wait_condition checks the list.
    This pattern must be copy-pasted per workflow class because
    Temporal requires @workflow.signal on class methods.
    """

    def __init__(self) -> None:
        self._batch_results: list[BatchResult] = []

    @workflow.signal
    async def batch_result_received(self, result: BatchResult) -> None:
        """Receive batch result from the poller."""
        self._batch_results.append(result)

    @workflow.run
    async def run(self, input: OcrStoreInput) -> OcrStoreResult:
        workflow.logger.info(
            "OcrStore started: document_id=%s",
            input.document_id,
        )

        # Step 1: Wait for batch result signal
        await workflow.wait_condition(
            lambda: len(self._batch_results) > 0,
            timeout=_BATCH_WAIT_TIMEOUT,
        )
        result = self._batch_results.pop(0)
        if result.error:
            raise ApplicationError(f"OCR batch error: {result.error}")
        if not result.raw_response_json:
            raise ApplicationError("OCR batch result has no response JSON")

        # Step 2: Parse OCR result
        parse_result = await workflow.execute_activity(
            "parse_ocr_result",
            result.raw_response_json,
            start_to_close_timeout=_PARSE_TIMEOUT,
            retry_policy=_LOCAL_RETRY,
            result_type=OcrParseResult,
        )

        # Step 3: Store OCR result
        store_data = json.dumps({
            "document_id": input.document_id,
            "file_path": input.file_path,
            "text": parse_result.text,
            "model_name": parse_result.model_name,
            "input_tokens": parse_result.input_tokens,
            "output_tokens": parse_result.output_tokens,
            "page_count": parse_result.page_count,
            "batch_id": result.batch_id,
            "workflow_id": workflow.info().workflow_id,
            "image_ids": parse_result.image_ids,
        })
        store_result = await workflow.execute_activity(
            "store_ocr_result",
            store_data,
            start_to_close_timeout=_STORE_TIMEOUT,
            retry_policy=_LOCAL_RETRY,
            result_type=OcrStoreResult,
        )

        # Step 4: Signal gather workflow if this is a chunk
        if input.gather_workflow_id:
            gather_handle = workflow.get_external_workflow_handle(
                input.gather_workflow_id,
            )
            await gather_handle.signal("chunk_completed", input.document_id)

        workflow.logger.info(
            "OcrStore done: document_id=%s text_length=%d",
            store_result.document_id,
            store_result.text_length,
        )

        return store_result
