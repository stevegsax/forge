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
    from forge.workflow_blocks import (
        _PERSIST_RETRY,
        _PERSIST_SCHEDULE_TO_CLOSE,
    )

_PARSE_TIMEOUT = timedelta(seconds=30)
_STORE_TIMEOUT = timedelta(seconds=30)
_BATCH_WAIT_TIMEOUT = timedelta(hours=25)
_STATUS_TIMEOUT = timedelta(seconds=15)
_LOCAL_RETRY = RetryPolicy(maximum_attempts=2)
# Status cleanup on the error branch must not loop — we want to record the
# failure once and then let the original exception propagate.
_STATUS_NO_RETRY = RetryPolicy(maximum_attempts=1)


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
        request_id = result.request_id

        # Early error branches — the poller already advanced the batch_jobs
        # row to STORING; promote it to ERRORED before re-raising so the
        # row doesn't get stuck in STORING forever.
        if result.error:
            await self._mark_errored(request_id, f"OCR batch error: {result.error}")
            raise ApplicationError(f"OCR batch error: {result.error}")
        if not result.raw_response_json:
            await self._mark_errored(request_id, "OCR batch result has no response JSON")
            raise ApplicationError("OCR batch result has no response JSON")

        try:
            # Step 2: Parse OCR result
            parse_result = await workflow.execute_activity(
                "parse_ocr_result",
                result.raw_response_json,
                start_to_close_timeout=_PARSE_TIMEOUT,
                retry_policy=_LOCAL_RETRY,
                result_type=OcrParseResult,
            )

            # Step 3: Store OCR result
            store_data = json.dumps(
                {
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
                }
            )
            store_result = await workflow.execute_activity(
                "store_ocr_result",
                store_data,
                start_to_close_timeout=_STORE_TIMEOUT,
                schedule_to_close_timeout=_PERSIST_SCHEDULE_TO_CLOSE,
                retry_policy=_PERSIST_RETRY,
                result_type=OcrStoreResult,
            )
        except Exception as exc:
            # Parse or store raised — record the failure on the batch_jobs
            # row before propagating so the list view doesn't leave this
            # chunk in STORING.
            await self._mark_errored(request_id, f"Parse/store failed: {exc}")
            raise

        # Step 4: Mark the batch_jobs row SUCCEEDED now that ocr_results
        # is committed. This is the only place SUCCEEDED is written — the
        # poller only advances to STORING on signal delivery.
        await workflow.execute_activity(
            "update_batch_job_status",
            json.dumps(
                {
                    "request_id": request_id,
                    "status": "succeeded",
                }
            ),
            start_to_close_timeout=_STATUS_TIMEOUT,
            retry_policy=_LOCAL_RETRY,
        )

        # Step 5: Signal gather workflow if this is a chunk
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

    async def _mark_errored(self, request_id: str, error_message: str) -> None:
        """Update the batch_jobs row to ERRORED. Never raises."""
        try:
            await workflow.execute_activity(
                "update_batch_job_status",
                json.dumps(
                    {
                        "request_id": request_id,
                        "status": "errored",
                        "error_message": error_message,
                    }
                ),
                start_to_close_timeout=_STATUS_TIMEOUT,
                retry_policy=_STATUS_NO_RETRY,
            )
        except Exception:
            workflow.logger.warning(
                "Failed to mark batch_jobs row as errored: request_id=%s",
                request_id,
            )
