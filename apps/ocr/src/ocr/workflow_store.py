"""OcrStoreWorkflow — wait for the platform batch result, then store it.

The platform poller signals this workflow (``batch_result_received``) when the
provider batch completes, delivering the result inline or by S3 pointer. This
workflow owns all OCR-side work: it resolves the result, stores images + text, and
writes its own terminal status (``ocr_job_status``). It never touches ``batch_jobs``
(the platform single-writer owns that).
"""

from __future__ import annotations

import json
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from forge_contracts.models import BatchResult

    from ocr.models import OcrStoreInput, OcrStoreResult
    from ocr.persist import _PERSIST_RETRY, _PERSIST_SCHEDULE_TO_CLOSE

_STORE_TIMEOUT = timedelta(seconds=60)
_BATCH_WAIT_TIMEOUT = timedelta(hours=25)
_STATUS_TIMEOUT = timedelta(seconds=15)
_STATUS_NO_RETRY = RetryPolicy(maximum_attempts=1)


@workflow.defn
class OcrStoreWorkflow:
    """Wait for the OCR batch result, store text + images, write terminal status."""

    def __init__(self) -> None:
        self._batch_results: dict[str, BatchResult] = {}

    @workflow.signal
    async def batch_result_received(self, result: BatchResult) -> None:
        """Receive the batch result from the platform poller.

        Keyed by request_id, first delivery wins: at-least-once signalling can
        redeliver a result, and a stale/duplicate for another request must not
        be mistaken for this one's (INTERIM; the signal path is deleted in
        Phase 4).
        """
        self._batch_results.setdefault(result.request_id, result)

    @workflow.run
    async def run(self, input: OcrStoreInput) -> OcrStoreResult:
        workflow.logger.info("OcrStore started: document_id=%s", input.document_id)

        # Wait for the result signal. A genuinely stuck provider batch raises
        # TimeoutError *inside* the workflow (catchable) — record a clean terminal
        # failure and surface an ApplicationError rather than dying with a raw
        # timeout that leaves the status row stuck.
        try:
            await workflow.wait_condition(
                lambda: input.request_id in self._batch_results,
                timeout=_BATCH_WAIT_TIMEOUT,
            )
        except TimeoutError as exc:
            await self._mark_failed(input, f"OCR batch wait timed out after {_BATCH_WAIT_TIMEOUT}")
            raise ApplicationError("OCR batch wait timed out") from exc

        result = self._batch_results[input.request_id]

        if result.error:
            await self._mark_failed(input, f"OCR batch error: {result.error}")
            raise ApplicationError(f"OCR batch error: {result.error}")
        if result.raw_response_json is None and result.s3_key is None:
            await self._mark_failed(input, "OCR batch result has no body")
            raise ApplicationError("OCR batch result has no body")

        try:
            store_result: OcrStoreResult = await workflow.execute_activity(
                "store_ocr_result",
                json.dumps(
                    {
                        "request_id": input.request_id,
                        "document_id": input.document_id,
                        "file_path": input.file_path,
                        "batch_id": result.batch_id,
                        "workflow_id": workflow.info().workflow_id,
                        "raw_response_json": result.raw_response_json,
                        "s3_key": result.s3_key,
                    }
                ),
                start_to_close_timeout=_STORE_TIMEOUT,
                schedule_to_close_timeout=_PERSIST_SCHEDULE_TO_CLOSE,
                retry_policy=_PERSIST_RETRY,
                result_type=OcrStoreResult,
            )
        except Exception as exc:
            await self._mark_failed(input, f"Parse/store failed: {exc}")
            raise

        # Signal the gather workflow if this is one chunk of a split document.
        if input.gather_workflow_id:
            gather_handle = workflow.get_external_workflow_handle(input.gather_workflow_id)
            await gather_handle.signal("chunk_completed", input.document_id)

        workflow.logger.info(
            "OcrStore done: document_id=%s text_length=%d",
            store_result.document_id,
            store_result.text_length,
        )
        return store_result

    async def _mark_failed(self, input: OcrStoreInput, error_message: str) -> None:
        """Write a terminal ``failed`` status to ocr_job_status. Never raises."""
        try:
            await workflow.execute_activity(
                "upsert_ocr_status",
                json.dumps(
                    {
                        "request_id": input.request_id,
                        "document_id": input.document_id,
                        "file_path": input.file_path,
                        "status": "failed",
                        "error_message": error_message,
                    }
                ),
                start_to_close_timeout=_STATUS_TIMEOUT,
                retry_policy=_STATUS_NO_RETRY,
            )
        except Exception:
            workflow.logger.warning(
                "Failed to write failed status: request_id=%s", input.request_id
            )
