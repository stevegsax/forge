"""OcrStoreWorkflow — poll one OCR batch on a timer, then fetch and store it.

Owns a single Mistral OCR batch end-to-end under the D88 timer-loop transport
(T4.2): it polls the provider's normalized status on a timer (the shared skeleton
in ``sax_platform.temporal.polling``) until the batch ends or the 25h ceiling
passes, then downloads and stores this batch's result in ONE activity — the
result bytes never transit workflow history. It writes its own terminal
``ocr_job_status`` and records the provider-lifecycle outcome on the platform
``batch_jobs`` ledger cross-queue (an activity call, not a signal). No signals.

Failure symmetry mirrors forge's ``batch_submit_and_wait``: a give-up at the
ceiling persists MISSING; a provider-terminal status persists FAILED/EXPIRED; an
error from the fetch/store activity persists FAILED. Every failure path also
marks ``ocr_job_status`` failed and raises a non-retryable ``ApplicationError``.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError

with workflow.unsafe.imports_passed_through():
    from sax_platform.contracts.constants import FORGE_TASK_QUEUE
    from sax_platform.contracts.models import BatchJobStatus
    from sax_platform.contracts.persist import (
        PERSIST_SCHEDULE_TO_CLOSE,
        PersistBatchOutcome,
        persist_block,
    )
    from sax_platform.temporal.polling import (
        BATCH_WAIT_CEILING,
        BackoffSchedule,
        wait_batch_ended,
    )
    from sax_platform.temporal.retries import IO_RETRY, PERSIST_RETRY

    from ocr.models import (
        OcrBatchStatusInput,
        OcrFetchStoreInput,
        OcrProcessingStatus,
        OcrStatusUpsertInput,
        OcrStoreInput,
        OcrStoreResult,
    )

_FETCH_STORE_TIMEOUT = timedelta(minutes=5)
_STATUS_TIMEOUT = timedelta(seconds=60)
_STATUS_UPSERT_TIMEOUT = timedelta(seconds=15)
_STATUS_NO_RETRY = RetryPolicy(maximum_attempts=1)
# Modest per-waiter jitter (10%) spreads a burst of concurrent waiters off the
# Mistral status endpoint; the backoff defaults (300s for the first hour, then
# doubling to a 1800s cap) come straight from the platform schedule (D88).
_POLL_JITTER_FRACTION = 0.1


@workflow.defn
class OcrStoreWorkflow:
    """Poll an OCR batch to completion, fetch + store it, write terminal status."""

    @workflow.run
    async def run(self, input: OcrStoreInput) -> OcrStoreResult:
        workflow.logger.info(
            "OcrStore started: document_id=%s batch_id=%s", input.document_id, input.batch_id
        )

        async def _poll_status() -> str:
            state: str = await workflow.execute_activity(
                "ocr_batch_status",
                OcrBatchStatusInput(batch_id=input.batch_id),
                start_to_close_timeout=_STATUS_TIMEOUT,
                retry_policy=IO_RETRY,
                result_type=str,
            )
            return state

        outcome = await wait_batch_ended(
            _poll_status,
            schedule=BackoffSchedule(jitter_fraction=_POLL_JITTER_FRACTION),
            ceiling=BATCH_WAIT_CEILING,
        )
        if outcome == "gave_up":
            await self._fail(
                input, BatchJobStatus.MISSING, f"OCR batch wait exceeded {BATCH_WAIT_CEILING}"
            )
            raise ApplicationError(
                f"OCR batch wait exceeded {BATCH_WAIT_CEILING} for request {input.request_id}",
                non_retryable=True,
            )
        if outcome in ("failed", "expired", "canceled"):
            terminal = BatchJobStatus.EXPIRED if outcome == "expired" else BatchJobStatus.FAILED
            await self._fail(input, terminal, f"provider batch {outcome}")
            raise ApplicationError(
                f"OCR batch {input.batch_id} {outcome} for request {input.request_id}",
                non_retryable=True,
            )

        try:
            store_result: OcrStoreResult = await workflow.execute_activity(
                "fetch_and_store_ocr_result",
                OcrFetchStoreInput(
                    batch_id=input.batch_id,
                    request_id=input.request_id,
                    document_id=input.document_id,
                    file_path=input.file_path,
                    workflow_id=workflow.info().workflow_id,
                ),
                start_to_close_timeout=_FETCH_STORE_TIMEOUT,
                schedule_to_close_timeout=PERSIST_SCHEDULE_TO_CLOSE,
                retry_policy=PERSIST_RETRY,
                result_type=OcrStoreResult,
            )
        except ActivityError as exc:
            await self._fail(input, BatchJobStatus.FAILED, f"fetch/store failed: {exc}")
            raise

        # Success: record the terminal ENDED outcome on the platform ledger.
        await persist_block(
            PersistBatchOutcome(request_id=input.request_id, status=BatchJobStatus.ENDED.value),
            task_queue=FORGE_TASK_QUEUE,
        )
        workflow.logger.info(
            "OcrStore done: document_id=%s text_length=%d",
            store_result.document_id,
            store_result.text_length,
        )
        return store_result

    async def _fail(self, input: OcrStoreInput, status: BatchJobStatus, message: str) -> None:
        """Mark ocr_job_status failed AND record the terminal outcome cross-queue.

        Symmetric with the success path: the OCR-side status write (best-effort,
        never raises) plus a ``PersistBatchOutcome`` on the platform ``batch_jobs``
        ledger cross-queue (survivable via ``persist_block``).
        """
        await self._mark_failed(input, message)
        await persist_block(
            PersistBatchOutcome(
                request_id=input.request_id, status=status.value, error_message=message
            ),
            task_queue=FORGE_TASK_QUEUE,
        )

    async def _mark_failed(self, input: OcrStoreInput, error_message: str) -> None:
        """Write a terminal ``failed`` status to ocr_job_status. Never raises."""
        try:
            await workflow.execute_activity(
                "upsert_ocr_status",
                OcrStatusUpsertInput(
                    request_id=input.request_id,
                    document_id=input.document_id,
                    file_path=input.file_path,
                    status=OcrProcessingStatus.FAILED,
                    error_message=error_message,
                ),
                start_to_close_timeout=_STATUS_UPSERT_TIMEOUT,
                retry_policy=_STATUS_NO_RETRY,
            )
        except Exception:
            workflow.logger.warning(
                "Failed to write failed status: request_id=%s", input.request_id
            )
