"""OcrStoreWorkflow — a signal-wait state machine for one OCR batch.

Owns a single Mistral OCR batch end-to-end under the T4.4 hint-signal transport.
Instead of polling the provider on its own timer, the workflow waits on a small
state machine that a separate stateless tracker advances by broadcasting
``ocr_status_hint`` signals — the sanctioned hint pattern: a hint carries only a
normalized status string, never a result payload. When the machine reaches a
terminal state the workflow acts: on ``ended`` it downloads and stores this
batch's result in ONE activity keyed by its own ``batch_id``/``request_id`` (the
result bytes never transit workflow history and never ride the signal); on a
provider-terminal status it fails. If no hint moves the machine to terminal
within the 25h ceiling, the wait times out and the batch is recorded MISSING.

The transition rules live in the pure module-level ``next_state`` function; the
signal handler is the only mutator of ``self._state``. ``@workflow.init`` seeds
the input and the initial ``pending`` state before ``run`` begins, so a hint that
races ahead of the wait is still honored.

Failure symmetry mirrors forge's ``batch_submit_and_wait``: a give-up at the
ceiling persists MISSING; a provider-terminal status persists FAILED/EXPIRED; an
error from the fetch/store activity persists FAILED. Every failure path also
marks ``ocr_job_status`` failed and raises a non-retryable ``ApplicationError``.
Nothing signals this workflow in production yet (the tracker is a later stage) —
an accepted gap; there is deliberately no fallback polling.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Final

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
    from sax_platform.temporal.polling import BATCH_WAIT_CEILING
    from sax_platform.temporal.retries import PERSIST_RETRY

    from ocr.models import (
        OcrFetchStoreInput,
        OcrProcessingStatus,
        OcrStatusHint,
        OcrStatusUpsertInput,
        OcrStoreInput,
        OcrStoreResult,
    )

_FETCH_STORE_TIMEOUT = timedelta(minutes=5)
_STATUS_UPSERT_TIMEOUT = timedelta(seconds=15)
_STATUS_NO_RETRY = RetryPolicy(maximum_attempts=1)

# The normalized batch-status vocabulary the tracker broadcasts. The four terminal
# states end the wait; ``pending``/``in_progress`` are the two in-flight states.
TERMINAL: Final = frozenset({"ended", "failed", "expired", "canceled"})
_IN_FLIGHT: Final = frozenset({"pending", "in_progress"})
_KNOWN: Final = TERMINAL | _IN_FLIGHT


def next_state(current: str, hint_state: str) -> str | None:
    """Advance the OCR store state machine by one hint (pure).

    Returns the state to adopt, or ``None`` when the hint is a same-state no-op or
    a rejected transition — the caller distinguishes the two by comparing
    ``hint_state`` to ``current``. Allowed moves: ``pending -> in_progress``,
    ``pending -> <terminal>``, and ``in_progress -> <terminal>``. Rejected (``None``):
    any move already at ``current``, any backward move (``in_progress -> pending``),
    any move out of a terminal state, and any unknown status string.
    """
    if hint_state not in _KNOWN or hint_state == current:
        return None
    if current == "pending" and (hint_state == "in_progress" or hint_state in TERMINAL):
        return hint_state
    if current == "in_progress" and hint_state in TERMINAL:
        return hint_state
    return None


@workflow.defn
class OcrStoreWorkflow:
    """Wait on tracker status hints, then fetch + store and write terminal status."""

    @workflow.init
    def __init__(self, input: OcrStoreInput) -> None:
        # Seed input + initial state before run() so a hint that arrives before the
        # wait begins is honored (the handler already sees self._input/self._state).
        self._input = input
        self._state = "pending"

    @workflow.signal(name="ocr_status_hint")
    def on_status_hint(self, hint: OcrStatusHint) -> None:
        """Advance the state machine from a tracker hint — the only state mutator."""
        if hint.batch_id != self._input.batch_id:
            workflow.logger.warning(
                "OcrStore ignoring hint for foreign batch: got=%s want=%s",
                hint.batch_id,
                self._input.batch_id,
            )
            return
        advanced = next_state(self._state, hint.state)
        if advanced is None:
            if hint.state != self._state:
                workflow.logger.warning(
                    "OcrStore rejected hint transition: current=%s hint=%s",
                    self._state,
                    hint.state,
                )
            return
        self._state = advanced

    @workflow.run
    async def run(self, input: OcrStoreInput) -> OcrStoreResult:
        workflow.logger.info(
            "OcrStore started: document_id=%s batch_id=%s", input.document_id, input.batch_id
        )

        try:
            await workflow.wait_condition(
                lambda: self._state in TERMINAL, timeout=BATCH_WAIT_CEILING
            )
        except TimeoutError as exc:
            await self._fail(
                input, BatchJobStatus.MISSING, f"OCR batch wait exceeded {BATCH_WAIT_CEILING}"
            )
            raise ApplicationError(
                f"OCR batch wait exceeded {BATCH_WAIT_CEILING} for request {input.request_id}",
                non_retryable=True,
            ) from exc

        if self._state in ("failed", "expired", "canceled"):
            terminal = BatchJobStatus.EXPIRED if self._state == "expired" else BatchJobStatus.FAILED
            await self._fail(input, terminal, f"provider batch {self._state}")
            raise ApplicationError(
                f"OCR batch {input.batch_id} {self._state} for request {input.request_id}",
                non_retryable=True,
            )

        # Terminal state is ``ended``: download + store this batch's result. A
        # duplicate/late ``ended`` hint is absorbed as a same-state no-op, so this
        # fetch runs exactly once.
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
