"""OcrBatchTrackerWorkflow — the stateless status-hint broadcast tracker (T4.4).

This is the sanctioned hint pattern's producer. A Schedule fires one short run
every couple of minutes; each run is a single cycle with NO memory of any other
run — it holds no state between runs, never continues-as-new, and is safe under
concurrent or duplicate execution. One cycle:

  1. list the in-flight OCR submissions worth sweeping (``list_live_ocr_jobs``);
  2. if there are none, record a ``(0, 0)`` heartbeat and return — an idle cycle
     makes ZERO provider calls (the Mistral sweep never runs);
  3. otherwise sweep Mistral's batch-list endpoint once (``sweep_mistral_batches``)
     for the current status of every recent batch;
  4. for each live job Mistral reported on, broadcast an ``ocr_status_hint`` signal
     to that job's waiting ``OcrStoreWorkflow`` — carrying only a normalized status
     string, NEVER a result payload. The receiver fetches its own result keyed by
     its ``batch_id``/``request_id``; the hint merely advances its state machine,
     which absorbs duplicate and out-of-order hints. Hints are sent unconditionally
     (no change detection) — the latest status, every cycle.
  5. record a heartbeat with the observed live-job count and the number of hints
     actually delivered.

A signal that bounces (the target store child closed the cycle it completed, or
was never found) is logged and skipped — an expected one-cycle overlap, not an
error — and excluded from the delivered-hint count. Because the receivers' state
machines are idempotent, a missed or duplicated hint is harmless: correctness
never depends on any single broadcast landing.

The heartbeat is infrastructure telemetry, not task state: a stale heartbeat is a
system-level alert for an operator, and no OCR task watches it or depends on it.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Final

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from ocr.models import (
        OcrStatusHint,
        TrackerHeartbeatInput,
        TrackerLiveJob,
    )

# Short IO activities; the file-conventional two-attempt IO retry preset (matching
# the sibling ocr workflows). All three tracker activities are idempotent reads or
# an upsert, so a single retry on a transient blip is safe.
_IO_RETRY: Final = RetryPolicy(maximum_attempts=2)

_LIST_TIMEOUT: Final = timedelta(seconds=60)  # DB join over live jobs
_SWEEP_TIMEOUT: Final = timedelta(seconds=120)  # Mistral list endpoint may page
_HEARTBEAT_TIMEOUT: Final = timedelta(seconds=30)  # single-row upsert

_STATUS_HINT_SIGNAL: Final = "ocr_status_hint"


@workflow.defn
class OcrBatchTrackerWorkflow:
    """One stateless cycle: list live jobs, sweep Mistral, broadcast status hints."""

    @workflow.run
    async def run(self) -> None:
        live_jobs: list[TrackerLiveJob] = await workflow.execute_activity(
            "list_live_ocr_jobs",
            start_to_close_timeout=_LIST_TIMEOUT,
            retry_policy=_IO_RETRY,
            result_type=list[TrackerLiveJob],
        )

        if not live_jobs:
            # Idle cycle: no live jobs → make ZERO provider calls (the sweep does
            # NOT run) and record a (0, 0) heartbeat so a wedged tracker still shows.
            await self._record_heartbeat(live_jobs=0, hints_sent=0)
            workflow.logger.info("OcrBatchTracker idle cycle: no live jobs")
            return

        sweep: dict[str, str] = await workflow.execute_activity(
            "sweep_mistral_batches",
            start_to_close_timeout=_SWEEP_TIMEOUT,
            retry_policy=_IO_RETRY,
            result_type=dict[str, str],
        )

        hints_sent = 0
        for job in live_jobs:
            state = sweep.get(job.batch_id)
            if state is None:
                # Mistral did not report this batch this cycle — not signaled now;
                # a later cycle picks it up once the provider lists it.
                continue
            if await self._broadcast(job, state):
                hints_sent += 1

        await self._record_heartbeat(live_jobs=len(live_jobs), hints_sent=hints_sent)
        workflow.logger.info(
            "OcrBatchTracker cycle done: live_jobs=%d hints_sent=%d",
            len(live_jobs),
            hints_sent,
        )

    async def _broadcast(self, job: TrackerLiveJob, state: str) -> bool:
        """Signal one store child with the latest status; True if it landed.

        A bounced signal (the target store child closed the cycle it completed, or
        was never found) surfaces as an ``ApplicationError`` from the external
        signal; it is logged and swallowed — an expected one-cycle overlap — and
        counts as not delivered. The receiver's state machine absorbs any hint a
        surviving child missed this cycle.
        """
        try:
            handle = workflow.get_external_workflow_handle(job.workflow_id)
            await handle.signal(
                _STATUS_HINT_SIGNAL,
                OcrStatusHint(batch_id=job.batch_id, state=state),
            )
        except ApplicationError:
            workflow.logger.warning(
                "OcrBatchTracker signal bounced: workflow_id=%s batch_id=%s state=%s",
                job.workflow_id,
                job.batch_id,
                state,
            )
            return False
        return True

    async def _record_heartbeat(self, *, live_jobs: int, hints_sent: int) -> None:
        await workflow.execute_activity(
            "record_tracker_heartbeat",
            TrackerHeartbeatInput(live_jobs=live_jobs, hints_sent=hints_sent),
            start_to_close_timeout=_HEARTBEAT_TIMEOUT,
            retry_policy=_IO_RETRY,
        )
