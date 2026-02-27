"""Temporal workflow for batch polling.

Thin wrapper that executes the poll_batch_results activity on a schedule.
The Temporal Schedule triggers this workflow at a configurable interval
(default 60 seconds).
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

_POLL_HEARTBEAT = timedelta(seconds=60)

with workflow.unsafe.imports_passed_through():
    from forge.models import BatchPollerInput, BatchPollerResult


@workflow.defn
class BatchPollerWorkflow:
    """Poll Anthropic for completed batches and signal waiting workflows."""

    @workflow.run
    async def run(self, input: BatchPollerInput) -> BatchPollerResult:
        return await workflow.execute_activity(
            "poll_batch_results",
            input,
            start_to_close_timeout=timedelta(minutes=5),
            heartbeat_timeout=_POLL_HEARTBEAT,
            result_type=BatchPollerResult,
        )
