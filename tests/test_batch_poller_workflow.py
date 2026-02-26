"""Tests for forge.batch_poller_workflow — BatchPollerWorkflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from temporalio.worker import Worker

from forge.batch_poller_workflow import BatchPollerWorkflow
from forge.models import BatchPollerInput, BatchPollerResult
from forge.workflows import FORGE_TASK_QUEUE

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment


@pytest.mark.asyncio
async def test_batch_poller_workflow_executes_activity(env: WorkflowEnvironment) -> None:
    """Verify BatchPollerWorkflow calls poll_batch_results and returns its result."""
    from temporalio import activity

    expected = BatchPollerResult(batches_checked=3, signals_sent=2, errors_found=1)

    @activity.defn(name="poll_batch_results")
    async def mock_poll(_input: BatchPollerInput) -> BatchPollerResult:
        return expected

    async with Worker(
        env.client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=[BatchPollerWorkflow],
        activities=[mock_poll],
    ):
        result = await env.client.execute_workflow(
            BatchPollerWorkflow.run,
            BatchPollerInput(),
            id="test-batch-poller",
            task_queue=FORGE_TASK_QUEUE,
        )

        assert result.batches_checked == 3
        assert result.signals_sent == 2
        assert result.errors_found == 1
