"""Tests for forge.batch_poller_workflow — BatchPollerWorkflow."""

from __future__ import annotations

import pytest
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from forge.batch_poller_workflow import BatchPollerWorkflow
from forge.models import BatchPollerInput, BatchPollerResult
from forge.workflows import FORGE_TASK_QUEUE


@pytest.mark.asyncio
async def test_batch_poller_workflow_executes_activity() -> None:
    """Verify BatchPollerWorkflow calls poll_batch_results and returns its result."""
    from temporalio import activity

    expected = BatchPollerResult(batches_checked=3, signals_sent=2, errors_found=1)

    @activity.defn(name="poll_batch_results")
    async def mock_poll(_input: BatchPollerInput) -> BatchPollerResult:
        return expected

    async with (
        await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter,
        ) as env,
        Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[BatchPollerWorkflow],
            activities=[mock_poll],
        ),
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
