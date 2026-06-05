"""Tests for survivable store writes (Phase C).

Two guarantees are proven here:

1. **Idempotency** — applying a ``PersistRequest`` twice writes one row; the second
   call reports ``applied=False`` (a retry never duplicates).
2. **Pause-and-retry** — when ``persist_to_store`` fails transiently, Temporal
   retries only that cheap activity; the expensive upstream call ran exactly once.
   A prolonged outage fails the workflow loudly (no hang).
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from temporalio import activity, workflow
from temporalio.client import WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.worker import Worker

from forge.activities.persist import persist_to_store
from forge.models import PlaybookEntry, TaskResult, TransitionSignal
from forge.persist_models import (
    PersistBatchStatus,
    PersistBatchSubmission,
    PersistInteraction,
    PersistPlaybooks,
    PersistRequest,
    PersistResult,
    PersistRun,
)
from forge.store import get_run, get_store_engine, run_migrations
from forge.workflows import FORGE_TASK_QUEUE

with workflow.unsafe.imports_passed_through():
    from forge_contracts.persist import persist_block

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment


@pytest.fixture
def migrated(forge_db_url: str) -> str:
    run_migrations(forge_db_url)
    return forge_db_url


# ---------------------------------------------------------------------------
# Idempotency — persist_to_store dispatched directly as a plain async function
# ---------------------------------------------------------------------------


class TestPersistIdempotency:
    @pytest.mark.asyncio
    async def test_run_idempotent(self, migrated: str) -> None:
        req = PersistRun(
            workflow_id="wf-1",
            task_result=TaskResult(task_id="t", status=TransitionSignal.SUCCESS),
        )
        first = await persist_to_store(req)
        second = await persist_to_store(req)
        assert (first.applied, second.applied) == (True, False)
        assert get_run(get_store_engine(), "wf-1") is not None

    @pytest.mark.asyncio
    async def test_interaction_idempotent(self, migrated: str) -> None:
        req = PersistInteraction(
            idempotency_key="wf:llm:1",
            task_id="t",
            role="llm",
            system_prompt="s",
            user_prompt="u",
            model_name="m",
            input_tokens=1,
            output_tokens=2,
            latency_ms=3.0,
        )
        first = await persist_to_store(req)
        second = await persist_to_store(req)
        assert (first.applied, second.applied) == (True, False)

    @pytest.mark.asyncio
    async def test_batch_submission_idempotent(self, migrated: str) -> None:
        req = PersistBatchSubmission(
            request_id="req-1", batch_id="b-1", workflow_id="wf", provider="mistral"
        )
        first = await persist_to_store(req)
        second = await persist_to_store(req)
        assert (first.applied, second.applied) == (True, False)

    @pytest.mark.asyncio
    async def test_playbooks_idempotent(self, migrated: str) -> None:
        entry = PlaybookEntry(
            title="Lesson",
            content="Do X.",
            tags=["py"],
            source_task_id="t",
            source_workflow_id="wf",
        )
        req = PersistPlaybooks(extraction_workflow_id="extract-1", entries=[entry])
        first = await persist_to_store(req)
        second = await persist_to_store(req)
        assert (first.applied, second.applied) == (True, False)

    @pytest.mark.asyncio
    async def test_batch_status_update_applies(self, migrated: str) -> None:
        await persist_to_store(
            PersistBatchSubmission(request_id="req-s", batch_id="b", workflow_id="wf")
        )
        # A status transition is a plain UPDATE (no dedupe) — always "applied".
        result = await persist_to_store(
            PersistBatchStatus(request_id="req-s", status="processing")
        )
        assert result.applied is True


# ---------------------------------------------------------------------------
# Pause-and-retry — a probe workflow runs one "expensive" activity, then a
# survivable persist_block. The flaky persist must retry without re-running the
# expensive call.
# ---------------------------------------------------------------------------


@workflow.defn
class _PersistRetryProbe:
    """Run an expensive activity once, then persist survivably via persist_block."""

    @workflow.run
    async def run(self, workflow_id: str) -> bool:
        await workflow.execute_activity(
            "expensive_op",
            start_to_close_timeout=timedelta(seconds=5),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        result = await persist_block(
            PersistRun(
                workflow_id=workflow_id,
                task_result=TaskResult(task_id="t", status=TransitionSignal.SUCCESS),
            )
        )
        return result.applied


def _probe_mocks(call_count: dict[str, int]) -> list:
    @activity.defn(name="expensive_op")
    async def expensive_op() -> None:
        call_count["expensive"] += 1

    return [expensive_op]


class TestPauseAndRetry:
    @pytest.mark.asyncio
    async def test_persist_retries_without_rerunning_expensive_call(
        self, env: WorkflowEnvironment
    ) -> None:
        call_count = {"expensive": 0, "persist": 0}

        @activity.defn(name="persist_to_store")
        async def flaky_persist(req: PersistRequest) -> PersistResult:
            call_count["persist"] += 1
            if call_count["persist"] <= 2:
                raise RuntimeError("transient DB outage")
            return PersistResult(kind=req.kind, applied=True)

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[_PersistRetryProbe],
            activities=[*_probe_mocks(call_count), flaky_persist],
        ):
            applied = await env.client.execute_workflow(
                _PersistRetryProbe.run,
                "probe-wf-1",
                id="test-persist-pause-retry",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert applied is True
        # The expensive call ran exactly once; only the cheap persist retried.
        assert call_count["expensive"] == 1
        assert call_count["persist"] == 3  # two failures + one success

    @pytest.mark.asyncio
    async def test_prolonged_outage_fails_workflow_without_hang(
        self, env: WorkflowEnvironment
    ) -> None:
        call_count = {"expensive": 0, "persist": 0}

        @activity.defn(name="persist_to_store")
        async def always_fail(req: PersistRequest) -> PersistResult:
            call_count["persist"] += 1
            raise RuntimeError("DB down for good")

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[_PersistRetryProbe],
            activities=[*_probe_mocks(call_count), always_fail],
        ):
            with pytest.raises(WorkflowFailureError):
                await env.client.execute_workflow(
                    _PersistRetryProbe.run,
                    "probe-wf-2",
                    id="test-persist-prolonged-outage",
                    task_queue=FORGE_TASK_QUEUE,
                )

        # The schedule-to-close cap was reached (time-skipped) and the workflow
        # failed loudly; the expensive call still ran only once.
        assert call_count["expensive"] == 1
        assert call_count["persist"] > 1
