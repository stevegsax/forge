"""Tests for OcrBatchTrackerWorkflow — the T4.4 stateless hint-broadcast tracker.

One tracker run is a single cycle: list live OCR jobs, sweep Mistral once, then
broadcast an ``ocr_status_hint`` signal to each live job's waiting
``OcrStoreWorkflow``. These tests drive the tracker on the time-skipping ``env``
with by-name mock activities (``list_live_ocr_jobs``, ``sweep_mistral_batches``,
``record_tracker_heartbeat``) and, where a broadcast must actually land, run REAL
``OcrStoreWorkflow`` children (their own fetch/upsert activities mocked) plus a
stand-in platform worker so the store children's cross-queue ``persist_to_store``
resolves.

The observable contracts asserted: an idle cycle makes ZERO provider calls (the
sweep never runs); a burst reaches every swept child's outcome; a live job whose
batch Mistral did not report is not signaled; a signal to a closed/absent child
bounces without failing the cycle and is excluded from the delivered-hint count;
and two concurrent cycles both targeting one child still fetch exactly once. The
``_ensure_schedule`` worker helper is unit-tested for both the fresh-create and
the reconcile-an-existing paths.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from sax_platform.contracts.constants import FORGE_TASK_QUEUE, OCR_TASK_QUEUE
from sax_platform.contracts.persist import PersistBatchOutcome, PersistResult
from temporalio import activity
from temporalio.client import (
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleAlreadyRunningError,
    ScheduleOverlapPolicy,
    ScheduleUpdate,
    WorkflowFailureError,
)
from temporalio.worker import Worker

import ocr.worker as worker_mod
from ocr.models import (
    OcrFetchStoreInput,
    OcrStatusUpsertInput,
    OcrStoreInput,
    OcrStoreResult,
    TrackerHeartbeatInput,
    TrackerLiveJob,
)
from ocr.workflow_store import OcrStoreWorkflow
from ocr.workflow_tracker import OcrBatchTrackerWorkflow

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from temporalio.client import WorkflowHandle
    from temporalio.testing import WorkflowEnvironment


# ---------------------------------------------------------------------------
# Mock activities + helpers (mirroring test_workflow_store.py)
# ---------------------------------------------------------------------------


def _new_calls() -> dict[str, list]:
    return {
        "list": [],
        "sweep": [],
        "heartbeat": [],
        "fetch": [],
        "status_upsert": [],
        "persist": [],
    }


def _tracker_activities(
    calls: dict[str, list],
    *,
    live_jobs: Sequence[TrackerLiveJob],
    sweep: Mapping[str, str],
) -> list:
    """The three tracker activities, returning canned list/sweep data."""

    @activity.defn(name="list_live_ocr_jobs")
    async def list_live_ocr_jobs() -> list[TrackerLiveJob]:
        calls["list"].append(True)
        return list(live_jobs)

    @activity.defn(name="sweep_mistral_batches")
    async def sweep_mistral_batches() -> dict[str, str]:
        calls["sweep"].append(True)
        return dict(sweep)

    @activity.defn(name="record_tracker_heartbeat")
    async def record_tracker_heartbeat(input: TrackerHeartbeatInput) -> None:
        calls["heartbeat"].append((input.live_jobs, input.hints_sent))

    return [list_live_ocr_jobs, sweep_mistral_batches, record_tracker_heartbeat]


def _store_activities(calls: dict[str, list]) -> list:
    """The OcrStoreWorkflow's own activities (fetch/store + terminal status upsert)."""

    @activity.defn(name="fetch_and_store_ocr_result")
    async def fetch_and_store_ocr_result(input: OcrFetchStoreInput) -> OcrStoreResult:
        calls["fetch"].append(input.request_id)
        return OcrStoreResult(document_id=input.document_id, text_length=5, page_count=1)

    @activity.defn(name="upsert_ocr_status")
    async def upsert_ocr_status(input: OcrStatusUpsertInput) -> None:
        calls["status_upsert"].append(input.status)

    return [fetch_and_store_ocr_result, upsert_ocr_status]


def _platform_activities(calls: dict[str, list]) -> list:
    @activity.defn(name="persist_to_store")
    async def persist_to_store(req: PersistBatchOutcome) -> PersistResult:
        calls["persist"].append(req)
        return PersistResult(kind=req.kind, applied=True)

    return [persist_to_store]


def _workers(
    env: WorkflowEnvironment,
    calls: dict[str, list],
    *,
    live_jobs: Sequence[TrackerLiveJob],
    sweep: Mapping[str, str],
) -> tuple[Worker, Worker]:
    """An OCR worker (tracker + store workflows and their activities) and a
    stand-in platform worker for the cross-queue ``persist_to_store``."""
    ocr_worker = Worker(
        env.client,
        task_queue=OCR_TASK_QUEUE,
        workflows=[OcrBatchTrackerWorkflow, OcrStoreWorkflow],
        activities=[
            *_tracker_activities(calls, live_jobs=live_jobs, sweep=sweep),
            *_store_activities(calls),
        ],
    )
    platform_worker = Worker(
        env.client,
        task_queue=FORGE_TASK_QUEUE,
        activities=_platform_activities(calls),
    )
    return ocr_worker, platform_worker


def _job(*, workflow_id: str, batch_id: str, request_id: str) -> TrackerLiveJob:
    return TrackerLiveJob(request_id=request_id, batch_id=batch_id, workflow_id=workflow_id)


async def _start_store(
    env: WorkflowEnvironment,
    *,
    wf_id: str,
    batch_id: str,
    request_id: str,
) -> WorkflowHandle:
    return await env.client.start_workflow(
        OcrStoreWorkflow.run,
        OcrStoreInput(
            batch_id=batch_id,
            request_id=request_id,
            document_id=f"doc-{request_id}",
            file_path=f"/{request_id}.pdf",
        ),
        id=wf_id,
        task_queue=OCR_TASK_QUEUE,
    )


async def _run_tracker(env: WorkflowEnvironment, *, wf_id: str) -> None:
    await env.client.execute_workflow(
        OcrBatchTrackerWorkflow.run, id=wf_id, task_queue=OCR_TASK_QUEUE
    )


# ---------------------------------------------------------------------------
# Broadcast-cycle tests
# ---------------------------------------------------------------------------


class TestOcrBatchTrackerWorkflow:
    @pytest.mark.asyncio
    async def test_idle_cycle_makes_zero_provider_calls(self, env: WorkflowEnvironment) -> None:
        """No live jobs → (0, 0) heartbeat and the sweep activity NEVER runs."""
        calls = _new_calls()
        ocr_worker, platform_worker = _workers(env, calls, live_jobs=[], sweep={})
        async with ocr_worker, platform_worker:
            await _run_tracker(env, wf_id="tracker-idle")

        assert calls["list"] == [True]
        assert calls["sweep"] == []  # the idle cycle short-circuits before any sweep
        assert calls["heartbeat"] == [(0, 0)]

    @pytest.mark.asyncio
    async def test_burst_broadcast_reaches_both_outcomes(self, env: WorkflowEnvironment) -> None:
        """Two live jobs, both swept → each store child reaches its outcome; (2, 2)."""
        calls = _new_calls()
        jobs = [
            _job(workflow_id="store-burst-a", batch_id="a", request_id="ra"),
            _job(workflow_id="store-burst-b", batch_id="b", request_id="rb"),
        ]
        sweep = {"a": "ended", "b": "failed"}
        ocr_worker, platform_worker = _workers(env, calls, live_jobs=jobs, sweep=sweep)
        async with ocr_worker, platform_worker:
            handle_a = await _start_store(env, wf_id="store-burst-a", batch_id="a", request_id="ra")
            handle_b = await _start_store(env, wf_id="store-burst-b", batch_id="b", request_id="rb")

            await _run_tracker(env, wf_id="tracker-burst")

            result_a = await handle_a.result()
            with pytest.raises(WorkflowFailureError):
                await handle_b.result()

        assert result_a.document_id == "doc-ra"
        assert calls["fetch"] == ["ra"]  # only the ENDED batch fetched
        assert calls["heartbeat"] == [(2, 2)]
        assert {p.status for p in calls["persist"]} == {"ended", "failed"}

    @pytest.mark.asyncio
    async def test_selective_send_skips_unswept_batch(self, env: WorkflowEnvironment) -> None:
        """A live job whose batch Mistral did not report is not signaled; hints=1."""
        calls = _new_calls()
        jobs = [
            _job(workflow_id="store-sel-a", batch_id="a", request_id="ra"),
            _job(workflow_id="store-sel-c", batch_id="c", request_id="rc"),
        ]
        sweep = {"a": "ended"}  # batch "c" absent → its child is never signaled
        ocr_worker, platform_worker = _workers(env, calls, live_jobs=jobs, sweep=sweep)
        async with ocr_worker, platform_worker:
            handle_a = await _start_store(env, wf_id="store-sel-a", batch_id="a", request_id="ra")
            await _run_tracker(env, wf_id="tracker-selective")
            await handle_a.result()

        assert calls["fetch"] == ["ra"]
        assert calls["heartbeat"] == [(2, 1)]  # 2 live, only 1 actually sent
        assert [p.status for p in calls["persist"]] == ["ended"]

    @pytest.mark.asyncio
    async def test_bounced_signal_does_not_fail_the_cycle(self, env: WorkflowEnvironment) -> None:
        """A signal to an absent workflow bounces (logged), the real child still lands."""
        calls = _new_calls()
        jobs = [
            _job(workflow_id="store-bounce-real", batch_id="a", request_id="ra"),
            _job(workflow_id="ocr-store-nonexistent", batch_id="g", request_id="rg"),
        ]
        sweep = {"a": "ended", "g": "ended"}  # both swept; only the ghost bounces
        ocr_worker, platform_worker = _workers(env, calls, live_jobs=jobs, sweep=sweep)
        async with ocr_worker, platform_worker:
            handle = await _start_store(
                env, wf_id="store-bounce-real", batch_id="a", request_id="ra"
            )
            # The tracker completes despite the bounced ghost signal.
            await _run_tracker(env, wf_id="tracker-bounce")
            await handle.result()

        assert calls["fetch"] == ["ra"]
        # hints_sent counts only the delivered signal — the bounce is excluded.
        assert calls["heartbeat"] == [(2, 1)]
        assert [p.status for p in calls["persist"]] == ["ended"]

    @pytest.mark.asyncio
    async def test_concurrent_trackers_fetch_once(self, env: WorkflowEnvironment) -> None:
        """Two cycles both sending ENDED to one child → exactly one fetch, no error."""
        calls = _new_calls()
        jobs = [_job(workflow_id="store-cc", batch_id="a", request_id="ra")]
        sweep = {"a": "ended"}
        ocr_worker, platform_worker = _workers(env, calls, live_jobs=jobs, sweep=sweep)
        async with ocr_worker, platform_worker:
            handle = await _start_store(env, wf_id="store-cc", batch_id="a", request_id="ra")

            # Both tracker cycles run against the same running store child.
            await asyncio.gather(
                _run_tracker(env, wf_id="tracker-cc-1"),
                _run_tracker(env, wf_id="tracker-cc-2"),
            )
            result = await handle.result()

        assert result.document_id == "doc-ra"
        assert calls["fetch"] == ["ra"]  # duplicate ENDED absorbed — one fetch only
        assert len(calls["heartbeat"]) == 2  # both cycles recorded a heartbeat
        assert [p.status for p in calls["persist"]] == ["ended"]  # one terminal outcome


# ---------------------------------------------------------------------------
# _ensure_schedule unit tests (the worker's Schedule wiring)
# ---------------------------------------------------------------------------


class TestEnsureSchedule:
    @pytest.mark.asyncio
    async def test_creates_fresh_schedule(self) -> None:
        client = MagicMock()
        client.create_schedule = AsyncMock()

        await worker_mod._ensure_schedule(
            client, "ocr-batch-tracker", "OcrBatchTrackerWorkflow", timedelta(seconds=120)
        )

        client.create_schedule.assert_awaited_once()
        schedule_id, schedule = client.create_schedule.await_args.args
        assert schedule_id == "ocr-batch-tracker"
        assert isinstance(schedule, Schedule)
        action = schedule.action
        assert isinstance(action, ScheduleActionStartWorkflow)
        assert action.workflow == "OcrBatchTrackerWorkflow"
        assert action.args == []  # the tracker takes NO argument
        assert action.id == "ocr-batch-tracker-run"
        assert action.task_queue == OCR_TASK_QUEUE
        assert action.execution_timeout == timedelta(minutes=5)
        assert schedule.spec.intervals[0].every == timedelta(seconds=120)
        assert schedule.policy.overlap == ScheduleOverlapPolicy.SKIP

    @pytest.mark.asyncio
    async def test_reconciles_existing_schedule(self) -> None:
        client = MagicMock()
        client.create_schedule = AsyncMock(side_effect=ScheduleAlreadyRunningError())
        handle = MagicMock()
        handle.update = AsyncMock()
        client.get_schedule_handle = MagicMock(return_value=handle)

        await worker_mod._ensure_schedule(
            client, "ocr-batch-tracker", "OcrBatchTrackerWorkflow", timedelta(seconds=120)
        )

        client.get_schedule_handle.assert_called_once_with("ocr-batch-tracker")
        handle.update.assert_awaited_once()

        # Drive the updater callback with a stand-in description and confirm it
        # overwrites spec/action/policy with the freshly built values.
        updater = handle.update.await_args.args[0]
        fake_schedule = MagicMock()
        fake_input = MagicMock()
        fake_input.description.schedule = fake_schedule

        result = await updater(fake_input)

        assert isinstance(result, ScheduleUpdate)
        assert result.schedule is fake_schedule
        assert fake_schedule.spec.intervals[0].every == timedelta(seconds=120)
        assert fake_schedule.action.workflow == "OcrBatchTrackerWorkflow"
        assert fake_schedule.action.task_queue == OCR_TASK_QUEUE
        assert fake_schedule.policy.overlap == ScheduleOverlapPolicy.SKIP
