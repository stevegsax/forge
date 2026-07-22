"""Tests for OcrStoreWorkflow — the T4.4 signal-wait state machine.

The store child no longer polls the provider (T4.4): it waits on a state machine
that a stateless tracker advances by broadcasting ``ocr_status_hint`` signals.
These tests start the workflow on the time-skipping ``env`` and drive it by
sending hints on the handle (or via signal-with-start). On an ``ended`` hint it
fetches + stores via ``fetch_and_store_ocr_result`` and records a
``PersistBatchOutcome`` on the platform ``batch_jobs`` ledger cross-queue
(``persist_to_store`` on ``forge-task-queue``); with no hint the wait times out at
the 25h ceiling and records MISSING. Both an OCR worker and a stand-in platform
worker run so the cross-queue persist resolves.

``next_state`` — the pure transition function — is exercised directly (no
Temporal) over the full transition table, including the forbidden moves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sax_platform.contracts.constants import FORGE_TASK_QUEUE, OCR_TASK_QUEUE
from sax_platform.contracts.persist import PersistBatchOutcome, PersistResult
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.exceptions import ApplicationError
from temporalio.worker import Worker

from ocr.models import (
    OcrFetchStoreInput,
    OcrProcessingStatus,
    OcrStatusHint,
    OcrStatusUpsertInput,
    OcrStoreInput,
    OcrStoreResult,
)
from ocr.workflow_store import OcrStoreWorkflow, next_state

if TYPE_CHECKING:
    from collections.abc import Sequence

    from temporalio.testing import WorkflowEnvironment


def _store_input() -> OcrStoreInput:
    return OcrStoreInput(batch_id="b1", request_id="r1", document_id="doc-1", file_path="/a.pdf")


def _ocr_activities(calls: dict[str, list], *, fetch_raises: bool = False) -> list:
    """OCR-queue mocks: fetch/store + the best-effort terminal status upsert."""

    @activity.defn(name="fetch_and_store_ocr_result")
    async def fetch_and_store_ocr_result(input: OcrFetchStoreInput) -> OcrStoreResult:
        calls["fetch"].append(input.request_id)
        if fetch_raises:
            raise ApplicationError("fetch/store boom", non_retryable=True)
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


async def _run_store(
    env: WorkflowEnvironment,
    calls: dict[str, list],
    *,
    wf_id: str,
    fetch_raises: bool = False,
    hints: Sequence[OcrStatusHint] = (),
    start_hint: OcrStatusHint | None = None,
) -> None:
    """Start the store workflow, deliver ``hints`` on its handle, await completion.

    ``start_hint`` is delivered via signal-with-start (before the wait begins);
    ``hints`` are sent one at a time on the running handle.
    """
    ocr_worker = Worker(
        env.client,
        task_queue=OCR_TASK_QUEUE,
        workflows=[OcrStoreWorkflow],
        activities=_ocr_activities(calls, fetch_raises=fetch_raises),
    )
    platform_worker = Worker(
        env.client,
        task_queue=FORGE_TASK_QUEUE,
        activities=_platform_activities(calls),
    )
    async with ocr_worker, platform_worker:
        start_kwargs: dict = {}
        if start_hint is not None:
            start_kwargs["start_signal"] = "ocr_status_hint"
            start_kwargs["start_signal_args"] = [start_hint]
        handle = await env.client.start_workflow(
            OcrStoreWorkflow.run,
            _store_input(),
            id=wf_id,
            task_queue=OCR_TASK_QUEUE,
            **start_kwargs,
        )
        for hint in hints:
            await handle.signal("ocr_status_hint", hint)
        await handle.result()


def _new_calls() -> dict[str, list]:
    return {"fetch": [], "status_upsert": [], "persist": []}


def _hint(state: str, *, batch_id: str = "b1") -> OcrStatusHint:
    return OcrStatusHint(batch_id=batch_id, state=state)


class TestNextState:
    """The pure transition function — no Temporal."""

    @pytest.mark.parametrize(
        ("current", "hint", "expected"),
        [
            # Allowed forward moves.
            ("pending", "in_progress", "in_progress"),
            ("pending", "ended", "ended"),
            ("pending", "failed", "failed"),
            ("pending", "expired", "expired"),
            ("pending", "canceled", "canceled"),
            ("in_progress", "ended", "ended"),
            ("in_progress", "failed", "failed"),
            ("in_progress", "expired", "expired"),
            ("in_progress", "canceled", "canceled"),
            # Same-state no-ops.
            ("pending", "pending", None),
            ("in_progress", "in_progress", None),
            ("ended", "ended", None),
            ("failed", "failed", None),
            ("expired", "expired", None),
            ("canceled", "canceled", None),
            # Backward move.
            ("in_progress", "pending", None),
            # Out of a terminal state.
            ("ended", "failed", None),
            ("ended", "in_progress", None),
            ("ended", "pending", None),
            ("failed", "ended", None),
            ("expired", "ended", None),
            ("canceled", "failed", None),
            # Unknown status strings.
            ("pending", "garbage", None),
            ("in_progress", "garbage", None),
            ("ended", "garbage", None),
        ],
    )
    def test_transition_table(self, current: str, hint: str, expected: str | None) -> None:
        assert next_state(current, hint) == expected


class TestOcrStoreWorkflow:
    @pytest.mark.asyncio
    async def test_ended_hint_fetches_stores_and_persists_ended(
        self, env: WorkflowEnvironment
    ) -> None:
        calls = _new_calls()
        await _run_store(env, calls, wf_id="store-ended", hints=[_hint("ended")])

        assert calls["fetch"] == ["r1"]
        assert [p.status for p in calls["persist"]] == ["ended"]
        assert calls["status_upsert"] == []

    @pytest.mark.asyncio
    async def test_in_progress_then_ended_completes_normally(
        self, env: WorkflowEnvironment
    ) -> None:
        calls = _new_calls()
        await _run_store(
            env, calls, wf_id="store-inprog", hints=[_hint("in_progress"), _hint("ended")]
        )

        assert calls["fetch"] == ["r1"]
        assert [p.status for p in calls["persist"]] == ["ended"]

    @pytest.mark.asyncio
    async def test_start_signal_before_wait_is_honored(self, env: WorkflowEnvironment) -> None:
        """A hint delivered via signal-with-start (before run() waits) is honored."""
        calls = _new_calls()
        await _run_store(env, calls, wf_id="store-startsig", start_hint=_hint("ended"))

        assert calls["fetch"] == ["r1"]
        assert [p.status for p in calls["persist"]] == ["ended"]

    @pytest.mark.asyncio
    async def test_duplicate_ended_hints_fetch_once(self, env: WorkflowEnvironment) -> None:
        """Two ``ended`` hints: the second is a same-state no-op — one fetch only."""
        calls = _new_calls()
        await _run_store(env, calls, wf_id="store-dupe", hints=[_hint("ended"), _hint("ended")])

        assert calls["fetch"] == ["r1"]
        assert [p.status for p in calls["persist"]] == ["ended"]

    @pytest.mark.asyncio
    async def test_rejected_hints_do_not_progress(self, env: WorkflowEnvironment) -> None:
        """A foreign-batch hint and an unknown-state hint are ignored; the following
        matching ``ended`` hint still completes the workflow with exactly one fetch."""
        calls = _new_calls()
        await _run_store(
            env,
            calls,
            wf_id="store-rejected",
            hints=[_hint("ended", batch_id="other"), _hint("garbage"), _hint("ended")],
        )

        assert calls["fetch"] == ["r1"]
        assert [p.status for p in calls["persist"]] == ["ended"]

    @pytest.mark.asyncio
    async def test_failed_hint_marks_failed_and_persists(self, env: WorkflowEnvironment) -> None:
        calls = _new_calls()
        with pytest.raises(WorkflowFailureError):
            await _run_store(env, calls, wf_id="store-failed", hints=[_hint("failed")])

        assert calls["fetch"] == []  # never fetched — terminal before ended
        assert calls["status_upsert"] == [OcrProcessingStatus.FAILED]
        assert [p.status for p in calls["persist"]] == ["failed"]

    @pytest.mark.asyncio
    async def test_expired_hint_persists_expired(self, env: WorkflowEnvironment) -> None:
        calls = _new_calls()
        with pytest.raises(WorkflowFailureError):
            await _run_store(env, calls, wf_id="store-expired", hints=[_hint("expired")])

        assert [p.status for p in calls["persist"]] == ["expired"]

    @pytest.mark.asyncio
    async def test_canceled_hint_persists_failed(self, env: WorkflowEnvironment) -> None:
        """CANCELED coarsens to FAILED on the ledger."""
        calls = _new_calls()
        with pytest.raises(WorkflowFailureError):
            await _run_store(env, calls, wf_id="store-canceled", hints=[_hint("canceled")])

        assert calls["fetch"] == []
        assert calls["status_upsert"] == [OcrProcessingStatus.FAILED]
        assert [p.status for p in calls["persist"]] == ["failed"]

    @pytest.mark.asyncio
    async def test_no_hint_times_out_and_persists_missing(self, env: WorkflowEnvironment) -> None:
        """No hint ever arrives: the wait time-skips to the 25h ceiling → MISSING."""
        calls = _new_calls()
        with pytest.raises(WorkflowFailureError):
            await _run_store(env, calls, wf_id="store-giveup")

        assert calls["fetch"] == []
        assert [p.status for p in calls["persist"]] == ["missing"]
        assert calls["status_upsert"] == [OcrProcessingStatus.FAILED]

    @pytest.mark.asyncio
    async def test_fetch_store_error_marks_failed(self, env: WorkflowEnvironment) -> None:
        calls = _new_calls()
        with pytest.raises(WorkflowFailureError):
            await _run_store(
                env, calls, wf_id="store-fetch-err", fetch_raises=True, hints=[_hint("ended")]
            )

        assert calls["fetch"] == ["r1"]
        assert calls["status_upsert"] == [OcrProcessingStatus.FAILED]
        assert [p.status for p in calls["persist"]] == ["failed"]
