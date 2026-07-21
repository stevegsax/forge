"""Tests for the ``persist_block`` survivable store-write primitive.

``persist_block`` dispatches through ``temporalio.workflow.execute_activity``,
which requires a live workflow execution context. Rather than spin up a real
Temporal time-skipping environment for a thin dispatch wrapper, these tests
monkeypatch ``workflow.execute_activity`` itself (the one genuinely external
call this module makes) and assert on what ``persist_block`` passed it —
activity name, retry policy, timeouts, and optional cross-queue routing.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import pytest

from sax_platform.contracts import persist as persist_module
from sax_platform.contracts.persist import (
    PersistBatchFailure,
    PersistBatchOutcome,
    PersistBatchSubmission,
    PersistResult,
    persist_block,
)

if TYPE_CHECKING:
    from pydantic import BaseModel


class TestPersistRequestModels:
    def test_batch_submission_defaults(self) -> None:
        req = PersistBatchSubmission(request_id="r1", batch_id="b1", workflow_id="wf1")
        assert req.kind == "batch_submission"
        assert req.provider == "anthropic"

    def test_batch_submission_explicit_provider(self) -> None:
        req = PersistBatchSubmission(
            request_id="r1", batch_id="b1", workflow_id="wf1", provider="mistral"
        )
        assert req.provider == "mistral"

    def test_batch_failure_defaults(self) -> None:
        req = PersistBatchFailure(request_id="r1", workflow_id="wf1", error_message="boom")
        assert req.kind == "batch_failure"
        assert req.provider == "anthropic"

    def test_batch_outcome_defaults(self) -> None:
        req = PersistBatchOutcome(request_id="r1", status="ended")
        assert req.kind == "batch_outcome"
        assert req.error_message is None

    def test_batch_outcome_carries_error_message(self) -> None:
        req = PersistBatchOutcome(request_id="r1", status="failed", error_message="provider failed")
        assert req.status == "failed"
        assert req.error_message == "provider failed"

    def test_persist_result_round_trips(self) -> None:
        result = PersistResult(kind="batch_submission", applied=True)
        assert PersistResult.model_validate_json(result.model_dump_json()) == result


class TestPersistBlock:
    @pytest.mark.asyncio
    async def test_dispatches_to_persist_to_store_on_caller_queue_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}

        async def fake_execute_activity(
            name: str, req: BaseModel, **kwargs: object
        ) -> PersistResult:
            captured["name"] = name
            captured["req"] = req
            captured["kwargs"] = kwargs
            return PersistResult(kind=req.kind, applied=True)

        monkeypatch.setattr(persist_module.workflow, "execute_activity", fake_execute_activity)

        req = PersistBatchSubmission(request_id="r1", batch_id="b1", workflow_id="wf1")
        result = await persist_block(req)

        assert result == PersistResult(kind="batch_submission", applied=True)
        assert captured["name"] == "persist_to_store"
        assert captured["req"] is req
        assert "task_queue" not in captured["kwargs"]
        assert captured["kwargs"]["result_type"] is PersistResult
        assert captured["kwargs"]["start_to_close_timeout"] == timedelta(seconds=30)
        assert captured["kwargs"]["schedule_to_close_timeout"] == timedelta(minutes=20)

    @pytest.mark.asyncio
    async def test_retry_policy_is_generous_but_bounded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}

        async def fake_execute_activity(
            name: str, req: BaseModel, **kwargs: object
        ) -> PersistResult:
            captured["kwargs"] = kwargs
            return PersistResult(kind=req.kind, applied=True)

        monkeypatch.setattr(persist_module.workflow, "execute_activity", fake_execute_activity)

        await persist_block(
            PersistBatchSubmission(request_id="r1", batch_id="b1", workflow_id="wf1")
        )

        retry_policy = captured["kwargs"]["retry_policy"]
        assert retry_policy.maximum_attempts == 20
        assert retry_policy.non_retryable_error_types == ["ValueError"]

    @pytest.mark.asyncio
    async def test_task_queue_threads_through_for_cross_queue_persist(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}

        async def fake_execute_activity(
            name: str, req: BaseModel, **kwargs: object
        ) -> PersistResult:
            captured["kwargs"] = kwargs
            return PersistResult(kind=req.kind, applied=False)

        monkeypatch.setattr(persist_module.workflow, "execute_activity", fake_execute_activity)

        req = PersistBatchFailure(request_id="r2", workflow_id="wf2", error_message="down")
        result = await persist_block(req, task_queue="forge-task-queue")

        assert result.applied is False
        assert captured["kwargs"]["task_queue"] == "forge-task-queue"
