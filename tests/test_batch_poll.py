"""Tests for forge.activities.batch_poll — batch poll activity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.activities.batch_poll import (
    _build_signal_from_entry,
    _ensure_utc,
    execute_poll_batch_results,
)
from forge.models import BatchPollerResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pending_job(
    *,
    request_id: str = "req-1",
    batch_id: str = "msgbatch_abc",
    workflow_id: str = "forge-task-test",
    created_at: datetime | None = None,
) -> dict:
    """Build a pending batch job dict matching store rows."""
    return {
        "id": request_id,
        "batch_id": batch_id,
        "workflow_id": workflow_id,
        "created_at": created_at or datetime.now(UTC),
        "status": "submitted",
    }


def _make_batch_response(*, processing_status: str = "ended") -> MagicMock:
    """Build a mock batch retrieve response."""
    batch = MagicMock()
    batch.processing_status = processing_status
    return batch


def _make_result_entry(
    *,
    custom_id: str = "req-1",
    result_type: str = "succeeded",
    message_json: str = '{"content": "hello"}',
    error: str = "some error",
) -> MagicMock:
    """Build a mock batch result entry."""
    entry = MagicMock()
    entry.custom_id = custom_id
    entry.result = MagicMock()
    entry.result.type = result_type

    if result_type == "succeeded":
        entry.result.message = MagicMock()
        entry.result.message.model_dump_json.return_value = message_json
    elif result_type == "errored":
        entry.result.error = error

    return entry


def _make_anthropic_client(
    *,
    batch: MagicMock | None = None,
    results: list[MagicMock] | None = None,
    retrieve_error: Exception | None = None,
    results_error: Exception | None = None,
) -> AsyncMock:
    """Build a mock AsyncAnthropic client."""
    client = AsyncMock()

    if retrieve_error:
        client.messages.batches.retrieve = AsyncMock(side_effect=retrieve_error)
    else:
        client.messages.batches.retrieve = AsyncMock(return_value=batch or _make_batch_response())

    if results_error:
        client.messages.batches.results = AsyncMock(side_effect=results_error)
    else:
        # results() returns an async iterable
        async def _async_iter():
            for r in results or []:
                yield r

        client.messages.batches.results = AsyncMock(return_value=_async_iter())

    return client


def _make_temporal_client(*, signal_error: Exception | None = None) -> AsyncMock:
    """Build a mock Temporal client."""
    client = AsyncMock()
    handle = AsyncMock()

    if signal_error:
        handle.signal = AsyncMock(side_effect=signal_error)
    else:
        handle.signal = AsyncMock()

    client.get_workflow_handle = MagicMock(return_value=handle)
    return client


def _noop_update(**_kwargs) -> None:
    """No-op status update function."""


# ---------------------------------------------------------------------------
# _ensure_utc
# ---------------------------------------------------------------------------


class TestEnsureUtc:
    def test_naive_datetime_gets_utc(self) -> None:
        naive = datetime(2024, 1, 1, 12, 0, 0)
        result = _ensure_utc(naive)
        assert result.tzinfo is UTC

    def test_aware_datetime_unchanged(self) -> None:
        aware = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = _ensure_utc(aware)
        assert result is aware


# ---------------------------------------------------------------------------
# _build_signal_from_entry
# ---------------------------------------------------------------------------


class TestBuildSignalFromEntry:
    def test_succeeded_entry(self) -> None:
        entry = _make_result_entry(result_type="succeeded", message_json='{"role": "assistant"}')
        signal = _build_signal_from_entry(entry, "msgbatch_x")

        assert signal.request_id == "req-1"
        assert signal.batch_id == "msgbatch_x"
        assert signal.raw_response_json == '{"role": "assistant"}'
        assert signal.error is None
        assert signal.result_type == "succeeded"

    def test_errored_entry(self) -> None:
        entry = _make_result_entry(result_type="errored", error="rate limited")
        signal = _build_signal_from_entry(entry, "msgbatch_x")

        assert signal.error is not None
        assert "rate limited" in signal.error
        assert signal.raw_response_json is None
        assert signal.result_type == "errored"

    def test_expired_entry(self) -> None:
        entry = _make_result_entry(result_type="expired")
        signal = _build_signal_from_entry(entry, "msgbatch_x")

        assert signal.error is not None
        assert "expired" in signal.error.lower()
        assert signal.result_type == "expired"

    def test_canceled_entry(self) -> None:
        entry = _make_result_entry(result_type="canceled")
        signal = _build_signal_from_entry(entry, "msgbatch_x")

        assert signal.error is not None
        assert "canceled" in signal.error.lower()
        assert signal.result_type == "canceled"

    def test_unknown_result_type(self) -> None:
        entry = _make_result_entry(result_type="something_new")
        signal = _build_signal_from_entry(entry, "msgbatch_x")

        assert signal.error is not None
        assert "something_new" in signal.error
        assert signal.result_type == "something_new"


# ---------------------------------------------------------------------------
# execute_poll_batch_results
# ---------------------------------------------------------------------------


class TestExecutePollBatchResults:
    @pytest.mark.asyncio
    async def test_no_pending_jobs_returns_zero_counts(self) -> None:
        client = _make_anthropic_client()
        temporal = _make_temporal_client()

        result = await execute_poll_batch_results([], client, temporal, _noop_update)

        assert result == BatchPollerResult(batches_checked=0, signals_sent=0, errors_found=0)

    @pytest.mark.asyncio
    async def test_succeeded_batch_sends_signal(self) -> None:
        entry = _make_result_entry(result_type="succeeded", message_json='{"text": "hi"}')
        client = _make_anthropic_client(results=[entry])
        temporal = _make_temporal_client()
        updates: list[dict] = []

        def track_update(**kwargs):
            updates.append(kwargs)

        job = _make_pending_job()
        result = await execute_poll_batch_results([job], client, temporal, track_update)

        assert result.batches_checked == 1
        assert result.signals_sent == 1
        assert result.errors_found == 0

        # Verify signal was sent
        temporal.get_workflow_handle.assert_called_once_with("forge-task-test")
        handle = temporal.get_workflow_handle.return_value
        handle.signal.assert_called_once()
        signal_args = handle.signal.call_args
        assert signal_args[0][0] == "batch_result_received"
        assert signal_args[0][1].raw_response_json == '{"text": "hi"}'

        # Verify status update
        assert len(updates) == 1
        assert updates[0]["status"] == "succeeded"

    @pytest.mark.asyncio
    async def test_errored_batch_sends_error_signal(self) -> None:
        entry = _make_result_entry(result_type="errored", error="invalid request")
        client = _make_anthropic_client(results=[entry])
        temporal = _make_temporal_client()

        job = _make_pending_job()
        result = await execute_poll_batch_results([job], client, temporal, _noop_update)

        assert result.signals_sent == 1
        handle = temporal.get_workflow_handle.return_value
        signal = handle.signal.call_args[0][1]
        assert signal.error is not None
        assert "invalid request" in signal.error

    @pytest.mark.asyncio
    async def test_expired_batch_sends_error_signal(self) -> None:
        entry = _make_result_entry(result_type="expired")
        client = _make_anthropic_client(results=[entry])
        temporal = _make_temporal_client()

        job = _make_pending_job()
        result = await execute_poll_batch_results([job], client, temporal, _noop_update)

        assert result.signals_sent == 1
        handle = temporal.get_workflow_handle.return_value
        signal = handle.signal.call_args[0][1]
        assert signal.error is not None
        assert "expired" in signal.error.lower()

    @pytest.mark.asyncio
    async def test_canceled_batch_sends_error_signal(self) -> None:
        entry = _make_result_entry(result_type="canceled")
        client = _make_anthropic_client(results=[entry])
        temporal = _make_temporal_client()

        job = _make_pending_job()
        result = await execute_poll_batch_results([job], client, temporal, _noop_update)

        assert result.signals_sent == 1
        handle = temporal.get_workflow_handle.return_value
        signal = handle.signal.call_args[0][1]
        assert "canceled" in signal.error.lower()

    @pytest.mark.asyncio
    async def test_still_processing_is_skipped(self) -> None:
        batch = _make_batch_response(processing_status="in_progress")
        client = _make_anthropic_client(batch=batch)
        temporal = _make_temporal_client()

        job = _make_pending_job()
        result = await execute_poll_batch_results([job], client, temporal, _noop_update)

        assert result.batches_checked == 1
        assert result.signals_sent == 0
        assert result.errors_found == 0

    @pytest.mark.asyncio
    async def test_retrieve_failure_logged_no_crash(self) -> None:
        client = _make_anthropic_client(retrieve_error=RuntimeError("network error"))
        temporal = _make_temporal_client()

        job = _make_pending_job()
        result = await execute_poll_batch_results([job], client, temporal, _noop_update)

        assert result.batches_checked == 1
        assert result.signals_sent == 0
        assert result.errors_found == 0  # Not old enough to be MISSING

    @pytest.mark.asyncio
    async def test_missing_batch_old_job_marks_missing(self) -> None:
        client = _make_anthropic_client(retrieve_error=RuntimeError("not found"))
        temporal = _make_temporal_client()
        updates: list[dict] = []

        def track_update(**kwargs):
            updates.append(kwargs)

        old_time = datetime.now(UTC) - timedelta(hours=25)
        job = _make_pending_job(created_at=old_time)
        result = await execute_poll_batch_results([job], client, temporal, track_update)

        assert result.errors_found == 1
        assert len(updates) == 1
        assert updates[0]["status"] == "missing"

    @pytest.mark.asyncio
    async def test_signal_delivery_failure_increments_errors(self) -> None:
        entry = _make_result_entry(result_type="succeeded")
        client = _make_anthropic_client(results=[entry])
        temporal = _make_temporal_client(signal_error=RuntimeError("workflow not found"))

        job = _make_pending_job()
        result = await execute_poll_batch_results([job], client, temporal, _noop_update)

        assert result.batches_checked == 1
        assert result.signals_sent == 0
        assert result.errors_found == 1

    @pytest.mark.asyncio
    async def test_multiple_pending_jobs_all_processed(self) -> None:
        entry1 = _make_result_entry(custom_id="req-1", result_type="succeeded")
        entry2 = _make_result_entry(custom_id="req-2", result_type="succeeded")

        # Create two separate clients that return different results
        # For simplicity, use a single client that's called twice
        batch = _make_batch_response()

        async def _results1():
            yield entry1

        async def _results2():
            yield entry2

        call_count = 0

        async def _retrieve(batch_id):
            return batch

        async def _results(batch_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _results1()
            return _results2()

        client = AsyncMock()
        client.messages.batches.retrieve = _retrieve
        client.messages.batches.results = _results

        temporal = _make_temporal_client()

        jobs = [
            _make_pending_job(request_id="req-1", batch_id="batch-1", workflow_id="wf-1"),
            _make_pending_job(request_id="req-2", batch_id="batch-2", workflow_id="wf-2"),
        ]

        result = await execute_poll_batch_results(jobs, client, temporal, _noop_update)

        assert result.batches_checked == 2
        assert result.signals_sent == 2

    @pytest.mark.asyncio
    async def test_results_retrieval_failure_increments_errors(self) -> None:
        client = _make_anthropic_client(results_error=RuntimeError("stream error"))
        temporal = _make_temporal_client()

        job = _make_pending_job()
        result = await execute_poll_batch_results([job], client, temporal, _noop_update)

        assert result.batches_checked == 1
        assert result.signals_sent == 0
        assert result.errors_found == 1
