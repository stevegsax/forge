"""Tests for forge.activities.batch_poll — batch poll activity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge.activities.batch_poll import (
    _ensure_utc,
    execute_poll_batch_results,
)
from forge.llm_providers.models import BatchPollResult as ProviderBatchPollResult
from forge.llm_providers.models import BatchPollStatus, BatchResultEntry
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
    provider: str = "anthropic",
) -> dict:
    """Build a pending batch job dict matching store rows."""
    return {
        "id": request_id,
        "batch_id": batch_id,
        "workflow_id": workflow_id,
        "created_at": created_at or datetime.now(UTC),
        "status": "submitted",
        "provider": provider,
    }


def _make_mock_provider(
    *,
    poll_result: ProviderBatchPollResult | None = None,
    poll_error: Exception | None = None,
) -> MagicMock:
    """Build a mock LLMProvider with poll_batch method."""
    provider = MagicMock()
    if poll_error:
        provider.poll_batch = AsyncMock(side_effect=poll_error)
    else:
        provider.poll_batch = AsyncMock(
            return_value=poll_result
            or ProviderBatchPollResult(status=BatchPollStatus.IN_PROGRESS)
        )
    return provider


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
# execute_poll_batch_results
# ---------------------------------------------------------------------------


class TestExecutePollBatchResults:
    @pytest.mark.asyncio
    async def test_no_pending_jobs_returns_zero_counts(self) -> None:
        temporal = _make_temporal_client()

        result = await execute_poll_batch_results([], temporal, _noop_update)

        assert result == BatchPollerResult(batches_checked=0, signals_sent=0, errors_found=0)

    @pytest.mark.asyncio
    async def test_succeeded_batch_sends_signal(self) -> None:
        poll_result = ProviderBatchPollResult(
            status=BatchPollStatus.ENDED,
            entries=[
                BatchResultEntry(
                    custom_id="req-1",
                    succeeded=True,
                    raw_response_json='{"text": "hi"}',
                )
            ],
        )
        provider = _make_mock_provider(poll_result=poll_result)
        temporal = _make_temporal_client()
        updates: list[dict] = []

        def track_update(**kwargs):
            updates.append(kwargs)

        job = _make_pending_job()
        with patch("forge.llm_providers.get_provider", return_value=provider):
            result = await execute_poll_batch_results([job], temporal, track_update)

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
    async def test_errored_entry_sends_error_signal(self) -> None:
        poll_result = ProviderBatchPollResult(
            status=BatchPollStatus.ENDED,
            entries=[
                BatchResultEntry(
                    custom_id="req-1",
                    succeeded=False,
                    error="invalid request",
                )
            ],
        )
        provider = _make_mock_provider(poll_result=poll_result)
        temporal = _make_temporal_client()

        job = _make_pending_job()
        with patch("forge.llm_providers.get_provider", return_value=provider):
            result = await execute_poll_batch_results([job], temporal, _noop_update)

        assert result.signals_sent == 1
        handle = temporal.get_workflow_handle.return_value
        signal = handle.signal.call_args[0][1]
        assert signal.error is not None
        assert "invalid request" in signal.error

    @pytest.mark.asyncio
    async def test_still_processing_is_skipped(self) -> None:
        poll_result = ProviderBatchPollResult(status=BatchPollStatus.IN_PROGRESS)
        provider = _make_mock_provider(poll_result=poll_result)
        temporal = _make_temporal_client()

        job = _make_pending_job()
        with patch("forge.llm_providers.get_provider", return_value=provider):
            result = await execute_poll_batch_results([job], temporal, _noop_update)

        assert result.batches_checked == 1
        assert result.signals_sent == 0
        assert result.errors_found == 0

    @pytest.mark.asyncio
    async def test_retrieve_failure_logged_no_crash(self) -> None:
        provider = _make_mock_provider(poll_error=RuntimeError("network error"))
        temporal = _make_temporal_client()

        job = _make_pending_job()
        with patch("forge.llm_providers.get_provider", return_value=provider):
            result = await execute_poll_batch_results([job], temporal, _noop_update)

        assert result.batches_checked == 1
        assert result.signals_sent == 0
        assert result.errors_found == 0  # Not old enough to be MISSING

    @pytest.mark.asyncio
    async def test_missing_batch_old_job_marks_missing(self) -> None:
        provider = _make_mock_provider(poll_error=RuntimeError("not found"))
        temporal = _make_temporal_client()
        updates: list[dict] = []

        def track_update(**kwargs):
            updates.append(kwargs)

        old_time = datetime.now(UTC) - timedelta(hours=25)
        job = _make_pending_job(created_at=old_time)
        with patch("forge.llm_providers.get_provider", return_value=provider):
            result = await execute_poll_batch_results([job], temporal, track_update)

        assert result.errors_found == 1
        assert len(updates) == 1
        assert updates[0]["status"] == "missing"

    @pytest.mark.asyncio
    async def test_signal_delivery_failure_increments_errors(self) -> None:
        poll_result = ProviderBatchPollResult(
            status=BatchPollStatus.ENDED,
            entries=[BatchResultEntry(custom_id="req-1", succeeded=True, raw_response_json="{}")],
        )
        provider = _make_mock_provider(poll_result=poll_result)
        temporal = _make_temporal_client(signal_error=RuntimeError("workflow not found"))

        job = _make_pending_job()
        with patch("forge.llm_providers.get_provider", return_value=provider):
            result = await execute_poll_batch_results([job], temporal, _noop_update)

        assert result.batches_checked == 1
        assert result.signals_sent == 0
        assert result.errors_found == 1

    @pytest.mark.asyncio
    async def test_multiple_pending_jobs_all_processed(self) -> None:
        poll_result = ProviderBatchPollResult(
            status=BatchPollStatus.ENDED,
            entries=[BatchResultEntry(custom_id="req-1", succeeded=True, raw_response_json="{}")],
        )
        provider = _make_mock_provider(poll_result=poll_result)
        temporal = _make_temporal_client()

        jobs = [
            _make_pending_job(request_id="req-1", batch_id="batch-1", workflow_id="wf-1"),
            _make_pending_job(request_id="req-2", batch_id="batch-2", workflow_id="wf-2"),
        ]

        with patch("forge.llm_providers.get_provider", return_value=provider):
            result = await execute_poll_batch_results(jobs, temporal, _noop_update)

        assert result.batches_checked == 2
        assert result.signals_sent == 2
