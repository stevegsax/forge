"""Tests for forge.activities.batch_poll — batch poll activity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sax_platform.contracts.models import parse_batch_result_payload
from sax_platform.llm.batch import BatchRequestFailed, BatchStatus
from sax_platform.ocr import BatchPollResult as ProviderBatchPollResult
from sax_platform.ocr import BatchPollStatus, BatchResultEntry, ExtractedImage

from forge.activities.batch_poll import (
    _ensure_utc,
    _poll_anthropic_batch,
    _poll_batch_for,
    execute_poll_batch_results,
)
from forge.models import BatchJobStatus, BatchPollerResult

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
        "status": BatchJobStatus.SUBMITTED,
        "provider": provider,
    }


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


def _put_blob(custom_id: str, data: bytes) -> str:
    """Fake blob upload: record nothing, return a deterministic key."""
    return f"blob-{custom_id}"


def _patch_poll(*, result: ProviderBatchPollResult | None = None, side_effect=None):
    """Patch the per-provider poll dispatch used by execute_poll_batch_results."""
    if side_effect is not None:
        return patch(
            "forge.activities.batch_poll._poll_batch_for", AsyncMock(side_effect=side_effect)
        )
    return patch("forge.activities.batch_poll._poll_batch_for", AsyncMock(return_value=result))


def _batch_status(processing_status: str) -> BatchStatus:
    """A BatchStatus with the given processing_status and zeroed counts."""
    return BatchStatus(
        batch_id="b-1",
        processing_status=processing_status,
        succeeded=0,
        errored=0,
        canceled=0,
        expired=0,
        processing=0,
    )


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

        result = await execute_poll_batch_results([], temporal, _noop_update, _put_blob)

        assert result == BatchPollerResult(batches_checked=0, signals_sent=0, errors_found=0)

    @pytest.mark.asyncio
    async def test_succeeded_small_batch_delivers_inline(self) -> None:
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
        temporal = _make_temporal_client()
        updates: list[dict] = []

        def track_update(**kwargs):
            updates.append(kwargs)

        job = _make_pending_job()
        with _patch_poll(result=poll_result):
            result = await execute_poll_batch_results([job], temporal, track_update, _put_blob)

        assert result.batches_checked == 1
        assert result.signals_sent == 1
        assert result.errors_found == 0

        # Verify signal was sent inline (small, image-free payload).
        temporal.get_workflow_handle.assert_called_once_with("forge-task-test")
        handle = temporal.get_workflow_handle.return_value
        handle.signal.assert_called_once()
        signal_args = handle.signal.call_args
        assert signal_args[0][0] == "batch_result_received"
        assert signal_args[0][1].raw_response_json == '{"text": "hi"}'
        assert signal_args[0][1].s3_key is None

        # On delivery the provider lifecycle is done from the platform's view:
        # the row advances to PROCESSING (handed to the consumer).
        assert len(updates) == 1
        assert updates[0]["status"] == BatchJobStatus.PROCESSING

    @pytest.mark.asyncio
    async def test_succeeded_with_images_delivers_pointer(self) -> None:
        captured: dict[str, bytes] = {}

        def capture_put(custom_id: str, data: bytes) -> str:
            key = f"blob-{custom_id}"
            captured[key] = data
            return key

        img = ExtractedImage(
            original_image_id="img-0.jpeg",
            page_index=0,
            image_base64="ZmFrZQ==",
            mime_type="image/jpeg",
        )
        poll_result = ProviderBatchPollResult(
            status=BatchPollStatus.ENDED,
            entries=[
                BatchResultEntry(
                    custom_id="req-1",
                    succeeded=True,
                    raw_response_json='{"pages": [{"markdown": "x"}]}',
                    extracted_images=[img],
                )
            ],
        )
        temporal = _make_temporal_client()

        job = _make_pending_job()
        with _patch_poll(result=poll_result):
            result = await execute_poll_batch_results([job], temporal, _noop_update, capture_put)

        assert result.signals_sent == 1
        signal = temporal.get_workflow_handle.return_value.signal.call_args[0][1]
        # Image-bearing result is delivered by pointer; the body stays out of the signal.
        assert signal.s3_key == "blob-req-1"
        assert signal.raw_response_json is None
        # The stashed envelope round-trips to body + images.
        body, images = parse_batch_result_payload(captured["blob-req-1"].decode("utf-8"))
        assert body == '{"pages": [{"markdown": "x"}]}'
        assert images[0]["original_image_id"] == "img-0.jpeg"

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
        temporal = _make_temporal_client()

        job = _make_pending_job()
        with _patch_poll(result=poll_result):
            result = await execute_poll_batch_results([job], temporal, _noop_update, _put_blob)

        assert result.signals_sent == 1
        handle = temporal.get_workflow_handle.return_value
        signal = handle.signal.call_args[0][1]
        assert signal.error is not None
        assert "invalid request" in signal.error

    @pytest.mark.asyncio
    async def test_still_processing_is_skipped(self) -> None:
        poll_result = ProviderBatchPollResult(status=BatchPollStatus.IN_PROGRESS)
        temporal = _make_temporal_client()

        job = _make_pending_job()
        with _patch_poll(result=poll_result):
            result = await execute_poll_batch_results([job], temporal, _noop_update, _put_blob)

        assert result.batches_checked == 1
        assert result.signals_sent == 0
        assert result.errors_found == 0

    @pytest.mark.asyncio
    async def test_retrieve_failure_reports_error_without_raising(self) -> None:
        """A poll failure on a fresh job is counted in errors_found; the poller
        no longer raises (that wedged the schedule — see T1.3)."""
        temporal = _make_temporal_client()

        job = _make_pending_job()
        with _patch_poll(side_effect=RuntimeError("network error")):
            result = await execute_poll_batch_results([job], temporal, _noop_update, _put_blob)

        assert result.batches_checked == 1
        assert result.errors_found == 1
        assert result.signals_sent == 0

    # -----------------------------------------------------------------------
    # Terminal failure statuses (FAILED / EXPIRED / CANCELED)
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (BatchPollStatus.FAILED, BatchJobStatus.FAILED),
            (BatchPollStatus.EXPIRED, BatchJobStatus.EXPIRED),
            (BatchPollStatus.CANCELED, BatchJobStatus.FAILED),
        ],
        ids=["failed", "expired", "canceled->failed"],
    )
    async def test_terminal_failure_signals_error_and_updates_status(
        self, status: BatchPollStatus, expected: BatchJobStatus
    ) -> None:
        """Terminal failure statuses should signal the waiting workflow with an
        error and update the batch_jobs DB status so the poller stops re-polling.
        CANCELED collapses to the generic FAILED state."""
        poll_result = ProviderBatchPollResult(status=status)
        temporal = _make_temporal_client()
        updates: list[dict] = []

        def track_update(**kwargs):
            updates.append(kwargs)

        job = _make_pending_job()
        with _patch_poll(result=poll_result):
            result = await execute_poll_batch_results([job], temporal, track_update, _put_blob)

        # Error signal was sent to the waiting workflow
        assert result.batches_checked == 1
        assert result.signals_sent == 1
        assert result.errors_found == 0

        handle = temporal.get_workflow_handle.return_value
        handle.signal.assert_called_once()
        signal = handle.signal.call_args[0][1]
        assert signal.error is not None
        assert status.value in signal.error
        assert signal.result_type == "errored"

        assert len(updates) == 1
        assert updates[0]["status"] == expected
        assert updates[0]["error_message"] is not None

    @pytest.mark.asyncio
    async def test_terminal_failure_with_signal_error_increments_errors(self) -> None:
        """When the workflow signal fails for a terminal batch, errors_found
        increments and the DB status is still updated (the batch failed at the
        provider, so it is terminal regardless of signal delivery). No raise."""
        poll_result = ProviderBatchPollResult(status=BatchPollStatus.FAILED)
        temporal = _make_temporal_client(signal_error=RuntimeError("workflow gone"))
        updates: list[dict] = []

        def track_update(**kwargs):
            updates.append(kwargs)

        job = _make_pending_job()
        with _patch_poll(result=poll_result):
            result = await execute_poll_batch_results([job], temporal, track_update, _put_blob)

        assert result.errors_found == 1
        assert len(updates) == 1
        assert updates[0]["status"] == BatchJobStatus.FAILED

    @pytest.mark.asyncio
    async def test_multiple_pending_jobs_all_processed(self) -> None:
        poll_result = ProviderBatchPollResult(
            status=BatchPollStatus.ENDED,
            entries=[BatchResultEntry(custom_id="req-1", succeeded=True, raw_response_json="{}")],
        )
        temporal = _make_temporal_client()

        jobs = [
            _make_pending_job(request_id="req-1", batch_id="batch-1", workflow_id="wf-1"),
            _make_pending_job(request_id="req-2", batch_id="batch-2", workflow_id="wf-2"),
        ]

        with _patch_poll(result=poll_result):
            result = await execute_poll_batch_results(jobs, temporal, _noop_update, _put_blob)

        assert result.batches_checked == 2
        assert result.signals_sent == 2

    # -----------------------------------------------------------------------
    # T1.3 acceptance criteria — INTERIM poller patch
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_signal_delivery_failure_leaves_row_submitted(self) -> None:
        """Criterion 1: a transient signal-delivery failure on a completed batch
        must NOT advance the row (which would lose the paid result). The row is
        left SUBMITTED — update_status_fn is never called — so the next cycle
        re-polls and re-delivers."""
        poll_result = ProviderBatchPollResult(
            status=BatchPollStatus.ENDED,
            entries=[BatchResultEntry(custom_id="req-1", succeeded=True, raw_response_json="{}")],
        )
        temporal = _make_temporal_client(signal_error=RuntimeError("workflow not found"))
        updates: list[dict] = []

        def track_update(**kwargs):
            updates.append(kwargs)

        job = _make_pending_job()
        with _patch_poll(result=poll_result):
            result = await execute_poll_batch_results([job], temporal, track_update, _put_blob)

        assert result.errors_found == 1
        assert result.signals_sent == 0
        # Row untouched => stays SUBMITTED => re-polled next cycle.
        assert updates == []

    @pytest.mark.asyncio
    async def test_next_cycle_retries_delivery_after_failure(self) -> None:
        """Criterion 1: the cycle after a delivery failure re-delivers and, on
        success, advances the (still-SUBMITTED) row to PROCESSING."""
        poll_result = ProviderBatchPollResult(
            status=BatchPollStatus.ENDED,
            entries=[BatchResultEntry(custom_id="req-1", succeeded=True, raw_response_json="{}")],
        )
        updates: list[dict] = []

        def track_update(**kwargs):
            updates.append(kwargs)

        job = _make_pending_job()
        with _patch_poll(result=poll_result):
            # Cycle 1: delivery fails, row left SUBMITTED.
            failing = _make_temporal_client(signal_error=RuntimeError("workflow not found"))
            first = await execute_poll_batch_results([job], failing, track_update, _put_blob)
            assert first.signals_sent == 0
            assert updates == []

            # Cycle 2: same still-SUBMITTED job re-polled; delivery succeeds.
            working = _make_temporal_client()
            second = await execute_poll_batch_results([job], working, track_update, _put_blob)

        assert second.signals_sent == 1
        assert second.errors_found == 0
        assert len(updates) == 1
        assert updates[0]["status"] == BatchJobStatus.PROCESSING

    @pytest.mark.asyncio
    async def test_missing_signals_waiter_with_error(self) -> None:
        """Criterion 2: a >24h unretrievable batch marks the row MISSING AND now
        sends the waiter an error-payload signal so it fails fast instead of
        burning the 25h wait timeout."""
        temporal = _make_temporal_client()
        updates: list[dict] = []

        def track_update(**kwargs):
            updates.append(kwargs)

        old_time = datetime.now(UTC) - timedelta(hours=25)
        job = _make_pending_job(created_at=old_time)
        with _patch_poll(side_effect=RuntimeError("not found")):
            result = await execute_poll_batch_results([job], temporal, track_update, _put_blob)

        # Row marked MISSING with an error message.
        assert len(updates) == 1
        assert updates[0]["status"] == BatchJobStatus.MISSING
        assert updates[0]["error_message"] is not None

        # Waiter signaled with an error payload (the fail-fast trigger).
        assert result.signals_sent == 1
        temporal.get_workflow_handle.assert_called_once_with("forge-task-test")
        signal = temporal.get_workflow_handle.return_value.signal.call_args[0][1]
        assert signal.request_id == "req-1"
        assert signal.error is not None
        assert signal.result_type == "errored"
        assert signal.raw_response_json is None

    @pytest.mark.asyncio
    async def test_one_errored_job_does_not_wedge_cycle(self) -> None:
        """Criterion 3: one errored job does not wedge the poller — the cycle
        completes (no raise), the other jobs in the same cycle are processed, and
        errors_found reports the failure so the next scheduled run is the retry."""

        def poll(_provider_name: str, batch_id: str) -> ProviderBatchPollResult:
            if batch_id == "batch-bad":
                raise RuntimeError("network error")
            return ProviderBatchPollResult(
                status=BatchPollStatus.ENDED,
                entries=[BatchResultEntry(custom_id="req", succeeded=True, raw_response_json="{}")],
            )

        temporal = _make_temporal_client()
        updates: list[dict] = []

        def track_update(**kwargs):
            updates.append(kwargs)

        jobs = [
            _make_pending_job(request_id="req-good-1", batch_id="batch-1", workflow_id="wf-1"),
            _make_pending_job(request_id="req-bad", batch_id="batch-bad", workflow_id="wf-bad"),
            _make_pending_job(request_id="req-good-2", batch_id="batch-2", workflow_id="wf-2"),
        ]
        with _patch_poll(side_effect=poll):
            result = await execute_poll_batch_results(jobs, temporal, track_update, _put_blob)

        # Cycle completed without raising; the bad job is reported, the two good
        # jobs are still delivered and advanced.
        assert result.batches_checked == 3
        assert result.signals_sent == 2
        assert result.errors_found == 1
        assert [u["status"] for u in updates] == [
            BatchJobStatus.PROCESSING,
            BatchJobStatus.PROCESSING,
        ]


# ---------------------------------------------------------------------------
# _poll_batch_for — the per-provider poll dispatch
# ---------------------------------------------------------------------------


class TestPollBatchFor:
    @pytest.mark.asyncio
    async def test_anthropic_dispatches_to_platform_adapter(self) -> None:
        """The anthropic route no longer goes through sax_llm — it polls via the
        platform batch lane adapter over the shared AsyncAnthropic client."""
        sentinel_client = object()
        expected = ProviderBatchPollResult(status=BatchPollStatus.IN_PROGRESS)
        adapter = AsyncMock(return_value=expected)

        with (
            patch("forge.activities.batch_poll._get_batch_client", return_value=sentinel_client),
            patch("forge.activities.batch_poll._poll_anthropic_batch", adapter),
        ):
            result = await _poll_batch_for("anthropic", "batch-1")

        adapter.assert_awaited_once_with(sentinel_client, "batch-1")
        assert result is expected

    @pytest.mark.asyncio
    async def test_mistral_returns_native_ocr_result(self) -> None:
        """mistral routes through sax_platform.ocr.MistralOcr and returns its
        native poll result directly — no conversion (the lingua franca IS the
        sax_platform.ocr family now)."""
        ocr_result = ProviderBatchPollResult(
            status=BatchPollStatus.ENDED,
            entries=[
                BatchResultEntry(
                    custom_id="req-1",
                    succeeded=True,
                    raw_response_json='{"pages": []}',
                    extracted_images=[
                        ExtractedImage(
                            original_image_id="img-0",
                            page_index=0,
                            image_base64="ZmFrZQ==",
                        )
                    ],
                )
            ],
        )
        mistral_provider = MagicMock()
        mistral_provider.poll_batch = AsyncMock(return_value=ocr_result)

        with (
            patch("sax_platform.ocr.make_mistral_client", return_value=MagicMock()),
            patch("sax_platform.ocr.MistralOcr", return_value=mistral_provider),
        ):
            result = await _poll_batch_for("mistral", "batch-mistral")

        mistral_provider.poll_batch.assert_awaited_once_with("batch-mistral")
        assert result is ocr_result


# ---------------------------------------------------------------------------
# _poll_anthropic_batch — status mapping + entry construction
# ---------------------------------------------------------------------------


class TestPollAnthropicBatch:
    @pytest.mark.asyncio
    async def test_in_progress_returns_without_fetching(self) -> None:
        client = MagicMock()
        get_status = AsyncMock(return_value=_batch_status("in_progress"))
        fetch = AsyncMock()

        with (
            patch("sax_platform.llm.batch.get_batch_status", get_status),
            patch("sax_platform.llm.batch.fetch_batch_result_lines", fetch),
        ):
            result = await _poll_anthropic_batch(client, "b-1")

        assert result.status == BatchPollStatus.IN_PROGRESS
        assert result.entries == []
        fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_canceling_maps_to_in_progress(self) -> None:
        # Anthropic's only non-ended statuses are in_progress / canceling; both
        # keep polling (IN_PROGRESS), never a batch-level terminal failure.
        client = MagicMock()
        get_status = AsyncMock(return_value=_batch_status("canceling"))
        with (
            patch("sax_platform.llm.batch.get_batch_status", get_status),
            patch("sax_platform.llm.batch.fetch_batch_result_lines", AsyncMock()),
        ):
            result = await _poll_anthropic_batch(client, "b-1")

        assert result.status == BatchPollStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_ended_builds_succeeded_entry_from_raw_line(self) -> None:
        client = MagicMock()
        get_status = AsyncMock(return_value=_batch_status("ended"))
        fetch = AsyncMock(return_value=[("req-1", '{"stop_reason": "end_turn"}')])

        with (
            patch("sax_platform.llm.batch.get_batch_status", get_status),
            patch("sax_platform.llm.batch.fetch_batch_result_lines", fetch),
        ):
            result = await _poll_anthropic_batch(client, "b-1")

        assert result.status == BatchPollStatus.ENDED
        assert len(result.entries) == 1
        entry = result.entries[0]
        assert entry.custom_id == "req-1"
        assert entry.succeeded is True
        assert entry.raw_response_json == '{"stop_reason": "end_turn"}'
        assert entry.error is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("kind", "fragment"),
        [
            ("errored", "Batch error"),
            ("expired", "expired"),
            ("canceled", "canceled"),
        ],
    )
    async def test_ended_maps_failed_line_to_error_entry(self, kind: str, fragment: str) -> None:
        client = MagicMock()
        get_status = AsyncMock(return_value=_batch_status("ended"))
        failed = BatchRequestFailed(kind=kind, detail="boom")
        fetch = AsyncMock(return_value=[("req-x", failed)])

        with (
            patch("sax_platform.llm.batch.get_batch_status", get_status),
            patch("sax_platform.llm.batch.fetch_batch_result_lines", fetch),
        ):
            result = await _poll_anthropic_batch(client, "b-1")

        entry = result.entries[0]
        assert entry.custom_id == "req-x"
        assert entry.succeeded is False
        assert entry.raw_response_json is None
        assert entry.error is not None
        assert fragment in entry.error
