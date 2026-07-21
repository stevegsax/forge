"""Tests for forge.activities.batch_fetch — timer-loop status/fetch cores (T4.1).

Forge submits anthropic only (T4.2 ST3): a non-anthropic provider raises before
the client is touched. The explicit provider threading stays (honest transport).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sax_platform.contracts.models import parse_batch_result_payload
from sax_platform.llm.batch import BatchRequestFailed, BatchStatus

from forge.activities.batch_fetch import execute_batch_status, execute_fetch_batch_result
from forge.models import BatchStatusInput, FetchBatchResultInput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _put_blob(custom_id: str, data: bytes) -> str:
    """Fake blob upload: record nothing, return a deterministic key."""
    return f"blob-{custom_id}"


def _capture_put() -> tuple[object, dict[str, bytes]]:
    """A put_result_blob that records the uploaded bytes by key."""
    captured: dict[str, bytes] = {}

    def put(custom_id: str, data: bytes) -> str:
        key = f"blob-{custom_id}"
        captured[key] = data
        return key

    return put, captured


# ---------------------------------------------------------------------------
# execute_batch_status — anthropic
# ---------------------------------------------------------------------------


class TestBatchStatusAnthropic:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("processing_status", "expected_state"),
        [
            ("in_progress", "in_progress"),
            # Anthropic's only non-ended statuses are in_progress / canceling; both
            # map to in_progress (never a batch-level terminal failure).
            ("canceling", "in_progress"),
            ("ended", "ended"),
        ],
    )
    async def test_status_mapping(self, processing_status: str, expected_state: str) -> None:
        client = MagicMock()
        get_status = AsyncMock(return_value=_batch_status(processing_status))
        with patch("sax_platform.llm.batch.get_batch_status", get_status):
            result = await execute_batch_status(
                BatchStatusInput(batch_id="b-1", provider="anthropic"),
                client=client,
            )

        assert result.batch_id == "b-1"
        assert result.state == expected_state
        get_status.assert_awaited_once_with(client, "b-1")


# ---------------------------------------------------------------------------
# execute_fetch_batch_result — anthropic
# ---------------------------------------------------------------------------


class TestFetchAnthropic:
    @pytest.mark.asyncio
    async def test_selects_matching_custom_id_inline(self) -> None:
        # Several custom_ids come back; the fetch selects THIS waiter's line and,
        # for a small image-free body, delivers it inline.
        lines: list[tuple[str, str | BatchRequestFailed]] = [
            ("req-other", '{"a": 1}'),
            ("req-1", '{"stop_reason": "end_turn"}'),
            ("req-third", '{"c": 3}'),
        ]
        fetch = AsyncMock(return_value=lines)
        with patch("sax_platform.llm.batch.fetch_batch_result_lines", fetch):
            result = await execute_fetch_batch_result(
                FetchBatchResultInput(batch_id="b-1", request_id="req-1", provider="anthropic"),
                client=MagicMock(),
                put_result_blob=_put_blob,
            )

        assert result.raw_response_json == '{"stop_reason": "end_turn"}'
        assert result.s3_key is None
        assert result.error is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("kind", "expected_error"),
        [
            ("errored", "Batch error: boom"),
            ("expired", "Batch request expired (24h limit)"),
            ("canceled", "Batch request was canceled"),
        ],
    )
    async def test_failed_line_error_matches_poller_format(
        self, kind: str, expected_error: str
    ) -> None:
        from forge.activities.batch_fetch import _format_request_failure

        failed = BatchRequestFailed(kind=kind, detail="boom")
        fetch = AsyncMock(return_value=[("req-1", failed)])
        with patch("sax_platform.llm.batch.fetch_batch_result_lines", fetch):
            result = await execute_fetch_batch_result(
                FetchBatchResultInput(batch_id="b-1", request_id="req-1", provider="anthropic"),
                client=MagicMock(),
                put_result_blob=_put_blob,
            )

        assert result.error == expected_error
        # The error string a waiter sees is produced by the fetch activity's own
        # formatter (_format_request_failure), pinned here against the same source.
        assert result.error == _format_request_failure(failed)
        assert result.raw_response_json is None
        assert result.s3_key is None

    @pytest.mark.asyncio
    async def test_missing_custom_id_returns_error(self) -> None:
        fetch = AsyncMock(return_value=[("req-other", "{}")])
        with patch("sax_platform.llm.batch.fetch_batch_result_lines", fetch):
            result = await execute_fetch_batch_result(
                FetchBatchResultInput(batch_id="b-1", request_id="req-1", provider="anthropic"),
                client=MagicMock(),
                put_result_blob=_put_blob,
            )

        assert result.error is not None
        assert "req-1" in result.error
        assert result.raw_response_json is None
        assert result.s3_key is None

    @pytest.mark.asyncio
    async def test_large_body_delivers_pointer(self) -> None:
        big = '{"x": "' + "a" * (256 * 1024) + '"}'
        fetch = AsyncMock(return_value=[("req-1", big)])
        put, captured = _capture_put()
        with patch("sax_platform.llm.batch.fetch_batch_result_lines", fetch):
            result = await execute_fetch_batch_result(
                FetchBatchResultInput(batch_id="b-1", request_id="req-1", provider="anthropic"),
                client=MagicMock(),
                put_result_blob=put,
            )

        # Over the 256KB inline threshold => stashed to a blob, pointer returned.
        assert result.s3_key == "blob-req-1"
        assert result.raw_response_json is None
        body, images = parse_batch_result_payload(captured["blob-req-1"].decode("utf-8"))
        assert body == big
        assert images == []


# ---------------------------------------------------------------------------
# Non-anthropic providers are rejected (forge submits anthropic only, T4.2 ST3)
# ---------------------------------------------------------------------------


class TestNonAnthropicRejected:
    @pytest.mark.asyncio
    async def test_status_rejects_non_anthropic_provider(self) -> None:
        """A non-anthropic batch is its owning app's concern — status raises before
        the client is touched (no provider call is made)."""
        get_status = AsyncMock()
        with (
            patch("sax_platform.llm.batch.get_batch_status", get_status),
            pytest.raises(RuntimeError, match="anthropic only"),
        ):
            await execute_batch_status(
                BatchStatusInput(batch_id="batch-m", provider="mistral"),
                client=MagicMock(),
            )
        get_status.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fetch_rejects_non_anthropic_provider(self) -> None:
        """A non-anthropic fetch raises before any provider download is attempted."""
        fetch = AsyncMock()
        with (
            patch("sax_platform.llm.batch.fetch_batch_result_lines", fetch),
            pytest.raises(RuntimeError, match="anthropic only"),
        ):
            await execute_fetch_batch_result(
                FetchBatchResultInput(batch_id="batch-m", request_id="req-1", provider="mistral"),
                client=MagicMock(),
                put_result_blob=_put_blob,
            )
        fetch.assert_not_awaited()
