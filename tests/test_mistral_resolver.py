"""Tests for forge.activities._mistral — shared lazily-cached MistralOcr resolver.

2026-07 Phase 3 code review, item 5a: batch_submit.py and batch_poll.py each
used to build their own ``MistralOcr(make_mistral_client())`` per call — a
fresh Mistral SDK client every poll cycle. This module caches the single pair
at module scope; these tests pin the caching contract (single construction
across repeated calls, shared across both call sites, and no error-swallowing
when construction itself raises — make_mistral_client() may raise ValueError
when MISTRAL_API_KEY is unset).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge.activities._mistral import get_mistral_ocr, reset_mistral_ocr_cache
from forge.activities.batch_poll import _poll_batch_for
from forge.activities.batch_submit import _resolve_blob_submit_provider


class TestGetMistralOcrCaching:
    def test_constructs_once_across_repeated_calls(self) -> None:
        sentinel_client = object()
        sentinel_provider = object()
        with (
            patch(
                "sax_platform.ocr.make_mistral_client", return_value=sentinel_client
            ) as mock_make_client,
            patch(
                "sax_platform.ocr.MistralOcr", return_value=sentinel_provider
            ) as mock_mistral_ocr,
        ):
            first = get_mistral_ocr()
            second = get_mistral_ocr()
            third = get_mistral_ocr()

        mock_make_client.assert_called_once_with()
        mock_mistral_ocr.assert_called_once_with(sentinel_client)
        assert first is second is third is sentinel_provider

    def test_shared_across_submit_and_poll_call_sites(self) -> None:
        """The submit-blob SPI and the poller dispatch through the same cache —
        one MistralOcr/client pair serves both, not one per call site."""
        sentinel_client = object()
        sentinel_provider = object()
        with (
            patch("sax_platform.ocr.make_mistral_client", return_value=sentinel_client),
            patch(
                "sax_platform.ocr.MistralOcr", return_value=sentinel_provider
            ) as mock_mistral_ocr,
        ):
            resolved = _resolve_blob_submit_provider("mistral")
            assert resolved is sentinel_provider
            # A second entry point (the poller) must reuse the same cached
            # instance rather than constructing its own.
            again = get_mistral_ocr()

        mock_mistral_ocr.assert_called_once()
        assert again is sentinel_provider

    def test_reset_forces_reconstruction(self) -> None:
        with (
            patch("sax_platform.ocr.make_mistral_client", return_value=object()),
            patch("sax_platform.ocr.MistralOcr", side_effect=[object(), object()]) as mock_ocr,
        ):
            first = get_mistral_ocr()
            reset_mistral_ocr_cache()
            second = get_mistral_ocr()

        assert mock_ocr.call_count == 2
        assert first is not second

    def test_missing_api_key_raises_and_is_not_cached(self) -> None:
        """make_mistral_client() may raise ValueError when MISTRAL_API_KEY is
        unset (the parallel libs/sax-llm + libs/sax-platform fix). The raise
        must propagate on every call — not be swallowed into a cached failure
        state that silently returns something else on retry."""
        with patch(
            "sax_platform.ocr.make_mistral_client",
            side_effect=ValueError("MISTRAL_API_KEY is not set"),
        ) as mock_make_client:
            with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
                get_mistral_ocr()
            # A second call retries construction rather than returning a
            # cached None/sentinel — the failed attempt left no cache entry.
            with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
                get_mistral_ocr()

        assert mock_make_client.call_count == 2


class TestPollBatchForUsesSharedResolver:
    @pytest.mark.asyncio
    async def test_poll_and_submit_share_one_cached_instance(self) -> None:
        """_poll_batch_for (batch_poll.py) and _resolve_blob_submit_provider
        (batch_submit.py) are the two former hand-rolled dispatch sites —
        after the dedup they must resolve through the identical cached
        MistralOcr, not two independently-constructed ones."""
        from sax_llm.models import BatchPollStatus as SaxLlmBatchPollStatus
        from sax_platform.ocr import BatchPollResult as PlatformBatchPollResult

        mistral_provider = MagicMock()
        mistral_provider.poll_batch = AsyncMock(
            return_value=PlatformBatchPollResult(status="ended", entries=[])
        )

        with (
            patch("sax_platform.ocr.make_mistral_client", return_value=object()),
            patch("sax_platform.ocr.MistralOcr", return_value=mistral_provider) as mock_mistral_ocr,
        ):
            submit_resolved = _resolve_blob_submit_provider("mistral")
            poll_result = await _poll_batch_for("mistral", "batch-mistral")

        mock_mistral_ocr.assert_called_once()
        assert submit_resolved is mistral_provider
        assert poll_result.status == SaxLlmBatchPollStatus.ENDED
        mistral_provider.poll_batch.assert_awaited_once_with("batch-mistral")
