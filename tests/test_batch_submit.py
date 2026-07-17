"""Tests for forge.activities.batch_submit — batch submit activity."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sax_platform.contracts.models import BatchSubmitSpiInput

from forge.activities.batch_submit import (
    _resolve_blob_submit_provider,
    execute_batch_submit,
    execute_submit_batch_blob,
)
from forge.models import AssembledContext, BatchSubmitInput, ThinkingPolicy
from tests.conftest import build_mock_provider


def _make_mock_provider(batch_id: str = "msgbatch_test123") -> MagicMock:
    """Build a mock LLMProvider with batch methods."""
    provider = build_mock_provider(
        tool_input={},
        model_name="test-model",
    )
    provider.build_batch_request = MagicMock(
        return_value={"custom_id": "mock-id", "params": {"model": "test"}}
    )
    provider.submit_batch = AsyncMock(return_value=batch_id)
    return provider


def _make_input(
    *,
    model_name: str = "",
    output_type_name: str = "LLMResponse",
    thinking: ThinkingPolicy | None = None,
    max_tokens: int = 4096,
) -> BatchSubmitInput:
    """Build a BatchSubmitInput with sensible defaults."""
    context = AssembledContext(
        task_id="test-task",
        system_prompt="You are a helpful assistant.",
        user_prompt="Do something.",
        model_name=model_name,
    )
    return BatchSubmitInput(
        context=context,
        output_type_name=output_type_name,
        workflow_id="wf-test-123",
        thinking=thinking or ThinkingPolicy(),
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# execute_batch_submit
# ---------------------------------------------------------------------------


class TestExecuteBatchSubmit:
    @pytest.mark.asyncio
    async def test_returns_batch_submit_result(self) -> None:
        provider = _make_mock_provider()
        input_data = _make_input()

        result = await execute_batch_submit(input_data, provider)

        assert result.batch_id == "msgbatch_test123"
        assert result.request_id  # non-empty UUID

    @pytest.mark.asyncio
    async def test_calls_provider_submit_batch(self) -> None:
        provider = _make_mock_provider()
        input_data = _make_input()

        await execute_batch_submit(input_data, provider)

        provider.build_request_params.assert_called_once()
        provider.build_batch_request.assert_called_once()
        provider.submit_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_id_is_uuid_format(self) -> None:
        provider = _make_mock_provider()
        input_data = _make_input()

        result = await execute_batch_submit(input_data, provider)

        # UUID format: 8-4-4-4-12 hex digits
        parts = result.request_id.split("-")
        assert len(parts) == 5
        assert [len(p) for p in parts] == [8, 4, 4, 4, 12]

    @pytest.mark.asyncio
    async def test_passes_thinking_policy_through(self) -> None:
        provider = _make_mock_provider()
        input_data = _make_input(
            model_name="claude-sonnet-4-5-20250929",
            thinking=ThinkingPolicy(enabled=True, effort="high"),
        )

        await execute_batch_submit(input_data, provider)

        call_kwargs = provider.build_request_params.call_args
        assert call_kwargs[1].get("thinking_enabled") is True
        assert call_kwargs[1].get("effort") == "high"

    @pytest.mark.asyncio
    async def test_passes_max_tokens_through(self) -> None:
        provider = _make_mock_provider()
        input_data = _make_input(max_tokens=8192)

        await execute_batch_submit(input_data, provider)

        call_kwargs = provider.build_request_params.call_args
        assert call_kwargs[1].get("max_tokens") == 8192


# ---------------------------------------------------------------------------
# execute_submit_batch_blob (opaque-blob submit SPI)
# ---------------------------------------------------------------------------


class TestExecuteSubmitBatchBlob:
    @pytest.mark.asyncio
    async def test_fetches_blob_and_submits_verbatim(self) -> None:
        provider = MagicMock()
        provider.submit_batch = AsyncMock(return_value="batch_abc")
        requests = [{"custom_id": "req-1", "body": {"document": "x"}}]
        import json

        blob = json.dumps(requests).encode("utf-8")

        def fetch(key: str) -> bytes:
            assert key == "blob-key-1"
            return blob

        spi = BatchSubmitSpiInput(
            s3_key="blob-key-1",
            model="mistral-ocr-latest",
            endpoint="/v1/ocr",
            provider="mistral",
            custom_id="req-1",
        )

        result = await execute_submit_batch_blob(spi, provider, fetch)

        assert result.batch_id == "batch_abc"
        assert result.request_id == "req-1"
        assert result.provider == "mistral"
        provider.submit_batch.assert_awaited_once_with(
            requests, "mistral-ocr-latest", endpoint="/v1/ocr"
        )

    @pytest.mark.asyncio
    async def test_writes_nothing_to_store(self) -> None:
        # The SPI never touches the store; passing a fetch + provider is the whole
        # dependency surface. A submit that raised before returning a batch_id must
        # propagate (no swallow) so the caller's record step is never reached.
        provider = MagicMock()
        provider.submit_batch = AsyncMock(side_effect=RuntimeError("provider down"))
        spi = BatchSubmitSpiInput(s3_key="k", model="m", provider="mistral", custom_id="c")
        with pytest.raises(RuntimeError, match="provider down"):
            await execute_submit_batch_blob(spi, provider, lambda _k: b"[]")


# ---------------------------------------------------------------------------
# _resolve_blob_submit_provider — the SPI's per-provider dispatch (T3.3)
# ---------------------------------------------------------------------------


class TestResolveBlobSubmitProvider:
    def test_anthropic_resolves_via_sax_llm_registry(self) -> None:
        """Every non-mistral provider name still goes through sax_llm's registry,
        unchanged by the mistral migration."""
        sentinel_provider = object()
        with patch(
            "sax_llm.get_provider_by_name", return_value=sentinel_provider
        ) as mock_get_provider:
            result = _resolve_blob_submit_provider("anthropic")

        mock_get_provider.assert_called_once_with("anthropic")
        assert result is sentinel_provider

    def test_mistral_resolves_via_sax_platform_ocr(self) -> None:
        """mistral no longer resolves through sax_llm at all (it carries no
        provider entry for it post-T3.3) — it routes through
        sax_platform.ocr.MistralOcr, built from make_mistral_client()."""
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
            result = _resolve_blob_submit_provider("mistral")

        mock_make_client.assert_called_once_with()
        mock_mistral_ocr.assert_called_once_with(sentinel_client)
        assert result is sentinel_provider
