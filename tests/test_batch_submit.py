"""Tests for forge.activities.batch_submit — batch submit activity."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sax_platform.contracts.models import BatchSubmitSpiInput
from sax_platform.llm.batch import BatchHandle

from forge.activities.batch_submit import (
    _AnthropicBlobSubmit,
    _resolve_blob_submit_provider,
    execute_batch_submit,
    execute_submit_batch_blob,
)
from forge.models import AssembledContext, BatchSubmitInput, ThinkingPolicy


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


def _submit_mock(batch_id: str = "msgbatch_test123") -> AsyncMock:
    """AsyncMock standing in for the platform ``submit_batch`` helper."""
    return AsyncMock(return_value=BatchHandle(batch_id=batch_id, processing_status="in_progress"))


def _submitted_params(submit: AsyncMock) -> dict:
    """Return the ``params`` of the single request handed to ``submit_batch``."""
    _client, requests = submit.await_args.args
    assert len(requests) == 1
    return requests[0]["params"]


# ---------------------------------------------------------------------------
# execute_batch_submit
# ---------------------------------------------------------------------------


class TestExecuteBatchSubmit:
    @pytest.mark.asyncio
    async def test_returns_batch_submit_result(self) -> None:
        client = MagicMock()
        submit = _submit_mock()
        with patch("sax_platform.llm.batch.submit_batch", submit):
            result = await execute_batch_submit(_make_input(), client)

        assert result.batch_id == "msgbatch_test123"
        assert result.request_id  # non-empty UUID
        called_client, requests = submit.await_args.args
        assert called_client is client
        assert requests[0]["custom_id"] == result.request_id

    @pytest.mark.asyncio
    async def test_request_id_is_uuid_format(self) -> None:
        client = MagicMock()
        with patch("sax_platform.llm.batch.submit_batch", _submit_mock()):
            result = await execute_batch_submit(_make_input(), client)

        parts = result.request_id.split("-")
        assert [len(p) for p in parts] == [8, 4, 4, 4, 12]

    @pytest.mark.asyncio
    async def test_resolves_output_type_into_structured_format(self) -> None:
        client = MagicMock()
        submit = _submit_mock()
        with patch("sax_platform.llm.batch.submit_batch", submit):
            await execute_batch_submit(_make_input(output_type_name="LLMResponse"), client)

        params = _submitted_params(submit)
        assert params["output_config"]["format"]["type"] == "json_schema"

    @pytest.mark.asyncio
    async def test_text_mode_omits_structured_format(self) -> None:
        client = MagicMock()
        submit = _submit_mock()
        with patch("sax_platform.llm.batch.submit_batch", submit):
            # Empty output_type_name -> text lane, no structured-output format.
            await execute_batch_submit(_make_input(output_type_name=""), client)

        params = _submitted_params(submit)
        assert "format" not in params.get("output_config", {})

    @pytest.mark.asyncio
    async def test_passes_thinking_through_for_non_haiku(self) -> None:
        client = MagicMock()
        submit = _submit_mock()
        input_data = _make_input(
            model_name="anthropic:claude-sonnet-5",
            thinking=ThinkingPolicy(enabled=True, effort="high"),
        )
        with patch("sax_platform.llm.batch.submit_batch", submit):
            await execute_batch_submit(input_data, client)

        params = _submitted_params(submit)
        assert params["model"] == "claude-sonnet-5"
        assert params["thinking"] == {"type": "adaptive"}
        assert params["output_config"]["effort"] == "high"

    @pytest.mark.asyncio
    async def test_haiku_guard_drops_thinking(self) -> None:
        # Old sax_llm silently dropped thinking for haiku; the platform builder
        # would otherwise emit a shape haiku 400s on, so batch_submit passes None.
        client = MagicMock()
        submit = _submit_mock()
        input_data = _make_input(
            model_name="anthropic:claude-haiku-4-5",
            thinking=ThinkingPolicy(enabled=True, effort="high"),
        )
        with patch("sax_platform.llm.batch.submit_batch", submit):
            await execute_batch_submit(input_data, client)

        params = _submitted_params(submit)
        assert "thinking" not in params
        assert "effort" not in params.get("output_config", {})

    @pytest.mark.asyncio
    async def test_passes_max_tokens_and_prompts_through(self) -> None:
        client = MagicMock()
        submit = _submit_mock()
        with patch("sax_platform.llm.batch.submit_batch", submit):
            await execute_batch_submit(_make_input(max_tokens=8192), client)

        params = _submitted_params(submit)
        assert params["max_tokens"] == 8192
        assert params["messages"] == [{"role": "user", "content": "Do something."}]
        # system is normalized to a text block by the builder.
        assert params["system"][0]["text"] == "You are a helpful assistant."


# ---------------------------------------------------------------------------
# execute_submit_batch_blob (opaque-blob submit SPI)
# ---------------------------------------------------------------------------


class TestExecuteSubmitBatchBlob:
    @pytest.mark.asyncio
    async def test_fetches_blob_and_submits_verbatim(self) -> None:
        provider = MagicMock()
        provider.submit_batch = AsyncMock(return_value="batch_abc")
        requests = [{"custom_id": "req-1", "body": {"document": "x"}}]
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
# _resolve_blob_submit_provider — the SPI's per-provider dispatch
# ---------------------------------------------------------------------------


class TestResolveBlobSubmitProvider:
    def test_anthropic_resolves_to_platform_adapter(self) -> None:
        """The anthropic route wraps the injected AsyncAnthropic client in the
        platform-batch adapter (client comes from the BatchActivities root)."""
        sentinel_client = object()
        result = _resolve_blob_submit_provider(
            "anthropic", client=sentinel_client, mistral_ocr=None
        )

        assert isinstance(result, _AnthropicBlobSubmit)
        assert result._client is sentinel_client

    @pytest.mark.asyncio
    async def test_anthropic_adapter_submits_via_platform(self) -> None:
        client = MagicMock()
        submit = _submit_mock("batch_via_adapter")
        adapter = _AnthropicBlobSubmit(client)
        requests = [{"custom_id": "c", "params": {"model": "m"}}]
        with patch("sax_platform.llm.batch.submit_batch", submit):
            # model / endpoint are accepted but ignored (they ride inside params).
            batch_id = await adapter.submit_batch(requests, "ignored-model", endpoint="/v1/x")

        assert batch_id == "batch_via_adapter"
        submit.assert_awaited_once_with(client, requests)

    def test_mistral_resolves_to_injected_ocr(self) -> None:
        """mistral routes through the injected MistralOcr (built once at the
        BatchActivities root when MISTRAL_API_KEY is set), not built per-call."""
        sentinel_provider = object()
        result = _resolve_blob_submit_provider(
            "mistral", client=object(), mistral_ocr=sentinel_provider
        )
        assert result is sentinel_provider

    def test_mistral_without_ocr_raises(self) -> None:
        """A mistral submit with no MistralOcr injected (MISTRAL_API_KEY unset at
        startup) raises a clear error at point of use."""
        with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
            _resolve_blob_submit_provider("mistral", client=object(), mistral_ocr=None)
