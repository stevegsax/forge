"""Tests for forge.activities.batch_submit — batch submit activity."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.activities.batch_submit import DEFAULT_MODEL, execute_batch_submit
from forge.models import AssembledContext, BatchSubmitInput


def _make_mock_client(batch_id: str = "msgbatch_test123") -> AsyncMock:
    """Build a mock AsyncAnthropic client with batches.create."""
    mock_batch = MagicMock()
    mock_batch.id = batch_id

    client = AsyncMock()
    client.messages.batches.create = AsyncMock(return_value=mock_batch)
    return client


def _make_input(
    *,
    model_name: str = "",
    output_type_name: str = "LLMResponse",
    thinking_budget_tokens: int = 0,
    thinking_effort: str = "high",
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
        thinking_budget_tokens=thinking_budget_tokens,
        thinking_effort=thinking_effort,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# execute_batch_submit
# ---------------------------------------------------------------------------


class TestExecuteBatchSubmit:
    @pytest.mark.asyncio
    async def test_returns_batch_submit_result(self) -> None:
        client = _make_mock_client()
        input = _make_input()

        result = await execute_batch_submit(input, client)

        assert result.batch_id == "msgbatch_test123"
        assert result.request_id  # non-empty UUID

    @pytest.mark.asyncio
    async def test_calls_batches_create(self) -> None:
        client = _make_mock_client()
        input = _make_input()

        await execute_batch_submit(input, client)

        client.messages.batches.create.assert_called_once()
        call_kwargs = client.messages.batches.create.call_args
        requests = call_kwargs.kwargs.get("requests") or call_kwargs[1].get("requests")
        assert len(requests) == 1
        assert "custom_id" in requests[0]
        assert "params" in requests[0]

    @pytest.mark.asyncio
    async def test_uses_context_model_name(self) -> None:
        client = _make_mock_client()
        input = _make_input(model_name="claude-opus-4-6")

        await execute_batch_submit(input, client)

        call_kwargs = client.messages.batches.create.call_args
        requests = call_kwargs.kwargs.get("requests") or call_kwargs[1].get("requests")
        params = requests[0]["params"]
        assert params["model"] == "claude-opus-4-6"

    @pytest.mark.asyncio
    async def test_falls_back_to_default_model(self) -> None:
        client = _make_mock_client()
        input = _make_input(model_name="")

        await execute_batch_submit(input, client)

        call_kwargs = client.messages.batches.create.call_args
        requests = call_kwargs.kwargs.get("requests") or call_kwargs[1].get("requests")
        params = requests[0]["params"]
        assert params["model"] == DEFAULT_MODEL

    @pytest.mark.asyncio
    async def test_request_id_is_uuid_format(self) -> None:
        client = _make_mock_client()
        input = _make_input()

        result = await execute_batch_submit(input, client)

        # UUID format: 8-4-4-4-12 hex digits
        parts = result.request_id.split("-")
        assert len(parts) == 5
        assert [len(p) for p in parts] == [8, 4, 4, 4, 12]

    @pytest.mark.asyncio
    async def test_passes_thinking_config_through(self) -> None:
        client = _make_mock_client()
        input = _make_input(
            model_name="claude-sonnet-4-5-20250929",
            thinking_budget_tokens=10_000,
            thinking_effort="high",
        )

        await execute_batch_submit(input, client)

        call_kwargs = client.messages.batches.create.call_args
        requests = call_kwargs.kwargs.get("requests") or call_kwargs[1].get("requests")
        params = requests[0]["params"]
        assert "thinking" in params
        assert params["thinking"]["budget_tokens"] >= 10_000

    @pytest.mark.asyncio
    async def test_passes_max_tokens_through(self) -> None:
        client = _make_mock_client()
        input = _make_input(max_tokens=8192)

        await execute_batch_submit(input, client)

        call_kwargs = client.messages.batches.create.call_args
        requests = call_kwargs.kwargs.get("requests") or call_kwargs[1].get("requests")
        params = requests[0]["params"]
        assert params["max_tokens"] == 8192
