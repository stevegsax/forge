"""Tests for forge.activities.batch_submit — batch submit activity."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.activities.batch_submit import execute_batch_submit
from forge.models import AssembledContext, BatchSubmitInput, ThinkingConfig
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
    thinking: ThinkingConfig | None = None,
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
        thinking=thinking or ThinkingConfig(),
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
    async def test_passes_thinking_config_through(self) -> None:
        provider = _make_mock_provider()
        input_data = _make_input(
            model_name="claude-sonnet-4-5-20250929",
            thinking=ThinkingConfig(budget_tokens=10_000, effort="high"),
        )

        await execute_batch_submit(input_data, provider)

        call_kwargs = provider.build_request_params.call_args
        assert call_kwargs[1].get("thinking_budget_tokens") == 10_000

    @pytest.mark.asyncio
    async def test_passes_max_tokens_through(self) -> None:
        provider = _make_mock_provider()
        input_data = _make_input(max_tokens=8192)

        await execute_batch_submit(input_data, provider)

        call_kwargs = provider.build_request_params.call_args
        assert call_kwargs[1].get("max_tokens") == 8192
