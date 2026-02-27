"""Tests for MistralProvider."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.llm_providers.mistral import MistralProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(
    tool_input: dict,
    *,
    model: str = "mistral-large-latest",
    prompt_tokens: int = 80,
    completion_tokens: int = 150,
) -> MagicMock:
    """Build a mock Mistral ChatCompletionResponse."""
    func = MagicMock()
    func.arguments = json.dumps(tool_input)

    tool_call = MagicMock()
    tool_call.function = func

    message = MagicMock()
    message.tool_calls = [tool_call]

    choice = MagicMock()
    choice.message = message

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model
    response.model_dump = MagicMock(return_value={"model": model, "choices": []})

    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildRequestParams:
    """Tests for MistralProvider.build_request_params."""

    def test_builds_mistral_format(self) -> None:
        from pydantic import BaseModel

        class TestOutput(BaseModel):
            """Test output model."""

            value: str

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        params = provider.build_request_params(
            system_prompt="You are helpful.",
            user_prompt="Do something.",
            output_type=TestOutput,
            model="mistral-large-latest",
            max_tokens=1024,
        )

        assert params["model"] == "mistral-large-latest"
        assert params["max_tokens"] == 1024
        assert params["tool_choice"] == "any"

        # Messages: system + user
        assert len(params["messages"]) == 2
        assert params["messages"][0]["role"] == "system"
        assert params["messages"][1]["role"] == "user"

        # Tool definition in function format
        tool = params["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "test_output"

    def test_no_cache_control(self) -> None:
        from pydantic import BaseModel

        class TestOutput(BaseModel):
            """Test output model."""

            value: str

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        params = provider.build_request_params(
            system_prompt="sys",
            user_prompt="user",
            output_type=TestOutput,
            model="mistral-large-latest",
            max_tokens=1024,
            cache_instructions=True,
            cache_tool_definitions=True,
        )

        # No cache_control on messages or tools
        for msg in params["messages"]:
            assert "cache_control" not in msg
        assert "cache_control" not in params["tools"][0]

    def test_thinking_budget_silently_ignored(self) -> None:
        from pydantic import BaseModel

        class TestOutput(BaseModel):
            """Test output model."""

            value: str

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        params = provider.build_request_params(
            system_prompt="sys",
            user_prompt="user",
            output_type=TestOutput,
            model="mistral-large-latest",
            max_tokens=1024,
            thinking_budget_tokens=5000,
        )

        # No thinking param in output
        assert "thinking" not in params


class TestCall:
    """Tests for MistralProvider.call."""

    @pytest.mark.asyncio
    async def test_extracts_tool_input(self) -> None:
        tool_input = {"value": "hello"}
        mock_response = _make_mock_response(tool_input)

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()
        provider._client.chat.complete_async = AsyncMock(return_value=mock_response)

        result = await provider.call({"model": "mistral-large-latest"})

        assert result.tool_input == tool_input

    @pytest.mark.asyncio
    async def test_maps_usage_tokens(self) -> None:
        mock_response = _make_mock_response(
            {"value": "ok"}, prompt_tokens=120, completion_tokens=300
        )

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()
        provider._client.chat.complete_async = AsyncMock(return_value=mock_response)

        result = await provider.call({"model": "test"})

        assert result.input_tokens == 120
        assert result.output_tokens == 300

    @pytest.mark.asyncio
    async def test_cache_tokens_are_zero(self) -> None:
        mock_response = _make_mock_response({"value": "ok"})

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()
        provider._client.chat.complete_async = AsyncMock(return_value=mock_response)

        result = await provider.call({"model": "test"})

        assert result.cache_creation_input_tokens == 0
        assert result.cache_read_input_tokens == 0

    @pytest.mark.asyncio
    async def test_model_name_from_response(self) -> None:
        mock_response = _make_mock_response({"value": "ok"}, model="mistral-large-2")

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()
        provider._client.chat.complete_async = AsyncMock(return_value=mock_response)

        result = await provider.call({"model": "mistral-large-latest"})

        assert result.model_name == "mistral-large-2"


class TestSupportsBatch:
    """Tests for batch support flag."""

    def test_supports_batch_is_true(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()
        assert provider.supports_batch is True


class TestBuildBatchRequest:
    """Tests for MistralProvider.build_batch_request."""

    def test_wraps_params_with_custom_id_and_body(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        result = provider.build_batch_request("req-456", {"model": "test"})

        assert result["custom_id"] == "req-456"
        assert result["body"]["model"] == "test"
