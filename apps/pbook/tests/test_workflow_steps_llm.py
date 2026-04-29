"""Tests for the generic llm_chat activity and its output-type registry."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel
from sax_llm.models import ProviderResponse

from pbook.llm import reset_provider, set_provider
from pbook.workflow_steps import (
    LLMChatInput,
    LLMChatResult,
    llm_chat,
    register_output_type,
    reset_registry,
    resolve_output_type,
)


class _ToyResult(BaseModel):
    answer: str
    score: int = 0


@pytest.fixture(autouse=True)
def _isolate_state():
    """Ensure a clean registry + provider between tests."""
    reset_registry()
    reset_provider()
    yield
    reset_registry()
    reset_provider()


def _mock_provider(tool_input: dict) -> MagicMock:
    """Build a MagicMock LLMProvider that returns a canned ProviderResponse."""
    provider = MagicMock()
    provider.build_request_params.return_value = {"mock": True}
    provider.call = AsyncMock(
        return_value=ProviderResponse(
            tool_input=tool_input,
            model_name="anthropic:test",
            input_tokens=42,
            output_tokens=17,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=5,
            raw_response_json="{}",
        ),
    )
    return provider


# ---------------------------------------------------------------------------
# Output type registry
# ---------------------------------------------------------------------------


class TestOutputTypeRegistry:
    def test_register_and_resolve_round_trip(self):
        register_output_type("ToyResult", _ToyResult)
        assert resolve_output_type("ToyResult") is _ToyResult

    def test_unknown_name_raises_keyerror_with_actionable_message(self):
        with pytest.raises(KeyError) as excinfo:
            resolve_output_type("DoesNotExist")
        assert "register_output_type" in str(excinfo.value)
        assert "DoesNotExist" in str(excinfo.value)

    def test_reset_clears(self):
        register_output_type("ToyResult", _ToyResult)
        reset_registry()
        with pytest.raises(KeyError):
            resolve_output_type("ToyResult")

    def test_re_registration_overwrites(self):
        class _Other(BaseModel):
            x: int = 0

        register_output_type("ToyResult", _ToyResult)
        register_output_type("ToyResult", _Other)
        assert resolve_output_type("ToyResult") is _Other


# ---------------------------------------------------------------------------
# llm_chat activity
# ---------------------------------------------------------------------------


class TestLLMChat:
    @pytest.mark.asyncio
    async def test_happy_path_returns_tool_input_and_telemetry(self):
        register_output_type("ToyResult", _ToyResult)
        set_provider(_mock_provider({"answer": "yes", "score": 7}))

        result = await llm_chat(
            LLMChatInput(
                system_prompt="sys", user_prompt="usr",
                output_type_name="ToyResult", model="anthropic:claude-x",
            ),
        )

        assert isinstance(result, LLMChatResult)
        assert result.tool_input == {"answer": "yes", "score": 7}
        assert result.model_name == "anthropic:test"
        assert result.input_tokens == 42
        assert result.output_tokens == 17
        assert result.cache_creation_input_tokens == 3
        assert result.cache_read_input_tokens == 5
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_workflow_can_validate_returned_tool_input(self):
        """Demonstrates the intended consumption pattern: workflow takes the
        raw tool_input dict and validates against its own Pydantic class."""
        register_output_type("ToyResult", _ToyResult)
        set_provider(_mock_provider({"answer": "ok", "score": 1}))

        result = await llm_chat(
            LLMChatInput(
                system_prompt="s", user_prompt="u",
                output_type_name="ToyResult", model="anthropic:m",
            ),
        )
        validated = _ToyResult.model_validate(result.tool_input)
        assert validated.answer == "ok"
        assert validated.score == 1

    @pytest.mark.asyncio
    async def test_unknown_output_type_raises_keyerror(self):
        set_provider(_mock_provider({}))
        with pytest.raises(KeyError):
            await llm_chat(
                LLMChatInput(
                    system_prompt="s", user_prompt="u",
                    output_type_name="NeverRegistered", model="anthropic:x",
                ),
            )

    @pytest.mark.asyncio
    async def test_empty_model_raises_value_error(self):
        register_output_type("ToyResult", _ToyResult)
        set_provider(_mock_provider({"answer": "ok"}))
        with pytest.raises(ValueError, match="empty model"):
            await llm_chat(
                LLMChatInput(
                    system_prompt="s", user_prompt="u",
                    output_type_name="ToyResult", model="",
                ),
            )

    @pytest.mark.asyncio
    async def test_max_tokens_is_plumbed_to_provider(self):
        register_output_type("ToyResult", _ToyResult)
        provider = _mock_provider({"answer": "ok"})
        set_provider(provider)

        await llm_chat(
            LLMChatInput(
                system_prompt="s", user_prompt="u",
                output_type_name="ToyResult", model="anthropic:m",
                max_tokens=512,
            ),
        )
        kwargs = provider.build_request_params.call_args.kwargs
        assert kwargs["max_tokens"] == 512
        # The `anthropic:` prefix is stripped before the SDK call.
        # provider SDKs (e.g., anthropic) expect the bare model name.
        assert kwargs["model"] == "m"
        assert kwargs["output_type"] is _ToyResult

    @pytest.mark.asyncio
    async def test_strips_provider_prefix_from_model(self):
        """Regression: resolve_model() returns 'anthropic:claude-...' but
        the Anthropic SDK rejects that string. llm_chat must pass only
        the bare model name to provider.build_request_params."""
        register_output_type("ToyResult", _ToyResult)
        provider = _mock_provider({"answer": "ok"})
        set_provider(provider)

        await llm_chat(
            LLMChatInput(
                system_prompt="s", user_prompt="u",
                output_type_name="ToyResult",
                model="anthropic:claude-haiku-4-5-20251001",
            ),
        )
        kwargs = provider.build_request_params.call_args.kwargs
        assert kwargs["model"] == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_bare_model_passes_through_unchanged(self):
        """When the caller already passed a bare model (no provider
        prefix), llm_chat should forward it unmodified."""
        register_output_type("ToyResult", _ToyResult)
        provider = _mock_provider({"answer": "ok"})
        set_provider(provider)

        await llm_chat(
            LLMChatInput(
                system_prompt="s", user_prompt="u",
                output_type_name="ToyResult",
                model="claude-haiku-4-5-20251001",
            ),
        )
        kwargs = provider.build_request_params.call_args.kwargs
        assert kwargs["model"] == "claude-haiku-4-5-20251001"
