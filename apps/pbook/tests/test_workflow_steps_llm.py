"""Tests for the generic llm_chat activity and its output-type mapping."""

from __future__ import annotations

from typing import Any

import pytest
from sax_platform.llm import (
    Completion,
    LLMRefused,
    LLMSchemaMismatch,
    LLMTruncated,
    Telemetry,
)
from temporalio.exceptions import ApplicationError

from pbook.llm import (
    ConsolidationResult,
    ExtractionResult,
    ReviewResult,
    reset_provider,
    set_provider,
)
from pbook.workflow_steps import (
    OUTPUT_TYPES,
    LLMChatInput,
    LLMChatResult,
    llm_chat,
    resolve_output_type,
)


def _telemetry(*, stop_reason: str) -> Telemetry:
    return Telemetry(
        model="claude-x",
        stop_reason=stop_reason,
        input_tokens=1,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        request_id=None,
    )


def _completion(output: Any) -> Completion[Any]:
    """A canned successful Completion carrying distinctive telemetry."""
    return Completion(
        output=output,
        model="anthropic-claude-x",
        stop_reason="end_turn",
        input_tokens=42,
        output_tokens=17,
        cache_creation_input_tokens=3,
        cache_read_input_tokens=5,
        request_id="req-1",
    )


class _StubProvider:
    """A ``SupportsComplete`` stub returning a canned Completion or raising.

    Records the kwargs of each ``complete`` call so tests can assert what
    ``llm_chat`` forwarded (bare model, system prompt, max_tokens, output_type).
    """

    def __init__(
        self,
        *,
        completion: Completion[Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._completion = completion
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: Any,
        *,
        output_type: Any,
        model: str,
        max_tokens: int,
        system: Any = None,
        cache: Any = None,
        thinking: Any = None,
    ) -> Completion[Any]:
        self.calls.append(
            {
                "messages": list(messages),
                "output_type": output_type,
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
            }
        )
        if self._error is not None:
            raise self._error
        assert self._completion is not None
        return self._completion


@pytest.fixture(autouse=True)
def _isolate_provider():
    """Ensure a clean provider between tests."""
    reset_provider()
    yield
    reset_provider()


# ---------------------------------------------------------------------------
# Output-type frozen mapping
# ---------------------------------------------------------------------------


class TestOutputTypeMapping:
    def test_resolves_the_three_known_types(self):
        assert resolve_output_type("ExtractionResult") is ExtractionResult
        assert resolve_output_type("ReviewResult") is ReviewResult
        assert resolve_output_type("ConsolidationResult") is ConsolidationResult

    def test_mapping_holds_exactly_the_known_types(self):
        assert set(OUTPUT_TYPES) == {
            "ExtractionResult",
            "ReviewResult",
            "ConsolidationResult",
        }

    def test_unknown_name_raises_keyerror_with_actionable_message(self):
        with pytest.raises(KeyError) as excinfo:
            resolve_output_type("DoesNotExist")
        message = str(excinfo.value)
        assert "DoesNotExist" in message
        # The message lists the known types so the fix is obvious.
        assert "ExtractionResult" in message

    def test_mapping_is_immutable(self):
        with pytest.raises(TypeError):
            OUTPUT_TYPES["New"] = ReviewResult  # type: ignore[index]


# ---------------------------------------------------------------------------
# llm_chat activity — happy path + telemetry
# ---------------------------------------------------------------------------


class TestLLMChat:
    @pytest.mark.asyncio
    async def test_happy_path_returns_tool_input_and_telemetry(self):
        review = ReviewResult(approved=True, suggested_title="Better title")
        set_provider(_StubProvider(completion=_completion(review)))

        result = await llm_chat(
            LLMChatInput(
                system_prompt="sys",
                user_prompt="usr",
                output_type_name="ReviewResult",
                model="anthropic:claude-x",
            ),
        )

        assert isinstance(result, LLMChatResult)
        # tool_input is exactly the structured output serialized.
        assert result.tool_input == review.model_dump()
        assert result.model_name == "anthropic-claude-x"
        assert result.input_tokens == 42
        assert result.output_tokens == 17
        assert result.cache_creation_input_tokens == 3
        assert result.cache_read_input_tokens == 5
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_workflow_can_validate_returned_tool_input(self):
        """The intended consumption pattern: the workflow takes the raw
        tool_input dict and validates it against its own Pydantic class."""
        review = ReviewResult(approved=True, rejection_reason="", suggested_title="ok")
        set_provider(_StubProvider(completion=_completion(review)))

        result = await llm_chat(
            LLMChatInput(
                system_prompt="s",
                user_prompt="u",
                output_type_name="ReviewResult",
                model="anthropic:m",
            ),
        )
        validated = ReviewResult.model_validate(result.tool_input)
        assert validated.approved is True
        assert validated.suggested_title == "ok"

    @pytest.mark.asyncio
    async def test_unknown_output_type_raises_keyerror(self):
        set_provider(_StubProvider(completion=_completion(ReviewResult())))
        with pytest.raises(KeyError):
            await llm_chat(
                LLMChatInput(
                    system_prompt="s",
                    user_prompt="u",
                    output_type_name="NeverRegistered",
                    model="anthropic:x",
                ),
            )

    @pytest.mark.asyncio
    async def test_empty_model_raises_value_error(self):
        set_provider(_StubProvider(completion=_completion(ReviewResult())))
        with pytest.raises(ValueError, match="empty model"):
            await llm_chat(
                LLMChatInput(
                    system_prompt="s",
                    user_prompt="u",
                    output_type_name="ReviewResult",
                    model="",
                ),
            )

    @pytest.mark.asyncio
    async def test_forwards_prompt_output_type_and_max_tokens_to_provider(self):
        provider = _StubProvider(completion=_completion(ReviewResult()))
        set_provider(provider)

        await llm_chat(
            LLMChatInput(
                system_prompt="the system prompt",
                user_prompt="the user prompt",
                output_type_name="ReviewResult",
                model="anthropic:m",
                max_tokens=512,
            ),
        )
        call = provider.calls[-1]
        assert call["max_tokens"] == 512
        assert call["output_type"] is ReviewResult
        assert call["system"] == "the system prompt"
        assert call["messages"] == [{"role": "user", "content": "the user prompt"}]

    @pytest.mark.asyncio
    async def test_strips_provider_prefix_from_model(self):
        """resolve_model() returns 'anthropic:claude-...' but the client
        expects the bare model name; llm_chat must strip the prefix."""
        provider = _StubProvider(completion=_completion(ReviewResult()))
        set_provider(provider)

        await llm_chat(
            LLMChatInput(
                system_prompt="s",
                user_prompt="u",
                output_type_name="ReviewResult",
                model="anthropic:claude-haiku-4-5",
            ),
        )
        assert provider.calls[-1]["model"] == "claude-haiku-4-5"

    @pytest.mark.asyncio
    async def test_bare_model_passes_through_unchanged(self):
        provider = _StubProvider(completion=_completion(ReviewResult()))
        set_provider(provider)

        await llm_chat(
            LLMChatInput(
                system_prompt="s",
                user_prompt="u",
                output_type_name="ReviewResult",
                model="claude-haiku-4-5",
            ),
        )
        assert provider.calls[-1]["model"] == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# llm_chat activity — typed failure handling
# ---------------------------------------------------------------------------


class TestLLMChatTypedFailures:
    """Refusal and truncation are terminal for the request and must surface
    as non-retryable ApplicationErrors; a schema mismatch may clear on retry
    and must propagate unchanged (retryable)."""

    @pytest.mark.asyncio
    async def test_refusal_maps_to_non_retryable_application_error(self):
        set_provider(
            _StubProvider(
                error=LLMRefused(
                    category="policy",
                    telemetry=_telemetry(stop_reason="refusal"),
                ),
            ),
        )

        with pytest.raises(ApplicationError) as excinfo:
            await llm_chat(
                LLMChatInput(
                    system_prompt="s",
                    user_prompt="u",
                    output_type_name="ReviewResult",
                    model="anthropic:m",
                ),
            )

        assert excinfo.value.non_retryable is True
        assert excinfo.value.type == "LLMRefused"
        assert isinstance(excinfo.value.__cause__, LLMRefused)
        # The message surfaces the stop_reason and refusal category.
        assert "refusal" in str(excinfo.value)
        assert "policy" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_truncation_maps_to_non_retryable_application_error(self):
        set_provider(
            _StubProvider(
                error=LLMTruncated(
                    partial_text="half a resu",
                    max_tokens=8,
                    telemetry=_telemetry(stop_reason="max_tokens"),
                ),
            ),
        )

        with pytest.raises(ApplicationError) as excinfo:
            await llm_chat(
                LLMChatInput(
                    system_prompt="s",
                    user_prompt="u",
                    output_type_name="ReviewResult",
                    model="anthropic:m",
                ),
            )

        assert excinfo.value.non_retryable is True
        assert excinfo.value.type == "LLMTruncated"
        assert isinstance(excinfo.value.__cause__, LLMTruncated)
        assert "max_tokens" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_schema_mismatch_propagates_retryable(self):
        """LLMSchemaMismatch is not caught: it propagates unwrapped so the
        activity's retry policy still governs it."""
        set_provider(
            _StubProvider(
                error=LLMSchemaMismatch(
                    raw_text="not json",
                    error="invalid",
                    telemetry=_telemetry(stop_reason="end_turn"),
                ),
            ),
        )

        with pytest.raises(LLMSchemaMismatch):
            await llm_chat(
                LLMChatInput(
                    system_prompt="s",
                    user_prompt="u",
                    output_type_name="ReviewResult",
                    model="anthropic:m",
                ),
            )


class TestLLMChatRetryClassification:
    """An auth/config failure must surface as a non-retryable
    ApplicationError so the bounded retry policy doesn't waste attempts;
    every other provider error propagates unchanged (and stays retryable)."""

    @pytest.mark.asyncio
    async def test_auth_error_raises_non_retryable_application_error(self):
        set_provider(
            _StubProvider(
                error=TypeError(
                    "Could not resolve authentication method. Expected "
                    "either api_key or auth_token to be set."
                ),
            ),
        )

        with pytest.raises(ApplicationError) as excinfo:
            await llm_chat(
                LLMChatInput(
                    system_prompt="s",
                    user_prompt="u",
                    output_type_name="ReviewResult",
                    model="anthropic:m",
                ),
            )

        assert excinfo.value.non_retryable is True
        assert excinfo.value.type == "TypeError"
        assert isinstance(excinfo.value.__cause__, TypeError)

    @pytest.mark.asyncio
    async def test_transient_error_propagates_unwrapped(self):
        set_provider(_StubProvider(error=ConnectionError("connection reset")))

        with pytest.raises(ConnectionError, match="connection reset"):
            await llm_chat(
                LLMChatInput(
                    system_prompt="s",
                    user_prompt="u",
                    output_type_name="ReviewResult",
                    model="anthropic:m",
                ),
            )
