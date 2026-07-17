"""Tests for sax_platform.llm.models — the pure classification core."""

import pytest
from anthropic.types import Message, TextBlock, Usage
from pydantic import ValidationError

from sax_platform.llm.models import (
    ClassifiedMessage,
    Completion,
    LLMOutcomeError,
    LLMRefused,
    LLMSchemaMismatch,
    LLMTruncated,
    MismatchOutcome,
    RefusedOutcome,
    Telemetry,
    TruncatedOutcome,
    classify_message,
    outcome_from_error,
    telemetry_from_message,
)


class _StopDetails:
    """Stand-in for the SDK's `stop_details` object — used because the
    installed anthropic SDK version doesn't model this field, but
    `Message.model_config` has `extra="allow"`, so it can be attached at
    construction time exactly as a live response would carry it."""

    def __init__(self, category: str | None) -> None:
        self.category = category


def _make_message(
    *,
    stop_reason: str | None = "end_turn",
    text: str | None = "hello",
    input_tokens: int = 100,
    output_tokens: int = 20,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
    request_id: str | None = None,
    stop_details: _StopDetails | None = None,
    content: list[TextBlock] | None = None,
) -> Message:
    if content is None:
        content = [TextBlock(type="text", text=text)] if text is not None else []
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )
    extra: dict[str, object] = {}
    if stop_details is not None:
        extra["stop_details"] = stop_details
    message = Message(
        id="msg_test",
        content=content,
        model="claude-opus-4-8",
        role="assistant",
        stop_reason=stop_reason,  # type: ignore[arg-type]
        stop_sequence=None,
        type="message",
        usage=usage,
        **extra,
    )
    if request_id is not None:
        object.__setattr__(message, "_request_id", request_id)
    return message


class TestTelemetryFromMessage:
    def test_reads_usage_model_and_stop_reason(self) -> None:
        message = _make_message(
            stop_reason="end_turn",
            input_tokens=42,
            output_tokens=7,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=5,
        )

        telemetry = telemetry_from_message(message)

        assert telemetry == Telemetry(
            model="claude-opus-4-8",
            stop_reason="end_turn",
            input_tokens=42,
            output_tokens=7,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=5,
            request_id=None,
        )

    def test_missing_cache_fields_default_to_zero(self) -> None:
        message = _make_message(cache_creation_input_tokens=None, cache_read_input_tokens=None)

        telemetry = telemetry_from_message(message)

        assert telemetry.cache_creation_input_tokens == 0
        assert telemetry.cache_read_input_tokens == 0

    def test_request_id_absent_defaults_to_none(self) -> None:
        message = _make_message()

        telemetry = telemetry_from_message(message)

        assert telemetry.request_id is None

    def test_request_id_present_is_read(self) -> None:
        message = _make_message(request_id="req_abc123")

        telemetry = telemetry_from_message(message)

        assert telemetry.request_id == "req_abc123"

    def test_missing_stop_reason_becomes_empty_string(self) -> None:
        message = _make_message(stop_reason=None)

        telemetry = telemetry_from_message(message)

        assert telemetry.stop_reason == ""


class TestClassifyMessageRefusal:
    def test_raises_llm_refused_on_refusal_stop_reason(self) -> None:
        message = _make_message(stop_reason="refusal", text=None)

        with pytest.raises(LLMRefused) as exc_info:
            classify_message(message, max_tokens=1024)

        assert exc_info.value.category is None
        assert exc_info.value.telemetry.stop_reason == "refusal"

    def test_reads_category_from_stop_details_when_present(self) -> None:
        message = _make_message(
            stop_reason="refusal",
            text=None,
            stop_details=_StopDetails(category="cyber"),
        )

        with pytest.raises(LLMRefused) as exc_info:
            classify_message(message, max_tokens=1024)

        assert exc_info.value.category == "cyber"

    def test_category_none_when_stop_details_present_but_category_none(self) -> None:
        message = _make_message(
            stop_reason="refusal",
            text=None,
            stop_details=_StopDetails(category=None),
        )

        with pytest.raises(LLMRefused) as exc_info:
            classify_message(message, max_tokens=1024)

        assert exc_info.value.category is None

    def test_telemetry_populated_from_message(self) -> None:
        message = _make_message(
            stop_reason="refusal",
            text=None,
            input_tokens=55,
            output_tokens=0,
            request_id="req_refused",
        )

        with pytest.raises(LLMRefused) as exc_info:
            classify_message(message, max_tokens=1024)

        telemetry = exc_info.value.telemetry
        assert telemetry.input_tokens == 55
        assert telemetry.output_tokens == 0
        assert telemetry.request_id == "req_refused"
        assert telemetry.model == "claude-opus-4-8"

    def test_str_is_informative(self) -> None:
        message = _make_message(
            stop_reason="refusal", text=None, stop_details=_StopDetails(category="bio")
        )

        with pytest.raises(LLMRefused) as exc_info:
            classify_message(message, max_tokens=1024)

        assert "refus" in str(exc_info.value).lower()
        assert "bio" in str(exc_info.value)

    def test_is_instance_of_llm_outcome_error(self) -> None:
        message = _make_message(stop_reason="refusal", text=None)

        with pytest.raises(LLMOutcomeError):
            classify_message(message, max_tokens=1024)


class TestClassifyMessageTruncation:
    def test_raises_llm_truncated_on_max_tokens_stop_reason(self) -> None:
        message = _make_message(stop_reason="max_tokens", text="partial output here")

        with pytest.raises(LLMTruncated) as exc_info:
            classify_message(message, max_tokens=256)

        assert exc_info.value.partial_text == "partial output here"
        assert exc_info.value.max_tokens == 256

    def test_partial_text_empty_when_no_text_block(self) -> None:
        message = _make_message(stop_reason="max_tokens", text=None)

        with pytest.raises(LLMTruncated) as exc_info:
            classify_message(message, max_tokens=256)

        assert exc_info.value.partial_text == ""

    def test_telemetry_populated(self) -> None:
        message = _make_message(
            stop_reason="max_tokens", text="x", input_tokens=10, output_tokens=256
        )

        with pytest.raises(LLMTruncated) as exc_info:
            classify_message(message, max_tokens=256)

        assert exc_info.value.telemetry.output_tokens == 256

    def test_str_is_informative(self) -> None:
        message = _make_message(stop_reason="max_tokens", text="partial")

        with pytest.raises(LLMTruncated) as exc_info:
            classify_message(message, max_tokens=99)

        assert "99" in str(exc_info.value)

    def test_is_instance_of_llm_outcome_error(self) -> None:
        message = _make_message(stop_reason="max_tokens", text="x")

        with pytest.raises(LLMOutcomeError):
            classify_message(message, max_tokens=1)


class TestClassifyMessageSuccess:
    @pytest.mark.parametrize("stop_reason", ["end_turn", "tool_use", "stop_sequence", "pause_turn"])
    def test_returns_classified_message_for_non_terminal_stop_reasons(
        self, stop_reason: str
    ) -> None:
        message = _make_message(stop_reason=stop_reason, text="the answer")

        classified = classify_message(message, max_tokens=1024)

        assert classified == ClassifiedMessage(
            text="the answer",
            stop_reason=stop_reason,
            telemetry=telemetry_from_message(message),
        )

    def test_text_is_empty_string_when_no_text_block(self) -> None:
        message = _make_message(stop_reason="tool_use", text=None)

        classified = classify_message(message, max_tokens=1024)

        assert classified.text == ""

    def test_returns_first_text_block_when_multiple_blocks(self) -> None:
        message = _make_message(
            stop_reason="end_turn",
            content=[
                TextBlock(type="text", text="first"),
                TextBlock(type="text", text="second"),
            ],
        )

        classified = classify_message(message, max_tokens=1024)

        assert classified.text == "first"

    def test_does_not_parse_or_validate_json(self) -> None:
        # Text that is not valid JSON is returned as-is — classify_message
        # has no opinion on shape, only on stop_reason.
        message = _make_message(stop_reason="end_turn", text="not json at all {")

        classified = classify_message(message, max_tokens=1024)

        assert classified.text == "not json at all {"


class TestOutcomeFromError:
    def test_refused_maps_to_refused_outcome(self) -> None:
        telemetry = Telemetry(
            model="claude-opus-4-8",
            stop_reason="refusal",
            input_tokens=1,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            request_id=None,
        )
        err = LLMRefused(category="cyber", telemetry=telemetry)

        outcome = outcome_from_error(err)

        assert outcome == RefusedOutcome(category="cyber", telemetry=telemetry)

    def test_truncated_maps_to_truncated_outcome(self) -> None:
        telemetry = Telemetry(
            model="claude-opus-4-8",
            stop_reason="max_tokens",
            input_tokens=1,
            output_tokens=99,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            request_id=None,
        )
        err = LLMTruncated(partial_text="partial", max_tokens=99, telemetry=telemetry)

        outcome = outcome_from_error(err)

        assert outcome == TruncatedOutcome(
            partial_text="partial", max_tokens=99, telemetry=telemetry
        )

    def test_mismatch_maps_to_mismatch_outcome(self) -> None:
        telemetry = Telemetry(
            model="claude-opus-4-8",
            stop_reason="end_turn",
            input_tokens=1,
            output_tokens=1,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            request_id=None,
        )
        err = LLMSchemaMismatch(raw_text="not json", error="invalid JSON", telemetry=telemetry)

        outcome = outcome_from_error(err)

        assert outcome == MismatchOutcome(
            raw_text="not json", error="invalid JSON", telemetry=telemetry
        )

    def test_unrecognized_subclass_raises_type_error(self) -> None:
        telemetry = Telemetry(
            model="claude-opus-4-8",
            stop_reason="end_turn",
            input_tokens=1,
            output_tokens=1,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            request_id=None,
        )

        class _NotARealOutcomeError(LLMOutcomeError):
            pass

        err = _NotARealOutcomeError("unexpected", telemetry=telemetry)

        with pytest.raises(TypeError, match="_NotARealOutcomeError"):
            outcome_from_error(err)


class TestLLMSchemaMismatch:
    def test_str_includes_error_detail(self) -> None:
        telemetry = Telemetry(
            model="claude-opus-4-8",
            stop_reason="end_turn",
            input_tokens=1,
            output_tokens=1,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            request_id=None,
        )
        err = LLMSchemaMismatch(raw_text="oops", error="missing field 'name'", telemetry=telemetry)

        assert "missing field 'name'" in str(err)
        assert isinstance(err, LLMOutcomeError)


class TestCompletionAndTelemetryAreFrozen:
    def test_completion_is_immutable(self) -> None:
        completion = Completion[str](
            output="done",
            model="claude-opus-4-8",
            stop_reason="end_turn",
            input_tokens=1,
            output_tokens=1,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            request_id=None,
        )

        with pytest.raises(ValidationError):
            completion.output = "changed"  # type: ignore[misc]

    def test_telemetry_is_immutable(self) -> None:
        telemetry = Telemetry(
            model="claude-opus-4-8",
            stop_reason="end_turn",
            input_tokens=1,
            output_tokens=1,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            request_id=None,
        )

        with pytest.raises(ValidationError):
            telemetry.model = "changed"  # type: ignore[misc]

    def test_completion_generic_over_output_type(self) -> None:
        completion = Completion[int](
            output=42,
            model="claude-opus-4-8",
            stop_reason="end_turn",
            input_tokens=1,
            output_tokens=1,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            request_id=None,
        )

        assert completion.output == 42
