"""Tests for forge.activities.batch_parse — batch parse activity."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from temporalio.exceptions import ApplicationError

from forge.activities.batch_parse import execute_parse_llm_response

# ---------------------------------------------------------------------------
# Helpers — real wire-shaped anthropic.types.Message JSON (see
# sax_platform tests' _message_json). Structured output rides in a *text*
# block (output_config.format), NOT a tool_use block.
# ---------------------------------------------------------------------------


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _message_json(
    *,
    content: list[dict[str, Any]],
    stop_reason: str = "end_turn",
    model: str = "claude-sonnet-5",
    input_tokens: int = 100,
    output_tokens: int = 200,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    extra: dict[str, Any] | None = None,
) -> str:
    """Build a minimal valid Anthropic Message JSON string for testing."""
    message: dict[str, Any] = {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
        },
    }
    if extra:
        message.update(extra)
    return json.dumps(message)


def _typed_message(payload: dict[str, Any], **kwargs: Any) -> str:
    """A Message whose single text block carries the structured-output JSON."""
    return _message_json(content=[_text_block(json.dumps(payload))], **kwargs)


# ---------------------------------------------------------------------------
# execute_parse_llm_response — success (typed + text lanes)
# ---------------------------------------------------------------------------


class TestExecuteParseLLMResponse:
    def test_parses_llm_response(self) -> None:
        raw = _typed_message({"files": [], "edits": [], "explanation": "Done."})

        result = execute_parse_llm_response(raw, "LLMResponse")

        from forge.models import LLMResponse

        parsed = LLMResponse.model_validate_json(result.parsed_json)
        assert parsed.explanation == "Done."
        assert result.model_name == "claude-sonnet-5"
        assert result.stop_reason == "end_turn"

    def test_parses_plan(self) -> None:
        raw = _typed_message(
            {
                "task_id": "t1",
                "steps": [
                    {
                        "step_id": "s1",
                        "description": "Do it.",
                        "target_files": ["a.py"],
                    }
                ],
                "explanation": "Single step.",
            }
        )

        result = execute_parse_llm_response(raw, "Plan")

        from forge.models import Plan

        parsed = Plan.model_validate_json(result.parsed_json)
        assert parsed.task_id == "t1"
        assert len(parsed.steps) == 1

    def test_returns_correct_usage_stats(self) -> None:
        raw = _typed_message(
            {"files": [], "edits": [], "explanation": "x"},
            input_tokens=500,
            output_tokens=300,
            cache_creation_input_tokens=50,
            cache_read_input_tokens=75,
        )

        result = execute_parse_llm_response(raw, "LLMResponse")

        assert result.input_tokens == 500
        assert result.output_tokens == 300
        assert result.cache_creation_input_tokens == 50
        assert result.cache_read_input_tokens == 75

    def test_text_lane_serializes_text(self) -> None:
        raw = _message_json(content=[_text_block("plain answer")], stop_reason="end_turn")

        result = execute_parse_llm_response(raw, None)

        assert result.parsed_json == json.dumps("plain answer")
        assert result.stop_reason == "end_turn"
        assert result.model_name == "claude-sonnet-5"

    def test_latency_defaults_to_zero(self) -> None:
        raw = _typed_message({"files": [], "edits": [], "explanation": "x"})

        result = execute_parse_llm_response(raw, "LLMResponse")

        assert result.latency_ms == 0.0

    def test_records_stop_reason(self) -> None:
        raw = _typed_message(
            {"files": [], "edits": [], "explanation": "x"}, stop_reason="stop_sequence"
        )

        result = execute_parse_llm_response(raw, "LLMResponse")

        assert result.stop_reason == "stop_sequence"

    def test_raises_key_error_for_unknown_type(self) -> None:
        raw = _typed_message({"files": [], "edits": [], "explanation": "x"})

        with pytest.raises(KeyError, match="Unknown output type"):
            execute_parse_llm_response(raw, "NonExistentType")


# ---------------------------------------------------------------------------
# execute_parse_llm_response — failure outcomes raise non-retryable errors
#
# The stored bytes are deterministic, so a refusal / truncation / mismatch can
# never resolve differently on a retry — each raises a non_retryable
# ApplicationError with a stable `type` so the workflow fails fast without
# burning attempts.
# ---------------------------------------------------------------------------


class TestParseFailureOutcomes:
    def test_refusal_raises_non_retryable(self) -> None:
        raw = _message_json(
            content=[],
            stop_reason="refusal",
            extra={"stop_details": {"type": "refusal", "category": "cyber"}},
        )

        with pytest.raises(ApplicationError) as exc_info:
            execute_parse_llm_response(raw, "LLMResponse")

        err = exc_info.value
        assert err.type == "LLMRefused"
        assert err.non_retryable is True
        assert "refusal" in str(err)

    def test_truncation_raises_non_retryable(self) -> None:
        raw = _message_json(content=[_text_block("partial")], stop_reason="max_tokens")

        with pytest.raises(ApplicationError) as exc_info:
            execute_parse_llm_response(raw, "LLMResponse")

        err = exc_info.value
        assert err.type == "LLMTruncated"
        assert err.non_retryable is True
        assert "max_tokens" in str(err)

    def test_schema_mismatch_raises_non_retryable(self) -> None:
        # stop_reason is a normal terminal one, but the text is not valid JSON for
        # the expected schema -> MismatchOutcome -> LLMSchemaMismatch.
        raw = _message_json(content=[_text_block("not json at all")], stop_reason="end_turn")

        with pytest.raises(ApplicationError) as exc_info:
            execute_parse_llm_response(raw, "LLMResponse")

        err = exc_info.value
        assert err.type == "LLMSchemaMismatch"
        assert err.non_retryable is True


# ---------------------------------------------------------------------------
# parse_llm_response activity wrapper
# ---------------------------------------------------------------------------


class TestParseLlmResponseActivity:
    @pytest.mark.asyncio
    async def test_activity_delegates_and_records_model(self) -> None:
        from forge.activities.roots import BatchActivities
        from forge.models import ParseResponseInput
        from forge.output_types import OUTPUT_TYPES

        raw = _typed_message({"files": [], "edits": [], "explanation": "done"})

        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        batch = BatchActivities(
            client=MagicMock(),
            output_types=OUTPUT_TYPES,
            engine=MagicMock(),
            blob_store=None,
            mistral_ocr=None,
        )
        with patch("forge.activities.roots.get_tracer", return_value=mock_tracer):
            result = await batch.parse_llm_response(
                ParseResponseInput(
                    raw_response_json=raw,
                    output_type_name="LLMResponse",
                    task_id="t-ok",
                )
            )

        assert result.model_name == "claude-sonnet-5"
        assert result.stop_reason == "end_turn"
