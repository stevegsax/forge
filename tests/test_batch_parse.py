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
        raw = _typed_message(
            {
                "files": [{"file_path": "a.py", "content": "pass"}],
                "edits": [],
                "explanation": "Done.",
            }
        )

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
            {"files": [{"file_path": "a.py", "content": "pass"}], "edits": [], "explanation": "x"},
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
        raw = _typed_message(
            {"files": [{"file_path": "a.py", "content": "pass"}], "edits": [], "explanation": "x"}
        )

        result = execute_parse_llm_response(raw, "LLMResponse")

        assert result.latency_ms == 0.0

    def test_records_stop_reason(self) -> None:
        raw = _typed_message(
            {"files": [{"file_path": "a.py", "content": "pass"}], "edits": [], "explanation": "x"},
            stop_reason="stop_sequence",
        )

        result = execute_parse_llm_response(raw, "LLMResponse")

        assert result.stop_reason == "stop_sequence"

    def test_raises_key_error_for_unknown_type(self) -> None:
        raw = _typed_message(
            {"files": [{"file_path": "a.py", "content": "pass"}], "edits": [], "explanation": "x"}
        )

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

    def test_do_nothing_response_is_a_schema_mismatch(self) -> None:
        """T5.6: an LLMResponse with no files and no edits fails at the parse seam.

        Before the model validator this parsed cleanly and the pipeline wrote
        nothing, validated zero files, and reported SUCCESS. Now it is a
        classification failure like any other off-schema completion.
        """
        raw = _typed_message({"files": [], "edits": [], "explanation": "Nothing to do."})

        with pytest.raises(ApplicationError) as exc_info:
            execute_parse_llm_response(raw, "LLMResponse")

        assert exc_info.value.type == "LLMSchemaMismatch"
        assert "produced no output" in str(exc_info.value)

    def test_schema_mismatch_stays_retryable_in_the_llm_preset(self) -> None:
        """The error-aware path: a mismatch is a sampling accident, so the
        activity retry gets a differently-sampled call. (The batch lane's own
        raise is non-retryable because re-parsing the same stored bytes cannot
        change the outcome — this pins the *policy*, not the raise.)"""
        from sax_platform.temporal.retries import LLM_RETRY

        assert LLM_RETRY.non_retryable_error_types is not None
        assert "LLMSchemaMismatch" not in LLM_RETRY.non_retryable_error_types


# ---------------------------------------------------------------------------
# parse_llm_response activity wrapper
# ---------------------------------------------------------------------------


class _StubBlobs:
    """Minimal S3Blobs stand-in: returns a canned envelope for any key."""

    def __init__(self, envelope: bytes) -> None:
        self._envelope = envelope
        self.requested: list[str] = []

    def get(self, key: str) -> bytes:
        self.requested.append(key)
        return self._envelope


def _silent_tracer() -> MagicMock:
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)
    tracer = MagicMock()
    tracer.start_as_current_span.return_value = span
    return tracer


class TestParseLlmResponseActivity:
    @pytest.mark.asyncio
    async def test_activity_delegates_and_records_model(self) -> None:
        from forge.activities.roots import BatchActivities
        from forge.models import ParseResponseInput
        from forge.output_types import OUTPUT_TYPES

        raw = _typed_message(
            {
                "files": [{"file_path": "a.py", "content": "pass"}],
                "edits": [],
                "explanation": "done",
            }
        )

        batch = BatchActivities(
            client=MagicMock(),
            output_types=OUTPUT_TYPES,
            engine=MagicMock(),
            blob_store=None,
        )
        with patch("forge.activities.roots.get_tracer", return_value=_silent_tracer()):
            result = await batch.parse_llm_response(
                ParseResponseInput(
                    raw_response_json=raw,
                    output_type_name="LLMResponse",
                    task_id="t-ok",
                )
            )

        assert result.model_name == "claude-sonnet-5"
        assert result.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_activity_resolves_s3_pointer_via_blob_store(self) -> None:
        """A pointer-delivered result is fetched from blobs and unwrapped directly.

        The parse activity reads the envelope via ``S3Blobs.get`` +
        ``parse_batch_result_payload`` (T4.2 ST1); the old signal-era envelope
        model and its resolver were deleted in ST3.
        """
        from sax_platform.contracts.models import dump_batch_result_payload

        from forge.activities.roots import BatchActivities
        from forge.models import ParseResponseInput
        from forge.output_types import OUTPUT_TYPES

        raw = _typed_message(
            {
                "files": [{"file_path": "a.py", "content": "pass"}],
                "edits": [],
                "explanation": "from s3",
            }
        )
        envelope = dump_batch_result_payload(raw, []).encode("utf-8")
        blobs = _StubBlobs(envelope)

        batch = BatchActivities(
            client=MagicMock(),
            output_types=OUTPUT_TYPES,
            engine=MagicMock(),
            blob_store=blobs,  # type: ignore[arg-type]
        )
        with patch("forge.activities.roots.get_tracer", return_value=_silent_tracer()):
            result = await batch.parse_llm_response(
                ParseResponseInput(
                    raw_response_json=None,
                    s3_key="result-blob-key",
                    output_type_name="LLMResponse",
                    task_id="t-s3",
                )
            )

        assert blobs.requested == ["result-blob-key"]

        from forge.models import LLMResponse

        parsed = LLMResponse.model_validate_json(result.parsed_json)
        assert parsed.explanation == "from s3"

    @pytest.mark.asyncio
    async def test_activity_raises_when_s3_pointer_but_no_blob_store(self) -> None:
        from forge.activities.roots import BatchActivities
        from forge.models import ParseResponseInput
        from forge.output_types import OUTPUT_TYPES

        batch = BatchActivities(
            client=MagicMock(),
            output_types=OUTPUT_TYPES,
            engine=MagicMock(),
            blob_store=None,
        )
        with (
            patch("forge.activities.roots.get_tracer", return_value=_silent_tracer()),
            pytest.raises(RuntimeError, match="S3 blob store not configured"),
        ):
            await batch.parse_llm_response(
                ParseResponseInput(
                    raw_response_json=None,
                    s3_key="result-blob-key",
                    output_type_name="LLMResponse",
                    task_id="t-missing-blobs",
                )
            )
