"""Tests for forge.activities.batch_parse — batch parse activity."""

from __future__ import annotations

import json

import pytest

from forge.activities.batch_parse import execute_parse_llm_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_message_json(
    tool_name: str,
    tool_input: dict,
    *,
    model: str = "claude-sonnet-4-5-20250929",
    input_tokens: int = 100,
    output_tokens: int = 200,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> str:
    """Build a minimal valid Anthropic Message JSON string for testing."""
    message = {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_test123",
                "name": tool_name,
                "input": tool_input,
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
        },
    }
    return json.dumps(message)


# ---------------------------------------------------------------------------
# execute_parse_llm_response
# ---------------------------------------------------------------------------


class TestExecuteParseLLMResponse:
    def test_parses_llm_response(self) -> None:
        raw = _build_message_json(
            "llm_response",
            {"files": [], "edits": [], "explanation": "Done."},
        )

        result = execute_parse_llm_response(raw, "LLMResponse")

        from forge.models import LLMResponse

        parsed = LLMResponse.model_validate_json(result.parsed_json)
        assert parsed.explanation == "Done."
        assert result.model_name == "claude-sonnet-4-5-20250929"

    def test_parses_plan(self) -> None:
        raw = _build_message_json(
            "plan",
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
            },
        )

        result = execute_parse_llm_response(raw, "Plan")

        from forge.models import Plan

        parsed = Plan.model_validate_json(result.parsed_json)
        assert parsed.task_id == "t1"
        assert len(parsed.steps) == 1

    def test_returns_correct_usage_stats(self) -> None:
        raw = _build_message_json(
            "llm_response",
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

    def test_raises_key_error_for_unknown_type(self) -> None:
        with pytest.raises(KeyError, match="Unknown output type"):
            execute_parse_llm_response("{}", "NonExistentType")

    def test_raises_value_error_for_no_tool_use(self) -> None:
        raw_dict = json.loads(
            _build_message_json("llm_response", {"files": [], "edits": [], "explanation": "x"})
        )
        raw_dict["content"] = [{"type": "text", "text": "hello"}]
        raw = json.dumps(raw_dict)

        with pytest.raises(ValueError, match="No tool_use block found"):
            execute_parse_llm_response(raw, "LLMResponse")

    def test_latency_defaults_to_zero(self) -> None:
        raw = _build_message_json(
            "llm_response",
            {"files": [], "edits": [], "explanation": "x"},
        )

        result = execute_parse_llm_response(raw, "LLMResponse")

        assert result.latency_ms == 0.0
