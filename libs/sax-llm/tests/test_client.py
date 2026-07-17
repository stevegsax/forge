"""Tests for sax_llm.client — request building and utility functions."""

from __future__ import annotations

from pydantic import BaseModel

from sax_llm.client import (
    build_batch_request,
    build_system_param,
    build_thinking_param,
    build_tool_definition,
)


class TestBuildToolDefinition:
    def test_basic(self):
        class MyTool(BaseModel):
            """A test tool."""

            value: str

        tool = build_tool_definition(MyTool)
        assert tool["name"] == "my_tool"
        assert tool["description"] == "A test tool."
        assert "input_schema" in tool
        assert "cache_control" in tool

    def test_no_cache(self):
        class Simple(BaseModel):
            x: int

        tool = build_tool_definition(Simple, cache_control=False)
        assert "cache_control" not in tool


class TestBuildSystemParam:
    def test_with_cache(self):
        result = build_system_param("hello")
        assert isinstance(result, list)
        assert result[0]["text"] == "hello"
        assert "cache_control" in result[0]

    def test_without_cache(self):
        result = build_system_param("hello", cache_control=False)
        assert result == "hello"


class TestBuildThinkingParam:
    def test_opus_enabled_is_adaptive(self):
        result = build_thinking_param("claude-opus-4-6", enabled=True)
        assert result == {"type": "adaptive"}

    def test_sonnet_enabled_is_adaptive(self):
        result = build_thinking_param("claude-sonnet-4-5", enabled=True)
        assert result == {"type": "adaptive"}

    def test_haiku_returns_none(self):
        assert build_thinking_param("claude-haiku-4-5", enabled=True) is None

    def test_haiku_returns_none_when_disabled(self):
        assert build_thinking_param("claude-haiku-4-5", enabled=False) is None

    def test_disabled_is_explicit_shape(self):
        """Omitting "thinking" runs adaptive BY DEFAULT — "off" must be explicit."""
        assert build_thinking_param("claude-opus-4-6", enabled=False) == {"type": "disabled"}

    def test_non_anthropic(self):
        assert build_thinking_param("mistral-large", enabled=True) is None


class TestBuildBatchRequest:
    def test_wraps_params(self):
        result = build_batch_request("req-1", {"model": "claude", "max_tokens": 100})
        assert result["custom_id"] == "req-1"
        assert result["params"]["model"] == "claude"
