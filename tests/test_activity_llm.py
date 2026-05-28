"""Tests for forge.activities.llm — LLM call activity."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from forge.activities.llm import execute_llm_call
from forge.models import AssembledContext, FileOutput, LLMResponse
from tests.conftest import build_mock_provider

# ---------------------------------------------------------------------------
# execute_llm_call
# ---------------------------------------------------------------------------


class TestExecuteLlmCall:
    def _make_context(self) -> AssembledContext:
        return AssembledContext(
            task_id="llm-task",
            system_prompt="You are a code generator.",
            user_prompt="Generate the code.",
        )

    def _make_provider(
        self,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> MagicMock:
        return build_mock_provider(
            tool_input=LLMResponse(
                files=[FileOutput(file_path="out.py", content="print('hi')")],
                explanation="Created output file.",
            ).model_dump(),
            model_name="claude-sonnet-4-5-20250929",
            input_tokens=100,
            output_tokens=200,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )

    @pytest.mark.asyncio
    async def test_returns_llm_call_result(self) -> None:
        context = self._make_context()
        provider = self._make_provider()

        result = await execute_llm_call(context, provider)

        assert result.task_id == "llm-task"
        assert result.model_name == "claude-sonnet-4-5-20250929"
        assert len(result.response.files) == 1
        assert result.response.files[0].file_path == "out.py"

    @pytest.mark.asyncio
    async def test_extracts_usage(self) -> None:
        context = self._make_context()
        provider = self._make_provider()

        result = await execute_llm_call(context, provider)

        assert result.input_tokens == 100
        assert result.output_tokens == 200

    @pytest.mark.asyncio
    async def test_latency_is_positive(self) -> None:
        context = self._make_context()
        provider = self._make_provider()

        result = await execute_llm_call(context, provider)

        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_propagates_task_id(self) -> None:
        context = AssembledContext(
            task_id="custom-id",
            system_prompt="sys",
            user_prompt="usr",
        )
        provider = self._make_provider()

        result = await execute_llm_call(context, provider)

        assert result.task_id == "custom-id"

    @pytest.mark.asyncio
    async def test_calls_provider_build_and_call(self) -> None:
        context = self._make_context()
        provider = self._make_provider()

        await execute_llm_call(context, provider)

        provider.build_request_params.assert_called_once()
        provider.call.assert_called_once()

    @pytest.mark.asyncio
    async def test_explanation_preserved(self) -> None:
        context = self._make_context()
        provider = self._make_provider()

        result = await execute_llm_call(context, provider)

        assert result.response.explanation == "Created output file."


# ---------------------------------------------------------------------------
# Phase 9: cache stats extraction
# ---------------------------------------------------------------------------


class TestCacheStatsExtraction:
    def _make_context(self) -> AssembledContext:
        return AssembledContext(
            task_id="cache-task",
            system_prompt="sys",
            user_prompt="usr",
        )

    @pytest.mark.asyncio
    async def test_extracts_cache_tokens(self) -> None:
        context = self._make_context()
        provider = build_mock_provider(
            tool_input=LLMResponse(
                files=[FileOutput(file_path="out.py", content="pass")],
                explanation="Done.",
            ).model_dump(),
            cache_creation_input_tokens=500,
            cache_read_input_tokens=1000,
        )
        result = await execute_llm_call(context, provider)
        assert result.cache_creation_input_tokens == 500
        assert result.cache_read_input_tokens == 1000

    @pytest.mark.asyncio
    async def test_zero_cache_tokens(self) -> None:
        context = self._make_context()
        provider = build_mock_provider(
            tool_input=LLMResponse(explanation="Done.").model_dump(),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        result = await execute_llm_call(context, provider)
        assert result.cache_creation_input_tokens == 0
        assert result.cache_read_input_tokens == 0


# ---------------------------------------------------------------------------
# Phase 11: model_name threading via call_llm activity
# ---------------------------------------------------------------------------


class TestCallLlmModelNameThreading:
    @pytest.mark.asyncio
    async def test_threads_model_name_to_provider(self) -> None:
        from forge.activities.llm import call_llm

        provider = build_mock_provider(
            tool_input=LLMResponse(explanation="done").model_dump(),
            model_name="custom-model",
            input_tokens=10,
            output_tokens=20,
        )

        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        with (
            patch("sax_llm.get_provider", return_value=provider),
            patch("forge.tracing.get_tracer", return_value=mock_tracer),
        ):
            context = AssembledContext(
                task_id="t1",
                system_prompt="sys",
                user_prompt="usr",
                model_name="custom-model",
            )
            llm_result = await call_llm(context)

            assert llm_result.model_name == "custom-model"

    @pytest.mark.asyncio
    async def test_uses_default_when_model_name_empty(self) -> None:
        from forge.activities.llm import DEFAULT_MODEL, call_llm

        provider = build_mock_provider(
            tool_input=LLMResponse(explanation="done").model_dump(),
            model_name=DEFAULT_MODEL,
            input_tokens=10,
            output_tokens=20,
        )

        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        with (
            patch("sax_llm.get_provider", return_value=provider),
            patch("forge.tracing.get_tracer", return_value=mock_tracer),
        ):
            context = AssembledContext(
                task_id="t1",
                system_prompt="sys",
                user_prompt="usr",
            )
            result = await call_llm(context)

            assert result.model_name == DEFAULT_MODEL
