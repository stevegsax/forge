"""Tests for forge.activities.llm — LLM call activity."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge.activities.llm import _persist_interaction, execute_llm_call
from forge.models import AssembledContext, FileOutput, LLMCallResult, LLMResponse
from tests.conftest import build_mock_message

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

    def _make_mock_client(
        self,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> MagicMock:
        mock_message = build_mock_message(
            tool_name="llm_response",
            tool_input=LLMResponse(
                files=[FileOutput(file_path="out.py", content="print('hi')")],
                explanation="Created output file.",
            ).model_dump(),
            input_tokens=100,
            output_tokens=200,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)
        return mock_client

    @pytest.mark.asyncio
    async def test_returns_llm_call_result(self) -> None:
        context = self._make_context()
        client = self._make_mock_client()

        result = await execute_llm_call(context, client)

        assert result.task_id == "llm-task"
        assert result.model_name == "claude-sonnet-4-5-20250929"
        assert len(result.response.files) == 1
        assert result.response.files[0].file_path == "out.py"

    @pytest.mark.asyncio
    async def test_extracts_usage(self) -> None:
        context = self._make_context()
        client = self._make_mock_client()

        result = await execute_llm_call(context, client)

        assert result.input_tokens == 100
        assert result.output_tokens == 200

    @pytest.mark.asyncio
    async def test_latency_is_positive(self) -> None:
        context = self._make_context()
        client = self._make_mock_client()

        result = await execute_llm_call(context, client)

        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_propagates_task_id(self) -> None:
        context = AssembledContext(
            task_id="custom-id",
            system_prompt="sys",
            user_prompt="usr",
        )
        client = self._make_mock_client()

        result = await execute_llm_call(context, client)

        assert result.task_id == "custom-id"

    @pytest.mark.asyncio
    async def test_passes_prompts_to_client(self) -> None:
        context = self._make_context()
        client = self._make_mock_client()

        await execute_llm_call(context, client)

        client.messages.create.assert_called_once()
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["messages"][0]["content"] == "Generate the code."

    @pytest.mark.asyncio
    async def test_explanation_preserved(self) -> None:
        context = self._make_context()
        client = self._make_mock_client()

        result = await execute_llm_call(context, client)

        assert result.response.explanation == "Created output file."


# ---------------------------------------------------------------------------
# _persist_interaction (Phase 5)
# ---------------------------------------------------------------------------


class TestPersistInteraction:
    def _make_context(self) -> AssembledContext:
        return AssembledContext(
            task_id="llm-task",
            system_prompt="sys",
            user_prompt="usr",
        )

    def _make_result(self) -> LLMCallResult:
        return LLMCallResult(
            task_id="llm-task",
            response=LLMResponse(
                files=[FileOutput(file_path="out.py", content="pass")],
                explanation="Done.",
            ),
            model_name="test-model",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200.0,
        )

    @patch("forge.store.save_interaction")
    @patch("forge.store.get_engine")
    @patch("forge.store.get_db_path")
    def test_calls_save_interaction(
        self,
        mock_get_db_path: MagicMock,
        mock_get_engine: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        from pathlib import Path

        mock_get_db_path.return_value = Path("/tmp/test.db")
        mock_get_engine.return_value = MagicMock()

        _persist_interaction(self._make_context(), self._make_result())
        mock_save.assert_called_once()

    @patch("forge.store.get_db_path")
    def test_skips_when_disabled(self, mock_get_db_path: MagicMock) -> None:
        mock_get_db_path.return_value = None
        # Should not raise
        _persist_interaction(self._make_context(), self._make_result())

    @patch("forge.store.save_interaction", side_effect=RuntimeError("db error"))
    @patch("forge.store.get_engine")
    @patch("forge.store.get_db_path")
    def test_catches_exceptions(
        self,
        mock_get_db_path: MagicMock,
        mock_get_engine: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        from pathlib import Path

        mock_get_db_path.return_value = Path("/tmp/test.db")
        mock_get_engine.return_value = MagicMock()

        # Should not raise despite save_interaction throwing
        _persist_interaction(self._make_context(), self._make_result())


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

    def _make_mock_client(self, cache_write: int = 500, cache_read: int = 1000) -> MagicMock:
        mock_message = build_mock_message(
            tool_name="llm_response",
            tool_input=LLMResponse(
                files=[FileOutput(file_path="out.py", content="pass")],
                explanation="Done.",
            ).model_dump(),
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)
        return mock_client

    @pytest.mark.asyncio
    async def test_extracts_cache_tokens(self) -> None:
        context = self._make_context()
        client = self._make_mock_client(cache_write=500, cache_read=1000)
        result = await execute_llm_call(context, client)
        assert result.cache_creation_input_tokens == 500
        assert result.cache_read_input_tokens == 1000

    @pytest.mark.asyncio
    async def test_zero_cache_tokens(self) -> None:
        context = self._make_context()
        client = self._make_mock_client(cache_write=0, cache_read=0)
        result = await execute_llm_call(context, client)
        assert result.cache_creation_input_tokens == 0
        assert result.cache_read_input_tokens == 0

    @pytest.mark.asyncio
    async def test_none_cache_tokens_become_zero(self) -> None:
        context = self._make_context()
        mock_message = build_mock_message(
            tool_name="llm_response",
            tool_input=LLMResponse(explanation="Done.").model_dump(),
            input_tokens=100,
            output_tokens=50,
        )
        mock_message.usage.cache_creation_input_tokens = None
        mock_message.usage.cache_read_input_tokens = None
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)
        result = await execute_llm_call(context, mock_client)
        assert result.cache_creation_input_tokens == 0
        assert result.cache_read_input_tokens == 0


# ---------------------------------------------------------------------------
# Phase 11: model_name threading via call_llm activity
# ---------------------------------------------------------------------------


class TestCallLlmModelNameThreading:
    @pytest.mark.asyncio
    async def test_threads_model_name_to_client(self) -> None:
        from forge.activities.llm import call_llm

        mock_message = build_mock_message(
            tool_name="llm_response",
            tool_input=LLMResponse(explanation="done").model_dump(),
            input_tokens=10,
            output_tokens=20,
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        with (
            patch("forge.llm_client.get_anthropic_client", return_value=mock_client),
            patch("forge.activities.llm._persist_interaction"),
            patch("forge.tracing.get_tracer", return_value=mock_tracer),
        ):
            context = AssembledContext(
                task_id="t1",
                system_prompt="sys",
                user_prompt="usr",
                model_name="custom-model",
            )
            llm_result = await call_llm(context)

            # Model name should be threaded through
            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["model"] == "custom-model"
            assert llm_result.model_name == "custom-model"

    @pytest.mark.asyncio
    async def test_uses_default_when_model_name_empty(self) -> None:
        from forge.activities.llm import DEFAULT_MODEL, call_llm

        mock_message = build_mock_message(
            tool_name="llm_response",
            tool_input=LLMResponse(explanation="done").model_dump(),
            input_tokens=10,
            output_tokens=20,
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        with (
            patch("forge.llm_client.get_anthropic_client", return_value=mock_client),
            patch("forge.activities.llm._persist_interaction"),
            patch("forge.tracing.get_tracer", return_value=mock_tracer),
        ):
            context = AssembledContext(
                task_id="t1",
                system_prompt="sys",
                user_prompt="usr",
            )
            await call_llm(context)

            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["model"] == DEFAULT_MODEL
