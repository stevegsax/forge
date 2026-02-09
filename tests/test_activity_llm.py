"""Tests for forge.activities.llm — LLM call activity."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge.activities.llm import _persist_interaction, create_agent, execute_llm_call
from forge.models import AssembledContext, FileOutput, LLMCallResult, LLMResponse

# ---------------------------------------------------------------------------
# create_agent
# ---------------------------------------------------------------------------


class TestCreateAgent:
    def test_returns_agent_with_correct_output_type(self) -> None:
        agent = create_agent("test")
        assert agent.output_type is LLMResponse


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

    def _make_mock_agent(self) -> MagicMock:
        mock_response = LLMResponse(
            files=[FileOutput(file_path="out.py", content="print('hi')")],
            explanation="Created output file.",
        )
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 200

        mock_result = MagicMock()
        mock_result.output = mock_response
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent.model = "test-model"
        return mock_agent

    @pytest.mark.asyncio
    async def test_returns_llm_call_result(self) -> None:
        context = self._make_context()
        agent = self._make_mock_agent()

        result = await execute_llm_call(context, agent)

        assert result.task_id == "llm-task"
        assert result.model_name == "test-model"
        assert len(result.response.files) == 1
        assert result.response.files[0].file_path == "out.py"

    @pytest.mark.asyncio
    async def test_extracts_usage(self) -> None:
        context = self._make_context()
        agent = self._make_mock_agent()

        result = await execute_llm_call(context, agent)

        assert result.input_tokens == 100
        assert result.output_tokens == 200

    @pytest.mark.asyncio
    async def test_latency_is_positive(self) -> None:
        context = self._make_context()
        agent = self._make_mock_agent()

        result = await execute_llm_call(context, agent)

        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_propagates_task_id(self) -> None:
        context = self._make_context()
        context = AssembledContext(
            task_id="custom-id",
            system_prompt="sys",
            user_prompt="usr",
        )
        agent = self._make_mock_agent()

        result = await execute_llm_call(context, agent)

        assert result.task_id == "custom-id"

    @pytest.mark.asyncio
    async def test_passes_prompts_to_agent(self) -> None:
        context = self._make_context()
        agent = self._make_mock_agent()

        await execute_llm_call(context, agent)

        agent.run.assert_called_once_with(
            "Generate the code.",
            instructions="You are a code generator.",
        )

    @pytest.mark.asyncio
    async def test_explanation_preserved(self) -> None:
        context = self._make_context()
        agent = self._make_mock_agent()

        result = await execute_llm_call(context, agent)

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
