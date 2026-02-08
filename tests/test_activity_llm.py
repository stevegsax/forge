"""Tests for forge.activities.llm — LLM call activity."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.activities.llm import create_agent, execute_llm_call
from forge.models import AssembledContext, FileOutput, LLMResponse

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
