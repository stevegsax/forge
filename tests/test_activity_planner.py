"""Tests for forge.activities.planner — planning activity."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.activities.planner import (
    assemble_planner_context,
    build_planner_system_prompt,
    build_planner_user_prompt,
    create_planner_agent,
    execute_planner_call,
)
from forge.models import (
    AssembleContextInput,
    Plan,
    PlannerInput,
    PlanStep,
    TaskDefinition,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# build_planner_system_prompt (pure function)
# ---------------------------------------------------------------------------


class TestBuildPlannerSystemPrompt:
    def test_includes_description(self) -> None:
        task = TaskDefinition(task_id="t1", description="Build a REST API.")
        prompt = build_planner_system_prompt(task, {})
        assert "Build a REST API." in prompt

    def test_includes_target_file_hints(self) -> None:
        task = TaskDefinition(
            task_id="t1",
            description="desc",
            target_files=["src/api.py", "src/models.py"],
        )
        prompt = build_planner_system_prompt(task, {})
        assert "- src/api.py" in prompt
        assert "- src/models.py" in prompt

    def test_omits_target_section_when_empty(self) -> None:
        task = TaskDefinition(task_id="t1", description="desc")
        prompt = build_planner_system_prompt(task, {})
        assert "## Target Files" not in prompt

    def test_includes_context_files(self) -> None:
        task = TaskDefinition(task_id="t1", description="desc")
        context = {"ref.py": "# reference code"}
        prompt = build_planner_system_prompt(task, context)
        assert "### ref.py" in prompt
        assert "# reference code" in prompt

    def test_includes_decomposition_instructions(self) -> None:
        task = TaskDefinition(task_id="t1", description="desc")
        prompt = build_planner_system_prompt(task, {})
        assert "Decompose the task into ordered steps" in prompt


# ---------------------------------------------------------------------------
# build_planner_user_prompt (pure function)
# ---------------------------------------------------------------------------


class TestBuildPlannerUserPrompt:
    def test_includes_task_description(self) -> None:
        task = TaskDefinition(task_id="t1", description="Build a REST API.")
        prompt = build_planner_user_prompt(task)
        assert "Build a REST API." in prompt

    def test_asks_for_plan(self) -> None:
        task = TaskDefinition(task_id="t1", description="desc")
        prompt = build_planner_user_prompt(task)
        assert "plan" in prompt.lower()


# ---------------------------------------------------------------------------
# execute_planner_call (testable function)
# ---------------------------------------------------------------------------


_TEST_PLAN = Plan(
    task_id="t1",
    steps=[
        PlanStep(step_id="s1", description="Create models.", target_files=["models.py"]),
        PlanStep(step_id="s2", description="Create API.", target_files=["api.py"]),
    ],
    explanation="Split into models and API layers.",
)


class TestExecutePlannerCall:
    @pytest.mark.asyncio
    async def test_returns_plan_call_result(self) -> None:
        mock_usage = MagicMock()
        mock_usage.input_tokens = 200
        mock_usage.output_tokens = 100

        mock_result = MagicMock()
        mock_result.output = _TEST_PLAN
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent.model = "mock-model"

        planner_input = PlannerInput(
            task_id="t1",
            system_prompt="sys",
            user_prompt="usr",
        )
        result = await execute_planner_call(planner_input, mock_agent)

        assert result.task_id == "t1"
        assert result.plan == _TEST_PLAN
        assert result.model_name == "mock-model"
        assert result.input_tokens == 200
        assert result.output_tokens == 100
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_calls_agent_with_correct_prompts(self) -> None:
        mock_usage = MagicMock()
        mock_usage.input_tokens = 0
        mock_usage.output_tokens = 0

        mock_result = MagicMock()
        mock_result.output = _TEST_PLAN
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent.model = "mock-model"

        planner_input = PlannerInput(
            task_id="t1",
            system_prompt="my system prompt",
            user_prompt="my user prompt",
        )
        await execute_planner_call(planner_input, mock_agent)

        mock_agent.run.assert_called_once_with(
            "my user prompt",
            instructions="my system prompt",
        )


# ---------------------------------------------------------------------------
# create_planner_agent (imperative shell)
# ---------------------------------------------------------------------------


class TestCreatePlannerAgent:
    def test_creates_agent_with_plan_output_type(self) -> None:
        agent = create_planner_agent("test")
        assert agent.output_type is Plan


# ---------------------------------------------------------------------------
# assemble_planner_context (activity)
# ---------------------------------------------------------------------------


class TestAssemblePlannerContext:
    @pytest.mark.asyncio
    async def test_assembles_from_files(self, tmp_path: Path) -> None:
        (tmp_path / "ref.py").write_text("# reference")
        task = TaskDefinition(
            task_id="plan-1",
            description="Build something.",
            target_files=["out.py"],
            context_files=["ref.py"],
        )
        input_data = AssembleContextInput(
            task=task,
            repo_root=str(tmp_path),
            worktree_path=str(tmp_path / "wt"),
        )
        result = await assemble_planner_context(input_data)
        assert result.task_id == "plan-1"
        assert "# reference" in result.system_prompt
        assert "Build something." in result.user_prompt

    @pytest.mark.asyncio
    async def test_skips_missing_context_files(self, tmp_path: Path) -> None:
        task = TaskDefinition(
            task_id="plan-2",
            description="Build something.",
            context_files=["nonexistent.py"],
        )
        input_data = AssembleContextInput(
            task=task,
            repo_root=str(tmp_path),
            worktree_path=str(tmp_path / "wt"),
        )
        result = await assemble_planner_context(input_data)
        assert "## Context Files" not in result.system_prompt
