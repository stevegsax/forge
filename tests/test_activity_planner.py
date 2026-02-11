"""Tests for forge.activities.planner — planning activity."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.activities.planner import (
    assemble_planner_context,
    build_planner_system_prompt,
    build_planner_user_prompt,
    build_thinking_settings,
    create_planner_agent,
    execute_planner_call,
)
from forge.models import (
    AssembleContextInput,
    ContextConfig,
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


# ---------------------------------------------------------------------------
# Phase 4: repo_map parameter
# ---------------------------------------------------------------------------


class TestBuildPlannerSystemPromptWithRepoMap:
    def test_includes_repo_map(self) -> None:
        task = TaskDefinition(task_id="t1", description="Build an API.")
        repo_map = "src/forge/models.py:\n│class Foo:"
        prompt = build_planner_system_prompt(task, {}, repo_map=repo_map)
        assert "## Repository Structure" in prompt
        assert "src/forge/models.py:" in prompt
        assert "│class Foo:" in prompt

    def test_repo_map_none_omits_section(self) -> None:
        task = TaskDefinition(task_id="t1", description="desc")
        prompt = build_planner_system_prompt(task, {}, repo_map=None)
        assert "## Repository Structure" not in prompt

    def test_repo_map_adds_context_guidance(self) -> None:
        task = TaskDefinition(task_id="t1", description="desc")
        prompt = build_planner_system_prompt(task, {}, repo_map="some map")
        assert "automatically discovered from import graphs" in prompt


class TestAssemblePlannerContextAutoDiscover:
    @pytest.mark.asyncio
    async def test_disabled_no_repo_map(self, tmp_path: Path) -> None:
        config = ContextConfig(auto_discover=False)
        task = TaskDefinition(
            task_id="plan-no-auto",
            description="Build something.",
            context=config,
        )
        input_data = AssembleContextInput(
            task=task,
            repo_root=str(tmp_path),
            worktree_path=str(tmp_path / "wt"),
        )
        result = await assemble_planner_context(input_data)
        assert "## Repository Structure" not in result.system_prompt


# ---------------------------------------------------------------------------
# Phase 9: cache stats + model settings
# ---------------------------------------------------------------------------


class TestPlannerCacheStats:
    @pytest.mark.asyncio
    async def test_extracts_cache_tokens(self) -> None:
        mock_usage = MagicMock()
        mock_usage.input_tokens = 200
        mock_usage.output_tokens = 100
        mock_usage.cache_creation_input_tokens = 500
        mock_usage.cache_read_input_tokens = 1000

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
        assert result.cache_creation_input_tokens == 500
        assert result.cache_read_input_tokens == 1000


class TestCreatePlannerAgentModelSettings:
    def test_has_cache_settings(self) -> None:
        agent = create_planner_agent("test")
        settings = agent.model_settings
        assert settings.get("anthropic_cache_instructions") is True
        assert settings.get("anthropic_cache_tool_definitions") is True


# ---------------------------------------------------------------------------
# Project instructions in planner
# ---------------------------------------------------------------------------


class TestBuildPlannerSystemPromptProjectInstructions:
    def test_includes_project_instructions(self) -> None:
        task = TaskDefinition(task_id="t1", description="Build an API.")
        instructions = "## Project Instructions\n\nUse SOLID principles."
        prompt = build_planner_system_prompt(task, {}, project_instructions=instructions)
        assert "## Project Instructions" in prompt
        assert "Use SOLID principles." in prompt

    def test_instructions_after_role_before_task(self) -> None:
        task = TaskDefinition(task_id="t1", description="Build an API.")
        instructions = "## Project Instructions\n\nUse SOLID."
        prompt = build_planner_system_prompt(task, {}, project_instructions=instructions)
        role_pos = prompt.index("task decomposition assistant")
        instr_pos = prompt.index("## Project Instructions")
        task_pos = prompt.index("## Task")
        assert role_pos < instr_pos < task_pos

    def test_omits_when_empty(self) -> None:
        task = TaskDefinition(task_id="t1", description="desc")
        prompt = build_planner_system_prompt(task, {}, project_instructions="")
        assert "## Project Instructions" not in prompt


class TestAssemblePlannerContextProjectInstructions:
    @pytest.mark.asyncio
    async def test_includes_claude_md(self, tmp_path: Path) -> None:
        (tmp_path / "CLAUDE.md").write_text("Apply Function Core / Imperative Shell.")
        task = TaskDefinition(
            task_id="plan-instr",
            description="Build something.",
        )
        input_data = AssembleContextInput(
            task=task,
            repo_root=str(tmp_path),
            worktree_path=str(tmp_path / "wt"),
        )
        result = await assemble_planner_context(input_data)
        assert "## Project Instructions" in result.system_prompt
        assert "Function Core / Imperative Shell" in result.system_prompt

    @pytest.mark.asyncio
    async def test_no_claude_md_omits_section(self, tmp_path: Path) -> None:
        task = TaskDefinition(
            task_id="plan-no-instr",
            description="Build something.",
        )
        input_data = AssembleContextInput(
            task=task,
            repo_root=str(tmp_path),
            worktree_path=str(tmp_path / "wt"),
        )
        result = await assemble_planner_context(input_data)
        assert "## Project Instructions" not in result.system_prompt


# ---------------------------------------------------------------------------
# Phase 11: Capability tier docs in planner prompt
# ---------------------------------------------------------------------------


class TestBuildPlannerSystemPromptCapabilityTier:
    def test_includes_capability_tier_section(self) -> None:
        task = TaskDefinition(task_id="t1", description="Build an API.")
        prompt = build_planner_system_prompt(task, {})
        assert "## Capability Tier" in prompt

    def test_includes_tier_names(self) -> None:
        task = TaskDefinition(task_id="t1", description="Build an API.")
        prompt = build_planner_system_prompt(task, {})
        assert "REASONING" in prompt
        assert "GENERATION" in prompt
        assert "SUMMARIZATION" in prompt
        assert "CLASSIFICATION" in prompt

    def test_tier_section_before_fan_out(self) -> None:
        task = TaskDefinition(task_id="t1", description="Build an API.")
        prompt = build_planner_system_prompt(task, {})
        tier_pos = prompt.index("## Capability Tier")
        fan_out_pos = prompt.index("## Fan-Out Sub-Tasks")
        assert tier_pos < fan_out_pos


# ---------------------------------------------------------------------------
# Phase 11: model_name threading via call_planner activity
# ---------------------------------------------------------------------------


class TestCallPlannerModelNameThreading:
    @pytest.mark.asyncio
    async def test_threads_model_name_to_create_planner_agent(self) -> None:
        from unittest.mock import patch

        from forge.activities.planner import call_planner

        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        mock_usage.cache_creation_input_tokens = 0
        mock_usage.cache_read_input_tokens = 0

        mock_result = MagicMock()
        mock_result.output = _TEST_PLAN
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent.model = "custom:planner"

        with (
            patch(
                "forge.activities.planner.create_planner_agent",
                return_value=mock_agent,
            ) as mock_create,
            patch("forge.activities.planner._persist_interaction"),
            patch("forge.tracing.get_tracer") as mock_get_tracer,
        ):
            mock_span = MagicMock()
            mock_span.__enter__ = MagicMock(return_value=mock_span)
            mock_span.__exit__ = MagicMock(return_value=False)
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value = mock_span
            mock_get_tracer.return_value = mock_tracer

            planner_input = PlannerInput(
                task_id="t1",
                system_prompt="sys",
                user_prompt="usr",
                model_name="custom:planner",
            )
            await call_planner(planner_input)

            mock_create.assert_called_once_with(
                "custom:planner",
                thinking_budget_tokens=0,
                thinking_effort="high",
            )

    @pytest.mark.asyncio
    async def test_uses_default_when_model_name_empty(self) -> None:
        from unittest.mock import patch

        from forge.activities.planner import call_planner

        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        mock_usage.cache_creation_input_tokens = 0
        mock_usage.cache_read_input_tokens = 0

        mock_result = MagicMock()
        mock_result.output = _TEST_PLAN
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent.model = "default-model"

        with (
            patch(
                "forge.activities.planner.create_planner_agent",
                return_value=mock_agent,
            ) as mock_create,
            patch("forge.activities.planner._persist_interaction"),
            patch("forge.tracing.get_tracer") as mock_get_tracer,
        ):
            mock_span = MagicMock()
            mock_span.__enter__ = MagicMock(return_value=mock_span)
            mock_span.__exit__ = MagicMock(return_value=False)
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value = mock_span
            mock_get_tracer.return_value = mock_tracer

            planner_input = PlannerInput(
                task_id="t1",
                system_prompt="sys",
                user_prompt="usr",
            )
            await call_planner(planner_input)

            # Empty string is falsy, so or None gives None
            mock_create.assert_called_once_with(
                None, thinking_budget_tokens=0, thinking_effort="high"
            )


# ---------------------------------------------------------------------------
# Phase 12: Extended thinking for planning
# ---------------------------------------------------------------------------


class TestBuildThinkingSettings:
    def test_opus_gets_adaptive(self) -> None:
        result = build_thinking_settings("anthropic:claude-opus-4-6", 10_000, "high")
        assert result["anthropic_thinking"] == {"type": "adaptive"}
        assert result["anthropic_effort"] == "high"

    def test_sonnet_gets_budget(self) -> None:
        result = build_thinking_settings(
            "anthropic:claude-sonnet-4-5-20250929", 10_000, "high"
        )
        assert result["anthropic_thinking"] == {"type": "enabled", "budget_tokens": 10_000}
        assert "anthropic_effort" not in result

    def test_haiku_returns_empty(self) -> None:
        result = build_thinking_settings("anthropic:claude-haiku-4-5-20251001", 10_000, "high")
        assert result == {}

    def test_non_anthropic_returns_empty(self) -> None:
        result = build_thinking_settings("openai:gpt-4o", 10_000, "high")
        assert result == {}

    def test_zero_budget_returns_empty(self) -> None:
        result = build_thinking_settings("anthropic:claude-opus-4-6", 0, "high")
        assert result == {}


class TestCreatePlannerAgentThinking:
    def test_thinking_settings_merged_with_cache(self) -> None:
        agent = create_planner_agent(
            "anthropic:claude-opus-4-6",
            thinking_budget_tokens=10_000,
            thinking_effort="high",
        )
        settings = agent.model_settings
        # Cache settings preserved
        assert settings.get("anthropic_cache_instructions") is True
        assert settings.get("anthropic_cache_tool_definitions") is True
        # Thinking settings added
        assert settings.get("anthropic_thinking") == {"type": "adaptive"}
        assert settings.get("anthropic_effort") == "high"

    def test_no_thinking_when_budget_zero(self) -> None:
        agent = create_planner_agent(
            "anthropic:claude-opus-4-6",
            thinking_budget_tokens=0,
        )
        settings = agent.model_settings
        # Cache settings preserved
        assert settings.get("anthropic_cache_instructions") is True
        assert settings.get("anthropic_cache_tool_definitions") is True
        # No thinking settings
        assert "anthropic_thinking" not in settings
        assert "anthropic_effort" not in settings


class TestCallPlannerThinkingThreading:
    @pytest.mark.asyncio
    async def test_threads_thinking_config_to_agent(self) -> None:
        from unittest.mock import patch

        from forge.activities.planner import call_planner

        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        mock_usage.cache_creation_input_tokens = 0
        mock_usage.cache_read_input_tokens = 0

        mock_result = MagicMock()
        mock_result.output = _TEST_PLAN
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent.model = "anthropic:claude-opus-4-6"

        with (
            patch(
                "forge.activities.planner.create_planner_agent",
                return_value=mock_agent,
            ) as mock_create,
            patch("forge.activities.planner._persist_interaction"),
            patch("forge.tracing.get_tracer") as mock_get_tracer,
        ):
            mock_span = MagicMock()
            mock_span.__enter__ = MagicMock(return_value=mock_span)
            mock_span.__exit__ = MagicMock(return_value=False)
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value = mock_span
            mock_get_tracer.return_value = mock_tracer

            planner_input = PlannerInput(
                task_id="t1",
                system_prompt="sys",
                user_prompt="usr",
                model_name="anthropic:claude-opus-4-6",
                thinking_budget_tokens=10_000,
                thinking_effort="high",
            )
            await call_planner(planner_input)

            mock_create.assert_called_once_with(
                "anthropic:claude-opus-4-6",
                thinking_budget_tokens=10_000,
                thinking_effort="high",
            )
