"""Tests for forge.activities.planner — planning activity."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge.activities.planner import (
    assemble_planner_context,
    build_planner_system_prompt,
    build_planner_user_prompt,
    execute_planner_call,
)
from forge.llm_client import build_thinking_param
from forge.models import (
    AssembleContextInput,
    ContextConfig,
    Plan,
    PlannerInput,
    PlanStep,
    TaskDefinition,
    TaskDomain,
)
from tests.conftest import build_mock_message

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
        mock_message = build_mock_message(
            tool_name="plan",
            tool_input=_TEST_PLAN.model_dump(),
            input_tokens=200,
            output_tokens=100,
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        planner_input = PlannerInput(
            task_id="t1",
            system_prompt="sys",
            user_prompt="usr",
        )
        result = await execute_planner_call(planner_input, mock_client)

        assert result.task_id == "t1"
        assert result.plan == _TEST_PLAN
        assert result.input_tokens == 200
        assert result.output_tokens == 100
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_calls_client_with_correct_prompts(self) -> None:
        mock_message = build_mock_message(
            tool_name="plan",
            tool_input=_TEST_PLAN.model_dump(),
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        planner_input = PlannerInput(
            task_id="t1",
            system_prompt="my system prompt",
            user_prompt="my user prompt",
        )
        await execute_planner_call(planner_input, mock_client)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["messages"][0]["content"] == "my user prompt"


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
            task_id=task.task_id,
            description=task.description,
            target_files=task.target_files,
            context_files=task.context_files,
            context_config=task.context,
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
            task_id=task.task_id,
            description=task.description,
            target_files=task.target_files,
            context_files=task.context_files,
            context_config=task.context,
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
            task_id=task.task_id,
            description=task.description,
            target_files=task.target_files,
            context_files=task.context_files,
            context_config=task.context,
            repo_root=str(tmp_path),
            worktree_path=str(tmp_path / "wt"),
        )
        result = await assemble_planner_context(input_data)
        assert "## Repository Structure" not in result.system_prompt


# ---------------------------------------------------------------------------
# Phase 9: cache stats
# ---------------------------------------------------------------------------


class TestPlannerCacheStats:
    @pytest.mark.asyncio
    async def test_extracts_cache_tokens(self) -> None:
        mock_message = build_mock_message(
            tool_name="plan",
            tool_input=_TEST_PLAN.model_dump(),
            input_tokens=200,
            output_tokens=100,
            cache_creation_input_tokens=500,
            cache_read_input_tokens=1000,
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        planner_input = PlannerInput(
            task_id="t1",
            system_prompt="sys",
            user_prompt="usr",
        )
        result = await execute_planner_call(planner_input, mock_client)
        assert result.cache_creation_input_tokens == 500
        assert result.cache_read_input_tokens == 1000


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
            task_id=task.task_id,
            description=task.description,
            target_files=task.target_files,
            context_files=task.context_files,
            context_config=task.context,
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
            task_id=task.task_id,
            description=task.description,
            target_files=task.target_files,
            context_files=task.context_files,
            context_config=task.context,
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
    async def test_threads_model_name_to_client(self) -> None:
        from forge.activities.planner import call_planner

        mock_message = build_mock_message(
            tool_name="plan",
            tool_input=_TEST_PLAN.model_dump(),
            input_tokens=100,
            output_tokens=50,
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        with (
            patch("forge.llm_client.get_anthropic_client", return_value=mock_client),
            patch("forge.store.persist_interaction"),
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
                model_name="custom-planner",
            )
            await call_planner(planner_input)

            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["model"] == "custom-planner"

    @pytest.mark.asyncio
    async def test_uses_default_when_model_name_empty(self) -> None:
        from forge.activities.planner import call_planner

        mock_message = build_mock_message(
            tool_name="plan",
            tool_input=_TEST_PLAN.model_dump(),
            input_tokens=100,
            output_tokens=50,
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        with (
            patch("forge.llm_client.get_anthropic_client", return_value=mock_client),
            patch("forge.store.persist_interaction"),
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

            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["model"] == "claude-sonnet-4-5-20250929"


# ---------------------------------------------------------------------------
# Phase 12: Extended thinking for planning (now in llm_client)
# ---------------------------------------------------------------------------


class TestBuildThinkingParam:
    def test_opus_gets_enabled(self) -> None:
        result = build_thinking_param("claude-opus-4-6", 10_000, "high")
        assert result == {"type": "enabled", "budget_tokens": 10_000}

    def test_sonnet_gets_budget(self) -> None:
        result = build_thinking_param("claude-sonnet-4-5-20250929", 10_000, "high")
        assert result == {"type": "enabled", "budget_tokens": 10_000}

    def test_haiku_returns_none(self) -> None:
        result = build_thinking_param("claude-haiku-4-5-20251001", 10_000, "high")
        assert result is None

    def test_zero_budget_returns_none(self) -> None:
        result = build_thinking_param("claude-opus-4-6", 0, "high")
        assert result is None


class TestCallPlannerThinkingThreading:
    @pytest.mark.asyncio
    async def test_threads_thinking_config_to_params(self) -> None:
        from forge.activities.planner import call_planner

        mock_message = build_mock_message(
            tool_name="plan",
            tool_input=_TEST_PLAN.model_dump(),
            input_tokens=100,
            output_tokens=50,
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        with (
            patch("forge.llm_client.get_anthropic_client", return_value=mock_client),
            patch("forge.store.persist_interaction"),
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
                model_name="claude-opus-4-6",
                thinking_budget_tokens=10_000,
                thinking_effort="high",
            )
            await call_planner(planner_input)

            call_kwargs = mock_client.messages.create.call_args[1]
            assert "thinking" in call_kwargs


# ---------------------------------------------------------------------------
# Domain-aware planner prompts
# ---------------------------------------------------------------------------


class TestBuildPlannerSystemPromptDomain:
    def test_includes_task_domain_section(self) -> None:
        task = TaskDefinition(task_id="t1", description="Build module.")
        prompt = build_planner_system_prompt(task, {})
        assert "## Task Domain" in prompt

    def test_code_generation_domain_instruction(self) -> None:
        task = TaskDefinition(
            task_id="t1",
            description="Build module.",
            domain=TaskDomain.CODE_GENERATION,
        )
        prompt = build_planner_system_prompt(task, {})
        assert "code generation" in prompt.lower()
        assert "## Task Domain" in prompt

    def test_research_domain_instruction(self) -> None:
        task = TaskDefinition(
            task_id="t1",
            description="Research topic.",
            domain=TaskDomain.RESEARCH,
        )
        prompt = build_planner_system_prompt(task, {})
        assert "## Task Domain" in prompt
        assert "research" in prompt.lower()
