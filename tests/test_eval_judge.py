"""Tests for forge.eval.judge — LLM-as-judge plan evaluation."""

from __future__ import annotations

import pytest

from forge.eval.judge import (
    build_judge_system_prompt,
    build_judge_user_prompt,
    execute_judge_call,
)
from forge.eval.models import (
    EvalCase,
    JudgeCriterion,
    JudgeScore,
    JudgeVerdict,
)
from forge.models import Plan, PlanStep, SubTask, TaskDefinition
from tests.conftest import build_mock_provider

_TASK = TaskDefinition(
    task_id="t1",
    description="Add authentication.",
    target_files=["src/auth.py"],
    context_files=["src/models.py"],
)
_CASE = EvalCase(case_id="case-1", task=_TASK, repo_root="/tmp/repo")
_PLAN = Plan(
    task_id="t1",
    steps=[
        PlanStep(step_id="s1", description="Create auth module.", target_files=["src/auth.py"]),
    ],
    explanation="Single step auth implementation.",
)


# ---------------------------------------------------------------------------
# build_judge_system_prompt
# ---------------------------------------------------------------------------


class TestBuildJudgeSystemPrompt:
    def test_includes_task_description(self) -> None:
        prompt = build_judge_system_prompt(_CASE, _PLAN)
        assert "Add authentication." in prompt

    def test_includes_target_files(self) -> None:
        prompt = build_judge_system_prompt(_CASE, _PLAN)
        assert "src/auth.py" in prompt

    def test_includes_context_files(self) -> None:
        prompt = build_judge_system_prompt(_CASE, _PLAN)
        assert "src/models.py" in prompt

    def test_includes_plan_steps(self) -> None:
        prompt = build_judge_system_prompt(_CASE, _PLAN)
        assert "s1" in prompt
        assert "Create auth module." in prompt

    def test_includes_all_criteria(self) -> None:
        prompt = build_judge_system_prompt(_CASE, _PLAN)
        for criterion in JudgeCriterion:
            assert criterion.value in prompt

    def test_includes_repo_context(self) -> None:
        prompt = build_judge_system_prompt(_CASE, _PLAN, repo_context="file tree here")
        assert "file tree here" in prompt

    def test_includes_subtask_details(self) -> None:
        plan = Plan(
            task_id="t1",
            steps=[
                PlanStep(
                    step_id="s1",
                    description="Fan out.",
                    target_files=[],
                    sub_tasks=[
                        SubTask(
                            sub_task_id="st1",
                            description="Sub A.",
                            target_files=["a.py"],
                        ),
                        SubTask(
                            sub_task_id="st2",
                            description="Sub B.",
                            target_files=["b.py"],
                        ),
                    ],
                ),
            ],
            explanation="Fan-out plan.",
        )
        prompt = build_judge_system_prompt(_CASE, plan)
        assert "st1" in prompt
        assert "st2" in prompt
        assert "Sub A." in prompt


# ---------------------------------------------------------------------------
# build_judge_user_prompt
# ---------------------------------------------------------------------------


class TestBuildJudgeUserPrompt:
    def test_returns_nonempty(self) -> None:
        prompt = build_judge_user_prompt()
        assert len(prompt) > 0
        assert "1" in prompt and "5" in prompt


# ---------------------------------------------------------------------------
# execute_judge_call
# ---------------------------------------------------------------------------


class TestExecuteJudgeCall:
    @pytest.mark.asyncio
    async def test_returns_verdict(self) -> None:
        verdict = JudgeVerdict(
            scores=[
                JudgeScore(
                    criterion=JudgeCriterion.COMPLETENESS,
                    score=5,
                    rationale="Covers all targets.",
                ),
            ],
            overall_assessment="Solid plan.",
        )

        provider = build_mock_provider(
            tool_input=verdict.model_dump(),
            input_tokens=500,
            output_tokens=200,
        )

        result = await execute_judge_call("system prompt", "user prompt", provider)

        assert result == verdict

    @pytest.mark.asyncio
    async def test_passes_prompts_to_provider(self) -> None:
        verdict = JudgeVerdict(
            scores=[],
            overall_assessment="OK.",
        )

        provider = build_mock_provider(
            tool_input=verdict.model_dump(),
            input_tokens=0,
            output_tokens=0,
        )

        await execute_judge_call("sys", "usr", provider)

        provider.build_request_params.assert_called_once()
        call_kwargs = provider.build_request_params.call_args[1]
        assert call_kwargs["system_prompt"] == "sys"
        assert call_kwargs["user_prompt"] == "usr"
