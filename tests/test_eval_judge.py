"""Tests for forge.eval.judge — LLM-as-judge plan evaluation."""

from __future__ import annotations

import pytest
from sax_platform.testing import FakeLLM

from forge.eval.judge import (
    DEFAULT_JUDGE_MAX_TOKENS,
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

        llm = FakeLLM(verdict, input_tokens=500, output_tokens=200)

        result = await execute_judge_call("system prompt", "user prompt", llm)

        assert result == verdict

    @pytest.mark.asyncio
    async def test_passes_prompts_to_llm(self) -> None:
        verdict = JudgeVerdict(
            scores=[],
            overall_assessment="OK.",
        )

        llm = FakeLLM(verdict, input_tokens=0, output_tokens=0)

        await execute_judge_call("sys", "usr", llm)

        assert len(llm.calls) == 1
        call = llm.calls[-1]
        assert call.method == "complete"
        messages = call.args[0]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "usr"
        assert call.kwargs["system"] == "sys"
        assert call.kwargs["output_type"] is JudgeVerdict
        assert call.kwargs["max_tokens"] == DEFAULT_JUDGE_MAX_TOKENS

    @pytest.mark.asyncio
    async def test_propagates_typed_failure(self) -> None:
        from sax_platform.llm import LLMTruncated, Telemetry

        telemetry = Telemetry(
            model="test-model",
            stop_reason="max_tokens",
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            request_id=None,
        )
        error = LLMTruncated(
            partial_text="", max_tokens=DEFAULT_JUDGE_MAX_TOKENS, telemetry=telemetry
        )
        llm = FakeLLM(error=error)

        with pytest.raises(LLMTruncated):
            await execute_judge_call("sys", "usr", llm)
