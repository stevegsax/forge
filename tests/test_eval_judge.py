"""Tests for forge.eval.judge — LLM-as-judge plan evaluation."""

from __future__ import annotations

import pytest
from sax_platform.testing import FakeLLM

from forge.eval.judge import (
    DEFAULT_JUDGE_MAX_TOKENS,
    build_judge_system_prompt,
    build_judge_user_prompt,
    execute_judge_call,
    format_repo_context,
    judge_plan,
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


def _full_verdict(*, overall: str = "OK.") -> JudgeVerdict:
    """A JudgeVerdict scoring every criterion — the shape validation requires."""
    return JudgeVerdict(
        scores=[JudgeScore(criterion=crit, score=4, rationale="ok") for crit in JudgeCriterion],
        overall_assessment=overall,
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
        verdict = _full_verdict(overall="Solid plan.")

        llm = FakeLLM(verdict, input_tokens=500, output_tokens=200)

        result = await execute_judge_call("system prompt", "user prompt", llm)

        assert result == verdict

    @pytest.mark.asyncio
    async def test_passes_prompts_to_llm(self) -> None:
        verdict = _full_verdict()

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


# ---------------------------------------------------------------------------
# format_repo_context
# ---------------------------------------------------------------------------


class TestFormatRepoContext:
    def test_lists_sorted_files_with_count(self) -> None:
        out = format_repo_context({"b.py", "a.py"})
        assert "2 tracked file(s)" in out
        assert out.index("a.py") < out.index("b.py")

    def test_caps_large_listing(self) -> None:
        files = {f"src/mod_{i:04d}.py" for i in range(50)}
        out = format_repo_context(files, max_files=10)
        assert "50 tracked file(s)" in out
        assert "40 more file(s) not shown" in out
        # Only the capped number of "- <path>" listing lines appear.
        assert out.count("\n- ") == 10

    def test_no_truncation_line_when_within_cap(self) -> None:
        out = format_repo_context({"a.py", "b.py"}, max_files=10)
        assert "not shown" not in out


# ---------------------------------------------------------------------------
# judge_plan — repo context wiring (T0.6)
# ---------------------------------------------------------------------------


class TestJudgePlanRepoContext:
    @pytest.mark.asyncio
    async def test_repo_context_reaches_judge_system_prompt(self) -> None:
        """When a repo context is supplied, it rides in the judge's system prompt."""
        llm = FakeLLM(_full_verdict())
        repo_ctx = format_repo_context({"src/auth.py", "src/models.py"})

        await judge_plan(_CASE, _PLAN, llm, repo_context=repo_ctx)

        assert len(llm.calls) == 1
        system = llm.calls[-1].kwargs["system"]
        assert "Repository Context" in system
        assert "src/auth.py" in system

    @pytest.mark.asyncio
    async def test_no_repo_context_omits_section(self) -> None:
        llm = FakeLLM(_full_verdict())

        await judge_plan(_CASE, _PLAN, llm)

        system = llm.calls[-1].kwargs["system"]
        assert "Repository Context" not in system
