"""Tests for forge.models — data model serialization and defaults."""

from __future__ import annotations

import pytest

from forge.models import (
    AssembleStepContextInput,
    CommitChangesInput,
    ForgeTaskInput,
    Plan,
    PlanCallResult,
    PlannerInput,
    PlanStep,
    ResetWorktreeInput,
    StepResult,
    TaskDefinition,
    TaskResult,
    TransitionSignal,
)

# ---------------------------------------------------------------------------
# Planning models
# ---------------------------------------------------------------------------


class TestPlanStep:
    def test_defaults(self) -> None:
        step = PlanStep(step_id="s1", description="Do something.", target_files=["a.py"])
        assert step.context_files == []

    def test_round_trip(self) -> None:
        step = PlanStep(
            step_id="s1",
            description="Create module.",
            target_files=["a.py"],
            context_files=["ref.py"],
        )
        rebuilt = PlanStep.model_validate_json(step.model_dump_json())
        assert rebuilt == step


class TestPlan:
    def test_min_one_step(self) -> None:
        with pytest.raises(ValueError):
            Plan(task_id="t1", steps=[], explanation="empty")

    def test_round_trip(self) -> None:
        plan = Plan(
            task_id="t1",
            steps=[PlanStep(step_id="s1", description="d", target_files=["a.py"])],
            explanation="One step plan.",
        )
        rebuilt = Plan.model_validate_json(plan.model_dump_json())
        assert rebuilt == plan


class TestStepResult:
    def test_defaults(self) -> None:
        sr = StepResult(step_id="s1", status=TransitionSignal.SUCCESS)
        assert sr.output_files == {}
        assert sr.validation_results == []
        assert sr.commit_sha is None
        assert sr.error is None

    def test_round_trip(self) -> None:
        sr = StepResult(
            step_id="s1",
            status=TransitionSignal.SUCCESS,
            commit_sha="a" * 40,
        )
        rebuilt = StepResult.model_validate_json(sr.model_dump_json())
        assert rebuilt == sr


# ---------------------------------------------------------------------------
# Modified models — backward compatibility
# ---------------------------------------------------------------------------


class TestTaskDefinitionBackwardCompat:
    def test_target_files_optional(self) -> None:
        td = TaskDefinition(task_id="t1", description="d")
        assert td.target_files == []

    def test_target_files_still_accepts_list(self) -> None:
        td = TaskDefinition(task_id="t1", description="d", target_files=["a.py"])
        assert td.target_files == ["a.py"]


class TestTaskResultBackwardCompat:
    def test_new_fields_default(self) -> None:
        result = TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        assert result.step_results == []
        assert result.plan is None

    def test_with_plan(self) -> None:
        plan = Plan(
            task_id="t",
            steps=[PlanStep(step_id="s1", description="d", target_files=["a.py"])],
            explanation="test",
        )
        result = TaskResult(task_id="t", status=TransitionSignal.SUCCESS, plan=plan)
        rebuilt = TaskResult.model_validate_json(result.model_dump_json())
        assert rebuilt.plan == plan


class TestForgeTaskInputBackwardCompat:
    def test_defaults(self) -> None:
        td = TaskDefinition(task_id="t", description="d", target_files=["a.py"])
        inp = ForgeTaskInput(task=td, repo_root="/tmp/repo")
        assert inp.plan is False
        assert inp.max_step_attempts == 2
        assert inp.max_attempts == 2


class TestCommitChangesInputMessage:
    def test_message_default_none(self) -> None:
        inp = CommitChangesInput(repo_root="/r", task_id="t", status="s")
        assert inp.message is None

    def test_message_override(self) -> None:
        inp = CommitChangesInput(repo_root="/r", task_id="t", status="s", message="custom msg")
        assert inp.message == "custom msg"


# ---------------------------------------------------------------------------
# New activity I/O models
# ---------------------------------------------------------------------------


class TestResetWorktreeInput:
    def test_fields(self) -> None:
        inp = ResetWorktreeInput(repo_root="/repo", task_id="t1")
        assert inp.repo_root == "/repo"
        assert inp.task_id == "t1"


class TestPlannerInput:
    def test_round_trip(self) -> None:
        inp = PlannerInput(task_id="t", system_prompt="sys", user_prompt="usr")
        rebuilt = PlannerInput.model_validate_json(inp.model_dump_json())
        assert rebuilt == inp


class TestPlanCallResult:
    def test_round_trip(self) -> None:
        plan = Plan(
            task_id="t",
            steps=[PlanStep(step_id="s1", description="d", target_files=["a.py"])],
            explanation="test",
        )
        result = PlanCallResult(
            task_id="t",
            plan=plan,
            model_name="mock",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200.0,
        )
        rebuilt = PlanCallResult.model_validate_json(result.model_dump_json())
        assert rebuilt == result


class TestAssembleStepContextInput:
    def test_defaults(self) -> None:
        td = TaskDefinition(task_id="t", description="d")
        step = PlanStep(step_id="s1", description="d", target_files=["a.py"])
        inp = AssembleStepContextInput(
            task=td,
            step=step,
            step_index=0,
            total_steps=1,
            repo_root="/repo",
            worktree_path="/wt",
        )
        assert inp.completed_steps == []
