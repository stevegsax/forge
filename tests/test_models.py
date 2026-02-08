"""Tests for forge.models — data model serialization and defaults."""

from __future__ import annotations

import pytest

from forge.models import (
    AssembleStepContextInput,
    AssembleSubTaskContextInput,
    CommitChangesInput,
    ForgeTaskInput,
    Plan,
    PlanCallResult,
    PlannerInput,
    PlanStep,
    ResetWorktreeInput,
    StepResult,
    SubTask,
    SubTaskInput,
    SubTaskResult,
    TaskDefinition,
    TaskResult,
    TransitionSignal,
    WriteFilesInput,
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


# ---------------------------------------------------------------------------
# Phase 3: Fan-out models
# ---------------------------------------------------------------------------


class TestSubTask:
    def test_defaults(self) -> None:
        st = SubTask(sub_task_id="st1", description="Analyze.", target_files=["a.py"])
        assert st.context_files == []

    def test_round_trip(self) -> None:
        st = SubTask(
            sub_task_id="st1",
            description="Analyze schema.",
            target_files=["schema.py"],
            context_files=["models.py"],
        )
        rebuilt = SubTask.model_validate_json(st.model_dump_json())
        assert rebuilt == st


class TestPlanStepWithSubTasks:
    def test_sub_tasks_default_none(self) -> None:
        step = PlanStep(step_id="s1", description="d", target_files=["a.py"])
        assert step.sub_tasks is None

    def test_sub_tasks_populated(self) -> None:
        st = SubTask(sub_task_id="st1", description="d", target_files=["a.py"])
        step = PlanStep(step_id="s1", description="d", target_files=[], sub_tasks=[st])
        assert step.sub_tasks is not None
        assert len(step.sub_tasks) == 1

    def test_round_trip(self) -> None:
        st = SubTask(sub_task_id="st1", description="d", target_files=["a.py"])
        step = PlanStep(step_id="s1", description="d", target_files=[], sub_tasks=[st])
        rebuilt = PlanStep.model_validate_json(step.model_dump_json())
        assert rebuilt == step

    def test_backward_compat_without_sub_tasks(self) -> None:
        """Existing plans without sub_tasks still deserialize correctly."""
        step = PlanStep(step_id="s1", description="d", target_files=["a.py"])
        rebuilt = PlanStep.model_validate_json(step.model_dump_json())
        assert rebuilt.sub_tasks is None


class TestSubTaskResult:
    def test_defaults(self) -> None:
        sr = SubTaskResult(sub_task_id="st1", status=TransitionSignal.SUCCESS)
        assert sr.output_files == {}
        assert sr.validation_results == []
        assert sr.digest == ""
        assert sr.error is None

    def test_round_trip(self) -> None:
        sr = SubTaskResult(
            sub_task_id="st1",
            status=TransitionSignal.SUCCESS,
            output_files={"a.py": "code"},
            digest="Created schema module.",
        )
        rebuilt = SubTaskResult.model_validate_json(sr.model_dump_json())
        assert rebuilt == sr


class TestStepResultWithSubTaskResults:
    def test_sub_task_results_default_empty(self) -> None:
        sr = StepResult(step_id="s1", status=TransitionSignal.SUCCESS)
        assert sr.sub_task_results == []

    def test_sub_task_results_populated(self) -> None:
        st_result = SubTaskResult(sub_task_id="st1", status=TransitionSignal.SUCCESS)
        sr = StepResult(
            step_id="s1",
            status=TransitionSignal.SUCCESS,
            sub_task_results=[st_result],
        )
        assert len(sr.sub_task_results) == 1


class TestSubTaskInput:
    def test_creation_and_defaults(self) -> None:
        st = SubTask(sub_task_id="st1", description="d", target_files=["a.py"])
        inp = SubTaskInput(
            parent_task_id="t1",
            parent_description="Build an API.",
            sub_task=st,
            repo_root="/repo",
            parent_branch="forge/t1",
        )
        assert inp.max_attempts == 2
        assert inp.parent_description == "Build an API."

    def test_round_trip(self) -> None:
        st = SubTask(sub_task_id="st1", description="d", target_files=["a.py"])
        inp = SubTaskInput(
            parent_task_id="t1",
            parent_description="Build an API.",
            sub_task=st,
            repo_root="/repo",
            parent_branch="forge/t1",
            max_attempts=3,
        )
        rebuilt = SubTaskInput.model_validate_json(inp.model_dump_json())
        assert rebuilt == inp


class TestWriteFilesInput:
    def test_creation(self) -> None:
        inp = WriteFilesInput(
            task_id="t1",
            worktree_path="/wt",
            files={"a.py": "code", "b.py": "more code"},
        )
        assert inp.files == {"a.py": "code", "b.py": "more code"}


class TestAssembleSubTaskContextInput:
    def test_creation(self) -> None:
        st = SubTask(sub_task_id="st1", description="d", target_files=["a.py"])
        inp = AssembleSubTaskContextInput(
            parent_task_id="t1",
            parent_description="Build an API.",
            sub_task=st,
            worktree_path="/wt",
        )
        assert inp.parent_task_id == "t1"
        assert inp.worktree_path == "/wt"


class TestForgeTaskInputMaxSubTaskAttempts:
    def test_default(self) -> None:
        td = TaskDefinition(task_id="t", description="d")
        inp = ForgeTaskInput(task=td, repo_root="/repo")
        assert inp.max_sub_task_attempts == 2

    def test_override(self) -> None:
        td = TaskDefinition(task_id="t", description="d")
        inp = ForgeTaskInput(task=td, repo_root="/repo", max_sub_task_attempts=3)
        assert inp.max_sub_task_attempts == 3
