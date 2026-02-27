"""Tests for forge.models — data model serialization and defaults."""

from __future__ import annotations

import pytest

from forge.models import (
    AssembledContext,
    AssembleStepContextInput,
    AssembleSubTaskContextInput,
    CommitChangesInput,
    ContextConfig,
    ContextStats,
    ExtractionCallResult,
    FileEdit,
    FileOutput,
    ForgeTaskInput,
    LLMCallResult,
    LLMResponse,
    LLMStats,
    Plan,
    PlanCallResult,
    PlannerInput,
    PlanStep,
    ResetWorktreeInput,
    SearchReplaceEdit,
    StepResult,
    SubTask,
    SubTaskInput,
    SubTaskResult,
    TaskDefinition,
    TaskDomain,
    TaskResult,
    TransitionSignal,
    WriteFilesInput,
    WriteResult,
    build_llm_stats,
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
# TaskDomain enum
# ---------------------------------------------------------------------------


class TestTaskDomain:
    def test_enum_values(self) -> None:
        assert TaskDomain.CODE_GENERATION == "code_generation"
        assert TaskDomain.RESEARCH == "research"
        assert TaskDomain.CODE_REVIEW == "code_review"
        assert TaskDomain.DOCUMENTATION == "documentation"

    def test_five_members(self) -> None:
        assert len(TaskDomain) == 5


class TestTaskDefinitionDomain:
    def test_default_domain(self) -> None:
        td = TaskDefinition(task_id="t", description="d")
        assert td.domain == TaskDomain.CODE_GENERATION

    def test_explicit_domain(self) -> None:
        td = TaskDefinition(task_id="t", description="d", domain=TaskDomain.RESEARCH)
        assert td.domain == TaskDomain.RESEARCH

    def test_backward_compat_no_domain(self) -> None:
        """Old JSON without domain field still deserializes."""
        data = '{"task_id": "t", "description": "d"}'
        td = TaskDefinition.model_validate_json(data)
        assert td.domain == TaskDomain.CODE_GENERATION

    def test_round_trip(self) -> None:
        td = TaskDefinition(task_id="t", description="d", domain=TaskDomain.DOCUMENTATION)
        rebuilt = TaskDefinition.model_validate_json(td.model_dump_json())
        assert rebuilt.domain == TaskDomain.DOCUMENTATION


class TestSubTaskInputDomain:
    def test_default_domain(self) -> None:
        st = SubTask(sub_task_id="st1", description="d", target_files=["a.py"])
        inp = SubTaskInput(
            parent_task_id="t",
            parent_description="desc",
            sub_task=st,
            repo_root="/repo",
            parent_branch="forge/t",
        )
        assert inp.domain == TaskDomain.CODE_GENERATION

    def test_explicit_domain(self) -> None:
        st = SubTask(sub_task_id="st1", description="d", target_files=["a.py"])
        inp = SubTaskInput(
            parent_task_id="t",
            parent_description="desc",
            sub_task=st,
            repo_root="/repo",
            parent_branch="forge/t",
            domain=TaskDomain.RESEARCH,
        )
        assert inp.domain == TaskDomain.RESEARCH


class TestAssembleSubTaskContextInputDomain:
    def test_default_domain(self) -> None:
        st = SubTask(sub_task_id="st1", description="d", target_files=["a.py"])
        inp = AssembleSubTaskContextInput(
            parent_task_id="t",
            parent_description="desc",
            sub_task=st,
            worktree_path="/wt",
        )
        assert inp.domain == TaskDomain.CODE_GENERATION

    def test_explicit_domain(self) -> None:
        st = SubTask(sub_task_id="st1", description="d", target_files=["a.py"])
        inp = AssembleSubTaskContextInput(
            parent_task_id="t",
            parent_description="desc",
            sub_task=st,
            worktree_path="/wt",
            domain=TaskDomain.RESEARCH,
        )
        assert inp.domain == TaskDomain.RESEARCH


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
            task_id=td.task_id,
            task_description=td.description,
            context_config=td.context,
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


# ---------------------------------------------------------------------------
# Phase 4: Context discovery models
# ---------------------------------------------------------------------------


class TestContextConfig:
    def test_defaults(self) -> None:
        config = ContextConfig()
        assert config.auto_discover is True
        assert config.token_budget == 100_000
        assert config.output_reserve == 16_000
        assert config.max_import_depth == 2
        assert config.include_repo_map is True
        assert config.repo_map_tokens == 2048
        assert config.package_name is None

    def test_round_trip(self) -> None:
        config = ContextConfig(auto_discover=False, token_budget=50_000, package_name="myapp")
        rebuilt = ContextConfig.model_validate_json(config.model_dump_json())
        assert rebuilt == config


class TestContextStats:
    def test_defaults(self) -> None:
        stats = ContextStats()
        assert stats.files_discovered == 0
        assert stats.files_included_full == 0
        assert stats.files_included_signatures == 0
        assert stats.files_truncated == 0
        assert stats.total_estimated_tokens == 0
        assert stats.budget_utilization == 0.0
        assert stats.repo_map_tokens == 0

    def test_round_trip(self) -> None:
        stats = ContextStats(
            files_discovered=10,
            files_included_full=5,
            files_included_signatures=3,
            files_truncated=2,
            total_estimated_tokens=5000,
            budget_utilization=0.75,
            repo_map_tokens=512,
        )
        rebuilt = ContextStats.model_validate_json(stats.model_dump_json())
        assert rebuilt == stats


class TestTaskDefinitionContextConfig:
    def test_default_context_config(self) -> None:
        td = TaskDefinition(task_id="t", description="d")
        assert td.context.auto_discover is True
        assert td.context.token_budget == 100_000

    def test_custom_context_config(self) -> None:
        config = ContextConfig(auto_discover=False, token_budget=50_000)
        td = TaskDefinition(task_id="t", description="d", context=config)
        assert td.context.auto_discover is False

    def test_backward_compat_no_context(self) -> None:
        """Old JSON without context field still deserializes."""
        data = '{"task_id": "t", "description": "d"}'
        td = TaskDefinition.model_validate_json(data)
        assert td.context.auto_discover is True

    def test_round_trip(self) -> None:
        config = ContextConfig(auto_discover=False, package_name="forge")
        td = TaskDefinition(task_id="t", description="d", context=config)
        rebuilt = TaskDefinition.model_validate_json(td.model_dump_json())
        assert rebuilt.context == config


class TestAssembledContextStats:
    def test_default_none(self) -> None:
        ctx = AssembledContext(task_id="t", system_prompt="sys", user_prompt="usr")
        assert ctx.context_stats is None

    def test_with_stats(self) -> None:
        stats = ContextStats(files_discovered=5, total_estimated_tokens=3000)
        ctx = AssembledContext(
            task_id="t",
            system_prompt="sys",
            user_prompt="usr",
            context_stats=stats,
        )
        assert ctx.context_stats is not None
        assert ctx.context_stats.files_discovered == 5

    def test_round_trip(self) -> None:
        stats = ContextStats(files_discovered=5)
        ctx = AssembledContext(
            task_id="t",
            system_prompt="sys",
            user_prompt="usr",
            context_stats=stats,
        )
        rebuilt = AssembledContext.model_validate_json(ctx.model_dump_json())
        assert rebuilt.context_stats == stats


# ---------------------------------------------------------------------------
# Phase 5: Observability store models
# ---------------------------------------------------------------------------


class TestLLMStats:
    def test_creation(self) -> None:
        stats = LLMStats(
            model_name="test-model",
            input_tokens=100,
            output_tokens=50,
            latency_ms=250.0,
        )
        assert stats.model_name == "test-model"
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50
        assert stats.latency_ms == 250.0

    def test_round_trip(self) -> None:
        stats = LLMStats(
            model_name="test-model",
            input_tokens=100,
            output_tokens=50,
            latency_ms=250.0,
        )
        rebuilt = LLMStats.model_validate_json(stats.model_dump_json())
        assert rebuilt == stats


class TestBuildLlmStats:
    def test_from_llm_call_result(self) -> None:
        result = LLMCallResult(
            task_id="t",
            response=LLMResponse(
                files=[FileOutput(file_path="a.py", content="pass")],
                explanation="Created.",
            ),
            model_name="test-model",
            input_tokens=100,
            output_tokens=50,
            latency_ms=250.0,
        )
        stats = build_llm_stats(result)
        assert stats.model_name == "test-model"
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50
        assert stats.latency_ms == 250.0


class TestBuildLlmStatsFromPlanCallResult:
    def test_from_plan_call_result(self) -> None:
        plan = Plan(
            task_id="t",
            steps=[PlanStep(step_id="s1", description="d", target_files=["a.py"])],
            explanation="test",
        )
        result = PlanCallResult(
            task_id="t",
            plan=plan,
            model_name="planner-model",
            input_tokens=200,
            output_tokens=100,
            latency_ms=500.0,
        )
        stats = build_llm_stats(result)
        assert stats.model_name == "planner-model"
        assert stats.input_tokens == 200
        assert stats.output_tokens == 100
        assert stats.latency_ms == 500.0


class TestTaskResultLLMStats:
    def test_default_none(self) -> None:
        result = TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        assert result.llm_stats is None
        assert result.planner_stats is None
        assert result.context_stats is None

    def test_with_stats(self) -> None:
        stats = LLMStats(model_name="m", input_tokens=1, output_tokens=2, latency_ms=3.0)
        result = TaskResult(
            task_id="t",
            status=TransitionSignal.SUCCESS,
            llm_stats=stats,
            planner_stats=stats,
            context_stats=ContextStats(files_discovered=5),
        )
        assert result.llm_stats == stats
        assert result.planner_stats == stats
        assert result.context_stats is not None

    def test_backward_compat(self) -> None:
        """Old results without llm_stats still deserialize."""
        data = '{"task_id": "t", "status": "success"}'
        result = TaskResult.model_validate_json(data)
        assert result.llm_stats is None


class TestStepResultLLMStats:
    def test_default_none(self) -> None:
        sr = StepResult(step_id="s1", status=TransitionSignal.SUCCESS)
        assert sr.llm_stats is None


class TestSubTaskResultLLMStats:
    def test_default_none(self) -> None:
        sr = SubTaskResult(sub_task_id="st1", status=TransitionSignal.SUCCESS)
        assert sr.llm_stats is None


class TestAssembledContextStepSubTaskId:
    def test_defaults_none(self) -> None:
        ctx = AssembledContext(task_id="t", system_prompt="sys", user_prompt="usr")
        assert ctx.step_id is None
        assert ctx.sub_task_id is None

    def test_with_ids(self) -> None:
        ctx = AssembledContext(
            task_id="t",
            system_prompt="sys",
            user_prompt="usr",
            step_id="step-1",
            sub_task_id="st-1",
        )
        assert ctx.step_id == "step-1"
        assert ctx.sub_task_id == "st-1"

    def test_round_trip(self) -> None:
        ctx = AssembledContext(
            task_id="t",
            system_prompt="sys",
            user_prompt="usr",
            step_id="step-1",
        )
        rebuilt = AssembledContext.model_validate_json(ctx.model_dump_json())
        assert rebuilt.step_id == "step-1"


# ---------------------------------------------------------------------------
# D50: SearchReplaceEdit, FileEdit, updated LLMResponse, WriteResult
# ---------------------------------------------------------------------------


class TestSearchReplaceEdit:
    def test_creation(self) -> None:
        edit = SearchReplaceEdit(search="old_func", replace="new_func")
        assert edit.search == "old_func"
        assert edit.replace == "new_func"

    def test_round_trip(self) -> None:
        edit = SearchReplaceEdit(search="old", replace="new")
        rebuilt = SearchReplaceEdit.model_validate_json(edit.model_dump_json())
        assert rebuilt == edit


class TestFileEdit:
    def test_creation(self) -> None:
        edit = FileEdit(
            file_path="src/main.py",
            edits=[SearchReplaceEdit(search="old", replace="new")],
        )
        assert edit.file_path == "src/main.py"
        assert len(edit.edits) == 1

    def test_round_trip(self) -> None:
        edit = FileEdit(
            file_path="src/main.py",
            edits=[
                SearchReplaceEdit(search="a", replace="b"),
                SearchReplaceEdit(search="c", replace="d"),
            ],
        )
        rebuilt = FileEdit.model_validate_json(edit.model_dump_json())
        assert rebuilt == edit


class TestLLMResponseEdits:
    def test_edits_defaults_to_empty(self) -> None:
        resp = LLMResponse(explanation="test")
        assert resp.edits == []
        assert resp.files == []

    def test_files_defaults_to_empty(self) -> None:
        resp = LLMResponse(
            edits=[
                FileEdit(
                    file_path="a.py",
                    edits=[SearchReplaceEdit(search="x", replace="y")],
                )
            ],
            explanation="edit only",
        )
        assert resp.files == []
        assert len(resp.edits) == 1

    def test_backward_compat_without_edits(self) -> None:
        """Old JSON payloads without edits field still deserialize."""
        data = '{"files": [{"file_path": "a.py", "content": "pass"}], "explanation": "test"}'
        resp = LLMResponse.model_validate_json(data)
        assert len(resp.files) == 1
        assert resp.edits == []

    def test_round_trip_with_both(self) -> None:
        resp = LLMResponse(
            files=[FileOutput(file_path="new.py", content="# new")],
            edits=[
                FileEdit(
                    file_path="old.py",
                    edits=[SearchReplaceEdit(search="old", replace="new")],
                )
            ],
            explanation="mixed",
        )
        rebuilt = LLMResponse.model_validate_json(resp.model_dump_json())
        assert rebuilt == resp


class TestWriteResultOutputFiles:
    def test_output_files_defaults_to_empty(self) -> None:
        wr = WriteResult(task_id="t", files_written=["a.py"])
        assert wr.output_files == {}

    def test_output_files_populated(self) -> None:
        wr = WriteResult(
            task_id="t",
            files_written=["a.py"],
            output_files={"a.py": "content"},
        )
        assert wr.output_files == {"a.py": "content"}

    def test_backward_compat_without_output_files(self) -> None:
        """Old JSON payloads without output_files still deserialize."""
        data = '{"task_id": "t", "files_written": ["a.py"]}'
        wr = WriteResult.model_validate_json(data)
        assert wr.output_files == {}


# ---------------------------------------------------------------------------
# Phase 9: Prompt caching — cache token fields
# ---------------------------------------------------------------------------


class TestCacheTokenFields:
    def test_llm_call_result_defaults(self) -> None:
        result = LLMCallResult(
            task_id="t",
            response=LLMResponse(
                files=[FileOutput(file_path="a.py", content="pass")],
                explanation="Done.",
            ),
            model_name="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200.0,
        )
        assert result.cache_creation_input_tokens == 0
        assert result.cache_read_input_tokens == 0

    def test_llm_call_result_with_cache(self) -> None:
        result = LLMCallResult(
            task_id="t",
            response=LLMResponse(explanation="Done."),
            model_name="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200.0,
            cache_creation_input_tokens=500,
            cache_read_input_tokens=1000,
        )
        assert result.cache_creation_input_tokens == 500
        assert result.cache_read_input_tokens == 1000

    def test_llm_stats_defaults(self) -> None:
        stats = LLMStats(
            model_name="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200.0,
        )
        assert stats.cache_creation_input_tokens == 0
        assert stats.cache_read_input_tokens == 0

    def test_llm_stats_with_cache(self) -> None:
        stats = LLMStats(
            model_name="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200.0,
            cache_creation_input_tokens=500,
            cache_read_input_tokens=1000,
        )
        assert stats.cache_creation_input_tokens == 500
        assert stats.cache_read_input_tokens == 1000

    def test_plan_call_result_defaults(self) -> None:
        plan = Plan(
            task_id="t",
            steps=[PlanStep(step_id="s1", description="d", target_files=["a.py"])],
            explanation="test",
        )
        result = PlanCallResult(
            task_id="t",
            plan=plan,
            model_name="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200.0,
        )
        assert result.cache_creation_input_tokens == 0
        assert result.cache_read_input_tokens == 0

    def test_extraction_call_result_defaults(self) -> None:
        from forge.models import ExtractionResult

        result = ExtractionCallResult(
            result=ExtractionResult(entries=[], summary="Nothing."),
            source_workflow_ids=["wf-1"],
            model_name="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200.0,
        )
        assert result.cache_creation_input_tokens == 0
        assert result.cache_read_input_tokens == 0

    def test_backward_compat_without_cache_fields(self) -> None:
        """Old JSON without cache fields still deserializes."""
        data = (
            '{"task_id": "t", "response": {"files": [], "edits": [], '
            '"explanation": "test"}, "model_name": "m", "input_tokens": 1, '
            '"output_tokens": 2, "latency_ms": 3.0}'
        )
        result = LLMCallResult.model_validate_json(data)
        assert result.cache_creation_input_tokens == 0
        assert result.cache_read_input_tokens == 0


class TestBuildLlmStatsCache:
    def test_propagates_cache_fields(self) -> None:
        result = LLMCallResult(
            task_id="t",
            response=LLMResponse(explanation="Done."),
            model_name="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200.0,
            cache_creation_input_tokens=500,
            cache_read_input_tokens=1000,
        )
        stats = build_llm_stats(result)
        assert stats.cache_creation_input_tokens == 500
        assert stats.cache_read_input_tokens == 1000

    def test_zero_cache_by_default(self) -> None:
        result = LLMCallResult(
            task_id="t",
            response=LLMResponse(explanation="Done."),
            model_name="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200.0,
        )
        stats = build_llm_stats(result)
        assert stats.cache_creation_input_tokens == 0
        assert stats.cache_read_input_tokens == 0


class TestBuildLlmStatsFromPlanCallResultCache:
    def test_propagates_cache_fields(self) -> None:
        plan = Plan(
            task_id="t",
            steps=[PlanStep(step_id="s1", description="d", target_files=["a.py"])],
            explanation="test",
        )
        result = PlanCallResult(
            task_id="t",
            plan=plan,
            model_name="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200.0,
            cache_creation_input_tokens=300,
            cache_read_input_tokens=700,
        )
        stats = build_llm_stats(result)
        assert stats.cache_creation_input_tokens == 300
        assert stats.cache_read_input_tokens == 700
