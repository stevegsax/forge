"""Tests for forge.workflows — ForgeTaskWorkflow with mocked activities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from forge.models import (
    AssembleContextInput,
    AssembledContext,
    AssembleStepContextInput,
    AssembleSubTaskContextInput,
    CommitChangesInput,
    CommitChangesOutput,
    CreateWorktreeInput,
    CreateWorktreeOutput,
    FileOutput,
    ForgeTaskInput,
    LLMCallResult,
    LLMResponse,
    Plan,
    PlanCallResult,
    PlannerInput,
    PlanStep,
    RemoveWorktreeInput,
    ResetWorktreeInput,
    SubTask,
    SubTaskInput,
    SubTaskResult,
    TaskDefinition,
    TaskResult,
    TransitionInput,
    TransitionSignal,
    ValidateOutputInput,
    ValidationResult,
    WriteFilesInput,
    WriteOutputInput,
    WriteResult,
)
from forge.workflows import FORGE_TASK_QUEUE, ForgeSubTaskWorkflow, ForgeTaskWorkflow

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TASK = TaskDefinition(
    task_id="test-task",
    description="Write a hello module.",
    target_files=["hello.py"],
)

_FORGE_INPUT = ForgeTaskInput(
    task=_TASK,
    repo_root="/tmp/repo",
    max_attempts=2,
)

_LLM_RESPONSE = LLMResponse(
    files=[FileOutput(file_path="hello.py", content="print('hello')\n")],
    explanation="Created hello module.",
)


# ---------------------------------------------------------------------------
# Mock activities — registered by name to match workflow string references
# ---------------------------------------------------------------------------

# Mutable state shared across mock activities within a single test.
# Each test gets a fresh worker so there's no cross-test contamination.
_call_log: list[str] = []
_attempt_counter: int = 0
_transition_sequence: list[str] = []


def _reset_mock_state(
    transitions: list[str] | None = None,
) -> None:
    global _attempt_counter
    _call_log.clear()
    _attempt_counter = 0
    _transition_sequence.clear()
    if transitions:
        _transition_sequence.extend(transitions)


@activity.defn(name="create_worktree_activity")
async def mock_create_worktree(input: CreateWorktreeInput) -> CreateWorktreeOutput:
    _call_log.append("create_worktree")
    return CreateWorktreeOutput(
        worktree_path=f"/tmp/repo/.forge-worktrees/{input.task_id}",
        branch_name=f"forge/{input.task_id}",
    )


@activity.defn(name="remove_worktree_activity")
async def mock_remove_worktree(input: RemoveWorktreeInput) -> None:
    _call_log.append("remove_worktree")


@activity.defn(name="commit_changes_activity")
async def mock_commit_changes(input: CommitChangesInput) -> CommitChangesOutput:
    _call_log.append(f"commit:{input.status}")
    return CommitChangesOutput(commit_sha="a" * 40)


@activity.defn(name="assemble_context")
async def mock_assemble_context(input: AssembleContextInput) -> AssembledContext:
    _call_log.append("assemble_context")
    return AssembledContext(
        task_id=input.task.task_id,
        system_prompt="system prompt",
        user_prompt="user prompt",
    )


@activity.defn(name="call_llm")
async def mock_call_llm(context: AssembledContext) -> LLMCallResult:
    _call_log.append("call_llm")
    return LLMCallResult(
        task_id=context.task_id,
        response=_LLM_RESPONSE,
        model_name="mock-model",
        input_tokens=100,
        output_tokens=50,
        latency_ms=200.0,
    )


@activity.defn(name="write_output")
async def mock_write_output(input: WriteOutputInput) -> WriteResult:
    _call_log.append("write_output")
    return WriteResult(
        task_id=input.llm_result.task_id,
        files_written=["hello.py"],
    )


@activity.defn(name="validate_output")
async def mock_validate_output(input: ValidateOutputInput) -> list[ValidationResult]:
    _call_log.append("validate_output")
    return [ValidationResult(check_name="ruff_lint", passed=True, summary="ruff_lint passed")]


@activity.defn(name="evaluate_transition")
async def mock_evaluate_transition(input: TransitionInput) -> str:
    global _attempt_counter
    _call_log.append("evaluate_transition")
    _attempt_counter += 1

    if _transition_sequence:
        return _transition_sequence.pop(0)
    return TransitionSignal.SUCCESS.value


# All mock activities in registration order
_MOCK_ACTIVITIES = [
    mock_create_worktree,
    mock_remove_worktree,
    mock_commit_changes,
    mock_assemble_context,
    mock_call_llm,
    mock_write_output,
    mock_validate_output,
    mock_evaluate_transition,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def env():
    from temporalio.testing import WorkflowEnvironment

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env:
        yield env


async def _run_workflow(
    env: WorkflowEnvironment,
    input: ForgeTaskInput = _FORGE_INPUT,
) -> TaskResult:
    """Helper to run the workflow with mock activities."""
    async with Worker(
        env.client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=[ForgeTaskWorkflow],
        activities=_MOCK_ACTIVITIES,
    ):
        result = await env.client.execute_workflow(
            ForgeTaskWorkflow.run,
            input,
            id=f"test-{input.task.task_id}",
            task_queue=FORGE_TASK_QUEUE,
        )
        return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSuccessPath:
    @pytest.mark.asyncio
    async def test_returns_success_status(self, env: WorkflowEnvironment) -> None:
        _reset_mock_state(transitions=[TransitionSignal.SUCCESS.value])
        result = await _run_workflow(env)
        assert result.status == TransitionSignal.SUCCESS

    @pytest.mark.asyncio
    async def test_commits_with_success(self, env: WorkflowEnvironment) -> None:
        _reset_mock_state(transitions=[TransitionSignal.SUCCESS.value])
        await _run_workflow(env)
        assert "commit:success" in _call_log

    @pytest.mark.asyncio
    async def test_output_files_collected(self, env: WorkflowEnvironment) -> None:
        _reset_mock_state(transitions=[TransitionSignal.SUCCESS.value])
        result = await _run_workflow(env)
        assert result.output_files == {"hello.py": "print('hello')\n"}

    @pytest.mark.asyncio
    async def test_worktree_metadata(self, env: WorkflowEnvironment) -> None:
        _reset_mock_state(transitions=[TransitionSignal.SUCCESS.value])
        result = await _run_workflow(env)
        assert result.worktree_path == "/tmp/repo/.forge-worktrees/test-task"
        assert result.worktree_branch == "forge/test-task"

    @pytest.mark.asyncio
    async def test_validation_results_populated(self, env: WorkflowEnvironment) -> None:
        _reset_mock_state(transitions=[TransitionSignal.SUCCESS.value])
        result = await _run_workflow(env)
        assert len(result.validation_results) == 1
        assert result.validation_results[0].passed is True


class TestRetryOnValidationFailure:
    @pytest.mark.asyncio
    async def test_retry_then_success(self, env: WorkflowEnvironment) -> None:
        _reset_mock_state(
            transitions=[
                TransitionSignal.FAILURE_RETRYABLE.value,
                TransitionSignal.SUCCESS.value,
            ]
        )
        result = await _run_workflow(env)
        assert result.status == TransitionSignal.SUCCESS

    @pytest.mark.asyncio
    async def test_worktree_removed_after_retry(self, env: WorkflowEnvironment) -> None:
        _reset_mock_state(
            transitions=[
                TransitionSignal.FAILURE_RETRYABLE.value,
                TransitionSignal.SUCCESS.value,
            ]
        )
        await _run_workflow(env)
        assert "remove_worktree" in _call_log

    @pytest.mark.asyncio
    async def test_creates_fresh_worktree_for_second_attempt(
        self, env: WorkflowEnvironment
    ) -> None:
        _reset_mock_state(
            transitions=[
                TransitionSignal.FAILURE_RETRYABLE.value,
                TransitionSignal.SUCCESS.value,
            ]
        )
        await _run_workflow(env)
        create_count = _call_log.count("create_worktree")
        assert create_count == 2


class TestTerminalFailure:
    @pytest.mark.asyncio
    async def test_terminal_failure_status(self, env: WorkflowEnvironment) -> None:
        _reset_mock_state(
            transitions=[
                TransitionSignal.FAILURE_TERMINAL.value,
            ]
        )
        result = await _run_workflow(env)
        assert result.status == TransitionSignal.FAILURE_TERMINAL

    @pytest.mark.asyncio
    async def test_commits_with_failure(self, env: WorkflowEnvironment) -> None:
        _reset_mock_state(
            transitions=[
                TransitionSignal.FAILURE_TERMINAL.value,
            ]
        )
        await _run_workflow(env)
        assert "commit:failure" in _call_log

    @pytest.mark.asyncio
    async def test_error_populated(self, env: WorkflowEnvironment) -> None:
        _reset_mock_state(
            transitions=[
                TransitionSignal.FAILURE_TERMINAL.value,
            ]
        )
        result = await _run_workflow(env)
        # Mock validate_output returns all-passed, but the transition was forced
        # to FAILURE_TERMINAL. Error is empty because no validations failed.
        # This tests the error-joining logic doesn't crash on no failures.
        assert result.error == ""

    @pytest.mark.asyncio
    async def test_both_attempts_fail(self, env: WorkflowEnvironment) -> None:
        _reset_mock_state(
            transitions=[
                TransitionSignal.FAILURE_RETRYABLE.value,
                TransitionSignal.FAILURE_TERMINAL.value,
            ]
        )
        result = await _run_workflow(env)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert "remove_worktree" in _call_log
        assert "commit:failure" in _call_log

    @pytest.mark.asyncio
    async def test_worktree_metadata_on_failure(self, env: WorkflowEnvironment) -> None:
        _reset_mock_state(
            transitions=[
                TransitionSignal.FAILURE_TERMINAL.value,
            ]
        )
        result = await _run_workflow(env)
        assert result.worktree_path is not None
        assert result.worktree_branch is not None


# ===========================================================================
# Phase 2: Planned workflow tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Mock activities for planning
# ---------------------------------------------------------------------------

_PLAN = Plan(
    task_id="test-task",
    steps=[
        PlanStep(step_id="step-1", description="Create models.", target_files=["models.py"]),
        PlanStep(step_id="step-2", description="Create API.", target_files=["api.py"]),
    ],
    explanation="Split into models and API layers.",
)

_PLAN_CALL_LOG: list[str] = []
_PLAN_TRANSITION_SEQUENCE: list[str] = []
_PLAN_LLM_CALL_COUNT: int = 0


def _reset_plan_mock_state(
    transitions: list[str] | None = None,
) -> None:
    global _PLAN_LLM_CALL_COUNT
    _PLAN_CALL_LOG.clear()
    _PLAN_TRANSITION_SEQUENCE.clear()
    _PLAN_LLM_CALL_COUNT = 0
    if transitions:
        _PLAN_TRANSITION_SEQUENCE.extend(transitions)


@activity.defn(name="assemble_planner_context")
async def mock_assemble_planner_context(input: AssembleContextInput) -> PlannerInput:
    _PLAN_CALL_LOG.append("assemble_planner_context")
    return PlannerInput(
        task_id=input.task.task_id,
        system_prompt="planner system prompt",
        user_prompt="planner user prompt",
    )


@activity.defn(name="call_planner")
async def mock_call_planner(input: PlannerInput) -> PlanCallResult:
    _PLAN_CALL_LOG.append("call_planner")
    return PlanCallResult(
        task_id=input.task_id,
        plan=_PLAN,
        model_name="mock-planner",
        input_tokens=300,
        output_tokens=150,
        latency_ms=500.0,
    )


@activity.defn(name="assemble_step_context")
async def mock_assemble_step_context(input: AssembleStepContextInput) -> AssembledContext:
    _PLAN_CALL_LOG.append(f"assemble_step_context:{input.step.step_id}")
    return AssembledContext(
        task_id=input.task.task_id,
        system_prompt=f"step system prompt for {input.step.step_id}",
        user_prompt=f"step user prompt for {input.step.step_id}",
    )


@activity.defn(name="reset_worktree_activity")
async def mock_reset_worktree(input: ResetWorktreeInput) -> None:
    _PLAN_CALL_LOG.append("reset_worktree")


# Step-level LLM mock that returns different code per step
@activity.defn(name="call_llm")
async def mock_plan_call_llm(context: AssembledContext) -> LLMCallResult:
    global _PLAN_LLM_CALL_COUNT
    _PLAN_LLM_CALL_COUNT += 1
    _PLAN_CALL_LOG.append(f"call_llm:{_PLAN_LLM_CALL_COUNT}")

    # Return different files per call
    if "step-1" in context.system_prompt:
        files = [FileOutput(file_path="models.py", content="class Model: pass\n")]
    else:
        files = [FileOutput(file_path="api.py", content="def endpoint(): pass\n")]

    return LLMCallResult(
        task_id=context.task_id,
        response=LLMResponse(files=files, explanation=f"LLM call #{_PLAN_LLM_CALL_COUNT}"),
        model_name="mock-model",
        input_tokens=100,
        output_tokens=50,
        latency_ms=200.0,
    )


@activity.defn(name="write_output")
async def mock_plan_write_output(input: WriteOutputInput) -> WriteResult:
    _PLAN_CALL_LOG.append("write_output")
    return WriteResult(
        task_id=input.llm_result.task_id,
        files_written=[f.file_path for f in input.llm_result.response.files],
    )


@activity.defn(name="validate_output")
async def mock_plan_validate_output(input: ValidateOutputInput) -> list[ValidationResult]:
    _PLAN_CALL_LOG.append("validate_output")
    return [ValidationResult(check_name="ruff_lint", passed=True, summary="ruff_lint passed")]


@activity.defn(name="evaluate_transition")
async def mock_plan_evaluate_transition(input: TransitionInput) -> str:
    _PLAN_CALL_LOG.append("evaluate_transition")
    if _PLAN_TRANSITION_SEQUENCE:
        return _PLAN_TRANSITION_SEQUENCE.pop(0)
    return TransitionSignal.SUCCESS.value


@activity.defn(name="commit_changes_activity")
async def mock_plan_commit_changes(input: CommitChangesInput) -> CommitChangesOutput:
    msg = input.message or input.status
    _PLAN_CALL_LOG.append(f"commit:{msg}")
    return CommitChangesOutput(commit_sha="b" * 40)


@activity.defn(name="create_worktree_activity")
async def mock_plan_create_worktree(input: CreateWorktreeInput) -> CreateWorktreeOutput:
    _PLAN_CALL_LOG.append("create_worktree")
    return CreateWorktreeOutput(
        worktree_path=f"/tmp/repo/.forge-worktrees/{input.task_id}",
        branch_name=f"forge/{input.task_id}",
    )


# Activities list for planned workflow tests
_PLAN_MOCK_ACTIVITIES = [
    mock_plan_create_worktree,
    mock_assemble_planner_context,
    mock_call_planner,
    mock_assemble_step_context,
    mock_plan_call_llm,
    mock_plan_write_output,
    mock_plan_validate_output,
    mock_plan_evaluate_transition,
    mock_plan_commit_changes,
    mock_reset_worktree,
]

_PLANNED_TASK = TaskDefinition(
    task_id="planned-task",
    description="Build a REST API with models and routes.",
)

_PLANNED_INPUT = ForgeTaskInput(
    task=_PLANNED_TASK,
    repo_root="/tmp/repo",
    plan=True,
    max_step_attempts=2,
)


async def _run_planned_workflow(
    env: WorkflowEnvironment,
    input: ForgeTaskInput = _PLANNED_INPUT,
) -> TaskResult:
    """Helper to run the planned workflow with mock activities."""
    async with Worker(
        env.client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=[ForgeTaskWorkflow],
        activities=_PLAN_MOCK_ACTIVITIES,
    ):
        return await env.client.execute_workflow(
            ForgeTaskWorkflow.run,
            input,
            id=f"test-planned-{input.task.task_id}",
            task_queue=FORGE_TASK_QUEUE,
        )


# ---------------------------------------------------------------------------
# Tests — planned workflow success
# ---------------------------------------------------------------------------


class TestPlannedWorkflowSuccess:
    """Two-step plan, both steps succeed."""

    @pytest.mark.asyncio
    async def test_returns_success(self, env: WorkflowEnvironment) -> None:
        _reset_plan_mock_state(
            transitions=[TransitionSignal.SUCCESS.value, TransitionSignal.SUCCESS.value]
        )
        result = await _run_planned_workflow(env)
        assert result.status == TransitionSignal.SUCCESS

    @pytest.mark.asyncio
    async def test_plan_populated(self, env: WorkflowEnvironment) -> None:
        _reset_plan_mock_state(
            transitions=[TransitionSignal.SUCCESS.value, TransitionSignal.SUCCESS.value]
        )
        result = await _run_planned_workflow(env)
        assert result.plan is not None
        assert len(result.plan.steps) == 2

    @pytest.mark.asyncio
    async def test_step_results_populated(self, env: WorkflowEnvironment) -> None:
        _reset_plan_mock_state(
            transitions=[TransitionSignal.SUCCESS.value, TransitionSignal.SUCCESS.value]
        )
        result = await _run_planned_workflow(env)
        assert len(result.step_results) == 2
        assert all(sr.status == TransitionSignal.SUCCESS for sr in result.step_results)
        assert result.step_results[0].step_id == "step-1"
        assert result.step_results[1].step_id == "step-2"

    @pytest.mark.asyncio
    async def test_two_commits(self, env: WorkflowEnvironment) -> None:
        _reset_plan_mock_state(
            transitions=[TransitionSignal.SUCCESS.value, TransitionSignal.SUCCESS.value]
        )
        await _run_planned_workflow(env)
        commit_entries = [e for e in _PLAN_CALL_LOG if e.startswith("commit:")]
        assert len(commit_entries) == 2

    @pytest.mark.asyncio
    async def test_worktree_created_once(self, env: WorkflowEnvironment) -> None:
        _reset_plan_mock_state(
            transitions=[TransitionSignal.SUCCESS.value, TransitionSignal.SUCCESS.value]
        )
        await _run_planned_workflow(env)
        assert _PLAN_CALL_LOG.count("create_worktree") == 1

    @pytest.mark.asyncio
    async def test_output_files_accumulated(self, env: WorkflowEnvironment) -> None:
        _reset_plan_mock_state(
            transitions=[TransitionSignal.SUCCESS.value, TransitionSignal.SUCCESS.value]
        )
        result = await _run_planned_workflow(env)
        assert "models.py" in result.output_files
        assert "api.py" in result.output_files

    @pytest.mark.asyncio
    async def test_step_commit_shas(self, env: WorkflowEnvironment) -> None:
        _reset_plan_mock_state(
            transitions=[TransitionSignal.SUCCESS.value, TransitionSignal.SUCCESS.value]
        )
        result = await _run_planned_workflow(env)
        for sr in result.step_results:
            assert sr.commit_sha is not None
            assert len(sr.commit_sha) == 40


# ---------------------------------------------------------------------------
# Tests — planned workflow step retry
# ---------------------------------------------------------------------------


class TestPlannedWorkflowStepRetry:
    """Step 1 succeeds, step 2 fails then succeeds on retry."""

    @pytest.mark.asyncio
    async def test_retry_then_success(self, env: WorkflowEnvironment) -> None:
        _reset_plan_mock_state(
            transitions=[
                TransitionSignal.SUCCESS.value,  # step 1
                TransitionSignal.FAILURE_RETRYABLE.value,  # step 2, attempt 1
                TransitionSignal.SUCCESS.value,  # step 2, attempt 2
            ]
        )
        result = await _run_planned_workflow(env)
        assert result.status == TransitionSignal.SUCCESS

    @pytest.mark.asyncio
    async def test_reset_worktree_on_retry(self, env: WorkflowEnvironment) -> None:
        _reset_plan_mock_state(
            transitions=[
                TransitionSignal.SUCCESS.value,
                TransitionSignal.FAILURE_RETRYABLE.value,
                TransitionSignal.SUCCESS.value,
            ]
        )
        await _run_planned_workflow(env)
        assert "reset_worktree" in _PLAN_CALL_LOG

    @pytest.mark.asyncio
    async def test_two_step_results_on_retry(self, env: WorkflowEnvironment) -> None:
        _reset_plan_mock_state(
            transitions=[
                TransitionSignal.SUCCESS.value,
                TransitionSignal.FAILURE_RETRYABLE.value,
                TransitionSignal.SUCCESS.value,
            ]
        )
        result = await _run_planned_workflow(env)
        assert len(result.step_results) == 2
        assert result.step_results[0].status == TransitionSignal.SUCCESS
        assert result.step_results[1].status == TransitionSignal.SUCCESS


# ---------------------------------------------------------------------------
# Tests — planned workflow step failure
# ---------------------------------------------------------------------------


class TestPlannedWorkflowStepFailure:
    """Step 1 succeeds, step 2 fails terminally."""

    @pytest.mark.asyncio
    async def test_terminal_failure(self, env: WorkflowEnvironment) -> None:
        _reset_plan_mock_state(
            transitions=[
                TransitionSignal.SUCCESS.value,  # step 1
                TransitionSignal.FAILURE_TERMINAL.value,  # step 2
            ]
        )
        result = await _run_planned_workflow(env)
        assert result.status == TransitionSignal.FAILURE_TERMINAL

    @pytest.mark.asyncio
    async def test_step_results_show_failure(self, env: WorkflowEnvironment) -> None:
        _reset_plan_mock_state(
            transitions=[
                TransitionSignal.SUCCESS.value,
                TransitionSignal.FAILURE_TERMINAL.value,
            ]
        )
        result = await _run_planned_workflow(env)
        assert len(result.step_results) == 2
        assert result.step_results[0].status == TransitionSignal.SUCCESS
        assert result.step_results[1].status == TransitionSignal.FAILURE_TERMINAL

    @pytest.mark.asyncio
    async def test_step1_commit_preserved(self, env: WorkflowEnvironment) -> None:
        _reset_plan_mock_state(
            transitions=[
                TransitionSignal.SUCCESS.value,
                TransitionSignal.FAILURE_TERMINAL.value,
            ]
        )
        result = await _run_planned_workflow(env)
        assert result.step_results[0].commit_sha is not None

    @pytest.mark.asyncio
    async def test_error_references_step(self, env: WorkflowEnvironment) -> None:
        _reset_plan_mock_state(
            transitions=[
                TransitionSignal.SUCCESS.value,
                TransitionSignal.FAILURE_TERMINAL.value,
            ]
        )
        result = await _run_planned_workflow(env)
        assert result.error is not None
        assert "step-2" in result.error

    @pytest.mark.asyncio
    async def test_plan_in_result(self, env: WorkflowEnvironment) -> None:
        _reset_plan_mock_state(
            transitions=[
                TransitionSignal.SUCCESS.value,
                TransitionSignal.FAILURE_TERMINAL.value,
            ]
        )
        result = await _run_planned_workflow(env)
        assert result.plan is not None


# ===========================================================================
# Phase 3: Sub-task workflow tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Mock activities for sub-task workflow
# ---------------------------------------------------------------------------

_SUBTASK_CALL_LOG: list[str] = []
_SUBTASK_TRANSITION_SEQUENCE: list[str] = []


def _reset_subtask_mock_state(transitions: list[str] | None = None) -> None:
    _SUBTASK_CALL_LOG.clear()
    _SUBTASK_TRANSITION_SEQUENCE.clear()
    if transitions:
        _SUBTASK_TRANSITION_SEQUENCE.extend(transitions)


@activity.defn(name="create_worktree_activity")
async def mock_subtask_create_worktree(input: CreateWorktreeInput) -> CreateWorktreeOutput:
    _SUBTASK_CALL_LOG.append(f"create_worktree:{input.task_id}")
    return CreateWorktreeOutput(
        worktree_path=f"/tmp/repo/.forge-worktrees/{input.task_id}",
        branch_name=f"forge/{input.task_id}",
    )


@activity.defn(name="remove_worktree_activity")
async def mock_subtask_remove_worktree(input: RemoveWorktreeInput) -> None:
    _SUBTASK_CALL_LOG.append(f"remove_worktree:{input.task_id}")


@activity.defn(name="assemble_sub_task_context")
async def mock_assemble_sub_task_context(
    input: AssembleSubTaskContextInput,
) -> AssembledContext:
    _SUBTASK_CALL_LOG.append(f"assemble_sub_task_context:{input.sub_task.sub_task_id}")
    return AssembledContext(
        task_id=input.parent_task_id,
        system_prompt=f"sub-task system prompt for {input.sub_task.sub_task_id}",
        user_prompt=f"sub-task user prompt for {input.sub_task.sub_task_id}",
    )


@activity.defn(name="call_llm")
async def mock_subtask_call_llm(context: AssembledContext) -> LLMCallResult:
    _SUBTASK_CALL_LOG.append("call_llm")
    # Determine which sub-task by looking at the prompt
    if "st1" in context.system_prompt:
        files = [FileOutput(file_path="schema.py", content="# schema\n")]
    else:
        files = [FileOutput(file_path="routes.py", content="# routes\n")]
    return LLMCallResult(
        task_id=context.task_id,
        response=LLMResponse(files=files, explanation="Sub-task output."),
        model_name="mock-model",
        input_tokens=50,
        output_tokens=25,
        latency_ms=100.0,
    )


@activity.defn(name="write_output")
async def mock_subtask_write_output(input: WriteOutputInput) -> WriteResult:
    _SUBTASK_CALL_LOG.append("write_output")
    return WriteResult(
        task_id=input.llm_result.task_id,
        files_written=[f.file_path for f in input.llm_result.response.files],
    )


@activity.defn(name="validate_output")
async def mock_subtask_validate_output(input: ValidateOutputInput) -> list[ValidationResult]:
    _SUBTASK_CALL_LOG.append("validate_output")
    return [ValidationResult(check_name="ruff_lint", passed=True, summary="passed")]


@activity.defn(name="evaluate_transition")
async def mock_subtask_evaluate_transition(input: TransitionInput) -> str:
    _SUBTASK_CALL_LOG.append("evaluate_transition")
    if _SUBTASK_TRANSITION_SEQUENCE:
        return _SUBTASK_TRANSITION_SEQUENCE.pop(0)
    return TransitionSignal.SUCCESS.value


_SUBTASK_MOCK_ACTIVITIES = [
    mock_subtask_create_worktree,
    mock_subtask_remove_worktree,
    mock_assemble_sub_task_context,
    mock_subtask_call_llm,
    mock_subtask_write_output,
    mock_subtask_validate_output,
    mock_subtask_evaluate_transition,
]


async def _run_subtask_workflow(
    env: WorkflowEnvironment,
    input: SubTaskInput,
) -> SubTaskResult:
    """Helper to run the sub-task workflow with mock activities."""
    async with Worker(
        env.client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=[ForgeSubTaskWorkflow],
        activities=_SUBTASK_MOCK_ACTIVITIES,
    ):
        return await env.client.execute_workflow(
            ForgeSubTaskWorkflow.run,
            input,
            id=f"test-subtask-{input.sub_task.sub_task_id}",
            task_queue=FORGE_TASK_QUEUE,
        )


# ---------------------------------------------------------------------------
# Tests — sub-task workflow
# ---------------------------------------------------------------------------


class TestSubTaskWorkflow:
    @pytest.fixture
    def sub_task_input(self) -> SubTaskInput:
        return SubTaskInput(
            parent_task_id="parent-task",
            parent_description="Build an API.",
            sub_task=SubTask(
                sub_task_id="st1",
                description="Analyze schema.",
                target_files=["schema.py"],
            ),
            repo_root="/tmp/repo",
            parent_branch="forge/parent-task",
            max_attempts=2,
        )

    @pytest.mark.asyncio
    async def test_success(self, env: WorkflowEnvironment, sub_task_input: SubTaskInput) -> None:
        _reset_subtask_mock_state(transitions=[TransitionSignal.SUCCESS.value])
        result = await _run_subtask_workflow(env, sub_task_input)
        assert result.status == TransitionSignal.SUCCESS
        assert result.sub_task_id == "st1"
        assert "schema.py" in result.output_files
        assert result.digest == "Sub-task output."

    @pytest.mark.asyncio
    async def test_worktree_created_and_removed(
        self, env: WorkflowEnvironment, sub_task_input: SubTaskInput
    ) -> None:
        _reset_subtask_mock_state(transitions=[TransitionSignal.SUCCESS.value])
        await _run_subtask_workflow(env, sub_task_input)
        assert any("create_worktree:" in e for e in _SUBTASK_CALL_LOG)
        assert any("remove_worktree:" in e for e in _SUBTASK_CALL_LOG)

    @pytest.mark.asyncio
    async def test_retry_then_success(
        self, env: WorkflowEnvironment, sub_task_input: SubTaskInput
    ) -> None:
        _reset_subtask_mock_state(
            transitions=[
                TransitionSignal.FAILURE_RETRYABLE.value,
                TransitionSignal.SUCCESS.value,
            ]
        )
        result = await _run_subtask_workflow(env, sub_task_input)
        assert result.status == TransitionSignal.SUCCESS
        # Should have created worktree twice (remove after retry, create again)
        create_count = sum(1 for e in _SUBTASK_CALL_LOG if e.startswith("create_worktree:"))
        assert create_count == 2

    @pytest.mark.asyncio
    async def test_terminal_failure(
        self, env: WorkflowEnvironment, sub_task_input: SubTaskInput
    ) -> None:
        _reset_subtask_mock_state(transitions=[TransitionSignal.FAILURE_TERMINAL.value])
        result = await _run_subtask_workflow(env, sub_task_input)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.sub_task_id == "st1"
        # Worktree should still be removed on failure
        assert any("remove_worktree:" in e for e in _SUBTASK_CALL_LOG)


# ===========================================================================
# Phase 3: Fan-out step tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Mock activities for fan-out tests (parent + child)
# ---------------------------------------------------------------------------

_FANOUT_CALL_LOG: list[str] = []
_FANOUT_STEP_TRANSITIONS: list[str] = []
_FANOUT_SUBTASK_TRANSITIONS: list[str] = []
_FANOUT_SUBTASK_LLM_RESPONSES: list[LLMResponse] = []


def _reset_fanout_mock_state(
    step_transitions: list[str] | None = None,
    subtask_transitions: list[str] | None = None,
    subtask_responses: list[LLMResponse] | None = None,
) -> None:
    _FANOUT_CALL_LOG.clear()
    _FANOUT_STEP_TRANSITIONS.clear()
    _FANOUT_SUBTASK_TRANSITIONS.clear()
    _FANOUT_SUBTASK_LLM_RESPONSES.clear()
    if step_transitions:
        _FANOUT_STEP_TRANSITIONS.extend(step_transitions)
    if subtask_transitions:
        _FANOUT_SUBTASK_TRANSITIONS.extend(subtask_transitions)
    if subtask_responses:
        _FANOUT_SUBTASK_LLM_RESPONSES.extend(subtask_responses)


@activity.defn(name="create_worktree_activity")
async def mock_fanout_create_worktree(input: CreateWorktreeInput) -> CreateWorktreeOutput:
    _FANOUT_CALL_LOG.append(f"create_worktree:{input.task_id}")
    return CreateWorktreeOutput(
        worktree_path=f"/tmp/repo/.forge-worktrees/{input.task_id}",
        branch_name=f"forge/{input.task_id}",
    )


@activity.defn(name="remove_worktree_activity")
async def mock_fanout_remove_worktree(input: RemoveWorktreeInput) -> None:
    _FANOUT_CALL_LOG.append(f"remove_worktree:{input.task_id}")


@activity.defn(name="assemble_planner_context")
async def mock_fanout_assemble_planner_context(input: AssembleContextInput) -> PlannerInput:
    _FANOUT_CALL_LOG.append("assemble_planner_context")
    return PlannerInput(
        task_id=input.task.task_id,
        system_prompt="planner system prompt",
        user_prompt="planner user prompt",
    )


@activity.defn(name="call_planner")
async def mock_fanout_call_planner(input: PlannerInput) -> PlanCallResult:
    _FANOUT_CALL_LOG.append("call_planner")
    plan = Plan(
        task_id=input.task_id,
        steps=[
            PlanStep(
                step_id="fan-step",
                description="Fan-out step.",
                target_files=[],
                sub_tasks=[
                    SubTask(
                        sub_task_id="st1",
                        description="Create schema.",
                        target_files=["schema.py"],
                    ),
                    SubTask(
                        sub_task_id="st2",
                        description="Create routes.",
                        target_files=["routes.py"],
                    ),
                ],
            ),
        ],
        explanation="Single fan-out step.",
    )
    return PlanCallResult(
        task_id=input.task_id,
        plan=plan,
        model_name="mock-planner",
        input_tokens=300,
        output_tokens=150,
        latency_ms=500.0,
    )


@activity.defn(name="assemble_sub_task_context")
async def mock_fanout_assemble_sub_task_context(
    input: AssembleSubTaskContextInput,
) -> AssembledContext:
    _FANOUT_CALL_LOG.append(f"assemble_sub_task_context:{input.sub_task.sub_task_id}")
    return AssembledContext(
        task_id=input.parent_task_id,
        system_prompt=f"sub-task prompt for {input.sub_task.sub_task_id}",
        user_prompt=f"execute {input.sub_task.sub_task_id}",
    )


@activity.defn(name="call_llm")
async def mock_fanout_call_llm(context: AssembledContext) -> LLMCallResult:
    _FANOUT_CALL_LOG.append("call_llm")
    if _FANOUT_SUBTASK_LLM_RESPONSES:
        response = _FANOUT_SUBTASK_LLM_RESPONSES.pop(0)
    elif "st1" in context.system_prompt:
        response = LLMResponse(
            files=[FileOutput(file_path="schema.py", content="# schema\n")],
            explanation="Created schema.",
        )
    else:
        response = LLMResponse(
            files=[FileOutput(file_path="routes.py", content="# routes\n")],
            explanation="Created routes.",
        )
    return LLMCallResult(
        task_id=context.task_id,
        response=response,
        model_name="mock-model",
        input_tokens=50,
        output_tokens=25,
        latency_ms=100.0,
    )


@activity.defn(name="write_output")
async def mock_fanout_write_output(input: WriteOutputInput) -> WriteResult:
    _FANOUT_CALL_LOG.append("write_output")
    return WriteResult(
        task_id=input.llm_result.task_id,
        files_written=[f.file_path for f in input.llm_result.response.files],
    )


@activity.defn(name="write_files")
async def mock_fanout_write_files(input: WriteFilesInput) -> WriteResult:
    _FANOUT_CALL_LOG.append(f"write_files:{len(input.files)}")
    return WriteResult(
        task_id=input.task_id,
        files_written=list(input.files.keys()),
    )


@activity.defn(name="validate_output")
async def mock_fanout_validate_output(input: ValidateOutputInput) -> list[ValidationResult]:
    _FANOUT_CALL_LOG.append("validate_output")
    return [ValidationResult(check_name="ruff_lint", passed=True, summary="passed")]


@activity.defn(name="evaluate_transition")
async def mock_fanout_evaluate_transition(input: TransitionInput) -> str:
    _FANOUT_CALL_LOG.append("evaluate_transition")
    # Use step transitions for parent validation, subtask transitions for children
    if _FANOUT_SUBTASK_TRANSITIONS:
        return _FANOUT_SUBTASK_TRANSITIONS.pop(0)
    if _FANOUT_STEP_TRANSITIONS:
        return _FANOUT_STEP_TRANSITIONS.pop(0)
    return TransitionSignal.SUCCESS.value


@activity.defn(name="commit_changes_activity")
async def mock_fanout_commit_changes(input: CommitChangesInput) -> CommitChangesOutput:
    msg = input.message or input.status
    _FANOUT_CALL_LOG.append(f"commit:{msg}")
    return CommitChangesOutput(commit_sha="c" * 40)


@activity.defn(name="reset_worktree_activity")
async def mock_fanout_reset_worktree(input: ResetWorktreeInput) -> None:
    _FANOUT_CALL_LOG.append("reset_worktree")


_FANOUT_MOCK_ACTIVITIES = [
    mock_fanout_create_worktree,
    mock_fanout_remove_worktree,
    mock_fanout_assemble_planner_context,
    mock_fanout_call_planner,
    mock_fanout_assemble_sub_task_context,
    mock_fanout_call_llm,
    mock_fanout_write_output,
    mock_fanout_write_files,
    mock_fanout_validate_output,
    mock_fanout_evaluate_transition,
    mock_fanout_commit_changes,
    mock_fanout_reset_worktree,
]


_FANOUT_TASK = TaskDefinition(
    task_id="fanout-task",
    description="Build schema and routes in parallel.",
)

_FANOUT_INPUT = ForgeTaskInput(
    task=_FANOUT_TASK,
    repo_root="/tmp/repo",
    plan=True,
    max_sub_task_attempts=2,
)


async def _run_fanout_workflow(
    env: WorkflowEnvironment,
    input: ForgeTaskInput = _FANOUT_INPUT,
) -> TaskResult:
    """Helper to run the fan-out workflow with mock activities."""
    async with Worker(
        env.client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=[ForgeTaskWorkflow, ForgeSubTaskWorkflow],
        activities=_FANOUT_MOCK_ACTIVITIES,
    ):
        return await env.client.execute_workflow(
            ForgeTaskWorkflow.run,
            input,
            id=f"test-fanout-{input.task.task_id}",
            task_queue=FORGE_TASK_QUEUE,
        )


# ---------------------------------------------------------------------------
# Tests — fan-out success
# ---------------------------------------------------------------------------


class TestFanOutStep:
    """Fan-out step with two sub-tasks, both succeed."""

    @pytest.mark.asyncio
    async def test_all_children_succeed(self, env: WorkflowEnvironment) -> None:
        _reset_fanout_mock_state()
        result = await _run_fanout_workflow(env)
        assert result.status == TransitionSignal.SUCCESS

    @pytest.mark.asyncio
    async def test_step_results_populated(self, env: WorkflowEnvironment) -> None:
        _reset_fanout_mock_state()
        result = await _run_fanout_workflow(env)
        assert len(result.step_results) == 1
        sr = result.step_results[0]
        assert sr.step_id == "fan-step"
        assert sr.status == TransitionSignal.SUCCESS

    @pytest.mark.asyncio
    async def test_sub_task_results_populated(self, env: WorkflowEnvironment) -> None:
        _reset_fanout_mock_state()
        result = await _run_fanout_workflow(env)
        sr = result.step_results[0]
        assert len(sr.sub_task_results) == 2
        ids = {r.sub_task_id for r in sr.sub_task_results}
        assert ids == {"st1", "st2"}

    @pytest.mark.asyncio
    async def test_merged_output_files(self, env: WorkflowEnvironment) -> None:
        _reset_fanout_mock_state()
        result = await _run_fanout_workflow(env)
        assert "schema.py" in result.output_files
        assert "routes.py" in result.output_files

    @pytest.mark.asyncio
    async def test_write_files_called(self, env: WorkflowEnvironment) -> None:
        _reset_fanout_mock_state()
        await _run_fanout_workflow(env)
        assert any(e.startswith("write_files:") for e in _FANOUT_CALL_LOG)

    @pytest.mark.asyncio
    async def test_commit_with_fan_out_message(self, env: WorkflowEnvironment) -> None:
        _reset_fanout_mock_state()
        await _run_fanout_workflow(env)
        commits = [e for e in _FANOUT_CALL_LOG if e.startswith("commit:")]
        assert any("fan-out gather" in c for c in commits)


# ---------------------------------------------------------------------------
# Tests — fan-out child failure
# ---------------------------------------------------------------------------


class TestFanOutChildFailure:
    """One child fails terminally → fan-out step fails."""

    @pytest.mark.asyncio
    async def test_one_child_fails(self, env: WorkflowEnvironment) -> None:
        _reset_fanout_mock_state(
            subtask_transitions=[
                TransitionSignal.SUCCESS.value,  # st1
                TransitionSignal.FAILURE_TERMINAL.value,  # st2
            ]
        )
        result = await _run_fanout_workflow(env)
        assert result.status == TransitionSignal.FAILURE_TERMINAL

    @pytest.mark.asyncio
    async def test_error_references_sub_task(self, env: WorkflowEnvironment) -> None:
        _reset_fanout_mock_state(
            subtask_transitions=[
                TransitionSignal.SUCCESS.value,
                TransitionSignal.FAILURE_TERMINAL.value,
            ]
        )
        result = await _run_fanout_workflow(env)
        assert result.error is not None
        assert "fan-out failed" in result.error


# ---------------------------------------------------------------------------
# Tests — fan-out file conflict
# ---------------------------------------------------------------------------


class TestFanOutFileConflict:
    """Two sub-tasks produce the same file → conflict detected."""

    @pytest.mark.asyncio
    async def test_file_conflict_detected(self, env: WorkflowEnvironment) -> None:
        # Both sub-tasks return the same file path
        _reset_fanout_mock_state(
            subtask_responses=[
                LLMResponse(
                    files=[FileOutput(file_path="conflict.py", content="# from st1\n")],
                    explanation="st1 output",
                ),
                LLMResponse(
                    files=[FileOutput(file_path="conflict.py", content="# from st2\n")],
                    explanation="st2 output",
                ),
            ]
        )
        result = await _run_fanout_workflow(env)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.error is not None
        assert "File conflict" in result.error


# ---------------------------------------------------------------------------
# Tests — mixed plan (sequential + fan-out + sequential)
# ---------------------------------------------------------------------------


@activity.defn(name="call_planner")
async def mock_mixed_call_planner(input: PlannerInput) -> PlanCallResult:
    """Planner returns a mix of sequential and fan-out steps."""
    plan = Plan(
        task_id=input.task_id,
        steps=[
            PlanStep(
                step_id="seq-1",
                description="Create models.",
                target_files=["models.py"],
            ),
            PlanStep(
                step_id="fan-step",
                description="Fan-out step.",
                target_files=[],
                sub_tasks=[
                    SubTask(
                        sub_task_id="st1",
                        description="Create schema.",
                        target_files=["schema.py"],
                    ),
                    SubTask(
                        sub_task_id="st2",
                        description="Create routes.",
                        target_files=["routes.py"],
                    ),
                ],
            ),
            PlanStep(
                step_id="seq-2",
                description="Create tests.",
                target_files=["tests.py"],
            ),
        ],
        explanation="Mixed plan.",
    )
    return PlanCallResult(
        task_id=input.task_id,
        plan=plan,
        model_name="mock-planner",
        input_tokens=300,
        output_tokens=150,
        latency_ms=500.0,
    )


# Step-level LLM mock for mixed plan
_MIXED_LLM_CALL_COUNT = 0


@activity.defn(name="call_llm")
async def mock_mixed_call_llm(context: AssembledContext) -> LLMCallResult:
    global _MIXED_LLM_CALL_COUNT
    _MIXED_LLM_CALL_COUNT += 1

    # Determine what file to return based on prompt content
    if "st1" in context.system_prompt:
        files = [FileOutput(file_path="schema.py", content="# schema\n")]
    elif "st2" in context.system_prompt:
        files = [FileOutput(file_path="routes.py", content="# routes\n")]
    elif "seq-1" in context.user_prompt or "models" in context.system_prompt:
        files = [FileOutput(file_path="models.py", content="# models\n")]
    else:
        files = [FileOutput(file_path="tests.py", content="# tests\n")]

    return LLMCallResult(
        task_id=context.task_id,
        response=LLMResponse(files=files, explanation=f"Call #{_MIXED_LLM_CALL_COUNT}"),
        model_name="mock-model",
        input_tokens=50,
        output_tokens=25,
        latency_ms=100.0,
    )


@activity.defn(name="assemble_step_context")
async def mock_mixed_assemble_step_context(
    input: AssembleStepContextInput,
) -> AssembledContext:
    return AssembledContext(
        task_id=input.task.task_id,
        system_prompt=f"step prompt for {input.step.step_id}",
        user_prompt=f"execute {input.step.step_id}",
    )


_MIXED_MOCK_ACTIVITIES = [
    mock_fanout_create_worktree,
    mock_fanout_remove_worktree,
    mock_fanout_assemble_planner_context,
    mock_mixed_call_planner,
    mock_mixed_assemble_step_context,
    mock_fanout_assemble_sub_task_context,
    mock_mixed_call_llm,
    mock_fanout_write_output,
    mock_fanout_write_files,
    mock_fanout_validate_output,
    mock_fanout_evaluate_transition,
    mock_fanout_commit_changes,
    mock_fanout_reset_worktree,
]


class TestMixedPlan:
    """Sequential step → fan-out step → sequential step."""

    @pytest.mark.asyncio
    async def test_mixed_plan_succeeds(self, env: WorkflowEnvironment) -> None:
        global _MIXED_LLM_CALL_COUNT
        _MIXED_LLM_CALL_COUNT = 0
        _reset_fanout_mock_state()

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeTaskWorkflow, ForgeSubTaskWorkflow],
            activities=_MIXED_MOCK_ACTIVITIES,
        ):
            result = await env.client.execute_workflow(
                ForgeTaskWorkflow.run,
                _FANOUT_INPUT,
                id="test-mixed-plan",
                task_queue=FORGE_TASK_QUEUE,
            )
        assert result.status == TransitionSignal.SUCCESS
        assert len(result.step_results) == 3
        assert result.step_results[0].step_id == "seq-1"
        assert result.step_results[1].step_id == "fan-step"
        assert result.step_results[2].step_id == "seq-2"


# ---------------------------------------------------------------------------
# Tests — backward compat: Phase 2 plans without sub_tasks
# ---------------------------------------------------------------------------


class TestPlannedBackwardCompat:
    """Existing Phase 2 plans (no sub_tasks) still work."""

    @pytest.mark.asyncio
    async def test_no_sub_tasks_works(self, env: WorkflowEnvironment) -> None:
        _reset_plan_mock_state(
            transitions=[TransitionSignal.SUCCESS.value, TransitionSignal.SUCCESS.value]
        )
        result = await _run_planned_workflow(env)
        assert result.status == TransitionSignal.SUCCESS
        assert len(result.step_results) == 2
