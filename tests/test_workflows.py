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
    TaskDefinition,
    TaskResult,
    TransitionInput,
    TransitionSignal,
    ValidateOutputInput,
    ValidationResult,
    WriteOutputInput,
    WriteResult,
)
from forge.workflows import FORGE_TASK_QUEUE, ForgeTaskWorkflow

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
