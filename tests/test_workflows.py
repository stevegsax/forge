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
    AssembleSanityCheckContextInput,
    AssembleStepContextInput,
    AssembleSubTaskContextInput,
    BatchResult,
    BatchSubmitInput,
    BatchSubmitResult,
    CommitChangesInput,
    CommitChangesOutput,
    ConflictResolutionCallInput,
    ConflictResolutionCallResult,
    ConflictResolutionInput,
    CreateWorktreeInput,
    CreateWorktreeOutput,
    FileOutput,
    ForgeTaskInput,
    LLMCallResult,
    LLMResponse,
    ParsedLLMResponse,
    ParseResponseInput,
    Plan,
    PlanCallResult,
    PlannerInput,
    PlanStep,
    RemoveWorktreeInput,
    ResetWorktreeInput,
    SanityCheckCallResult,
    SanityCheckInput,
    SanityCheckResponse,
    SanityCheckVerdict,
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
    max_exploration_rounds=0,
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
    files = input.llm_result.response.files
    return WriteResult(
        task_id=input.llm_result.task_id,
        files_written=[f.file_path for f in files],
        output_files={f.file_path: f.content for f in files},
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
    files = input.llm_result.response.files
    return WriteResult(
        task_id=input.llm_result.task_id,
        files_written=[f.file_path for f in files],
        output_files={f.file_path: f.content for f in files},
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
    max_exploration_rounds=0,
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
    files = input.llm_result.response.files
    return WriteResult(
        task_id=input.llm_result.task_id,
        files_written=[f.file_path for f in files],
        output_files={f.file_path: f.content for f in files},
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
_FANOUT_CONFLICT_RESOLUTION_RESPONSES: list[ConflictResolutionCallResult] = []


def _reset_fanout_mock_state(
    step_transitions: list[str] | None = None,
    subtask_transitions: list[str] | None = None,
    subtask_responses: list[LLMResponse] | None = None,
    conflict_responses: list[ConflictResolutionCallResult] | None = None,
) -> None:
    _FANOUT_CALL_LOG.clear()
    _FANOUT_STEP_TRANSITIONS.clear()
    _FANOUT_SUBTASK_TRANSITIONS.clear()
    _FANOUT_SUBTASK_LLM_RESPONSES.clear()
    _FANOUT_CONFLICT_RESOLUTION_RESPONSES.clear()
    if step_transitions:
        _FANOUT_STEP_TRANSITIONS.extend(step_transitions)
    if subtask_transitions:
        _FANOUT_SUBTASK_TRANSITIONS.extend(subtask_transitions)
    if subtask_responses:
        _FANOUT_SUBTASK_LLM_RESPONSES.extend(subtask_responses)
    if conflict_responses:
        _FANOUT_CONFLICT_RESOLUTION_RESPONSES.extend(conflict_responses)


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
    files = input.llm_result.response.files
    return WriteResult(
        task_id=input.llm_result.task_id,
        files_written=[f.file_path for f in files],
        output_files={f.file_path: f.content for f in files},
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


@activity.defn(name="assemble_conflict_resolution_context")
async def mock_fanout_assemble_cr_context(
    input: ConflictResolutionInput,
) -> ConflictResolutionCallInput:
    _FANOUT_CALL_LOG.append("assemble_conflict_resolution_context")
    return ConflictResolutionCallInput(
        task_id=input.task_id,
        step_id=input.step_id,
        system_prompt="conflict resolution system prompt",
        user_prompt="conflict resolution user prompt",
    )


@activity.defn(name="call_conflict_resolution")
async def mock_fanout_call_conflict_resolution(
    input: ConflictResolutionCallInput,
) -> ConflictResolutionCallResult:
    _FANOUT_CALL_LOG.append("call_conflict_resolution")
    if _FANOUT_CONFLICT_RESOLUTION_RESPONSES:
        return _FANOUT_CONFLICT_RESOLUTION_RESPONSES.pop(0)
    return ConflictResolutionCallResult(
        task_id=input.task_id,
        resolved_files={},
        explanation="No conflicts resolved (default mock).",
        model_name="mock-reasoning",
        input_tokens=200,
        output_tokens=100,
        latency_ms=300.0,
    )


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
    mock_fanout_assemble_cr_context,
    mock_fanout_call_conflict_resolution,
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
    max_exploration_rounds=0,
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
    """Two sub-tasks produce the same file with resolve_conflicts=False → D27 terminal error."""

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
        no_resolve_input = ForgeTaskInput(
            task=_FANOUT_TASK,
            repo_root="/tmp/repo",
            plan=True,
            max_sub_task_attempts=2,
            max_exploration_rounds=0,
            resolve_conflicts=False,
        )
        result = await _run_fanout_workflow(env, no_resolve_input)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.error is not None
        assert "File conflict" in result.error


# ---------------------------------------------------------------------------
# Tests — fan-out conflict resolution
# ---------------------------------------------------------------------------


class TestFanOutConflictResolution:
    """Two sub-tasks produce same file, LLM resolves the conflict."""

    @pytest.mark.asyncio
    async def test_resolution_succeeds(self, env: WorkflowEnvironment) -> None:
        """Conflict is resolved, merged output passes validation, step succeeds."""
        _reset_fanout_mock_state(
            subtask_responses=[
                LLMResponse(
                    files=[
                        FileOutput(file_path="shared.py", content="# from st1\ndef foo(): pass\n")
                    ],
                    explanation="st1 output",
                ),
                LLMResponse(
                    files=[
                        FileOutput(file_path="shared.py", content="# from st2\ndef bar(): pass\n")
                    ],
                    explanation="st2 output",
                ),
            ],
            conflict_responses=[
                ConflictResolutionCallResult(
                    task_id="fanout-task",
                    resolved_files={"shared.py": "# merged\ndef foo(): pass\ndef bar(): pass\n"},
                    explanation="Combined both functions.",
                    model_name="mock-reasoning",
                    input_tokens=200,
                    output_tokens=100,
                    latency_ms=300.0,
                ),
            ],
        )
        result = await _run_fanout_workflow(env)
        assert result.status == TransitionSignal.SUCCESS
        assert "assemble_conflict_resolution_context" in _FANOUT_CALL_LOG
        assert "call_conflict_resolution" in _FANOUT_CALL_LOG
        sr = result.step_results[0]
        assert sr.conflict_resolution is not None
        assert "shared.py" in sr.output_files
        assert "merged" in sr.output_files["shared.py"]

    @pytest.mark.asyncio
    async def test_resolution_missing_path_fails(self, env: WorkflowEnvironment) -> None:
        """Resolution LLM omits a conflict path → step fails terminal."""
        _reset_fanout_mock_state(
            subtask_responses=[
                LLMResponse(
                    files=[FileOutput(file_path="shared.py", content="# from st1\n")],
                    explanation="st1 output",
                ),
                LLMResponse(
                    files=[FileOutput(file_path="shared.py", content="# from st2\n")],
                    explanation="st2 output",
                ),
            ],
            conflict_responses=[
                ConflictResolutionCallResult(
                    task_id="fanout-task",
                    resolved_files={},  # Missing shared.py!
                    explanation="Oops, forgot.",
                    model_name="mock-reasoning",
                    input_tokens=200,
                    output_tokens=100,
                    latency_ms=300.0,
                ),
            ],
        )
        result = await _run_fanout_workflow(env)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.error is not None
        assert "Conflict resolution incomplete" in result.error
        assert "shared.py" in result.error

    @pytest.mark.asyncio
    async def test_mixed_conflicting_and_non_conflicting(self, env: WorkflowEnvironment) -> None:
        """Sub-tasks produce one conflicting and two non-conflicting files."""
        _reset_fanout_mock_state(
            subtask_responses=[
                LLMResponse(
                    files=[
                        FileOutput(file_path="shared.py", content="# from st1\n"),
                        FileOutput(file_path="unique_a.py", content="# unique a\n"),
                    ],
                    explanation="st1 output",
                ),
                LLMResponse(
                    files=[
                        FileOutput(file_path="shared.py", content="# from st2\n"),
                        FileOutput(file_path="unique_b.py", content="# unique b\n"),
                    ],
                    explanation="st2 output",
                ),
            ],
            conflict_responses=[
                ConflictResolutionCallResult(
                    task_id="fanout-task",
                    resolved_files={"shared.py": "# merged shared\n"},
                    explanation="Merged shared.py.",
                    model_name="mock-reasoning",
                    input_tokens=200,
                    output_tokens=100,
                    latency_ms=300.0,
                ),
            ],
        )
        result = await _run_fanout_workflow(env)
        assert result.status == TransitionSignal.SUCCESS
        sr = result.step_results[0]
        assert sr.output_files["shared.py"] == "# merged shared\n"
        assert sr.output_files["unique_a.py"] == "# unique a\n"
        assert sr.output_files["unique_b.py"] == "# unique b\n"

    @pytest.mark.asyncio
    async def test_resolution_disabled_falls_back_to_terminal(
        self, env: WorkflowEnvironment
    ) -> None:
        """resolve_conflicts=False, falls back to D27 terminal error."""
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
            ],
        )
        no_resolve_input = ForgeTaskInput(
            task=_FANOUT_TASK,
            repo_root="/tmp/repo",
            plan=True,
            max_sub_task_attempts=2,
            max_exploration_rounds=0,
            resolve_conflicts=False,
        )
        result = await _run_fanout_workflow(env, no_resolve_input)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert "File conflict" in (result.error or "")
        # Conflict resolution activities should NOT be called
        assert "assemble_conflict_resolution_context" not in _FANOUT_CALL_LOG
        assert "call_conflict_resolution" not in _FANOUT_CALL_LOG

    @pytest.mark.asyncio
    async def test_validation_failure_after_resolution(self, env: WorkflowEnvironment) -> None:
        """Resolution succeeds but merged output fails validation → terminal error."""
        _reset_fanout_mock_state(
            subtask_responses=[
                LLMResponse(
                    files=[FileOutput(file_path="shared.py", content="# from st1\n")],
                    explanation="st1 output",
                ),
                LLMResponse(
                    files=[FileOutput(file_path="shared.py", content="# from st2\n")],
                    explanation="st2 output",
                ),
            ],
            conflict_responses=[
                ConflictResolutionCallResult(
                    task_id="fanout-task",
                    resolved_files={"shared.py": "# bad merge\n"},
                    explanation="Merged.",
                    model_name="mock-reasoning",
                    input_tokens=200,
                    output_tokens=100,
                    latency_ms=300.0,
                ),
            ],
            # Sub-tasks succeed (2 transitions for children), then parent
            # validation fails (1 transition for merged output).
            subtask_transitions=[
                TransitionSignal.SUCCESS.value,
                TransitionSignal.SUCCESS.value,
            ],
            step_transitions=[TransitionSignal.FAILURE_TERMINAL.value],
        )
        result = await _run_fanout_workflow(env)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.error is not None
        assert "Merged output validation failed" in result.error


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


# ===========================================================================
# Phase 8: Error-aware retry tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Mock activities that capture context assembly inputs
# ---------------------------------------------------------------------------

_P8_CALL_LOG: list[str] = []
_P8_TRANSITION_SEQUENCE: list[str] = []
_P8_ASSEMBLE_CONTEXT_INPUTS: list[AssembleContextInput] = []
_P8_VALIDATE_RESPONSES: list[list[ValidationResult]] = []


def _reset_p8_state(
    transitions: list[str] | None = None,
    validate_responses: list[list[ValidationResult]] | None = None,
) -> None:
    _P8_CALL_LOG.clear()
    _P8_TRANSITION_SEQUENCE.clear()
    _P8_ASSEMBLE_CONTEXT_INPUTS.clear()
    _P8_VALIDATE_RESPONSES.clear()
    if transitions:
        _P8_TRANSITION_SEQUENCE.extend(transitions)
    if validate_responses:
        _P8_VALIDATE_RESPONSES.extend(validate_responses)


@activity.defn(name="create_worktree_activity")
async def mock_p8_create_worktree(input: CreateWorktreeInput) -> CreateWorktreeOutput:
    _P8_CALL_LOG.append("create_worktree")
    return CreateWorktreeOutput(
        worktree_path=f"/tmp/repo/.forge-worktrees/{input.task_id}",
        branch_name=f"forge/{input.task_id}",
    )


@activity.defn(name="remove_worktree_activity")
async def mock_p8_remove_worktree(input: RemoveWorktreeInput) -> None:
    _P8_CALL_LOG.append("remove_worktree")


@activity.defn(name="commit_changes_activity")
async def mock_p8_commit(input: CommitChangesInput) -> CommitChangesOutput:
    _P8_CALL_LOG.append(f"commit:{input.status}")
    return CommitChangesOutput(commit_sha="d" * 40)


@activity.defn(name="assemble_context")
async def mock_p8_assemble_context(input: AssembleContextInput) -> AssembledContext:
    _P8_CALL_LOG.append("assemble_context")
    _P8_ASSEMBLE_CONTEXT_INPUTS.append(input)
    return AssembledContext(
        task_id=input.task.task_id,
        system_prompt="system prompt",
        user_prompt="user prompt",
    )


@activity.defn(name="call_llm")
async def mock_p8_call_llm(context: AssembledContext) -> LLMCallResult:
    _P8_CALL_LOG.append("call_llm")
    return LLMCallResult(
        task_id=context.task_id,
        response=LLMResponse(
            files=[FileOutput(file_path="hello.py", content="print('hello')\n")],
            explanation="output",
        ),
        model_name="mock-model",
        input_tokens=100,
        output_tokens=50,
        latency_ms=200.0,
    )


@activity.defn(name="write_output")
async def mock_p8_write_output(input: WriteOutputInput) -> WriteResult:
    _P8_CALL_LOG.append("write_output")
    files = input.llm_result.response.files
    return WriteResult(
        task_id=input.llm_result.task_id,
        files_written=[f.file_path for f in files],
        output_files={f.file_path: f.content for f in files},
    )


@activity.defn(name="validate_output")
async def mock_p8_validate_output(input: ValidateOutputInput) -> list[ValidationResult]:
    _P8_CALL_LOG.append("validate_output")
    if _P8_VALIDATE_RESPONSES:
        return _P8_VALIDATE_RESPONSES.pop(0)
    return [ValidationResult(check_name="ruff_lint", passed=True, summary="passed")]


@activity.defn(name="evaluate_transition")
async def mock_p8_evaluate_transition(input: TransitionInput) -> str:
    _P8_CALL_LOG.append("evaluate_transition")
    if _P8_TRANSITION_SEQUENCE:
        return _P8_TRANSITION_SEQUENCE.pop(0)
    return TransitionSignal.SUCCESS.value


_P8_MOCK_ACTIVITIES = [
    mock_p8_create_worktree,
    mock_p8_remove_worktree,
    mock_p8_commit,
    mock_p8_assemble_context,
    mock_p8_call_llm,
    mock_p8_write_output,
    mock_p8_validate_output,
    mock_p8_evaluate_transition,
]


class TestSingleStepErrorAwareRetry:
    """Phase 8: prior_errors are passed through single-step retry loop."""

    @pytest.mark.asyncio
    async def test_first_attempt_has_no_prior_errors(self, env: WorkflowEnvironment) -> None:
        _reset_p8_state(transitions=[TransitionSignal.SUCCESS.value])
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeTaskWorkflow],
            activities=_P8_MOCK_ACTIVITIES,
        ):
            await env.client.execute_workflow(
                ForgeTaskWorkflow.run,
                ForgeTaskInput(
                    task=_TASK,
                    repo_root="/tmp/repo",
                    max_attempts=2,
                    max_exploration_rounds=0,
                ),
                id="test-p8-first-attempt",
                task_queue=FORGE_TASK_QUEUE,
            )
        assert len(_P8_ASSEMBLE_CONTEXT_INPUTS) == 1
        first = _P8_ASSEMBLE_CONTEXT_INPUTS[0]
        assert first.prior_errors == []
        assert first.attempt == 1

    @pytest.mark.asyncio
    async def test_retry_passes_prior_errors(self, env: WorkflowEnvironment) -> None:
        lint_errors = [
            ValidationResult(
                check_name="ruff_lint",
                passed=False,
                summary="ruff_lint failed",
                details="hello.py:1:1: F401 unused import",
            )
        ]
        _reset_p8_state(
            transitions=[
                TransitionSignal.FAILURE_RETRYABLE.value,
                TransitionSignal.SUCCESS.value,
            ],
            validate_responses=[
                lint_errors,
                [ValidationResult(check_name="ruff_lint", passed=True, summary="passed")],
            ],
        )
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeTaskWorkflow],
            activities=_P8_MOCK_ACTIVITIES,
        ):
            result = await env.client.execute_workflow(
                ForgeTaskWorkflow.run,
                ForgeTaskInput(
                    task=_TASK,
                    repo_root="/tmp/repo",
                    max_attempts=2,
                    max_exploration_rounds=0,
                ),
                id="test-p8-retry-errors",
                task_queue=FORGE_TASK_QUEUE,
            )
        assert result.status == TransitionSignal.SUCCESS
        assert len(_P8_ASSEMBLE_CONTEXT_INPUTS) == 2

        # First attempt: no prior errors
        first = _P8_ASSEMBLE_CONTEXT_INPUTS[0]
        assert first.prior_errors == []
        assert first.attempt == 1

        # Second attempt: prior errors from first attempt
        second = _P8_ASSEMBLE_CONTEXT_INPUTS[1]
        assert len(second.prior_errors) == 1
        assert second.prior_errors[0].check_name == "ruff_lint"
        assert second.attempt == 2
        assert second.max_attempts == 2


# ---------------------------------------------------------------------------
# Phase 8: Planned step error-aware retry
# ---------------------------------------------------------------------------

_P8_STEP_CALL_LOG: list[str] = []
_P8_STEP_TRANSITION_SEQUENCE: list[str] = []
_P8_STEP_CONTEXT_INPUTS: list[AssembleStepContextInput] = []
_P8_STEP_VALIDATE_RESPONSES: list[list[ValidationResult]] = []


def _reset_p8_step_state(
    transitions: list[str] | None = None,
    validate_responses: list[list[ValidationResult]] | None = None,
) -> None:
    _P8_STEP_CALL_LOG.clear()
    _P8_STEP_TRANSITION_SEQUENCE.clear()
    _P8_STEP_CONTEXT_INPUTS.clear()
    _P8_STEP_VALIDATE_RESPONSES.clear()
    if transitions:
        _P8_STEP_TRANSITION_SEQUENCE.extend(transitions)
    if validate_responses:
        _P8_STEP_VALIDATE_RESPONSES.extend(validate_responses)


@activity.defn(name="create_worktree_activity")
async def mock_p8s_create_worktree(input: CreateWorktreeInput) -> CreateWorktreeOutput:
    _P8_STEP_CALL_LOG.append("create_worktree")
    return CreateWorktreeOutput(
        worktree_path=f"/tmp/repo/.forge-worktrees/{input.task_id}",
        branch_name=f"forge/{input.task_id}",
    )


@activity.defn(name="assemble_planner_context")
async def mock_p8s_assemble_planner_context(input: AssembleContextInput) -> PlannerInput:
    _P8_STEP_CALL_LOG.append("assemble_planner_context")
    return PlannerInput(
        task_id=input.task.task_id,
        system_prompt="planner prompt",
        user_prompt="planner user",
    )


@activity.defn(name="call_planner")
async def mock_p8s_call_planner(input: PlannerInput) -> PlanCallResult:
    _P8_STEP_CALL_LOG.append("call_planner")
    plan = Plan(
        task_id=input.task_id,
        steps=[PlanStep(step_id="step-1", description="Create.", target_files=["a.py"])],
        explanation="One step.",
    )
    return PlanCallResult(
        task_id=input.task_id,
        plan=plan,
        model_name="mock-planner",
        input_tokens=300,
        output_tokens=150,
        latency_ms=500.0,
    )


@activity.defn(name="assemble_step_context")
async def mock_p8s_assemble_step_context(input: AssembleStepContextInput) -> AssembledContext:
    _P8_STEP_CALL_LOG.append(f"assemble_step_context:{input.step.step_id}")
    _P8_STEP_CONTEXT_INPUTS.append(input)
    return AssembledContext(
        task_id=input.task.task_id,
        system_prompt=f"step prompt for {input.step.step_id}",
        user_prompt=f"step user for {input.step.step_id}",
    )


@activity.defn(name="call_llm")
async def mock_p8s_call_llm(context: AssembledContext) -> LLMCallResult:
    _P8_STEP_CALL_LOG.append("call_llm")
    return LLMCallResult(
        task_id=context.task_id,
        response=LLMResponse(
            files=[FileOutput(file_path="a.py", content="# code\n")],
            explanation="step output",
        ),
        model_name="mock-model",
        input_tokens=100,
        output_tokens=50,
        latency_ms=200.0,
    )


@activity.defn(name="write_output")
async def mock_p8s_write_output(input: WriteOutputInput) -> WriteResult:
    _P8_STEP_CALL_LOG.append("write_output")
    files = input.llm_result.response.files
    return WriteResult(
        task_id=input.llm_result.task_id,
        files_written=[f.file_path for f in files],
        output_files={f.file_path: f.content for f in files},
    )


@activity.defn(name="validate_output")
async def mock_p8s_validate_output(input: ValidateOutputInput) -> list[ValidationResult]:
    _P8_STEP_CALL_LOG.append("validate_output")
    if _P8_STEP_VALIDATE_RESPONSES:
        return _P8_STEP_VALIDATE_RESPONSES.pop(0)
    return [ValidationResult(check_name="ruff_lint", passed=True, summary="passed")]


@activity.defn(name="evaluate_transition")
async def mock_p8s_evaluate_transition(input: TransitionInput) -> str:
    _P8_STEP_CALL_LOG.append("evaluate_transition")
    if _P8_STEP_TRANSITION_SEQUENCE:
        return _P8_STEP_TRANSITION_SEQUENCE.pop(0)
    return TransitionSignal.SUCCESS.value


@activity.defn(name="commit_changes_activity")
async def mock_p8s_commit(input: CommitChangesInput) -> CommitChangesOutput:
    _P8_STEP_CALL_LOG.append(f"commit:{input.status}")
    return CommitChangesOutput(commit_sha="e" * 40)


@activity.defn(name="reset_worktree_activity")
async def mock_p8s_reset_worktree(input: ResetWorktreeInput) -> None:
    _P8_STEP_CALL_LOG.append("reset_worktree")


_P8_STEP_MOCK_ACTIVITIES = [
    mock_p8s_create_worktree,
    mock_p8s_assemble_planner_context,
    mock_p8s_call_planner,
    mock_p8s_assemble_step_context,
    mock_p8s_call_llm,
    mock_p8s_write_output,
    mock_p8s_validate_output,
    mock_p8s_evaluate_transition,
    mock_p8s_commit,
    mock_p8s_reset_worktree,
]


class TestPlannedStepErrorAwareRetry:
    """Phase 8: prior_errors are passed through planned step retry loop."""

    @pytest.mark.asyncio
    async def test_step_retry_passes_prior_errors(self, env: WorkflowEnvironment) -> None:
        lint_errors = [
            ValidationResult(
                check_name="ruff_format",
                passed=False,
                summary="ruff_format failed",
                details="a.py:10:1: formatting error",
            )
        ]
        _reset_p8_step_state(
            transitions=[
                TransitionSignal.FAILURE_RETRYABLE.value,
                TransitionSignal.SUCCESS.value,
            ],
            validate_responses=[
                lint_errors,
                [ValidationResult(check_name="ruff_format", passed=True, summary="passed")],
            ],
        )
        task = TaskDefinition(task_id="p8-step-task", description="Build.")
        input_data = ForgeTaskInput(
            task=task,
            repo_root="/tmp/repo",
            plan=True,
            max_step_attempts=2,
            max_exploration_rounds=0,
        )
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeTaskWorkflow],
            activities=_P8_STEP_MOCK_ACTIVITIES,
        ):
            result = await env.client.execute_workflow(
                ForgeTaskWorkflow.run,
                input_data,
                id="test-p8-step-retry",
                task_queue=FORGE_TASK_QUEUE,
            )
        assert result.status == TransitionSignal.SUCCESS
        assert len(_P8_STEP_CONTEXT_INPUTS) == 2

        # First attempt: no prior errors
        first = _P8_STEP_CONTEXT_INPUTS[0]
        assert first.prior_errors == []
        assert first.attempt == 1

        # Second attempt: errors from first
        second = _P8_STEP_CONTEXT_INPUTS[1]
        assert len(second.prior_errors) == 1
        assert second.prior_errors[0].check_name == "ruff_format"
        assert second.attempt == 2


# ---------------------------------------------------------------------------
# Phase 8: Sub-task error-aware retry
# ---------------------------------------------------------------------------

_P8_ST_CALL_LOG: list[str] = []
_P8_ST_TRANSITION_SEQUENCE: list[str] = []
_P8_ST_CONTEXT_INPUTS: list[AssembleSubTaskContextInput] = []
_P8_ST_VALIDATE_RESPONSES: list[list[ValidationResult]] = []


def _reset_p8_st_state(
    transitions: list[str] | None = None,
    validate_responses: list[list[ValidationResult]] | None = None,
) -> None:
    _P8_ST_CALL_LOG.clear()
    _P8_ST_TRANSITION_SEQUENCE.clear()
    _P8_ST_CONTEXT_INPUTS.clear()
    _P8_ST_VALIDATE_RESPONSES.clear()
    if transitions:
        _P8_ST_TRANSITION_SEQUENCE.extend(transitions)
    if validate_responses:
        _P8_ST_VALIDATE_RESPONSES.extend(validate_responses)


@activity.defn(name="create_worktree_activity")
async def mock_p8st_create_worktree(input: CreateWorktreeInput) -> CreateWorktreeOutput:
    _P8_ST_CALL_LOG.append(f"create_worktree:{input.task_id}")
    return CreateWorktreeOutput(
        worktree_path=f"/tmp/repo/.forge-worktrees/{input.task_id}",
        branch_name=f"forge/{input.task_id}",
    )


@activity.defn(name="remove_worktree_activity")
async def mock_p8st_remove_worktree(input: RemoveWorktreeInput) -> None:
    _P8_ST_CALL_LOG.append(f"remove_worktree:{input.task_id}")


@activity.defn(name="assemble_sub_task_context")
async def mock_p8st_assemble_sub_task_context(
    input: AssembleSubTaskContextInput,
) -> AssembledContext:
    _P8_ST_CALL_LOG.append(f"assemble_sub_task_context:{input.sub_task.sub_task_id}")
    _P8_ST_CONTEXT_INPUTS.append(input)
    return AssembledContext(
        task_id=input.parent_task_id,
        system_prompt=f"sub-task prompt for {input.sub_task.sub_task_id}",
        user_prompt=f"sub-task user for {input.sub_task.sub_task_id}",
    )


@activity.defn(name="call_llm")
async def mock_p8st_call_llm(context: AssembledContext) -> LLMCallResult:
    _P8_ST_CALL_LOG.append("call_llm")
    return LLMCallResult(
        task_id=context.task_id,
        response=LLMResponse(
            files=[FileOutput(file_path="schema.py", content="# schema\n")],
            explanation="sub-task output",
        ),
        model_name="mock-model",
        input_tokens=50,
        output_tokens=25,
        latency_ms=100.0,
    )


@activity.defn(name="write_output")
async def mock_p8st_write_output(input: WriteOutputInput) -> WriteResult:
    _P8_ST_CALL_LOG.append("write_output")
    files = input.llm_result.response.files
    return WriteResult(
        task_id=input.llm_result.task_id,
        files_written=[f.file_path for f in files],
        output_files={f.file_path: f.content for f in files},
    )


@activity.defn(name="validate_output")
async def mock_p8st_validate_output(input: ValidateOutputInput) -> list[ValidationResult]:
    _P8_ST_CALL_LOG.append("validate_output")
    if _P8_ST_VALIDATE_RESPONSES:
        return _P8_ST_VALIDATE_RESPONSES.pop(0)
    return [ValidationResult(check_name="ruff_lint", passed=True, summary="passed")]


@activity.defn(name="evaluate_transition")
async def mock_p8st_evaluate_transition(input: TransitionInput) -> str:
    _P8_ST_CALL_LOG.append("evaluate_transition")
    if _P8_ST_TRANSITION_SEQUENCE:
        return _P8_ST_TRANSITION_SEQUENCE.pop(0)
    return TransitionSignal.SUCCESS.value


_P8_ST_MOCK_ACTIVITIES = [
    mock_p8st_create_worktree,
    mock_p8st_remove_worktree,
    mock_p8st_assemble_sub_task_context,
    mock_p8st_call_llm,
    mock_p8st_write_output,
    mock_p8st_validate_output,
    mock_p8st_evaluate_transition,
]


class TestSubTaskErrorAwareRetry:
    """Phase 8: prior_errors are passed through sub-task retry loop."""

    @pytest.mark.asyncio
    async def test_subtask_retry_passes_prior_errors(self, env: WorkflowEnvironment) -> None:
        test_errors = [
            ValidationResult(
                check_name="tests",
                passed=False,
                summary="tests failed",
                details="FAILED test_schema.py::test_parse - AssertionError",
            )
        ]
        _reset_p8_st_state(
            transitions=[
                TransitionSignal.FAILURE_RETRYABLE.value,
                TransitionSignal.SUCCESS.value,
            ],
            validate_responses=[
                test_errors,
                [ValidationResult(check_name="tests", passed=True, summary="passed")],
            ],
        )
        st_input = SubTaskInput(
            parent_task_id="parent-task",
            parent_description="Build API.",
            sub_task=SubTask(
                sub_task_id="st1",
                description="Create schema.",
                target_files=["schema.py"],
            ),
            repo_root="/tmp/repo",
            parent_branch="forge/parent-task",
            max_attempts=2,
        )
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeSubTaskWorkflow],
            activities=_P8_ST_MOCK_ACTIVITIES,
        ):
            result = await env.client.execute_workflow(
                ForgeSubTaskWorkflow.run,
                st_input,
                id="test-p8-subtask-retry",
                task_queue=FORGE_TASK_QUEUE,
            )
        assert result.status == TransitionSignal.SUCCESS
        assert len(_P8_ST_CONTEXT_INPUTS) == 2

        # First attempt: no prior errors
        first = _P8_ST_CONTEXT_INPUTS[0]
        assert first.prior_errors == []
        assert first.attempt == 1

        # Second attempt: errors from first
        second = _P8_ST_CONTEXT_INPUTS[1]
        assert len(second.prior_errors) == 1
        assert second.prior_errors[0].check_name == "tests"
        assert second.attempt == 2
        assert second.max_attempts == 2


# ===========================================================================
# Recursive fan-out tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Mock activities for recursive fan-out tests
# ---------------------------------------------------------------------------

_RECURSIVE_CALL_LOG: list[str] = []
_RECURSIVE_TRANSITION_SEQUENCE: list[str] = []
_RECURSIVE_LLM_RESPONSES: list[LLMResponse] = []
_RECURSIVE_CONFLICT_RESPONSES: list[ConflictResolutionCallResult] = []


def _reset_recursive_mock_state(
    transitions: list[str] | None = None,
    llm_responses: list[LLMResponse] | None = None,
    conflict_responses: list[ConflictResolutionCallResult] | None = None,
) -> None:
    _RECURSIVE_CALL_LOG.clear()
    _RECURSIVE_TRANSITION_SEQUENCE.clear()
    _RECURSIVE_LLM_RESPONSES.clear()
    _RECURSIVE_CONFLICT_RESPONSES.clear()
    if transitions:
        _RECURSIVE_TRANSITION_SEQUENCE.extend(transitions)
    if llm_responses:
        _RECURSIVE_LLM_RESPONSES.extend(llm_responses)
    if conflict_responses:
        _RECURSIVE_CONFLICT_RESPONSES.extend(conflict_responses)


@activity.defn(name="create_worktree_activity")
async def mock_recursive_create_worktree(input: CreateWorktreeInput) -> CreateWorktreeOutput:
    _RECURSIVE_CALL_LOG.append(f"create_worktree:{input.task_id}")
    return CreateWorktreeOutput(
        worktree_path=f"/tmp/repo/.forge-worktrees/{input.task_id}",
        branch_name=f"forge/{input.task_id}",
    )


@activity.defn(name="remove_worktree_activity")
async def mock_recursive_remove_worktree(input: RemoveWorktreeInput) -> None:
    _RECURSIVE_CALL_LOG.append(f"remove_worktree:{input.task_id}")


@activity.defn(name="assemble_sub_task_context")
async def mock_recursive_assemble_sub_task_context(
    input: AssembleSubTaskContextInput,
) -> AssembledContext:
    _RECURSIVE_CALL_LOG.append(f"assemble_sub_task_context:{input.sub_task.sub_task_id}")
    return AssembledContext(
        task_id=input.parent_task_id,
        system_prompt=f"sub-task prompt for {input.sub_task.sub_task_id}",
        user_prompt=f"execute {input.sub_task.sub_task_id}",
    )


@activity.defn(name="call_llm")
async def mock_recursive_call_llm(context: AssembledContext) -> LLMCallResult:
    _RECURSIVE_CALL_LOG.append("call_llm")
    if _RECURSIVE_LLM_RESPONSES:
        response = _RECURSIVE_LLM_RESPONSES.pop(0)
    elif "gc1" in context.system_prompt:
        response = LLMResponse(
            files=[FileOutput(file_path="gc1.py", content="# gc1\n")],
            explanation="Grandchild 1 output.",
        )
    elif "gc2" in context.system_prompt:
        response = LLMResponse(
            files=[FileOutput(file_path="gc2.py", content="# gc2\n")],
            explanation="Grandchild 2 output.",
        )
    else:
        response = LLMResponse(
            files=[FileOutput(file_path="leaf.py", content="# leaf\n")],
            explanation="Leaf output.",
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
async def mock_recursive_write_output(input: WriteOutputInput) -> WriteResult:
    _RECURSIVE_CALL_LOG.append("write_output")
    files = input.llm_result.response.files
    return WriteResult(
        task_id=input.llm_result.task_id,
        files_written=[f.file_path for f in files],
        output_files={f.file_path: f.content for f in files},
    )


@activity.defn(name="write_files")
async def mock_recursive_write_files(input: WriteFilesInput) -> WriteResult:
    _RECURSIVE_CALL_LOG.append(f"write_files:{len(input.files)}")
    return WriteResult(
        task_id=input.task_id,
        files_written=list(input.files.keys()),
    )


@activity.defn(name="validate_output")
async def mock_recursive_validate_output(input: ValidateOutputInput) -> list[ValidationResult]:
    _RECURSIVE_CALL_LOG.append("validate_output")
    return [ValidationResult(check_name="ruff_lint", passed=True, summary="passed")]


@activity.defn(name="evaluate_transition")
async def mock_recursive_evaluate_transition(input: TransitionInput) -> str:
    _RECURSIVE_CALL_LOG.append("evaluate_transition")
    if _RECURSIVE_TRANSITION_SEQUENCE:
        return _RECURSIVE_TRANSITION_SEQUENCE.pop(0)
    return TransitionSignal.SUCCESS.value


@activity.defn(name="assemble_conflict_resolution_context")
async def mock_recursive_assemble_cr_context(
    input: ConflictResolutionInput,
) -> ConflictResolutionCallInput:
    _RECURSIVE_CALL_LOG.append("assemble_conflict_resolution_context")
    return ConflictResolutionCallInput(
        task_id=input.task_id,
        step_id=input.step_id,
        system_prompt="conflict resolution system prompt",
        user_prompt="conflict resolution user prompt",
    )


@activity.defn(name="call_conflict_resolution")
async def mock_recursive_call_conflict_resolution(
    input: ConflictResolutionCallInput,
) -> ConflictResolutionCallResult:
    _RECURSIVE_CALL_LOG.append("call_conflict_resolution")
    if _RECURSIVE_CONFLICT_RESPONSES:
        return _RECURSIVE_CONFLICT_RESPONSES.pop(0)
    return ConflictResolutionCallResult(
        task_id=input.task_id,
        resolved_files={},
        explanation="No conflicts resolved (default mock).",
        model_name="mock-reasoning",
        input_tokens=200,
        output_tokens=100,
        latency_ms=300.0,
    )


_RECURSIVE_MOCK_ACTIVITIES = [
    mock_recursive_create_worktree,
    mock_recursive_remove_worktree,
    mock_recursive_assemble_sub_task_context,
    mock_recursive_call_llm,
    mock_recursive_write_output,
    mock_recursive_write_files,
    mock_recursive_validate_output,
    mock_recursive_evaluate_transition,
    mock_recursive_assemble_cr_context,
    mock_recursive_call_conflict_resolution,
]


async def _run_recursive_subtask_workflow(
    env: WorkflowEnvironment,
    input: SubTaskInput,
) -> SubTaskResult:
    """Helper to run the sub-task workflow with recursive mock activities."""
    async with Worker(
        env.client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=[ForgeSubTaskWorkflow],
        activities=_RECURSIVE_MOCK_ACTIVITIES,
    ):
        return await env.client.execute_workflow(
            ForgeSubTaskWorkflow.run,
            input,
            id=f"test-recursive-{input.sub_task.sub_task_id}",
            task_queue=FORGE_TASK_QUEUE,
        )


# ---------------------------------------------------------------------------
# Tests — recursive fan-out success (2-level)
# ---------------------------------------------------------------------------


class TestRecursiveFanOut:
    """2-level fan-out success. Sub-task has nested sub-tasks, all succeed."""

    @pytest.fixture
    def recursive_input(self) -> SubTaskInput:
        return SubTaskInput(
            parent_task_id="parent-task",
            parent_description="Build an API.",
            sub_task=SubTask(
                sub_task_id="st1",
                description="Create schema components.",
                target_files=[],
                sub_tasks=[
                    SubTask(
                        sub_task_id="gc1",
                        description="Create models.",
                        target_files=["gc1.py"],
                    ),
                    SubTask(
                        sub_task_id="gc2",
                        description="Create validators.",
                        target_files=["gc2.py"],
                    ),
                ],
            ),
            repo_root="/tmp/repo",
            parent_branch="forge/parent-task",
            max_attempts=2,
            depth=0,
            max_depth=2,
        )

    @pytest.mark.asyncio
    async def test_recursive_success(
        self, env: WorkflowEnvironment, recursive_input: SubTaskInput
    ) -> None:
        _reset_recursive_mock_state()
        result = await _run_recursive_subtask_workflow(env, recursive_input)
        assert result.status == TransitionSignal.SUCCESS
        assert result.sub_task_id == "st1"

    @pytest.mark.asyncio
    async def test_merged_output_files_propagate(
        self, env: WorkflowEnvironment, recursive_input: SubTaskInput
    ) -> None:
        _reset_recursive_mock_state()
        result = await _run_recursive_subtask_workflow(env, recursive_input)
        assert "gc1.py" in result.output_files
        assert "gc2.py" in result.output_files

    @pytest.mark.asyncio
    async def test_nested_sub_task_results_populated(
        self, env: WorkflowEnvironment, recursive_input: SubTaskInput
    ) -> None:
        _reset_recursive_mock_state()
        result = await _run_recursive_subtask_workflow(env, recursive_input)
        assert len(result.sub_task_results) == 2
        ids = {r.sub_task_id for r in result.sub_task_results}
        assert ids == {"gc1", "gc2"}

    @pytest.mark.asyncio
    async def test_worktrees_created_and_removed(
        self, env: WorkflowEnvironment, recursive_input: SubTaskInput
    ) -> None:
        _reset_recursive_mock_state()
        await _run_recursive_subtask_workflow(env, recursive_input)
        # Parent sub-task worktree + 2 grandchild worktrees
        create_count = sum(1 for e in _RECURSIVE_CALL_LOG if e.startswith("create_worktree:"))
        remove_count = sum(1 for e in _RECURSIVE_CALL_LOG if e.startswith("remove_worktree:"))
        assert create_count == 3
        assert remove_count == 3


# ---------------------------------------------------------------------------
# Tests — recursive fan-out depth limit
# ---------------------------------------------------------------------------


class TestRecursiveFanOutDepthLimit:
    """max_depth=1, depth=1 with nested sub-tasks.

    The sub-task has nested sub_tasks but depth >= max_depth, so it
    runs single-step (ignores its sub_tasks).
    """

    @pytest.fixture
    def depth_limited_input(self) -> SubTaskInput:
        return SubTaskInput(
            parent_task_id="parent-task",
            parent_description="Build an API.",
            sub_task=SubTask(
                sub_task_id="st1",
                description="Create schema.",
                target_files=["leaf.py"],
                sub_tasks=[
                    SubTask(
                        sub_task_id="gc1",
                        description="Nested child (should be ignored).",
                        target_files=["gc1.py"],
                    ),
                ],
            ),
            repo_root="/tmp/repo",
            parent_branch="forge/parent-task",
            max_attempts=2,
            depth=1,
            max_depth=1,
        )

    @pytest.mark.asyncio
    async def test_runs_single_step(
        self, env: WorkflowEnvironment, depth_limited_input: SubTaskInput
    ) -> None:
        _reset_recursive_mock_state()
        result = await _run_recursive_subtask_workflow(env, depth_limited_input)
        assert result.status == TransitionSignal.SUCCESS
        # Should have run single-step: LLM was called, not nested fan-out
        assert "call_llm" in _RECURSIVE_CALL_LOG
        # Only one worktree created (leaf, not grandchild)
        create_count = sum(1 for e in _RECURSIVE_CALL_LOG if e.startswith("create_worktree:"))
        assert create_count == 1

    @pytest.mark.asyncio
    async def test_no_nested_sub_task_results(
        self, env: WorkflowEnvironment, depth_limited_input: SubTaskInput
    ) -> None:
        _reset_recursive_mock_state()
        result = await _run_recursive_subtask_workflow(env, depth_limited_input)
        assert result.sub_task_results == []


# ---------------------------------------------------------------------------
# Tests — recursive fan-out nested failure
# ---------------------------------------------------------------------------


class TestRecursiveFanOutNestedFailure:
    """Grandchild fails terminal. Verify failure propagates up through all levels."""

    @pytest.fixture
    def nested_failure_input(self) -> SubTaskInput:
        return SubTaskInput(
            parent_task_id="parent-task",
            parent_description="Build an API.",
            sub_task=SubTask(
                sub_task_id="st1",
                description="Create schema components.",
                target_files=[],
                sub_tasks=[
                    SubTask(
                        sub_task_id="gc1",
                        description="Create models.",
                        target_files=["gc1.py"],
                    ),
                    SubTask(
                        sub_task_id="gc2",
                        description="Create validators.",
                        target_files=["gc2.py"],
                    ),
                ],
            ),
            repo_root="/tmp/repo",
            parent_branch="forge/parent-task",
            max_attempts=1,
            depth=0,
            max_depth=2,
        )

    @pytest.mark.asyncio
    async def test_failure_propagates(
        self, env: WorkflowEnvironment, nested_failure_input: SubTaskInput
    ) -> None:
        _reset_recursive_mock_state(
            transitions=[
                TransitionSignal.SUCCESS.value,  # gc1
                TransitionSignal.FAILURE_TERMINAL.value,  # gc2
            ]
        )
        result = await _run_recursive_subtask_workflow(env, nested_failure_input)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.error is not None
        assert "gc2" in result.error

    @pytest.mark.asyncio
    async def test_worktrees_cleaned_up(
        self, env: WorkflowEnvironment, nested_failure_input: SubTaskInput
    ) -> None:
        _reset_recursive_mock_state(
            transitions=[
                TransitionSignal.SUCCESS.value,
                TransitionSignal.FAILURE_TERMINAL.value,
            ]
        )
        await _run_recursive_subtask_workflow(env, nested_failure_input)
        # All worktrees should be removed even on failure
        remove_count = sum(1 for e in _RECURSIVE_CALL_LOG if e.startswith("remove_worktree:"))
        assert remove_count >= 3  # parent + 2 grandchildren


# ---------------------------------------------------------------------------
# Tests — recursive fan-out nested file conflict
# ---------------------------------------------------------------------------


class TestRecursiveFanOutNestedFileConflict:
    """Two grandchildren produce the same file → conflict resolution attempted."""

    @pytest.fixture
    def conflict_input(self) -> SubTaskInput:
        return SubTaskInput(
            parent_task_id="parent-task",
            parent_description="Build an API.",
            sub_task=SubTask(
                sub_task_id="st1",
                description="Create components.",
                target_files=[],
                sub_tasks=[
                    SubTask(
                        sub_task_id="gc1",
                        description="Create module.",
                        target_files=["conflict.py"],
                    ),
                    SubTask(
                        sub_task_id="gc2",
                        description="Create module.",
                        target_files=["conflict.py"],
                    ),
                ],
            ),
            repo_root="/tmp/repo",
            parent_branch="forge/parent-task",
            max_attempts=1,
            depth=0,
            max_depth=2,
        )

    @pytest.mark.asyncio
    async def test_conflict_resolution_attempted(
        self, env: WorkflowEnvironment, conflict_input: SubTaskInput
    ) -> None:
        """Nested conflict triggers LLM resolution; incomplete resolution fails."""
        _reset_recursive_mock_state(
            llm_responses=[
                LLMResponse(
                    files=[FileOutput(file_path="conflict.py", content="# from gc1\n")],
                    explanation="gc1 output",
                ),
                LLMResponse(
                    files=[FileOutput(file_path="conflict.py", content="# from gc2\n")],
                    explanation="gc2 output",
                ),
            ]
            # Default mock returns empty resolved_files → incomplete
        )
        result = await _run_recursive_subtask_workflow(env, conflict_input)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.error is not None
        assert "Conflict resolution incomplete" in result.error
        assert "conflict.py" in result.error
        assert "assemble_conflict_resolution_context" in _RECURSIVE_CALL_LOG
        assert "call_conflict_resolution" in _RECURSIVE_CALL_LOG

    @pytest.mark.asyncio
    async def test_nested_conflict_resolution_succeeds(
        self, env: WorkflowEnvironment, conflict_input: SubTaskInput
    ) -> None:
        """Nested conflict resolved successfully → sub-task succeeds."""
        _reset_recursive_mock_state(
            llm_responses=[
                LLMResponse(
                    files=[FileOutput(file_path="conflict.py", content="# from gc1\n")],
                    explanation="gc1 output",
                ),
                LLMResponse(
                    files=[FileOutput(file_path="conflict.py", content="# from gc2\n")],
                    explanation="gc2 output",
                ),
            ],
            conflict_responses=[
                ConflictResolutionCallResult(
                    task_id="parent-task.sub.st1",
                    resolved_files={"conflict.py": "# merged gc1+gc2\n"},
                    explanation="Combined both.",
                    model_name="mock-reasoning",
                    input_tokens=200,
                    output_tokens=100,
                    latency_ms=300.0,
                ),
            ],
        )
        result = await _run_recursive_subtask_workflow(env, conflict_input)
        assert result.status == TransitionSignal.SUCCESS
        assert result.output_files["conflict.py"] == "# merged gc1+gc2\n"
        assert result.conflict_resolution is not None


# ---------------------------------------------------------------------------
# Tests — backward compat: flat fan-out with default max_fan_out_depth
# ---------------------------------------------------------------------------


class TestRecursiveBackwardCompat:
    """Existing flat fan-out works unchanged with default max_fan_out_depth=1."""

    @pytest.mark.asyncio
    async def test_flat_fanout_still_works(self, env: WorkflowEnvironment) -> None:
        _reset_fanout_mock_state()
        result = await _run_fanout_workflow(env)
        assert result.status == TransitionSignal.SUCCESS
        assert len(result.step_results) == 1
        sr = result.step_results[0]
        assert len(sr.sub_task_results) == 2

    @pytest.mark.asyncio
    async def test_default_max_fan_out_depth(self) -> None:
        """ForgeTaskInput defaults to max_fan_out_depth=1."""
        task_input = ForgeTaskInput(
            task=TaskDefinition(task_id="t", description="d"),
            repo_root="/tmp/repo",
        )
        assert task_input.max_fan_out_depth == 1

    @pytest.mark.asyncio
    async def test_subtask_input_default_depth(self) -> None:
        """SubTaskInput defaults to depth=0, max_depth=1."""
        st_input = SubTaskInput(
            parent_task_id="p",
            parent_description="d",
            sub_task=SubTask(sub_task_id="s", description="d", target_files=["f.py"]),
            repo_root="/tmp/repo",
            parent_branch="main",
        )
        assert st_input.depth == 0
        assert st_input.max_depth == 1


# ===========================================================================
# Sanity check workflow tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Mock activities for sanity check tests
# ---------------------------------------------------------------------------

_SC_CALL_LOG: list[str] = []
_SC_TRANSITION_SEQUENCE: list[str] = []
_SC_LLM_CALL_COUNT: int = 0
_SC_SANITY_RESPONSES: list[SanityCheckCallResult] = []

_SC_PLAN = Plan(
    task_id="sc-task",
    steps=[
        PlanStep(step_id="step-1", description="Create models.", target_files=["models.py"]),
        PlanStep(step_id="step-2", description="Create API.", target_files=["api.py"]),
        PlanStep(step_id="step-3", description="Add tests.", target_files=["test_api.py"]),
        PlanStep(step_id="step-4", description="Add docs.", target_files=["docs.py"]),
    ],
    explanation="Four-step plan.",
)


def _reset_sc_mock_state(
    transitions: list[str] | None = None,
    sanity_responses: list[SanityCheckCallResult] | None = None,
    plan: Plan | None = None,
) -> None:
    global _SC_LLM_CALL_COUNT, _SC_PLAN
    _SC_CALL_LOG.clear()
    _SC_TRANSITION_SEQUENCE.clear()
    _SC_LLM_CALL_COUNT = 0
    _SC_SANITY_RESPONSES.clear()
    if transitions:
        _SC_TRANSITION_SEQUENCE.extend(transitions)
    if sanity_responses:
        _SC_SANITY_RESPONSES.extend(sanity_responses)
    if plan is not None:
        _SC_PLAN = plan


@activity.defn(name="create_worktree_activity")
async def mock_sc_create_worktree(input: CreateWorktreeInput) -> CreateWorktreeOutput:
    _SC_CALL_LOG.append("create_worktree")
    return CreateWorktreeOutput(
        worktree_path=f"/tmp/repo/.forge-worktrees/{input.task_id}",
        branch_name=f"forge/{input.task_id}",
    )


@activity.defn(name="assemble_planner_context")
async def mock_sc_assemble_planner_context(input: AssembleContextInput) -> PlannerInput:
    _SC_CALL_LOG.append("assemble_planner_context")
    return PlannerInput(
        task_id=input.task.task_id,
        system_prompt="planner system prompt",
        user_prompt="planner user prompt",
    )


@activity.defn(name="call_planner")
async def mock_sc_call_planner(input: PlannerInput) -> PlanCallResult:
    _SC_CALL_LOG.append("call_planner")
    return PlanCallResult(
        task_id=input.task_id,
        plan=_SC_PLAN,
        model_name="mock-planner",
        input_tokens=300,
        output_tokens=150,
        latency_ms=500.0,
    )


@activity.defn(name="assemble_step_context")
async def mock_sc_assemble_step_context(input: AssembleStepContextInput) -> AssembledContext:
    _SC_CALL_LOG.append(f"assemble_step_context:{input.step.step_id}")
    return AssembledContext(
        task_id=input.task.task_id,
        system_prompt=f"step system prompt for {input.step.step_id}",
        user_prompt=f"step user prompt for {input.step.step_id}",
    )


@activity.defn(name="call_llm")
async def mock_sc_call_llm(context: AssembledContext) -> LLMCallResult:
    global _SC_LLM_CALL_COUNT
    _SC_LLM_CALL_COUNT += 1
    _SC_CALL_LOG.append(f"call_llm:{_SC_LLM_CALL_COUNT}")
    files = [FileOutput(file_path=f"file{_SC_LLM_CALL_COUNT}.py", content="# code\n")]
    return LLMCallResult(
        task_id=context.task_id,
        response=LLMResponse(files=files, explanation=f"LLM call #{_SC_LLM_CALL_COUNT}"),
        model_name="mock-model",
        input_tokens=100,
        output_tokens=50,
        latency_ms=200.0,
    )


@activity.defn(name="write_output")
async def mock_sc_write_output(input: WriteOutputInput) -> WriteResult:
    _SC_CALL_LOG.append("write_output")
    files = input.llm_result.response.files
    return WriteResult(
        task_id=input.llm_result.task_id,
        files_written=[f.file_path for f in files],
        output_files={f.file_path: f.content for f in files},
    )


@activity.defn(name="validate_output")
async def mock_sc_validate_output(input: ValidateOutputInput) -> list[ValidationResult]:
    _SC_CALL_LOG.append("validate_output")
    return [ValidationResult(check_name="ruff_lint", passed=True, summary="passed")]


@activity.defn(name="evaluate_transition")
async def mock_sc_evaluate_transition(input: TransitionInput) -> str:
    _SC_CALL_LOG.append("evaluate_transition")
    if _SC_TRANSITION_SEQUENCE:
        return _SC_TRANSITION_SEQUENCE.pop(0)
    return TransitionSignal.SUCCESS.value


@activity.defn(name="commit_changes_activity")
async def mock_sc_commit_changes(input: CommitChangesInput) -> CommitChangesOutput:
    msg = input.message or input.status
    _SC_CALL_LOG.append(f"commit:{msg}")
    return CommitChangesOutput(commit_sha="c" * 40)


@activity.defn(name="reset_worktree_activity")
async def mock_sc_reset_worktree(input: ResetWorktreeInput) -> None:
    _SC_CALL_LOG.append("reset_worktree")


@activity.defn(name="assemble_sanity_check_context")
async def mock_assemble_sanity_check_context(
    input: AssembleSanityCheckContextInput,
) -> SanityCheckInput:
    _SC_CALL_LOG.append("assemble_sanity_check_context")
    return SanityCheckInput(
        task_id=input.task.task_id,
        system_prompt="sanity check system prompt",
        user_prompt="sanity check user prompt",
    )


@activity.defn(name="call_sanity_check")
async def mock_call_sanity_check(input: SanityCheckInput) -> SanityCheckCallResult:
    _SC_CALL_LOG.append("call_sanity_check")
    if _SC_SANITY_RESPONSES:
        return _SC_SANITY_RESPONSES.pop(0)
    return SanityCheckCallResult(
        task_id=input.task_id,
        response=SanityCheckResponse(
            verdict=SanityCheckVerdict.CONTINUE,
            explanation="Plan looks good.",
        ),
        model_name="mock-reasoning",
        input_tokens=200,
        output_tokens=100,
        latency_ms=300.0,
    )


_SC_MOCK_ACTIVITIES = [
    mock_sc_create_worktree,
    mock_sc_assemble_planner_context,
    mock_sc_call_planner,
    mock_sc_assemble_step_context,
    mock_sc_call_llm,
    mock_sc_write_output,
    mock_sc_validate_output,
    mock_sc_evaluate_transition,
    mock_sc_commit_changes,
    mock_sc_reset_worktree,
    mock_assemble_sanity_check_context,
    mock_call_sanity_check,
]

_SC_TASK = TaskDefinition(
    task_id="sc-task",
    description="Build a full API.",
)


async def _run_sc_workflow(
    env: WorkflowEnvironment,
    input: ForgeTaskInput | None = None,
) -> TaskResult:
    """Helper to run the planned workflow with sanity check mock activities."""
    if input is None:
        input = ForgeTaskInput(
            task=_SC_TASK,
            repo_root="/tmp/repo",
            plan=True,
            max_step_attempts=2,
            max_exploration_rounds=0,
            sanity_check_interval=2,
        )
    async with Worker(
        env.client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=[ForgeTaskWorkflow],
        activities=_SC_MOCK_ACTIVITIES,
    ):
        return await env.client.execute_workflow(
            ForgeTaskWorkflow.run,
            input,
            id=f"test-sc-{input.task.task_id}",
            task_queue=FORGE_TASK_QUEUE,
        )


# ---------------------------------------------------------------------------
# Tests — sanity check continue
# ---------------------------------------------------------------------------


class TestSanityCheckContinue:
    """interval=2, 4 steps, sanity check fires after step 2, returns 'continue'."""

    @pytest.mark.asyncio
    async def test_all_steps_complete(self, env: WorkflowEnvironment) -> None:
        _reset_sc_mock_state(
            transitions=[TransitionSignal.SUCCESS.value] * 4,
        )
        result = await _run_sc_workflow(env)
        assert result.status == TransitionSignal.SUCCESS
        assert len(result.step_results) == 4

    @pytest.mark.asyncio
    async def test_sanity_check_count(self, env: WorkflowEnvironment) -> None:
        _reset_sc_mock_state(
            transitions=[TransitionSignal.SUCCESS.value] * 4,
        )
        result = await _run_sc_workflow(env)
        # Fires after step 2 (2 % 2 == 0, not last step)
        # Does NOT fire after step 4 (last step)
        assert result.sanity_check_count == 1

    @pytest.mark.asyncio
    async def test_sanity_check_activities_called(self, env: WorkflowEnvironment) -> None:
        _reset_sc_mock_state(
            transitions=[TransitionSignal.SUCCESS.value] * 4,
        )
        await _run_sc_workflow(env)
        assert "assemble_sanity_check_context" in _SC_CALL_LOG
        assert "call_sanity_check" in _SC_CALL_LOG


# ---------------------------------------------------------------------------
# Tests — sanity check abort
# ---------------------------------------------------------------------------


class TestSanityCheckAbort:
    """interval=1, 3 steps, sanity check fires after step 1, returns 'abort'."""

    @pytest.mark.asyncio
    async def test_abort_returns_failure(self, env: WorkflowEnvironment) -> None:
        abort_response = SanityCheckCallResult(
            task_id="sc-task",
            response=SanityCheckResponse(
                verdict=SanityCheckVerdict.ABORT,
                explanation="Fundamental issue found.",
            ),
            model_name="mock-reasoning",
            input_tokens=200,
            output_tokens=100,
            latency_ms=300.0,
        )
        three_step_plan = Plan(
            task_id="sc-task",
            steps=[
                PlanStep(step_id="s1", description="Step 1.", target_files=["a.py"]),
                PlanStep(step_id="s2", description="Step 2.", target_files=["b.py"]),
                PlanStep(step_id="s3", description="Step 3.", target_files=["c.py"]),
            ],
            explanation="Three steps.",
        )
        _reset_sc_mock_state(
            transitions=[TransitionSignal.SUCCESS.value],
            sanity_responses=[abort_response],
            plan=three_step_plan,
        )
        input = ForgeTaskInput(
            task=_SC_TASK,
            repo_root="/tmp/repo",
            plan=True,
            max_exploration_rounds=0,
            sanity_check_interval=1,
        )
        result = await _run_sc_workflow(env, input)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert "Sanity check aborted" in (result.error or "")

    @pytest.mark.asyncio
    async def test_abort_only_one_step_result(self, env: WorkflowEnvironment) -> None:
        abort_response = SanityCheckCallResult(
            task_id="sc-task",
            response=SanityCheckResponse(
                verdict=SanityCheckVerdict.ABORT,
                explanation="Stop now.",
            ),
            model_name="mock-reasoning",
            input_tokens=200,
            output_tokens=100,
            latency_ms=300.0,
        )
        three_step_plan = Plan(
            task_id="sc-task",
            steps=[
                PlanStep(step_id="s1", description="Step 1.", target_files=["a.py"]),
                PlanStep(step_id="s2", description="Step 2.", target_files=["b.py"]),
                PlanStep(step_id="s3", description="Step 3.", target_files=["c.py"]),
            ],
            explanation="Three steps.",
        )
        _reset_sc_mock_state(
            transitions=[TransitionSignal.SUCCESS.value],
            sanity_responses=[abort_response],
            plan=three_step_plan,
        )
        input = ForgeTaskInput(
            task=_SC_TASK,
            repo_root="/tmp/repo",
            plan=True,
            max_exploration_rounds=0,
            sanity_check_interval=1,
        )
        result = await _run_sc_workflow(env, input)
        assert len(result.step_results) == 1
        assert result.sanity_check_count == 1


# ---------------------------------------------------------------------------
# Tests — sanity check revise
# ---------------------------------------------------------------------------


class TestSanityCheckRevise:
    """interval=1, 3 steps, sanity check fires after step 1, returns 'revise' with 1 step."""

    @pytest.mark.asyncio
    async def test_revise_replaces_remaining_steps(self, env: WorkflowEnvironment) -> None:
        revised_step = PlanStep(
            step_id="revised-1", description="Revised step.", target_files=["revised.py"]
        )
        revise_response = SanityCheckCallResult(
            task_id="sc-task",
            response=SanityCheckResponse(
                verdict=SanityCheckVerdict.REVISE,
                explanation="Need to adjust approach.",
                revised_steps=[revised_step],
            ),
            model_name="mock-reasoning",
            input_tokens=200,
            output_tokens=100,
            latency_ms=300.0,
        )
        three_step_plan = Plan(
            task_id="sc-task",
            steps=[
                PlanStep(step_id="s1", description="Step 1.", target_files=["a.py"]),
                PlanStep(step_id="s2", description="Step 2.", target_files=["b.py"]),
                PlanStep(step_id="s3", description="Step 3.", target_files=["c.py"]),
            ],
            explanation="Three steps.",
        )
        _reset_sc_mock_state(
            # step 1 succeeds, then sanity check revises.
            # revised-1 succeeds (no more sanity check since it's the last step).
            transitions=[TransitionSignal.SUCCESS.value, TransitionSignal.SUCCESS.value],
            sanity_responses=[revise_response],
            plan=three_step_plan,
        )
        input = ForgeTaskInput(
            task=_SC_TASK,
            repo_root="/tmp/repo",
            plan=True,
            max_exploration_rounds=0,
            sanity_check_interval=1,
        )
        result = await _run_sc_workflow(env, input)
        assert result.status == TransitionSignal.SUCCESS
        assert len(result.step_results) == 2
        assert result.step_results[0].step_id == "s1"
        assert result.step_results[1].step_id == "revised-1"

    @pytest.mark.asyncio
    async def test_revise_updates_plan_in_result(self, env: WorkflowEnvironment) -> None:
        revised_step = PlanStep(
            step_id="revised-1", description="Revised step.", target_files=["revised.py"]
        )
        revise_response = SanityCheckCallResult(
            task_id="sc-task",
            response=SanityCheckResponse(
                verdict=SanityCheckVerdict.REVISE,
                explanation="Need to adjust.",
                revised_steps=[revised_step],
            ),
            model_name="mock-reasoning",
            input_tokens=200,
            output_tokens=100,
            latency_ms=300.0,
        )
        three_step_plan = Plan(
            task_id="sc-task",
            steps=[
                PlanStep(step_id="s1", description="Step 1.", target_files=["a.py"]),
                PlanStep(step_id="s2", description="Step 2.", target_files=["b.py"]),
                PlanStep(step_id="s3", description="Step 3.", target_files=["c.py"]),
            ],
            explanation="Three steps.",
        )
        _reset_sc_mock_state(
            transitions=[TransitionSignal.SUCCESS.value, TransitionSignal.SUCCESS.value],
            sanity_responses=[revise_response],
            plan=three_step_plan,
        )
        input = ForgeTaskInput(
            task=_SC_TASK,
            repo_root="/tmp/repo",
            plan=True,
            max_exploration_rounds=0,
            sanity_check_interval=1,
        )
        result = await _run_sc_workflow(env, input)
        assert result.plan is not None
        # Plan should have 2 steps: original s1 + revised-1
        assert len(result.plan.steps) == 2
        assert result.plan.steps[1].step_id == "revised-1"


# ---------------------------------------------------------------------------
# Tests — sanity check disabled
# ---------------------------------------------------------------------------


class TestSanityCheckDisabled:
    """interval=0 (default), verify no sanity check activities called."""

    @pytest.mark.asyncio
    async def test_no_sanity_check_when_disabled(self, env: WorkflowEnvironment) -> None:
        _reset_sc_mock_state(
            transitions=[TransitionSignal.SUCCESS.value] * 4,
        )
        input = ForgeTaskInput(
            task=_SC_TASK,
            repo_root="/tmp/repo",
            plan=True,
            max_exploration_rounds=0,
            sanity_check_interval=0,
        )
        result = await _run_sc_workflow(env, input)
        assert result.status == TransitionSignal.SUCCESS
        assert result.sanity_check_count == 0
        assert "assemble_sanity_check_context" not in _SC_CALL_LOG
        assert "call_sanity_check" not in _SC_CALL_LOG


# ---------------------------------------------------------------------------
# Tests — sanity check skips last step
# ---------------------------------------------------------------------------


class TestSanityCheckSkipsLastStep:
    """interval=1, 2 steps, verify sanity check fires after step 1 but not after step 2."""

    @pytest.mark.asyncio
    async def test_fires_after_first_not_last(self, env: WorkflowEnvironment) -> None:
        two_step_plan = Plan(
            task_id="sc-task",
            steps=[
                PlanStep(step_id="s1", description="Step 1.", target_files=["a.py"]),
                PlanStep(step_id="s2", description="Step 2.", target_files=["b.py"]),
            ],
            explanation="Two steps.",
        )
        _reset_sc_mock_state(
            transitions=[TransitionSignal.SUCCESS.value, TransitionSignal.SUCCESS.value],
            plan=two_step_plan,
        )
        input = ForgeTaskInput(
            task=_SC_TASK,
            repo_root="/tmp/repo",
            plan=True,
            max_exploration_rounds=0,
            sanity_check_interval=1,
        )
        result = await _run_sc_workflow(env, input)
        assert result.status == TransitionSignal.SUCCESS
        assert result.sanity_check_count == 1
        # Only one sanity check call, not two
        assert _SC_CALL_LOG.count("call_sanity_check") == 1


# ===========================================================================
# Phase 14b: Batch path tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Mock activities for batch path
# ---------------------------------------------------------------------------

_BATCH_CALL_LOG: list[str] = []
_BATCH_TRANSITION_SEQUENCE: list[str] = []
_BATCH_PARSE_RESPONSES: list[ParsedLLMResponse] = []


def _reset_batch_mock_state(
    transitions: list[str] | None = None,
    parse_responses: list[ParsedLLMResponse] | None = None,
) -> None:
    _BATCH_CALL_LOG.clear()
    _BATCH_TRANSITION_SEQUENCE.clear()
    _BATCH_PARSE_RESPONSES.clear()
    if transitions:
        _BATCH_TRANSITION_SEQUENCE.extend(transitions)
    if parse_responses:
        _BATCH_PARSE_RESPONSES.extend(parse_responses)


@activity.defn(name="create_worktree_activity")
async def mock_batch_create_worktree(input: CreateWorktreeInput) -> CreateWorktreeOutput:
    _BATCH_CALL_LOG.append("create_worktree")
    return CreateWorktreeOutput(
        worktree_path=f"/tmp/repo/.forge-worktrees/{input.task_id}",
        branch_name=f"forge/{input.task_id}",
    )


@activity.defn(name="remove_worktree_activity")
async def mock_batch_remove_worktree(input: RemoveWorktreeInput) -> None:
    _BATCH_CALL_LOG.append("remove_worktree")


@activity.defn(name="commit_changes_activity")
async def mock_batch_commit_changes(input: CommitChangesInput) -> CommitChangesOutput:
    _BATCH_CALL_LOG.append(f"commit:{input.status}")
    return CommitChangesOutput(commit_sha="d" * 40)


@activity.defn(name="assemble_context")
async def mock_batch_assemble_context(input: AssembleContextInput) -> AssembledContext:
    _BATCH_CALL_LOG.append("assemble_context")
    return AssembledContext(
        task_id=input.task.task_id,
        system_prompt="system prompt",
        user_prompt="user prompt",
    )


@activity.defn(name="submit_batch_request")
async def mock_batch_submit(input: BatchSubmitInput) -> BatchSubmitResult:
    _BATCH_CALL_LOG.append("submit_batch_request")
    return BatchSubmitResult(
        request_id="req-test-123",
        batch_id="msgbatch_test123",
    )


@activity.defn(name="parse_llm_response")
async def mock_batch_parse(input: ParseResponseInput) -> ParsedLLMResponse:
    _BATCH_CALL_LOG.append(f"parse_llm_response:{input.output_type_name}")
    if _BATCH_PARSE_RESPONSES:
        return _BATCH_PARSE_RESPONSES.pop(0)
    # Default: return a valid LLMResponse
    llm_resp = LLMResponse(
        files=[FileOutput(file_path="hello.py", content="print('hello')\n")],
        explanation="Created hello module.",
    )
    return ParsedLLMResponse(
        parsed_json=llm_resp.model_dump_json(),
        model_name="mock-batch-model",
        input_tokens=100,
        output_tokens=50,
    )


@activity.defn(name="write_output")
async def mock_batch_write_output(input: WriteOutputInput) -> WriteResult:
    _BATCH_CALL_LOG.append("write_output")
    files = input.llm_result.response.files
    return WriteResult(
        task_id=input.llm_result.task_id,
        files_written=[f.file_path for f in files],
        output_files={f.file_path: f.content for f in files},
    )


@activity.defn(name="validate_output")
async def mock_batch_validate_output(input: ValidateOutputInput) -> list[ValidationResult]:
    _BATCH_CALL_LOG.append("validate_output")
    return [ValidationResult(check_name="ruff_lint", passed=True, summary="passed")]


@activity.defn(name="evaluate_transition")
async def mock_batch_evaluate_transition(input: TransitionInput) -> str:
    _BATCH_CALL_LOG.append("evaluate_transition")
    if _BATCH_TRANSITION_SEQUENCE:
        return _BATCH_TRANSITION_SEQUENCE.pop(0)
    return TransitionSignal.SUCCESS.value


_BATCH_MOCK_ACTIVITIES = [
    mock_batch_create_worktree,
    mock_batch_remove_worktree,
    mock_batch_commit_changes,
    mock_batch_assemble_context,
    mock_batch_submit,
    mock_batch_parse,
    mock_batch_write_output,
    mock_batch_validate_output,
    mock_batch_evaluate_transition,
]


# ---------------------------------------------------------------------------
# Tests — batch single step
# ---------------------------------------------------------------------------


class TestBatchSingleStep:
    """Single-step workflow with sync_mode=False uses batch path."""

    @pytest.mark.asyncio
    async def test_batch_generation_success(self, env: WorkflowEnvironment) -> None:
        _reset_batch_mock_state(transitions=[TransitionSignal.SUCCESS.value])

        task = TaskDefinition(
            task_id="batch-test",
            description="Write a hello module.",
            target_files=["hello.py"],
        )
        input = ForgeTaskInput(
            task=task,
            repo_root="/tmp/repo",
            max_attempts=2,
            max_exploration_rounds=0,
            sync_mode=False,
        )

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeTaskWorkflow],
            activities=_BATCH_MOCK_ACTIVITIES,
        ):
            handle = await env.client.start_workflow(
                ForgeTaskWorkflow.run,
                input,
                id="test-batch-single",
                task_queue=FORGE_TASK_QUEUE,
            )

            # Send signal with batch result
            batch_result = BatchResult(
                request_id="req-test-123",
                batch_id="msgbatch_test123",
                raw_response_json='{"dummy": "json"}',
                result_type="LLMResponse",
            )
            await handle.signal(ForgeTaskWorkflow.batch_result_received, batch_result)

            result = await handle.result()

        assert result.status == TransitionSignal.SUCCESS
        assert "submit_batch_request" in _BATCH_CALL_LOG
        assert "parse_llm_response:LLMResponse" in _BATCH_CALL_LOG
        # Verify sync path was NOT called
        assert "call_llm" not in _BATCH_CALL_LOG
        assert result.output_files == {"hello.py": "print('hello')\n"}

    @pytest.mark.asyncio
    async def test_batch_error_in_signal_raises(self, env: WorkflowEnvironment) -> None:
        _reset_batch_mock_state()

        task = TaskDefinition(
            task_id="batch-err",
            description="Error test.",
            target_files=["x.py"],
        )
        input = ForgeTaskInput(
            task=task,
            repo_root="/tmp/repo",
            max_attempts=1,
            max_exploration_rounds=0,
            sync_mode=False,
        )

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeTaskWorkflow],
            activities=_BATCH_MOCK_ACTIVITIES,
        ):
            handle = await env.client.start_workflow(
                ForgeTaskWorkflow.run,
                input,
                id="test-batch-error",
                task_queue=FORGE_TASK_QUEUE,
            )

            # Send signal with error
            batch_result = BatchResult(
                request_id="req-test-123",
                batch_id="msgbatch_test123",
                error="Batch expired",
                result_type="LLMResponse",
            )
            await handle.signal(ForgeTaskWorkflow.batch_result_received, batch_result)

            from temporalio.client import WorkflowFailureError

            with pytest.raises(WorkflowFailureError):
                await handle.result()


# ---------------------------------------------------------------------------
# Tests — batch planned workflow
# ---------------------------------------------------------------------------

# Additional mock activities needed for planned batch tests

_BATCH_PLAN_CALL_LOG: list[str] = []
_BATCH_PLAN_TRANSITION_SEQUENCE: list[str] = []
_BATCH_PLAN_PARSE_QUEUE: list[ParsedLLMResponse] = []


def _reset_batch_plan_mock_state(
    transitions: list[str] | None = None,
    parse_queue: list[ParsedLLMResponse] | None = None,
) -> None:
    _BATCH_PLAN_CALL_LOG.clear()
    _BATCH_PLAN_TRANSITION_SEQUENCE.clear()
    _BATCH_PLAN_PARSE_QUEUE.clear()
    if transitions:
        _BATCH_PLAN_TRANSITION_SEQUENCE.extend(transitions)
    if parse_queue:
        _BATCH_PLAN_PARSE_QUEUE.extend(parse_queue)


@activity.defn(name="create_worktree_activity")
async def mock_bp_create_worktree(input: CreateWorktreeInput) -> CreateWorktreeOutput:
    _BATCH_PLAN_CALL_LOG.append("create_worktree")
    return CreateWorktreeOutput(
        worktree_path=f"/tmp/repo/.forge-worktrees/{input.task_id}",
        branch_name=f"forge/{input.task_id}",
    )


@activity.defn(name="assemble_planner_context")
async def mock_bp_assemble_planner(input: AssembleContextInput) -> PlannerInput:
    _BATCH_PLAN_CALL_LOG.append("assemble_planner_context")
    return PlannerInput(
        task_id=input.task.task_id,
        system_prompt="planner system",
        user_prompt="planner user",
    )


@activity.defn(name="submit_batch_request")
async def mock_bp_submit_batch(input: BatchSubmitInput) -> BatchSubmitResult:
    _BATCH_PLAN_CALL_LOG.append(f"submit_batch:{input.output_type_name}")
    return BatchSubmitResult(request_id="req-bp-123", batch_id="msgbatch_bp123")


@activity.defn(name="parse_llm_response")
async def mock_bp_parse(input: ParseResponseInput) -> ParsedLLMResponse:
    _BATCH_PLAN_CALL_LOG.append(f"parse:{input.output_type_name}")
    if _BATCH_PLAN_PARSE_QUEUE:
        return _BATCH_PLAN_PARSE_QUEUE.pop(0)
    msg = "No parse response queued"
    raise RuntimeError(msg)


@activity.defn(name="assemble_step_context")
async def mock_bp_assemble_step(input: AssembleStepContextInput) -> AssembledContext:
    _BATCH_PLAN_CALL_LOG.append(f"assemble_step:{input.step.step_id}")
    return AssembledContext(
        task_id=input.task.task_id,
        system_prompt=f"step system for {input.step.step_id}",
        user_prompt=f"step user for {input.step.step_id}",
    )


@activity.defn(name="write_output")
async def mock_bp_write_output(input: WriteOutputInput) -> WriteResult:
    _BATCH_PLAN_CALL_LOG.append("write_output")
    files = input.llm_result.response.files
    return WriteResult(
        task_id=input.llm_result.task_id,
        files_written=[f.file_path for f in files],
        output_files={f.file_path: f.content for f in files},
    )


@activity.defn(name="validate_output")
async def mock_bp_validate_output(input: ValidateOutputInput) -> list[ValidationResult]:
    _BATCH_PLAN_CALL_LOG.append("validate_output")
    return [ValidationResult(check_name="ruff_lint", passed=True, summary="passed")]


@activity.defn(name="evaluate_transition")
async def mock_bp_evaluate_transition(input: TransitionInput) -> str:
    _BATCH_PLAN_CALL_LOG.append("evaluate_transition")
    if _BATCH_PLAN_TRANSITION_SEQUENCE:
        return _BATCH_PLAN_TRANSITION_SEQUENCE.pop(0)
    return TransitionSignal.SUCCESS.value


@activity.defn(name="commit_changes_activity")
async def mock_bp_commit(input: CommitChangesInput) -> CommitChangesOutput:
    _BATCH_PLAN_CALL_LOG.append(f"commit:{input.status}")
    return CommitChangesOutput(commit_sha="e" * 40)


@activity.defn(name="reset_worktree_activity")
async def mock_bp_reset_worktree(input: ResetWorktreeInput) -> None:
    _BATCH_PLAN_CALL_LOG.append("reset_worktree")


_BATCH_PLAN_MOCK_ACTIVITIES = [
    mock_bp_create_worktree,
    mock_bp_assemble_planner,
    mock_bp_submit_batch,
    mock_bp_parse,
    mock_bp_assemble_step,
    mock_bp_write_output,
    mock_bp_validate_output,
    mock_bp_evaluate_transition,
    mock_bp_commit,
    mock_bp_reset_worktree,
]


class TestBatchPlanned:
    """Planned workflow with sync_mode=False uses batch path for planner + generation."""

    @pytest.mark.asyncio
    async def test_batch_planner_and_generation(self, env: WorkflowEnvironment) -> None:
        plan = Plan(
            task_id="batch-plan-task",
            steps=[
                PlanStep(step_id="s1", description="Create it.", target_files=["a.py"]),
            ],
            explanation="One step.",
        )
        plan_parsed = ParsedLLMResponse(
            parsed_json=plan.model_dump_json(),
            model_name="mock-planner",
            input_tokens=300,
            output_tokens=150,
        )
        gen_resp = LLMResponse(
            files=[FileOutput(file_path="a.py", content="# step1\n")],
            explanation="Created a.py.",
        )
        gen_parsed = ParsedLLMResponse(
            parsed_json=gen_resp.model_dump_json(),
            model_name="mock-gen",
            input_tokens=100,
            output_tokens=50,
        )
        _reset_batch_plan_mock_state(
            transitions=[TransitionSignal.SUCCESS.value],
            parse_queue=[plan_parsed, gen_parsed],
        )

        task = TaskDefinition(
            task_id="batch-plan-task",
            description="Build a thing.",
        )
        input = ForgeTaskInput(
            task=task,
            repo_root="/tmp/repo",
            plan=True,
            max_exploration_rounds=0,
            sync_mode=False,
        )

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeTaskWorkflow],
            activities=_BATCH_PLAN_MOCK_ACTIVITIES,
        ):
            handle = await env.client.start_workflow(
                ForgeTaskWorkflow.run,
                input,
                id="test-batch-planned",
                task_queue=FORGE_TASK_QUEUE,
            )

            # Signal for planner batch
            planner_signal = BatchResult(
                request_id="req-bp-123",
                batch_id="msgbatch_bp123",
                raw_response_json='{"dummy": "plan"}',
                result_type="Plan",
            )
            await handle.signal(ForgeTaskWorkflow.batch_result_received, planner_signal)

            # Signal for generation batch
            gen_signal = BatchResult(
                request_id="req-bp-123",
                batch_id="msgbatch_bp123",
                raw_response_json='{"dummy": "gen"}',
                result_type="LLMResponse",
            )
            await handle.signal(ForgeTaskWorkflow.batch_result_received, gen_signal)

            result = await handle.result()

        assert result.status == TransitionSignal.SUCCESS
        assert "submit_batch:Plan" in _BATCH_PLAN_CALL_LOG
        assert "submit_batch:LLMResponse" in _BATCH_PLAN_CALL_LOG
        assert "parse:Plan" in _BATCH_PLAN_CALL_LOG
        assert "parse:LLMResponse" in _BATCH_PLAN_CALL_LOG
        assert result.plan is not None
        assert len(result.step_results) == 1


# ---------------------------------------------------------------------------
# Tests — existing sync_mode=True backward compatibility
# ---------------------------------------------------------------------------


class TestSyncModeDefaultBackwardCompat:
    """Verify that sync_mode defaults to True and existing tests still pass."""

    def test_default_sync_mode_is_true(self) -> None:
        task = TaskDefinition(task_id="t1", description="Test.")
        input = ForgeTaskInput(task=task, repo_root="/repo")
        assert input.sync_mode is True

    def test_subtask_default_sync_mode_is_true(self) -> None:
        input = SubTaskInput(
            parent_task_id="p",
            parent_description="Parent.",
            sub_task=SubTask(sub_task_id="s", description="Sub.", target_files=["x.py"]),
            repo_root="/repo",
            parent_branch="main",
        )
        assert input.sync_mode is True
