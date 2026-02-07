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
    CommitChangesInput,
    CommitChangesOutput,
    CreateWorktreeInput,
    CreateWorktreeOutput,
    FileOutput,
    ForgeTaskInput,
    LLMCallResult,
    LLMResponse,
    RemoveWorktreeInput,
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
