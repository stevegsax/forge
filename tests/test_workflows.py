"""Tests for forge.workflows — ForgeTaskWorkflow with mocked activities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from temporalio import activity
from temporalio.worker import Worker

from forge.activities.conflict_resolution import classify_file_conflicts
from forge.models import (
    AssembleContextInput,
    AssembledContext,
    AssembleSanityCheckContextInput,
    AssembleStepContextInput,
    AssembleSubTaskContextInput,
    BatchFetchResult,
    BatchStatusInput,
    BatchStatusResult,
    BatchSubmitInput,
    BatchSubmitResult,
    CommitChangesInput,
    CommitChangesOutput,
    ConflictResolutionCallInput,
    ConflictResolutionCallResult,
    ConflictResolutionInput,
    ConflictResolutionResponse,
    CreateWorktreeInput,
    CreateWorktreeOutput,
    DetectFileConflictsInput,
    DetectFileConflictsOutput,
    FetchBatchResultInput,
    FileOutput,
    ForgeTaskInput,
    LLMCallResult,
    LLMResponse,
    ModelConfig,
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
    ThinkingPolicy,
    TransitionInput,
    TransitionSignal,
    ValidateOutputInput,
    ValidationResult,
    WriteFilesInput,
    WriteOutputInput,
    WriteResult,
)
from forge.persist_models import PersistRequest, PersistResult
from forge.workflow_blocks import THINKING_MAX_TOKENS
from forge.workflows import (
    FORGE_TASK_QUEUE,
    ForgeSubTaskWorkflow,
    ForgeTaskWorkflow,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel
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
# Shared batch mock infrastructure
# ---------------------------------------------------------------------------

_parse_handler: Callable[[ParseResponseInput], ParsedLLMResponse] | None = None
# Captured submit_batch_request inputs, keyed by output_type_name — opt-in
# regression coverage for the shared thinking fallback and the
# thinking-enabled max_tokens bump (workflow_blocks.THINKING_MAX_TOKENS).
# Additive only: tests that don't read this incur no behavior change.
_CAPTURED_SUBMIT_INPUTS: dict[str, BatchSubmitInput] = {}

# Recorded request_id → output_type_name, populated by the submit mock. The
# timer-loop fetch input carries only request_id, so the fetch mock recovers
# the output type from here (kept for symmetry with the real transport; the
# canned parse handler dispatches on output_type regardless).
_SUBMIT_OUTPUT_TYPES: dict[str, str] = {}


@activity.defn(name="submit_batch_request")
async def mock_submit_batch(input: BatchSubmitInput) -> BatchSubmitResult:
    """Echo the workflow-minted request_id (T4.1: the workflow always passes it)."""
    _CAPTURED_SUBMIT_INPUTS[input.output_type_name] = input
    _SUBMIT_OUTPUT_TYPES[input.request_id] = input.output_type_name
    return BatchSubmitResult(
        request_id=input.request_id,
        batch_id="msgbatch_mock123",
        provider="anthropic",
    )


@activity.defn(name="batch_status")
async def mock_batch_status_ended(input: BatchStatusInput) -> BatchStatusResult:
    """Report the batch as immediately ended — the timer loop breaks to fetch."""
    return BatchStatusResult(batch_id=input.batch_id, state="ended")


@activity.defn(name="fetch_batch_result")
async def mock_fetch_batch(input: FetchBatchResultInput) -> BatchFetchResult:
    """Return this waiter's inline canned body; parse dispatches on output_type."""
    return BatchFetchResult(raw_response_json='{"mock": true}')


@activity.defn(name="persist_to_store")
async def mock_persist_to_store(req: PersistRequest) -> PersistResult:
    """No-op survivable-write mock: workflows now persist after each LLM call."""
    return PersistResult(kind=req.kind, applied=True)


@activity.defn(name="parse_llm_response")
async def mock_parse_response(input: ParseResponseInput) -> ParsedLLMResponse:
    """Dispatch to section-specific parse handler."""
    assert _parse_handler is not None, "No parse handler set — call _reset_*() first"
    return _parse_handler(input)


def _make_parsed(
    model: BaseModel,
    *,
    model_name: str = "mock-model",
    input_tokens: int = 100,
    output_tokens: int = 50,
    latency_ms: float = 200.0,
) -> ParsedLLMResponse:
    """Build a ParsedLLMResponse from any Pydantic model."""
    return ParsedLLMResponse(
        parsed_json=model.model_dump_json(),
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )


@activity.defn(name="detect_file_conflicts_activity")
async def mock_detect_file_conflicts(
    input: DetectFileConflictsInput,
) -> DetectFileConflictsOutput:
    """Mock conflict detection using the pure classify_file_conflicts function."""
    non_conflicting, conflicts = classify_file_conflicts(input.sub_task_results)
    return DetectFileConflictsOutput(
        non_conflicting_files=non_conflicting,
        conflicts=conflicts,
    )


# ---------------------------------------------------------------------------
# Mock activities — registered by name to match workflow string references
# ---------------------------------------------------------------------------

# Mutable state shared across mock activities within a single test.
# Each test gets a fresh worker so there's no cross-test contamination.
_call_log: list[str] = []
_attempt_counter: int = 0
_transition_sequence: list[str] = []


def _single_step_parse_handler(input: ParseResponseInput) -> ParsedLLMResponse:
    _call_log.append("call_llm")
    return _make_parsed(_LLM_RESPONSE)


def _reset_mock_state(
    transitions: list[str] | None = None,
) -> None:
    global _attempt_counter, _parse_handler
    _call_log.clear()
    _attempt_counter = 0
    _transition_sequence.clear()
    _parse_handler = _single_step_parse_handler
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
        task_id=input.task_id,
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
    mock_persist_to_store,
    mock_create_worktree,
    mock_remove_worktree,
    mock_commit_changes,
    mock_assemble_context,
    mock_call_llm,
    mock_write_output,
    mock_validate_output,
    mock_evaluate_transition,
    mock_submit_batch,
    mock_batch_status_ended,
    mock_fetch_batch,
    mock_parse_response,
]


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


def _planned_parse_handler(input: ParseResponseInput) -> ParsedLLMResponse:
    global _PLAN_LLM_CALL_COUNT
    if input.output_type_name == "Plan":
        _PLAN_CALL_LOG.append("call_planner")
        return _make_parsed(_PLAN, model_name="mock-planner", input_tokens=300, output_tokens=150)
    # LLMResponse — counter-based (odd=models.py, even=api.py)
    _PLAN_LLM_CALL_COUNT += 1
    _PLAN_CALL_LOG.append(f"call_llm:{_PLAN_LLM_CALL_COUNT}")
    if _PLAN_LLM_CALL_COUNT % 2 == 1:
        files = [FileOutput(file_path="models.py", content="class Model: pass\n")]
    else:
        files = [FileOutput(file_path="api.py", content="def endpoint(): pass\n")]
    return _make_parsed(LLMResponse(files=files, explanation=f"LLM call #{_PLAN_LLM_CALL_COUNT}"))


def _reset_plan_mock_state(
    transitions: list[str] | None = None,
) -> None:
    global _PLAN_LLM_CALL_COUNT, _parse_handler
    _PLAN_CALL_LOG.clear()
    _PLAN_TRANSITION_SEQUENCE.clear()
    _PLAN_LLM_CALL_COUNT = 0
    _parse_handler = _planned_parse_handler
    if transitions:
        _PLAN_TRANSITION_SEQUENCE.extend(transitions)


@activity.defn(name="assemble_planner_context")
async def mock_assemble_planner_context(input: AssembleContextInput) -> PlannerInput:
    _PLAN_CALL_LOG.append("assemble_planner_context")
    return PlannerInput(
        task_id=input.task_id,
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
        task_id=input.task_id,
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
    mock_persist_to_store,
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
    mock_submit_batch,
    mock_batch_status_ended,
    mock_fetch_batch,
    mock_parse_response,
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
_SUBTASK_LLM_PARSE_COUNT: int = 0


def _subtask_parse_handler(input: ParseResponseInput) -> ParsedLLMResponse:
    global _SUBTASK_LLM_PARSE_COUNT
    _SUBTASK_CALL_LOG.append("call_llm")
    _SUBTASK_LLM_PARSE_COUNT += 1
    if _SUBTASK_LLM_PARSE_COUNT % 2 == 1:
        files = [FileOutput(file_path="schema.py", content="# schema\n")]
    else:
        files = [FileOutput(file_path="routes.py", content="# routes\n")]
    return _make_parsed(
        LLMResponse(files=files, explanation="Sub-task output."),
        input_tokens=50,
        output_tokens=25,
        latency_ms=100.0,
    )


def _reset_subtask_mock_state(transitions: list[str] | None = None) -> None:
    global _SUBTASK_LLM_PARSE_COUNT, _parse_handler
    _SUBTASK_CALL_LOG.clear()
    _SUBTASK_TRANSITION_SEQUENCE.clear()
    _SUBTASK_LLM_PARSE_COUNT = 0
    _parse_handler = _subtask_parse_handler
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
    mock_persist_to_store,
    mock_subtask_create_worktree,
    mock_subtask_remove_worktree,
    mock_assemble_sub_task_context,
    mock_subtask_call_llm,
    mock_subtask_write_output,
    mock_subtask_validate_output,
    mock_subtask_evaluate_transition,
    mock_submit_batch,
    mock_batch_status_ended,
    mock_fetch_batch,
    mock_parse_response,
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


_FANOUT_PARSE_LLM_COUNT: int = 0


def _fanout_parse_handler(input: ParseResponseInput) -> ParsedLLMResponse:
    global _FANOUT_PARSE_LLM_COUNT
    if input.output_type_name == "Plan":
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
        return _make_parsed(plan, model_name="mock-planner", input_tokens=300, output_tokens=150)
    if input.output_type_name == "ConflictResolutionResponse":
        _FANOUT_CALL_LOG.append("call_conflict_resolution")
        if _FANOUT_CONFLICT_RESOLUTION_RESPONSES:
            cr = _FANOUT_CONFLICT_RESOLUTION_RESPONSES.pop(0)
            response = ConflictResolutionResponse(
                resolved_files=[
                    FileOutput(file_path=k, content=v) for k, v in cr.resolved_files.items()
                ],
                explanation=cr.explanation,
            )
            return _make_parsed(
                response,
                model_name=cr.model_name,
                input_tokens=cr.input_tokens,
                output_tokens=cr.output_tokens,
                latency_ms=cr.latency_ms,
            )
        return _make_parsed(
            ConflictResolutionResponse(
                resolved_files=[],
                explanation="No conflicts resolved (default mock).",
            ),
            model_name="mock-reasoning",
            input_tokens=200,
            output_tokens=100,
            latency_ms=300.0,
        )
    # LLMResponse
    _FANOUT_CALL_LOG.append("call_llm")
    if _FANOUT_SUBTASK_LLM_RESPONSES:
        response = _FANOUT_SUBTASK_LLM_RESPONSES.pop(0)
    else:
        _FANOUT_PARSE_LLM_COUNT += 1
        if _FANOUT_PARSE_LLM_COUNT % 2 == 1:
            response = LLMResponse(
                files=[FileOutput(file_path="schema.py", content="# schema\n")],
                explanation="Created schema.",
            )
        else:
            response = LLMResponse(
                files=[FileOutput(file_path="routes.py", content="# routes\n")],
                explanation="Created routes.",
            )
    return _make_parsed(response, input_tokens=50, output_tokens=25, latency_ms=100.0)


def _reset_fanout_mock_state(
    step_transitions: list[str] | None = None,
    subtask_transitions: list[str] | None = None,
    subtask_responses: list[LLMResponse] | None = None,
    conflict_responses: list[ConflictResolutionCallResult] | None = None,
) -> None:
    global _FANOUT_PARSE_LLM_COUNT, _parse_handler
    _FANOUT_CALL_LOG.clear()
    _FANOUT_STEP_TRANSITIONS.clear()
    _FANOUT_SUBTASK_TRANSITIONS.clear()
    _FANOUT_SUBTASK_LLM_RESPONSES.clear()
    _FANOUT_CONFLICT_RESOLUTION_RESPONSES.clear()
    _FANOUT_PARSE_LLM_COUNT = 0
    _parse_handler = _fanout_parse_handler
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
        task_id=input.task_id,
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
    mock_persist_to_store,
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
    mock_detect_file_conflicts,
    mock_submit_batch,
    mock_batch_status_ended,
    mock_fetch_batch,
    mock_parse_response,
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
        task_id=input.task_id,
        system_prompt=f"step prompt for {input.step.step_id}",
        user_prompt=f"execute {input.step.step_id}",
    )


_MIXED_PARSE_COUNT: int = 0


def _mixed_parse_handler(input: ParseResponseInput) -> ParsedLLMResponse:
    global _MIXED_PARSE_COUNT
    if input.output_type_name == "Plan":
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
        return _make_parsed(plan, model_name="mock-planner", input_tokens=300, output_tokens=150)
    # LLMResponse — counter-based order: models, schema, routes, tests
    _MIXED_PARSE_COUNT += 1
    files_map = {
        1: [FileOutput(file_path="models.py", content="# models\n")],
        2: [FileOutput(file_path="schema.py", content="# schema\n")],
        3: [FileOutput(file_path="routes.py", content="# routes\n")],
        4: [FileOutput(file_path="tests.py", content="# tests\n")],
    }
    files = files_map.get(
        _MIXED_PARSE_COUNT,
        [FileOutput(file_path=f"unknown{_MIXED_PARSE_COUNT}.py", content="# unknown\n")],
    )
    return _make_parsed(
        LLMResponse(files=files, explanation=f"Call #{_MIXED_PARSE_COUNT}"),
        input_tokens=50,
        output_tokens=25,
        latency_ms=100.0,
    )


_MIXED_MOCK_ACTIVITIES = [
    mock_persist_to_store,
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
    mock_detect_file_conflicts,
    mock_submit_batch,
    mock_batch_status_ended,
    mock_fetch_batch,
    mock_parse_response,
]


class TestMixedPlan:
    """Sequential step → fan-out step → sequential step."""

    @pytest.mark.asyncio
    async def test_mixed_plan_succeeds(self, env: WorkflowEnvironment) -> None:
        global _MIXED_LLM_CALL_COUNT, _MIXED_PARSE_COUNT, _parse_handler
        _MIXED_LLM_CALL_COUNT = 0
        _MIXED_PARSE_COUNT = 0
        _reset_fanout_mock_state()
        _parse_handler = _mixed_parse_handler

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


def _p8_parse_handler(input: ParseResponseInput) -> ParsedLLMResponse:
    _P8_CALL_LOG.append("call_llm")
    return _make_parsed(
        LLMResponse(
            files=[FileOutput(file_path="hello.py", content="print('hello')\n")],
            explanation="output",
        ),
    )


def _reset_p8_state(
    transitions: list[str] | None = None,
    validate_responses: list[list[ValidationResult]] | None = None,
) -> None:
    global _parse_handler
    _P8_CALL_LOG.clear()
    _P8_TRANSITION_SEQUENCE.clear()
    _P8_ASSEMBLE_CONTEXT_INPUTS.clear()
    _P8_VALIDATE_RESPONSES.clear()
    _parse_handler = _p8_parse_handler
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
        task_id=input.task_id,
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
    mock_persist_to_store,
    mock_p8_create_worktree,
    mock_p8_remove_worktree,
    mock_p8_commit,
    mock_p8_assemble_context,
    mock_p8_call_llm,
    mock_p8_write_output,
    mock_p8_validate_output,
    mock_p8_evaluate_transition,
    mock_submit_batch,
    mock_batch_status_ended,
    mock_fetch_batch,
    mock_parse_response,
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


def _p8_step_parse_handler(input: ParseResponseInput) -> ParsedLLMResponse:
    if input.output_type_name == "Plan":
        _P8_STEP_CALL_LOG.append("call_planner")
        plan = Plan(
            task_id=input.task_id,
            steps=[PlanStep(step_id="step-1", description="Create.", target_files=["a.py"])],
            explanation="One step.",
        )
        return _make_parsed(plan, model_name="mock-planner", input_tokens=300, output_tokens=150)
    _P8_STEP_CALL_LOG.append("call_llm")
    return _make_parsed(
        LLMResponse(
            files=[FileOutput(file_path="a.py", content="# code\n")],
            explanation="step output",
        ),
    )


def _reset_p8_step_state(
    transitions: list[str] | None = None,
    validate_responses: list[list[ValidationResult]] | None = None,
) -> None:
    global _parse_handler
    _P8_STEP_CALL_LOG.clear()
    _P8_STEP_TRANSITION_SEQUENCE.clear()
    _P8_STEP_CONTEXT_INPUTS.clear()
    _P8_STEP_VALIDATE_RESPONSES.clear()
    _parse_handler = _p8_step_parse_handler
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
        task_id=input.task_id,
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
        task_id=input.task_id,
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
    mock_persist_to_store,
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
    mock_submit_batch,
    mock_batch_status_ended,
    mock_fetch_batch,
    mock_parse_response,
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


def _p8_st_parse_handler(input: ParseResponseInput) -> ParsedLLMResponse:
    _P8_ST_CALL_LOG.append("call_llm")
    return _make_parsed(
        LLMResponse(
            files=[FileOutput(file_path="schema.py", content="# schema\n")],
            explanation="sub-task output",
        ),
        input_tokens=50,
        output_tokens=25,
        latency_ms=100.0,
    )


def _reset_p8_st_state(
    transitions: list[str] | None = None,
    validate_responses: list[list[ValidationResult]] | None = None,
) -> None:
    global _parse_handler
    _P8_ST_CALL_LOG.clear()
    _P8_ST_TRANSITION_SEQUENCE.clear()
    _P8_ST_CONTEXT_INPUTS.clear()
    _P8_ST_VALIDATE_RESPONSES.clear()
    _parse_handler = _p8_st_parse_handler
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
    mock_persist_to_store,
    mock_p8st_create_worktree,
    mock_p8st_remove_worktree,
    mock_p8st_assemble_sub_task_context,
    mock_p8st_call_llm,
    mock_p8st_write_output,
    mock_p8st_validate_output,
    mock_p8st_evaluate_transition,
    mock_submit_batch,
    mock_batch_status_ended,
    mock_fetch_batch,
    mock_parse_response,
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


_RECURSIVE_PARSE_LLM_COUNT: int = 0


def _recursive_parse_handler(input: ParseResponseInput) -> ParsedLLMResponse:
    global _RECURSIVE_PARSE_LLM_COUNT
    if input.output_type_name == "ConflictResolutionResponse":
        _RECURSIVE_CALL_LOG.append("call_conflict_resolution")
        if _RECURSIVE_CONFLICT_RESPONSES:
            cr = _RECURSIVE_CONFLICT_RESPONSES.pop(0)
            response = ConflictResolutionResponse(
                resolved_files=[
                    FileOutput(file_path=k, content=v) for k, v in cr.resolved_files.items()
                ],
                explanation=cr.explanation,
            )
            return _make_parsed(
                response,
                model_name=cr.model_name,
                input_tokens=cr.input_tokens,
                output_tokens=cr.output_tokens,
                latency_ms=cr.latency_ms,
            )
        return _make_parsed(
            ConflictResolutionResponse(
                resolved_files=[],
                explanation="No conflicts resolved (default mock).",
            ),
            model_name="mock-reasoning",
            input_tokens=200,
            output_tokens=100,
            latency_ms=300.0,
        )
    # LLMResponse
    _RECURSIVE_CALL_LOG.append("call_llm")
    if _RECURSIVE_LLM_RESPONSES:
        response = _RECURSIVE_LLM_RESPONSES.pop(0)
    else:
        _RECURSIVE_PARSE_LLM_COUNT += 1
        if _RECURSIVE_PARSE_LLM_COUNT % 3 == 1:
            response = LLMResponse(
                files=[FileOutput(file_path="gc1.py", content="# gc1\n")],
                explanation="Grandchild 1 output.",
            )
        elif _RECURSIVE_PARSE_LLM_COUNT % 3 == 2:
            response = LLMResponse(
                files=[FileOutput(file_path="gc2.py", content="# gc2\n")],
                explanation="Grandchild 2 output.",
            )
        else:
            response = LLMResponse(
                files=[FileOutput(file_path="leaf.py", content="# leaf\n")],
                explanation="Leaf output.",
            )
    return _make_parsed(response, input_tokens=50, output_tokens=25, latency_ms=100.0)


def _reset_recursive_mock_state(
    transitions: list[str] | None = None,
    llm_responses: list[LLMResponse] | None = None,
    conflict_responses: list[ConflictResolutionCallResult] | None = None,
) -> None:
    global _RECURSIVE_PARSE_LLM_COUNT, _parse_handler
    _RECURSIVE_CALL_LOG.clear()
    _RECURSIVE_TRANSITION_SEQUENCE.clear()
    _RECURSIVE_LLM_RESPONSES.clear()
    _RECURSIVE_CONFLICT_RESPONSES.clear()
    _RECURSIVE_PARSE_LLM_COUNT = 0
    _parse_handler = _recursive_parse_handler
    _CAPTURED_SUBMIT_INPUTS.clear()
    _SUBMIT_OUTPUT_TYPES.clear()
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
    mock_persist_to_store,
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
    mock_detect_file_conflicts,
    mock_submit_batch,
    mock_batch_status_ended,
    mock_fetch_batch,
    mock_parse_response,
]


async def _run_recursive_subtask_workflow(
    env: WorkflowEnvironment,
    input: SubTaskInput,
    activities: list[Callable[..., object]] | None = None,
) -> SubTaskResult:
    """Helper to run the sub-task workflow with recursive mock activities.

    Pass ``activities`` to override the default mock set (e.g. to swap in a
    capturing closure for one activity) while reusing the client wiring.
    """
    async with Worker(
        env.client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=[ForgeSubTaskWorkflow],
        activities=activities if activities is not None else _RECURSIVE_MOCK_ACTIVITIES,
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
# Tests — nested fan-out honors resolve_conflicts=False (D71/D27, T1.5)
# ---------------------------------------------------------------------------


class TestRecursiveFanOutNoResolveConflicts:
    """Regression (T1.5): a nested fan-out with resolve_conflicts=False and a
    real conflict must terminate via the D27 terminal fallback, never resolve.

    Before T1.5 the nested gather ignored resolve_conflicts and always ran LLM
    resolution — this asserts the flag is now honored at depth >= 1.
    """

    @pytest.fixture
    def no_resolve_conflict_input(self) -> SubTaskInput:
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
            resolve_conflicts=False,
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_d27_terminal(
        self, env: WorkflowEnvironment, no_resolve_conflict_input: SubTaskInput
    ) -> None:
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
        )
        result = await _run_recursive_subtask_workflow(env, no_resolve_conflict_input)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.error is not None
        assert "File conflict" in result.error
        assert "conflict.py" in result.error
        # D27 fallback: LLM resolution must NOT be invoked.
        assert "assemble_conflict_resolution_context" not in _RECURSIVE_CALL_LOG
        assert "call_conflict_resolution" not in _RECURSIVE_CALL_LOG

    @pytest.mark.asyncio
    async def test_worktrees_cleaned_up_on_terminal(
        self, env: WorkflowEnvironment, no_resolve_conflict_input: SubTaskInput
    ) -> None:
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
        )
        await _run_recursive_subtask_workflow(env, no_resolve_conflict_input)
        # Nested node + 2 grandchildren all created and removed (D16).
        create_count = sum(1 for e in _RECURSIVE_CALL_LOG if e.startswith("create_worktree:"))
        remove_count = sum(1 for e in _RECURSIVE_CALL_LOG if e.startswith("remove_worktree:"))
        assert create_count == 3
        assert remove_count == 3


# ---------------------------------------------------------------------------
# Tests — nested fan-out propagates thinking + model_routing (T1.5)
# ---------------------------------------------------------------------------


class TestRecursiveFanOutPropagatesThinkingAndRouting:
    """thinking + model_routing propagate to a depth-1 nested node and reach its
    conflict-resolution LLM-call inputs (not the pre-T1.5 hardcoded defaults).

    Uses a 3-level tree (depth 0 -> 1 -> 2) so the resolving node sits at
    depth 1: st1 (depth 0) nests mid (depth 1), which nests gc1/gc2 (depth 2)
    that both target the same file.
    """

    @pytest.fixture
    def deep_conflict_input(self) -> SubTaskInput:
        return SubTaskInput(
            parent_task_id="parent-task",
            parent_description="Build an API.",
            sub_task=SubTask(
                sub_task_id="st1",
                description="Top nested node.",
                target_files=[],
                sub_tasks=[
                    SubTask(
                        sub_task_id="mid",
                        description="Mid nested node.",
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
                ],
            ),
            repo_root="/tmp/repo",
            parent_branch="forge/parent-task",
            max_attempts=1,
            depth=0,
            max_depth=2,
            # model_name is the leaf-generation model; conflict resolution must
            # route via model_routing.reasoning instead (distinct value).
            model_name="anthropic:sub-generation-model",
            model_routing=ModelConfig(reasoning="anthropic:custom-reasoning-model"),
            thinking=ThinkingPolicy(enabled=True, effort="max"),
            # Runs at the default 600s poll interval: the mode-aware batch
            # _child_timeout (T4.1 ST3c) now sizes the depth-1 ``mid`` node from its
            # permitted batch-wait budget ((max_attempts + remaining) * 25h), so the
            # two sequential batch phases (await grandchildren, then conflict
            # resolution) fit comfortably. This is the natural regression proof for
            # the ST3a fixture pin removal.
        )

    @pytest.mark.asyncio
    async def test_thinking_and_routing_reach_depth_one_resolution(
        self, env: WorkflowEnvironment, deep_conflict_input: SubTaskInput
    ) -> None:
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
                    task_id="parent-task.sub.st1.sub.mid",
                    resolved_files={"conflict.py": "# merged\n"},
                    explanation="Combined both.",
                    model_name="mock-reasoning",
                    input_tokens=200,
                    output_tokens=100,
                    latency_ms=300.0,
                ),
            ],
        )

        # Capture the ConflictResolutionInput reaching the assemble activity via
        # a test-local closure (no new module global; T5.5 will retire the rest).
        captured: list[ConflictResolutionInput] = []

        @activity.defn(name="assemble_conflict_resolution_context")
        async def capturing_assemble_cr(
            input: ConflictResolutionInput,
        ) -> ConflictResolutionCallInput:
            captured.append(input)
            _RECURSIVE_CALL_LOG.append("assemble_conflict_resolution_context")
            # Mirror the real assemble_conflict_resolution_context activity
            # (activities/conflict_resolution.py), which threads model_name and
            # thinking through from the input — a mock that dropped them would
            # silently mask propagation bugs downstream of this activity.
            return ConflictResolutionCallInput(
                task_id=input.task_id,
                step_id=input.step_id,
                system_prompt="conflict resolution system prompt",
                user_prompt="conflict resolution user prompt",
                model_name=input.model_name,
                thinking=input.thinking,
            )

        activities = [
            a for a in _RECURSIVE_MOCK_ACTIVITIES if a is not mock_recursive_assemble_cr_context
        ]
        activities.append(capturing_assemble_cr)

        result = await _run_recursive_subtask_workflow(env, deep_conflict_input, activities)

        assert result.status == TransitionSignal.SUCCESS
        # Exactly one resolution, at the depth-1 node ("...st1.sub.mid").
        assert len(captured) == 1
        cr_input = captured[0]
        assert cr_input.task_id == "parent-task.sub.st1.sub.mid"
        # thinking propagated (parent's, not the pre-T1.5 hardcoded default).
        assert cr_input.thinking == ThinkingPolicy(enabled=True, effort="max")
        # model_routing propagated: REASONING tier resolved from the parent's
        # ModelConfig, not the pre-T1.5 ModelConfig()/model_name override.
        assert cr_input.model_name == "anthropic:custom-reasoning-model"
        # Conflict resolution is thinking-enabled here, so its batch submit
        # must carry the explicit adaptive-thinking cap, not the generic
        # batch_submit_and_wait default (4096).
        submitted = _CAPTURED_SUBMIT_INPUTS["ConflictResolutionResponse"]
        assert submitted.thinking == ThinkingPolicy(enabled=True, effort="max")
        assert submitted.max_tokens == THINKING_MAX_TOKENS


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


def _sc_parse_handler(input: ParseResponseInput) -> ParsedLLMResponse:
    global _SC_LLM_CALL_COUNT
    if input.output_type_name == "Plan":
        _SC_CALL_LOG.append("call_planner")
        return _make_parsed(
            _SC_PLAN, model_name="mock-planner", input_tokens=300, output_tokens=150
        )
    if input.output_type_name == "SanityCheckResponse":
        _SC_CALL_LOG.append("call_sanity_check")
        if _SC_SANITY_RESPONSES:
            sc = _SC_SANITY_RESPONSES.pop(0)
            return _make_parsed(
                sc.response,
                model_name=sc.model_name,
                input_tokens=sc.input_tokens,
                output_tokens=sc.output_tokens,
                latency_ms=sc.latency_ms,
            )
        return _make_parsed(
            SanityCheckResponse(
                verdict=SanityCheckVerdict.CONTINUE,
                explanation="Plan looks good.",
            ),
            model_name="mock-reasoning",
            input_tokens=200,
            output_tokens=100,
            latency_ms=300.0,
        )
    # LLMResponse
    _SC_LLM_CALL_COUNT += 1
    _SC_CALL_LOG.append(f"call_llm:{_SC_LLM_CALL_COUNT}")
    files = [FileOutput(file_path=f"file{_SC_LLM_CALL_COUNT}.py", content="# code\n")]
    return _make_parsed(
        LLMResponse(files=files, explanation=f"LLM call #{_SC_LLM_CALL_COUNT}"),
    )


def _reset_sc_mock_state(
    transitions: list[str] | None = None,
    sanity_responses: list[SanityCheckCallResult] | None = None,
    plan: Plan | None = None,
) -> None:
    global _SC_LLM_CALL_COUNT, _SC_PLAN, _parse_handler
    _SC_CALL_LOG.clear()
    _SC_TRANSITION_SEQUENCE.clear()
    _SC_LLM_CALL_COUNT = 0
    _SC_SANITY_RESPONSES.clear()
    _parse_handler = _sc_parse_handler
    _CAPTURED_SUBMIT_INPUTS.clear()
    _SUBMIT_OUTPUT_TYPES.clear()
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
        task_id=input.task_id,
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
        task_id=input.task_id,
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
        task_id=input.task_id,
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
    mock_persist_to_store,
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
    mock_submit_batch,
    mock_batch_status_ended,
    mock_fetch_batch,
    mock_parse_response,
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
        # Sanity-check is thinking-enabled, so its batch submit must carry the
        # explicit adaptive-thinking cap, not the generic
        # batch_submit_and_wait default (4096).
        assert _CAPTURED_SUBMIT_INPUTS["SanityCheckResponse"].max_tokens == THINKING_MAX_TOKENS


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
_BATCH_PERSISTED: list[PersistRequest] = []
# Captured per submit_batch_request call — regression coverage for the shared
# thinking fallback (workflow_blocks.py) and the thinking-enabled max_tokens
# bump (both owner-adjudicated per the 2026-07 Phase 3 code review).
_BATCH_SUBMIT_INPUTS: list[BatchSubmitInput] = []


def _reset_batch_mock_state(
    transitions: list[str] | None = None,
    parse_responses: list[ParsedLLMResponse] | None = None,
) -> None:
    _BATCH_CALL_LOG.clear()
    _BATCH_TRANSITION_SEQUENCE.clear()
    _BATCH_PARSE_RESPONSES.clear()
    _BATCH_PERSISTED.clear()
    _BATCH_SUBMIT_INPUTS.clear()
    if transitions:
        _BATCH_TRANSITION_SEQUENCE.extend(transitions)
    if parse_responses:
        _BATCH_PARSE_RESPONSES.extend(parse_responses)


@activity.defn(name="persist_to_store")
async def mock_batch_persist_to_store(req: PersistRequest) -> PersistResult:
    """Capturing survivable-write mock: records each request so tests can assert
    that a FAILURE_TERMINAL run row was written on the batch-failure path (T1.6b)."""
    _BATCH_PERSISTED.append(req)
    return PersistResult(kind=req.kind, applied=True)


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
        task_id=input.task_id,
        system_prompt="system prompt",
        user_prompt="user prompt",
    )


@activity.defn(name="submit_batch_request")
async def mock_batch_submit(input: BatchSubmitInput) -> BatchSubmitResult:
    _BATCH_CALL_LOG.append("submit_batch_request")
    _BATCH_SUBMIT_INPUTS.append(input)
    # Echo the workflow-minted request_id (T4.1: the workflow always passes it).
    return BatchSubmitResult(
        request_id=input.request_id,
        batch_id="msgbatch_test123",
        provider="anthropic",
    )


@activity.defn(name="batch_status")
async def mock_batch_status(input: BatchStatusInput) -> BatchStatusResult:
    """Report the batch ended — the timer loop breaks straight to fetch."""
    _BATCH_CALL_LOG.append("batch_status")
    return BatchStatusResult(batch_id=input.batch_id, state="ended")


@activity.defn(name="batch_status")
async def mock_batch_status_pending(input: BatchStatusInput) -> BatchStatusResult:
    """Never end: the batch stays in_progress so the poll loop runs to the 25h
    ceiling (fast-forwarded by the time-skipping env) and gives up with MISSING."""
    _BATCH_CALL_LOG.append("batch_status")
    return BatchStatusResult(batch_id=input.batch_id, state="in_progress")


@activity.defn(name="batch_status")
async def mock_batch_status_failed(input: BatchStatusInput) -> BatchStatusResult:
    """Report a provider-terminal failure — the poll loop persists FAILED and
    raises a non-retryable ApplicationError (the terminal-status fast-fail path)."""
    _BATCH_CALL_LOG.append("batch_status")
    return BatchStatusResult(batch_id=input.batch_id, state="failed")


@activity.defn(name="fetch_batch_result")
async def mock_batch_fetch(input: FetchBatchResultInput) -> BatchFetchResult:
    """Return this waiter's inline body; the parse mock produces the parsed result."""
    _BATCH_CALL_LOG.append("fetch_batch_result")
    return BatchFetchResult(raw_response_json='{"dummy": "json"}')


@activity.defn(name="fetch_batch_result")
async def mock_batch_fetch_error(input: FetchBatchResultInput) -> BatchFetchResult:
    """Return an error-bearing fetch — a failed result line / absent custom_id. The
    waiter turns it into a non-retryable ApplicationError (T4.1) that run()'s
    failure-symmetry handler catches instead of crashing the workflow."""
    _BATCH_CALL_LOG.append("fetch_batch_result")
    return BatchFetchResult(error="Batch expired")


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
    mock_batch_persist_to_store,
    mock_batch_create_worktree,
    mock_batch_remove_worktree,
    mock_batch_commit_changes,
    mock_batch_assemble_context,
    mock_batch_submit,
    mock_batch_status,
    mock_batch_fetch,
    mock_batch_parse,
    mock_batch_write_output,
    mock_batch_validate_output,
    mock_batch_evaluate_transition,
]

# Same as above but the batch never ends, so the poll loop runs to the 25h
# ceiling (fast-forwarded by the time-skipping env) — the wait-timeout path.
_BATCH_TIMEOUT_ACTIVITIES = [
    mock_batch_persist_to_store,
    mock_batch_create_worktree,
    mock_batch_remove_worktree,
    mock_batch_commit_changes,
    mock_batch_assemble_context,
    mock_batch_submit,
    mock_batch_status_pending,
    mock_batch_fetch,
    mock_batch_parse,
    mock_batch_write_output,
    mock_batch_validate_output,
    mock_batch_evaluate_transition,
]

# The batch ends but the fetch returns an error — the fast-failure path.
_BATCH_FETCH_ERROR_ACTIVITIES = [
    mock_batch_persist_to_store,
    mock_batch_create_worktree,
    mock_batch_remove_worktree,
    mock_batch_commit_changes,
    mock_batch_assemble_context,
    mock_batch_submit,
    mock_batch_status,
    mock_batch_fetch_error,
    mock_batch_parse,
    mock_batch_write_output,
    mock_batch_validate_output,
    mock_batch_evaluate_transition,
]

# The provider reports the batch FAILED — the terminal-status fast-fail path.
_BATCH_FAILED_STATUS_ACTIVITIES = [
    mock_batch_persist_to_store,
    mock_batch_create_worktree,
    mock_batch_remove_worktree,
    mock_batch_commit_changes,
    mock_batch_assemble_context,
    mock_batch_submit,
    mock_batch_status_failed,
    mock_batch_fetch,
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
            # The timer loop polls batch_status (mocked "ended") then fetches the
            # result — no signal needed.
            result = await env.client.execute_workflow(
                ForgeTaskWorkflow.run,
                input,
                id="test-batch-single",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.status == TransitionSignal.SUCCESS
        assert "submit_batch_request" in _BATCH_CALL_LOG
        assert "parse_llm_response:LLMResponse" in _BATCH_CALL_LOG
        # Verify sync path was NOT called
        assert "call_llm" not in _BATCH_CALL_LOG
        assert result.output_files == {"hello.py": "print('hello')\n"}
        # generation_dispatch omits `thinking`; the shared fallback in
        # batch_submit_and_wait must resolve it to disabled — not to
        # ThinkingPolicy()'s own enabled=True default (D94) and not to the
        # task-level ForgeTaskInput.thinking (enabled=True here).
        assert len(_BATCH_SUBMIT_INPUTS) == 1
        assert _BATCH_SUBMIT_INPUTS[0].thinking == ThinkingPolicy(enabled=False)
        # Generation is thinking-disabled, so its cap stays the untouched
        # batch_submit_and_wait default — not the thinking-enabled bump.
        assert _BATCH_SUBMIT_INPUTS[0].max_tokens == 4096

    @pytest.mark.asyncio
    async def test_batch_generation_persists_tokens_and_stop_reason(
        self, env: WorkflowEnvironment
    ) -> None:
        """The interactions row built for a batch result carries the parsed
        response's token counts and stop_reason through end to end (2026-07
        Phase 3 code review, item 3a/3b) — not silently dropped or zeroed
        anywhere between parse_llm_response and the persisted row."""
        distinctive_parsed = ParsedLLMResponse(
            parsed_json=LLMResponse(
                files=[FileOutput(file_path="hello.py", content="print('hello')\n")],
                explanation="Created hello module.",
            ).model_dump_json(),
            model_name="mock-batch-model",
            input_tokens=777,
            output_tokens=888,
            cache_creation_input_tokens=13,
            cache_read_input_tokens=17,
            stop_reason="end_turn",
        )
        _reset_batch_mock_state(
            transitions=[TransitionSignal.SUCCESS.value],
            parse_responses=[distinctive_parsed],
        )

        task = TaskDefinition(
            task_id="batch-token-test",
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
            result = await env.client.execute_workflow(
                ForgeTaskWorkflow.run,
                input,
                id="test-batch-token-check",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.status == TransitionSignal.SUCCESS
        interactions = [r for r in _BATCH_PERSISTED if r.kind == "interaction" and r.role == "llm"]
        assert len(interactions) == 1
        row = interactions[0]
        assert row.input_tokens == 777
        assert row.output_tokens == 888
        assert row.cache_creation_input_tokens == 13
        assert row.cache_read_input_tokens == 17
        assert row.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_batch_fetch_error_records_failure(self, env: WorkflowEnvironment) -> None:
        """An error-bearing fetch (T4.1 fast failure) ends in a graceful
        FAILURE_TERMINAL run row + cleaned worktree, not a raw workflow crash (T1.6b)."""
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
            activities=_BATCH_FETCH_ERROR_ACTIVITIES,
        ):
            # The batch ends but its fetch carries an error; the waiter raises a
            # non-retryable ApplicationError that run()'s failure-symmetry handler
            # catches instead of letting it crash the workflow.
            result = await env.client.execute_workflow(
                ForgeTaskWorkflow.run,
                input,
                id="test-batch-error",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert "ApplicationError" in (result.error or "")
        assert "Batch expired" in (result.error or "")
        # Worktree was cleaned — no orphan left behind.
        assert "remove_worktree" in _BATCH_CALL_LOG
        # Exactly one FAILURE_TERMINAL run row was persisted (same PersistRun the
        # success path uses, keyed on (workflow_id, run_id)).
        runs = [r for r in _BATCH_PERSISTED if r.kind == "run"]
        assert len(runs) == 1
        assert runs[0].task_result.status == TransitionSignal.FAILURE_TERMINAL


# ---------------------------------------------------------------------------
# Tests — batch-wait failure symmetry (T1.6b)
# ---------------------------------------------------------------------------

# Sub-task workflow batch activities whose batch never ends, so the 25h batch
# wait runs to the ceiling (fast-forwarded by the time-skipping env) inside the
# sub-task and gives up with MISSING.
_SUBTASK_TIMEOUT_ACTIVITIES = [
    mock_persist_to_store,
    mock_subtask_create_worktree,
    mock_subtask_remove_worktree,
    mock_assemble_sub_task_context,
    mock_subtask_write_output,
    mock_subtask_validate_output,
    mock_subtask_evaluate_transition,
    mock_batch_submit,  # echoes request_id
    mock_batch_status_pending,  # in_progress forever → 25h ceiling
    mock_fetch_batch,  # never reached
    mock_parse_response,
]


class TestBatchWaitFailure:
    """A batch wait that times out or errors leaves a run row and no orphan (T1.6b)."""

    @pytest.mark.asyncio
    async def test_wait_timeout_records_failure_and_cleans_worktree(
        self, env: WorkflowEnvironment
    ) -> None:
        """The batch never ends, so the poll loop runs to the 25h ceiling and gives
        up with a non-retryable ApplicationError; run() records a terminal row and
        removes the worktree instead of crashing out with an orphan."""
        _reset_batch_mock_state()

        task = TaskDefinition(
            task_id="batch-timeout",
            description="Timeout test.",
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
            activities=_BATCH_TIMEOUT_ACTIVITIES,
        ):
            # batch_status returns in_progress forever, so the poll loop runs to
            # the 25h ceiling (the time-skipping env fast-forwards every sleep) and
            # gives up.
            result = await env.client.execute_workflow(
                ForgeTaskWorkflow.run,
                input,
                id="test-batch-wait-timeout",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert "ApplicationError" in (result.error or "")
        # Worktree was cleaned — no orphan.
        assert "remove_worktree" in _BATCH_CALL_LOG
        # Exactly one FAILURE_TERMINAL run row was persisted (same PersistRun the
        # success path uses).
        runs = [r for r in _BATCH_PERSISTED if r.kind == "run"]
        assert len(runs) == 1
        assert runs[0].task_result.status == TransitionSignal.FAILURE_TERMINAL
        # The waiter recorded a terminal MISSING outcome before giving up.
        outcomes = [
            r for r in _BATCH_PERSISTED if r.kind == "batch_outcome" and r.status == "missing"
        ]
        assert len(outcomes) == 1

    @pytest.mark.asyncio
    async def test_provider_terminal_status_records_failure_and_cleans_worktree(
        self, env: WorkflowEnvironment
    ) -> None:
        """A provider-terminal batch status (FAILED) records a terminal FAILED
        outcome and raises a non-retryable ApplicationError; run() records a
        terminal row and removes the worktree instead of crashing (T1.6b)."""
        _reset_batch_mock_state()

        task = TaskDefinition(
            task_id="batch-provider-fail",
            description="Provider-failure test.",
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
            activities=_BATCH_FAILED_STATUS_ACTIVITIES,
        ):
            result = await env.client.execute_workflow(
                ForgeTaskWorkflow.run,
                input,
                id="test-batch-provider-fail",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert "ApplicationError" in (result.error or "")
        assert "remove_worktree" in _BATCH_CALL_LOG
        runs = [r for r in _BATCH_PERSISTED if r.kind == "run"]
        assert len(runs) == 1
        assert runs[0].task_result.status == TransitionSignal.FAILURE_TERMINAL
        # The waiter recorded a terminal FAILED outcome before raising.
        outcomes = [
            r for r in _BATCH_PERSISTED if r.kind == "batch_outcome" and r.status == "failed"
        ]
        assert len(outcomes) == 1

    @pytest.mark.asyncio
    async def test_subtask_wait_timeout_returns_terminal_and_cleans_worktree(
        self, env: WorkflowEnvironment
    ) -> None:
        """A sub-task batch wait timing out returns a FAILURE_TERMINAL SubTaskResult
        (so the parent records the run row) and removes its own worktree (T1.6b)."""
        _reset_subtask_mock_state()

        input = SubTaskInput(
            parent_task_id="parent-task",
            parent_description="Build an API.",
            sub_task=SubTask(
                sub_task_id="st1",
                description="Analyze schema.",
                target_files=["schema.py"],
            ),
            repo_root="/tmp/repo",
            parent_branch="forge/parent-task",
            max_attempts=1,
            sync_mode=False,
        )

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeSubTaskWorkflow],
            activities=_SUBTASK_TIMEOUT_ACTIVITIES,
        ):
            result = await env.client.execute_workflow(
                ForgeSubTaskWorkflow.run,
                input,
                id="test-subtask-wait-timeout",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.sub_task_id == "st1"
        assert "ApplicationError" in (result.error or "")
        # The sub-task's own compound-id worktree was removed — no orphan.
        assert "remove_worktree:parent-task.sub.st1" in _SUBTASK_CALL_LOG


# ---------------------------------------------------------------------------
# Tests — batch planned workflow
# ---------------------------------------------------------------------------

# Additional mock activities needed for planned batch tests

_BATCH_PLAN_CALL_LOG: list[str] = []
_BATCH_PLAN_TRANSITION_SEQUENCE: list[str] = []
_BATCH_PLAN_PARSE_QUEUE: list[ParsedLLMResponse] = []
# Captured submit_batch_request inputs keyed by output_type_name — regression
# coverage for the shared thinking fallback: the planner call passes an
# explicit (non-None) thinking policy through unchanged, while the generation
# call omits it and must land disabled.
_BATCH_PLAN_SUBMIT_INPUTS: dict[str, BatchSubmitInput] = {}

# Distinctive planner thinking policy — proves passthrough rather than
# coincidentally matching some other default.
_PLANNER_THINKING = ThinkingPolicy(enabled=True, effort="high")


def _reset_batch_plan_mock_state(
    transitions: list[str] | None = None,
    parse_queue: list[ParsedLLMResponse] | None = None,
) -> None:
    _BATCH_PLAN_CALL_LOG.clear()
    _BATCH_PLAN_TRANSITION_SEQUENCE.clear()
    _BATCH_PLAN_PARSE_QUEUE.clear()
    _BATCH_PLAN_SUBMIT_INPUTS.clear()
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
        task_id=input.task_id,
        system_prompt="planner system",
        user_prompt="planner user",
        thinking=_PLANNER_THINKING,
    )


@activity.defn(name="submit_batch_request")
async def mock_bp_submit_batch(input: BatchSubmitInput) -> BatchSubmitResult:
    _BATCH_PLAN_CALL_LOG.append(f"submit_batch:{input.output_type_name}")
    _BATCH_PLAN_SUBMIT_INPUTS[input.output_type_name] = input
    # Echo the workflow-minted request_id (T4.1). The planner and generation
    # calls run sequentially, so their fetch/parse pairs stay ordered.
    return BatchSubmitResult(
        request_id=input.request_id, batch_id="msgbatch_bp123", provider="anthropic"
    )


@activity.defn(name="batch_status")
async def mock_bp_batch_status(input: BatchStatusInput) -> BatchStatusResult:
    return BatchStatusResult(batch_id=input.batch_id, state="ended")


@activity.defn(name="fetch_batch_result")
async def mock_bp_fetch(input: FetchBatchResultInput) -> BatchFetchResult:
    return BatchFetchResult(raw_response_json='{"mock": true}')


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
        task_id=input.task_id,
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
    mock_persist_to_store,
    mock_bp_create_worktree,
    mock_bp_assemble_planner,
    mock_bp_submit_batch,
    mock_bp_batch_status,
    mock_bp_fetch,
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
            # Two sequential batch calls (planner then generation); each polls
            # batch_status (mocked "ended") and fetches — no signals.
            result = await env.client.execute_workflow(
                ForgeTaskWorkflow.run,
                input,
                id="test-batch-planned",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.status == TransitionSignal.SUCCESS
        assert "submit_batch:Plan" in _BATCH_PLAN_CALL_LOG
        assert "submit_batch:LLMResponse" in _BATCH_PLAN_CALL_LOG
        assert "parse:Plan" in _BATCH_PLAN_CALL_LOG
        assert "parse:LLMResponse" in _BATCH_PLAN_CALL_LOG
        assert result.plan is not None
        assert len(result.step_results) == 1
        # Shared thinking fallback (workflow_blocks.py): the planner call
        # passes an explicit thinking policy through unchanged...
        assert _BATCH_PLAN_SUBMIT_INPUTS["Plan"].thinking == _PLANNER_THINKING
        # ...while the generation call omits `thinking` entirely and must
        # land disabled via the shared fallback, not enabled-by-default.
        assert _BATCH_PLAN_SUBMIT_INPUTS["LLMResponse"].thinking == ThinkingPolicy(enabled=False)
        # Planner is thinking-enabled, so it carries the explicit
        # adaptive-thinking cap; generation stays thinking-disabled and keeps
        # the untouched default.
        assert _BATCH_PLAN_SUBMIT_INPUTS["Plan"].max_tokens == THINKING_MAX_TOKENS
        assert _BATCH_PLAN_SUBMIT_INPUTS["LLMResponse"].max_tokens == 4096


# ---------------------------------------------------------------------------
# Tests — sync_mode defaults to False (batch mode)
# ---------------------------------------------------------------------------


class TestSyncModeDefaultBatchMode:
    """Verify that sync_mode defaults to False (batch mode is default)."""

    def test_default_sync_mode_is_false(self) -> None:
        task = TaskDefinition(task_id="t1", description="Test.")
        input = ForgeTaskInput(task=task, repo_root="/repo")
        assert input.sync_mode is False

    def test_subtask_default_sync_mode_is_false(self) -> None:
        input = SubTaskInput(
            parent_task_id="p",
            parent_description="Parent.",
            sub_task=SubTask(sub_task_id="s", description="Sub.", target_files=["x.py"]),
            repo_root="/repo",
            parent_branch="main",
        )
        assert input.sync_mode is False


# ===========================================================================
# T4.1 ST3c — batch-mode fan-out child execution-timeout derivation
# ===========================================================================
#
# A real ForgeTaskWorkflow (planned, one single-child fan-out step) spawns a real
# ForgeSubTaskWorkflow child, in batch mode. The mode-aware _child_timeout now sizes
# the child from its permitted batch-wait budget, so:
#   (a) a child survives a multi-poll (>20 min) batch turnaround the old fixed
#       15-20 min ceiling would have killed; and
#   (b) a child whose batch never ends still hits its own 25h wait ceiling, cleans its
#       worktree, and returns FAILURE_TERMINAL, so the live parent records a run row
#       (T1.6b failure symmetry extended to a spawned child).
#
# The planner batch always ends immediately, so the workflow reaches the fan-out step;
# only the child's generation batch ("LLMResponse") is delayed/stalled. The status mock
# recovers the waiter's output type from the request_id the submit mock encodes into the
# batch_id, and mutates module-level scenario dicts (no ``global`` statements).

_ST3C_CALL_LOG: list[str] = []
_ST3C_PERSISTED: list[PersistRequest] = []
# request_id -> output_type_name, recorded at submit so the status mock decides per
# waiter whether (and how long) its batch stays in_progress.
_ST3C_REQUEST_TYPES: dict[str, str] = {}
# batch_id -> number of batch_status polls seen so far.
_ST3C_STATUS_POLLS: dict[str, int] = {}
# Scenario knob: how many in_progress polls the child's generation batch returns before
# ``ended``. A value larger than 25h / 600s (= 150) stalls the child to its 25h ceiling.
_ST3C_CONFIG: dict[str, int] = {"gen_in_progress_polls": 0}

_ST3C_STALL_POLLS = 10_000  # >> 150, so the generation batch never ends within 25h


def _reset_st3c_mock_state(gen_in_progress_polls: int) -> None:
    _ST3C_CALL_LOG.clear()
    _ST3C_PERSISTED.clear()
    _ST3C_REQUEST_TYPES.clear()
    _ST3C_STATUS_POLLS.clear()
    _ST3C_CONFIG["gen_in_progress_polls"] = gen_in_progress_polls


@activity.defn(name="persist_to_store")
async def mock_st3c_persist(req: PersistRequest) -> PersistResult:
    _ST3C_PERSISTED.append(req)
    return PersistResult(kind=req.kind, applied=True)


@activity.defn(name="create_worktree_activity")
async def mock_st3c_create_worktree(input: CreateWorktreeInput) -> CreateWorktreeOutput:
    _ST3C_CALL_LOG.append(f"create_worktree:{input.task_id}")
    return CreateWorktreeOutput(
        worktree_path=f"/tmp/repo/.forge-worktrees/{input.task_id}",
        branch_name=f"forge/{input.task_id}",
    )


@activity.defn(name="remove_worktree_activity")
async def mock_st3c_remove_worktree(input: RemoveWorktreeInput) -> None:
    _ST3C_CALL_LOG.append(f"remove_worktree:{input.task_id}")


@activity.defn(name="commit_changes_activity")
async def mock_st3c_commit(input: CommitChangesInput) -> CommitChangesOutput:
    _ST3C_CALL_LOG.append(f"commit:{input.status}")
    return CommitChangesOutput(commit_sha="c" * 40)


@activity.defn(name="assemble_planner_context")
async def mock_st3c_assemble_planner(input: AssembleContextInput) -> PlannerInput:
    return PlannerInput(
        task_id=input.task_id,
        system_prompt="planner system",
        user_prompt="planner user",
    )


@activity.defn(name="assemble_sub_task_context")
async def mock_st3c_assemble_sub_task(input: AssembleSubTaskContextInput) -> AssembledContext:
    return AssembledContext(
        task_id=input.parent_task_id,
        system_prompt=f"sub-task prompt for {input.sub_task.sub_task_id}",
        user_prompt=f"execute {input.sub_task.sub_task_id}",
    )


@activity.defn(name="submit_batch_request")
async def mock_st3c_submit(input: BatchSubmitInput) -> BatchSubmitResult:
    _ST3C_REQUEST_TYPES[input.request_id] = input.output_type_name
    # Encode the request_id in the batch_id so the thin batch_status input (batch_id
    # only) can recover this waiter's output type.
    return BatchSubmitResult(
        request_id=input.request_id,
        batch_id=f"batch-{input.request_id}",
        provider="anthropic",
    )


@activity.defn(name="batch_status")
async def mock_st3c_batch_status(input: BatchStatusInput) -> BatchStatusResult:
    request_id = input.batch_id.removeprefix("batch-")
    output_type = _ST3C_REQUEST_TYPES.get(request_id, "")
    seen = _ST3C_STATUS_POLLS.get(input.batch_id, 0)
    _ST3C_STATUS_POLLS[input.batch_id] = seen + 1
    # Only the child's generation batch is delayed; the planner batch ends at once so
    # the workflow reaches the fan-out step.
    if output_type == "LLMResponse" and seen < _ST3C_CONFIG["gen_in_progress_polls"]:
        return BatchStatusResult(batch_id=input.batch_id, state="in_progress")
    return BatchStatusResult(batch_id=input.batch_id, state="ended")


@activity.defn(name="fetch_batch_result")
async def mock_st3c_fetch(input: FetchBatchResultInput) -> BatchFetchResult:
    return BatchFetchResult(raw_response_json='{"mock": true}')


@activity.defn(name="parse_llm_response")
async def mock_st3c_parse(input: ParseResponseInput) -> ParsedLLMResponse:
    if input.output_type_name == "Plan":
        plan = Plan(
            task_id=input.task_id,
            steps=[
                PlanStep(
                    step_id="fan-step",
                    description="Single-child fan-out step.",
                    target_files=[],
                    sub_tasks=[
                        SubTask(
                            sub_task_id="st1",
                            description="Do the thing.",
                            target_files=["a.py"],
                        ),
                    ],
                ),
            ],
            explanation="One fan-out step.",
        )
        return _make_parsed(plan, model_name="mock-planner")
    return _make_parsed(
        LLMResponse(
            files=[FileOutput(file_path="a.py", content="# child\n")],
            explanation="Child output.",
        )
    )


@activity.defn(name="write_output")
async def mock_st3c_write_output(input: WriteOutputInput) -> WriteResult:
    files = input.llm_result.response.files
    return WriteResult(
        task_id=input.llm_result.task_id,
        files_written=[f.file_path for f in files],
        output_files={f.file_path: f.content for f in files},
    )


@activity.defn(name="write_files")
async def mock_st3c_write_files(input: WriteFilesInput) -> WriteResult:
    return WriteResult(task_id=input.task_id, files_written=list(input.files.keys()))


@activity.defn(name="validate_output")
async def mock_st3c_validate(input: ValidateOutputInput) -> list[ValidationResult]:
    return [ValidationResult(check_name="ruff_lint", passed=True, summary="passed")]


@activity.defn(name="evaluate_transition")
async def mock_st3c_transition(input: TransitionInput) -> str:
    return TransitionSignal.SUCCESS.value


_ST3C_MOCK_ACTIVITIES = [
    mock_st3c_persist,
    mock_st3c_create_worktree,
    mock_st3c_remove_worktree,
    mock_st3c_commit,
    mock_st3c_assemble_planner,
    mock_st3c_assemble_sub_task,
    mock_st3c_submit,
    mock_st3c_batch_status,
    mock_st3c_fetch,
    mock_st3c_parse,
    mock_st3c_write_output,
    mock_st3c_write_files,
    mock_st3c_validate,
    mock_st3c_transition,
    mock_detect_file_conflicts,
]


async def _run_st3c_workflow(env: WorkflowEnvironment, task_id: str) -> TaskResult:
    """Run a planned, single-fan-out-step ForgeTaskWorkflow (batch mode) that spawns a
    real ForgeSubTaskWorkflow child, with the child execution timeout derived by the
    mode-aware _child_timeout — no execution_timeout override on the parent."""
    task = TaskDefinition(task_id=task_id, description="Build a thing.")
    task_input = ForgeTaskInput(
        task=task,
        repo_root="/tmp/repo",
        plan=True,
        max_exploration_rounds=0,
        sync_mode=False,  # batch mode
    )
    async with Worker(
        env.client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=[ForgeTaskWorkflow, ForgeSubTaskWorkflow],
        activities=_ST3C_MOCK_ACTIVITIES,
    ):
        return await env.client.execute_workflow(
            ForgeTaskWorkflow.run,
            task_input,
            id=f"test-{task_id}",
            task_queue=FORGE_TASK_QUEUE,
        )


class TestBatchFanOutChildTimeoutDerivation:
    """The child execution timeout is derived from its batch-wait budget (T4.1 ST3c)."""

    @pytest.mark.asyncio
    async def test_child_survives_slow_batch_turnaround(self, env: WorkflowEnvironment) -> None:
        """AC (a): the child's generation batch stays in_progress for 3 polls (>20 min
        of workflow time at the 600s default) before ending. The old fixed 15-20-min
        _child_timeout would have killed the child; the derived batch budget lets it
        finish, so the fan-out step and the workflow succeed."""
        _reset_st3c_mock_state(gen_in_progress_polls=3)

        result = await _run_st3c_workflow(env, "st3c-slow")

        assert result.status == TransitionSignal.SUCCESS
        # The single fan-out step succeeded.
        assert len(result.step_results) == 1
        assert result.step_results[0].status == TransitionSignal.SUCCESS
        assert result.step_results[0].sub_task_results[0].status == TransitionSignal.SUCCESS
        # The slow path really engaged: the child's generation batch was polled at
        # least 4 times (3 in_progress + 1 ended) — >20 min at the 600s poll floor.
        assert max(_ST3C_STATUS_POLLS.values()) >= 4

    @pytest.mark.asyncio
    async def test_child_ceiling_expiry_cleans_worktree_and_records_run(
        self, env: WorkflowEnvironment
    ) -> None:
        """AC (b): the child's generation batch never ends, so the child hits its own
        25h wait ceiling, cleans its compound-id worktree, and returns FAILURE_TERMINAL
        to the parent; the live parent then records a terminal run row (T1.6b symmetry
        extended to a spawned child)."""
        _reset_st3c_mock_state(gen_in_progress_polls=_ST3C_STALL_POLLS)

        result = await _run_st3c_workflow(env, "st3c-stall")

        # 1) The parent workflow ended terminally...
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        # 2) ...because the child returned FAILURE_TERMINAL from the fan-out step.
        assert len(result.step_results) == 1
        assert result.step_results[0].status == TransitionSignal.FAILURE_TERMINAL
        child_results = result.step_results[0].sub_task_results
        assert len(child_results) == 1
        assert child_results[0].status == TransitionSignal.FAILURE_TERMINAL
        # 3) The child cleaned its own compound-id worktree — no orphan.
        assert "remove_worktree:st3c-stall.sub.st1" in _ST3C_CALL_LOG
        # 4) The live parent persisted exactly one FAILURE_TERMINAL run row.
        runs = [r for r in _ST3C_PERSISTED if r.kind == "run"]
        assert len(runs) == 1
        assert runs[0].task_result.status == TransitionSignal.FAILURE_TERMINAL
        # The child recorded a terminal MISSING batch outcome before giving up.
        missing = [
            r for r in _ST3C_PERSISTED if r.kind == "batch_outcome" and r.status == "missing"
        ]
        assert len(missing) == 1
