"""End-to-end integration tests for the Forge workflow.

Exercises the full pipeline with real git operations, real file I/O, and real
validation — only mocking the LLM call — to prove Phase 1 Definition of Done.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from forge.activities import (
    assemble_context,
    assemble_step_context,
    assemble_sub_task_context,
    commit_changes_activity,
    create_worktree_activity,
    evaluate_transition,
    remove_worktree_activity,
    reset_worktree_activity,
    validate_output,
    write_files,
    write_output,
)
from forge.models import (
    AssembleContextInput,
    AssembledContext,
    FileOutput,
    ForgeTaskInput,
    LLMCallResult,
    LLMResponse,
    Plan,
    PlanCallResult,
    PlannerInput,
    PlanStep,
    SubTask,
    TaskDefinition,
    TaskResult,
    TransitionSignal,
)
from forge.workflows import FORGE_TASK_QUEUE, ForgeSubTaskWorkflow, ForgeTaskWorkflow

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment


# ---------------------------------------------------------------------------
# Valid Python code that passes ruff check + ruff format --check
# ---------------------------------------------------------------------------

_VALID_PYTHON = '''\
def greet(name: str) -> str:
    """Return a greeting for *name*."""
    return f"Hello, {name}!"
'''

_INVALID_PYTHON = "def greet(name):\n  x=1\n  return f'Hello, {name}!'\n  y = 2\n"

# Code with auto-fixable issues: typing.List instead of list, trailing whitespace
_FIXABLE_PYTHON = '''\
from typing import List


def names() -> List[str]:
    """Return a list of names."""
    return ["Alice", "Bob"]
'''

# Expected content after ruff auto-fix
_FIXED_PYTHON = '''\
def names() -> list[str]:
    """Return a list of names."""
    return ["Alice", "Bob"]
'''


# ---------------------------------------------------------------------------
# Mock LLM activities
# ---------------------------------------------------------------------------


@activity.defn(name="call_llm")
async def mock_call_llm_valid(context: AssembledContext) -> LLMCallResult:
    """Return valid, ruff-clean Python code."""
    return LLMCallResult(
        task_id=context.task_id,
        response=LLMResponse(
            files=[FileOutput(file_path="hello.py", content=_VALID_PYTHON)],
            explanation="Created greeting module.",
        ),
        model_name="mock-model",
        input_tokens=100,
        output_tokens=50,
        latency_ms=200.0,
    )


@activity.defn(name="call_llm")
async def mock_call_llm_invalid(context: AssembledContext) -> LLMCallResult:
    """Return Python code that will fail ruff validation."""
    return LLMCallResult(
        task_id=context.task_id,
        response=LLMResponse(
            files=[FileOutput(file_path="hello.py", content=_INVALID_PYTHON)],
            explanation="Created greeting module (with issues).",
        ),
        model_name="mock-model",
        input_tokens=100,
        output_tokens=50,
        latency_ms=200.0,
    )


@activity.defn(name="call_llm")
async def mock_call_llm_fixable(context: AssembledContext) -> LLMCallResult:
    """Return Python code with auto-fixable cosmetic issues."""
    return LLMCallResult(
        task_id=context.task_id,
        response=LLMResponse(
            files=[FileOutput(file_path="hello.py", content=_FIXABLE_PYTHON)],
            explanation="Created names module (with typing.List).",
        ),
        model_name="mock-model",
        input_tokens=100,
        output_tokens=50,
        latency_ms=200.0,
    )


# ---------------------------------------------------------------------------
# Activity lists
# ---------------------------------------------------------------------------

_REAL_ACTIVITIES = [
    create_worktree_activity,
    remove_worktree_activity,
    commit_changes_activity,
    assemble_context,
    write_output,
    validate_output,
    evaluate_transition,
]

_ACTIVITIES_VALID = [*_REAL_ACTIVITIES, mock_call_llm_valid]
_ACTIVITIES_INVALID = [*_REAL_ACTIVITIES, mock_call_llm_invalid]
_ACTIVITIES_FIXABLE = [*_REAL_ACTIVITIES, mock_call_llm_fixable]


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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _run_e2e_workflow(
    env: WorkflowEnvironment,
    git_repo: Path,
    *,
    activities: list | None = None,
    task: TaskDefinition | None = None,
    max_attempts: int = 2,
) -> TaskResult:
    """Run the ForgeTaskWorkflow with real activities and a mock LLM."""
    if task is None:
        task = TaskDefinition(
            task_id="e2e-task",
            description="Write a greeting module.",
            target_files=["hello.py"],
        )
    if activities is None:
        activities = _ACTIVITIES_VALID

    forge_input = ForgeTaskInput(
        task=task,
        repo_root=str(git_repo),
        max_attempts=max_attempts,
    )

    async with Worker(
        env.client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=[ForgeTaskWorkflow],
        activities=activities,
    ):
        return await env.client.execute_workflow(
            ForgeTaskWorkflow.run,
            forge_input,
            id=f"e2e-{task.task_id}",
            task_queue=FORGE_TASK_QUEUE,
        )


# ---------------------------------------------------------------------------
# Tests — happy path
# ---------------------------------------------------------------------------


class TestEndToEndSuccess:
    """LLM produces valid code → full pipeline succeeds."""

    @pytest.fixture
    async def result(self, env: WorkflowEnvironment, git_repo: Path) -> TaskResult:
        return await _run_e2e_workflow(env, git_repo)

    @pytest.mark.asyncio
    async def test_returns_success_status(self, result: TaskResult) -> None:
        assert result.status == TransitionSignal.SUCCESS

    @pytest.mark.asyncio
    async def test_worktree_exists_on_disk(self, result: TaskResult) -> None:
        assert result.worktree_path is not None
        assert Path(result.worktree_path).is_dir()

    @pytest.mark.asyncio
    async def test_output_file_written(self, result: TaskResult) -> None:
        assert result.worktree_path is not None
        output_file = Path(result.worktree_path) / "hello.py"
        assert output_file.is_file()
        assert output_file.read_text() == _VALID_PYTHON

    @pytest.mark.asyncio
    async def test_branch_exists(self, result: TaskResult, git_repo: Path) -> None:
        branches = subprocess.run(
            ["git", "branch", "--list", "forge/e2e-task"],
            cwd=git_repo,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "forge/e2e-task" in branches.stdout

    @pytest.mark.asyncio
    async def test_commit_exists(self, result: TaskResult) -> None:
        assert result.worktree_path is not None
        log = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=result.worktree_path,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "forge(e2e-task): success" in log.stdout

    @pytest.mark.asyncio
    async def test_validation_all_passed(self, result: TaskResult) -> None:
        assert len(result.validation_results) > 0
        assert all(v.passed for v in result.validation_results)

    @pytest.mark.asyncio
    async def test_output_files_in_result(self, result: TaskResult) -> None:
        assert "hello.py" in result.output_files
        assert result.output_files["hello.py"] == _VALID_PYTHON


# ---------------------------------------------------------------------------
# Tests — validation failure
# ---------------------------------------------------------------------------


class TestEndToEndValidationFailure:
    """LLM produces bad code → validation fails → terminal failure."""

    @pytest.fixture
    async def result(self, env: WorkflowEnvironment, git_repo: Path) -> TaskResult:
        return await _run_e2e_workflow(
            env,
            git_repo,
            activities=_ACTIVITIES_INVALID,
            max_attempts=1,
        )

    @pytest.mark.asyncio
    async def test_returns_failure_terminal(self, result: TaskResult) -> None:
        assert result.status == TransitionSignal.FAILURE_TERMINAL

    @pytest.mark.asyncio
    async def test_error_populated(self, result: TaskResult) -> None:
        assert result.error is not None
        assert result.error != ""

    @pytest.mark.asyncio
    async def test_worktree_still_exists(self, result: TaskResult) -> None:
        """Failed worktree is left for debugging per spec."""
        assert result.worktree_path is not None
        assert Path(result.worktree_path).is_dir()


# ---------------------------------------------------------------------------
# Tests — context files
# ---------------------------------------------------------------------------


class TestEndToEndWithContextFiles:
    """Verifies context assembly reads real files from the repo."""

    @pytest.mark.asyncio
    async def test_workflow_succeeds_with_context_files(
        self, env: WorkflowEnvironment, git_repo: Path
    ) -> None:
        context_file = git_repo / "context.txt"
        context_file.write_text("This is reference context for the LLM.\n")
        subprocess.run(
            ["git", "add", "context.txt"],
            cwd=git_repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Add context file"],
            cwd=git_repo,
            check=True,
            capture_output=True,
        )

        task = TaskDefinition(
            task_id="e2e-ctx",
            description="Write a greeting module with context.",
            target_files=["hello.py"],
            context_files=["context.txt"],
        )

        result = await _run_e2e_workflow(env, git_repo, task=task)
        assert result.status == TransitionSignal.SUCCESS


# ---------------------------------------------------------------------------
# Tests — auto-fix
# ---------------------------------------------------------------------------


class TestEndToEndAutoFix:
    """LLM produces code with cosmetic issues → auto-fix cleans it → pipeline succeeds."""

    @pytest.fixture
    async def result(self, env: WorkflowEnvironment, git_repo: Path) -> TaskResult:
        return await _run_e2e_workflow(
            env,
            git_repo,
            activities=_ACTIVITIES_FIXABLE,
        )

    @pytest.mark.asyncio
    async def test_returns_success_status(self, result: TaskResult) -> None:
        assert result.status == TransitionSignal.SUCCESS

    @pytest.mark.asyncio
    async def test_output_file_was_fixed(self, result: TaskResult) -> None:
        assert result.worktree_path is not None
        output_file = Path(result.worktree_path) / "hello.py"
        content = output_file.read_text()
        assert "List[str]" not in content
        assert "list[str]" in content

    @pytest.mark.asyncio
    async def test_validation_all_passed(self, result: TaskResult) -> None:
        assert len(result.validation_results) > 0
        assert all(v.passed for v in result.validation_results)


# ===========================================================================
# Phase 2: Planned end-to-end tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Valid Python code for multi-step plans
# ---------------------------------------------------------------------------

_STEP1_PYTHON = 'class Model:\n    """A simple data model."""\n\n    name: str\n    value: int\n'

_STEP2_PYTHON = (
    "from models import Model\n\n\n"
    "def create_model(name: str, value: int) -> Model:\n"
    '    """Create a Model instance."""\n'
    "    return Model(name=name, value=value)\n"
)

_STEP2_INVALID_PYTHON = "def create_model(name):\n  x=1\n  return name\n  y = 2\n"


# ---------------------------------------------------------------------------
# Mock planner + step LLM activities for planned e2e
# ---------------------------------------------------------------------------

_E2E_PLAN_LLM_CALL_COUNT = 0
_E2E_PLAN_LLM_RESPONSES: list[LLMResponse] = []


def _reset_e2e_plan_state(responses: list[LLMResponse] | None = None) -> None:
    global _E2E_PLAN_LLM_CALL_COUNT
    _E2E_PLAN_LLM_CALL_COUNT = 0
    _E2E_PLAN_LLM_RESPONSES.clear()
    if responses:
        _E2E_PLAN_LLM_RESPONSES.extend(responses)


@activity.defn(name="assemble_planner_context")
async def mock_e2e_assemble_planner_context(input: AssembleContextInput) -> PlannerInput:
    return PlannerInput(
        task_id=input.task.task_id,
        system_prompt="planner prompt",
        user_prompt="plan it",
    )


@activity.defn(name="call_planner")
async def mock_e2e_call_planner(input: PlannerInput) -> PlanCallResult:
    plan = Plan(
        task_id=input.task_id,
        steps=[
            PlanStep(step_id="step-1", description="Create model.", target_files=["models.py"]),
            PlanStep(
                step_id="step-2",
                description="Create API.",
                target_files=["api.py"],
                context_files=["models.py"],
            ),
        ],
        explanation="Two-step plan.",
    )
    return PlanCallResult(
        task_id=input.task_id,
        plan=plan,
        model_name="mock-planner",
        input_tokens=300,
        output_tokens=150,
        latency_ms=500.0,
    )


@activity.defn(name="call_llm")
async def mock_e2e_plan_call_llm(context: AssembledContext) -> LLMCallResult:
    global _E2E_PLAN_LLM_CALL_COUNT
    _E2E_PLAN_LLM_CALL_COUNT += 1

    response = _E2E_PLAN_LLM_RESPONSES.pop(0)

    return LLMCallResult(
        task_id=context.task_id,
        response=response,
        model_name="mock-model",
        input_tokens=100,
        output_tokens=50,
        latency_ms=200.0,
    )


_PLANNED_REAL_ACTIVITIES = [
    create_worktree_activity,
    remove_worktree_activity,
    reset_worktree_activity,
    commit_changes_activity,
    assemble_step_context,
    write_output,
    validate_output,
    evaluate_transition,
    mock_e2e_assemble_planner_context,
    mock_e2e_call_planner,
]

_PLANNED_ACTIVITIES_VALID = [*_PLANNED_REAL_ACTIVITIES, mock_e2e_plan_call_llm]


async def _run_planned_e2e(
    env: WorkflowEnvironment,
    git_repo: Path,
    *,
    activities: list | None = None,
    task: TaskDefinition | None = None,
    max_step_attempts: int = 2,
) -> TaskResult:
    """Run the planned workflow with real git/validation and mock LLM+planner."""
    if task is None:
        task = TaskDefinition(
            task_id="e2e-planned",
            description="Build a model and API.",
        )
    if activities is None:
        activities = _PLANNED_ACTIVITIES_VALID

    forge_input = ForgeTaskInput(
        task=task,
        repo_root=str(git_repo),
        plan=True,
        max_step_attempts=max_step_attempts,
    )

    async with Worker(
        env.client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=[ForgeTaskWorkflow],
        activities=activities,
    ):
        return await env.client.execute_workflow(
            ForgeTaskWorkflow.run,
            forge_input,
            id=f"e2e-{task.task_id}",
            task_queue=FORGE_TASK_QUEUE,
        )


# ---------------------------------------------------------------------------
# Tests — planned happy path
# ---------------------------------------------------------------------------


class TestEndToEndPlanned:
    """Two-step plan, both steps produce valid code."""

    @pytest.fixture
    async def result(self, env: WorkflowEnvironment, git_repo: Path) -> TaskResult:
        _reset_e2e_plan_state(
            responses=[
                LLMResponse(
                    files=[FileOutput(file_path="models.py", content=_STEP1_PYTHON)],
                    explanation="Created model.",
                ),
                LLMResponse(
                    files=[FileOutput(file_path="api.py", content=_STEP2_PYTHON)],
                    explanation="Created API.",
                ),
            ]
        )
        return await _run_planned_e2e(env, git_repo)

    @pytest.mark.asyncio
    async def test_returns_success(self, result: TaskResult) -> None:
        assert result.status == TransitionSignal.SUCCESS

    @pytest.mark.asyncio
    async def test_two_step_results(self, result: TaskResult) -> None:
        assert len(result.step_results) == 2
        assert result.step_results[0].step_id == "step-1"
        assert result.step_results[1].step_id == "step-2"

    @pytest.mark.asyncio
    async def test_plan_attached(self, result: TaskResult) -> None:
        assert result.plan is not None
        assert len(result.plan.steps) == 2

    @pytest.mark.asyncio
    async def test_two_commits_in_history(self, result: TaskResult) -> None:
        assert result.worktree_path is not None
        log = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=result.worktree_path,
            capture_output=True,
            text=True,
            check=True,
        )
        # Two step commits + initial commit from git_repo fixture
        lines = [line for line in log.stdout.strip().split("\n") if line]
        assert len(lines) >= 3  # initial + step-1 + step-2

    @pytest.mark.asyncio
    async def test_both_output_files_on_disk(self, result: TaskResult) -> None:
        assert result.worktree_path is not None
        wt = Path(result.worktree_path)
        assert (wt / "models.py").is_file()
        assert (wt / "api.py").is_file()

    @pytest.mark.asyncio
    async def test_step_commit_shas(self, result: TaskResult) -> None:
        for sr in result.step_results:
            assert sr.commit_sha is not None
            assert len(sr.commit_sha) == 40


# ---------------------------------------------------------------------------
# Tests — planned step failure
# ---------------------------------------------------------------------------


class TestEndToEndPlannedStepFailure:
    """Step 1 succeeds, step 2 produces invalid code → terminal failure."""

    @pytest.fixture
    async def result(self, env: WorkflowEnvironment, git_repo: Path) -> TaskResult:
        _reset_e2e_plan_state(
            responses=[
                # Step 1 valid
                LLMResponse(
                    files=[FileOutput(file_path="models.py", content=_STEP1_PYTHON)],
                    explanation="Created model.",
                ),
                # Step 2 invalid
                LLMResponse(
                    files=[FileOutput(file_path="api.py", content=_STEP2_INVALID_PYTHON)],
                    explanation="Created broken API.",
                ),
            ]
        )
        return await _run_planned_e2e(env, git_repo, max_step_attempts=1)

    @pytest.mark.asyncio
    async def test_returns_failure(self, result: TaskResult) -> None:
        assert result.status == TransitionSignal.FAILURE_TERMINAL

    @pytest.mark.asyncio
    async def test_step1_succeeded(self, result: TaskResult) -> None:
        assert result.step_results[0].status == TransitionSignal.SUCCESS
        assert result.step_results[0].commit_sha is not None

    @pytest.mark.asyncio
    async def test_step2_failed(self, result: TaskResult) -> None:
        assert result.step_results[1].status == TransitionSignal.FAILURE_TERMINAL

    @pytest.mark.asyncio
    async def test_step1_commit_preserved(self, result: TaskResult) -> None:
        """Step 1's commit should still exist in the worktree history."""
        assert result.worktree_path is not None
        log = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=result.worktree_path,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "step-1" in log.stdout


# ---------------------------------------------------------------------------
# Tests — planned step retry
# ---------------------------------------------------------------------------


class TestEndToEndPlannedStepRetry:
    """Step 1 succeeds, step 2 fails then succeeds on retry."""

    @pytest.fixture
    async def result(self, env: WorkflowEnvironment, git_repo: Path) -> TaskResult:
        _reset_e2e_plan_state(
            responses=[
                # Step 1 valid
                LLMResponse(
                    files=[FileOutput(file_path="models.py", content=_STEP1_PYTHON)],
                    explanation="Created model.",
                ),
                # Step 2, attempt 1: invalid
                LLMResponse(
                    files=[FileOutput(file_path="api.py", content=_STEP2_INVALID_PYTHON)],
                    explanation="Created broken API.",
                ),
                # Step 2, attempt 2: valid
                LLMResponse(
                    files=[FileOutput(file_path="api.py", content=_STEP2_PYTHON)],
                    explanation="Created API (fixed).",
                ),
            ]
        )
        return await _run_planned_e2e(env, git_repo, max_step_attempts=2)

    @pytest.mark.asyncio
    async def test_returns_success(self, result: TaskResult) -> None:
        assert result.status == TransitionSignal.SUCCESS

    @pytest.mark.asyncio
    async def test_two_step_results(self, result: TaskResult) -> None:
        assert len(result.step_results) == 2
        assert result.step_results[0].status == TransitionSignal.SUCCESS
        assert result.step_results[1].status == TransitionSignal.SUCCESS

    @pytest.mark.asyncio
    async def test_both_files_on_disk(self, result: TaskResult) -> None:
        assert result.worktree_path is not None
        wt = Path(result.worktree_path)
        assert (wt / "models.py").is_file()
        assert (wt / "api.py").is_file()


# ===========================================================================
# Phase 3: Fan-out end-to-end tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Valid Python code for fan-out sub-tasks
# ---------------------------------------------------------------------------

_SUBTASK1_PYTHON = 'class Schema:\n    """A simple schema."""\n\n    name: str\n    value: int\n'

_SUBTASK2_PYTHON = (
    "def build_routes() -> list[str]:\n"
    '    """Return a list of route paths."""\n'
    '    return ["/api/v1/items", "/api/v1/users"]\n'
)

_SUBTASK2_INVALID_PYTHON = "def build_routes(x):\n  y=1\n  return x\n  z = 2\n"


# ---------------------------------------------------------------------------
# Mock planner + LLM for fan-out e2e
# ---------------------------------------------------------------------------

_E2E_FANOUT_LLM_RESPONSES: list[LLMResponse] = []


def _reset_e2e_fanout_state(responses: list[LLMResponse] | None = None) -> None:
    _E2E_FANOUT_LLM_RESPONSES.clear()
    if responses:
        _E2E_FANOUT_LLM_RESPONSES.extend(responses)


@activity.defn(name="assemble_planner_context")
async def mock_e2e_fanout_assemble_planner_context(
    input: AssembleContextInput,
) -> PlannerInput:
    return PlannerInput(
        task_id=input.task.task_id,
        system_prompt="planner prompt",
        user_prompt="plan it",
    )


@activity.defn(name="call_planner")
async def mock_e2e_fanout_call_planner(input: PlannerInput) -> PlanCallResult:
    plan = Plan(
        task_id=input.task_id,
        steps=[
            PlanStep(
                step_id="fan-step",
                description="Create schema and routes in parallel.",
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
        explanation="Fan-out plan.",
    )
    return PlanCallResult(
        task_id=input.task_id,
        plan=plan,
        model_name="mock-planner",
        input_tokens=300,
        output_tokens=150,
        latency_ms=500.0,
    )


@activity.defn(name="call_llm")
async def mock_e2e_fanout_call_llm(context: AssembledContext) -> LLMCallResult:
    response = _E2E_FANOUT_LLM_RESPONSES.pop(0)
    return LLMCallResult(
        task_id=context.task_id,
        response=response,
        model_name="mock-model",
        input_tokens=100,
        output_tokens=50,
        latency_ms=200.0,
    )


_FANOUT_REAL_ACTIVITIES = [
    create_worktree_activity,
    remove_worktree_activity,
    reset_worktree_activity,
    commit_changes_activity,
    assemble_sub_task_context,
    write_output,
    write_files,
    validate_output,
    evaluate_transition,
    mock_e2e_fanout_assemble_planner_context,
    mock_e2e_fanout_call_planner,
    mock_e2e_fanout_call_llm,
]


async def _run_fanout_e2e(
    env: WorkflowEnvironment,
    git_repo: Path,
    *,
    activities: list | None = None,
    task: TaskDefinition | None = None,
    max_sub_task_attempts: int = 2,
) -> TaskResult:
    """Run the fan-out workflow with real git/validation and mock LLM+planner."""
    if task is None:
        task = TaskDefinition(
            task_id="e2e-fanout",
            description="Build schema and routes.",
        )
    if activities is None:
        activities = _FANOUT_REAL_ACTIVITIES

    forge_input = ForgeTaskInput(
        task=task,
        repo_root=str(git_repo),
        plan=True,
        max_sub_task_attempts=max_sub_task_attempts,
    )

    async with Worker(
        env.client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=[ForgeTaskWorkflow, ForgeSubTaskWorkflow],
        activities=activities,
    ):
        return await env.client.execute_workflow(
            ForgeTaskWorkflow.run,
            forge_input,
            id=f"e2e-{task.task_id}",
            task_queue=FORGE_TASK_QUEUE,
        )


# ---------------------------------------------------------------------------
# Tests — fan-out happy path
# ---------------------------------------------------------------------------


class TestEndToEndFanOut:
    """Fan-out with 2 sub-tasks, both produce valid code."""

    @pytest.fixture
    async def result(self, env: WorkflowEnvironment, git_repo: Path) -> TaskResult:
        _reset_e2e_fanout_state(
            responses=[
                LLMResponse(
                    files=[FileOutput(file_path="schema.py", content=_SUBTASK1_PYTHON)],
                    explanation="Created schema.",
                ),
                LLMResponse(
                    files=[FileOutput(file_path="routes.py", content=_SUBTASK2_PYTHON)],
                    explanation="Created routes.",
                ),
            ]
        )
        return await _run_fanout_e2e(env, git_repo)

    @pytest.mark.asyncio
    async def test_returns_success(self, result: TaskResult) -> None:
        assert result.status == TransitionSignal.SUCCESS

    @pytest.mark.asyncio
    async def test_step_with_sub_tasks(self, result: TaskResult) -> None:
        assert len(result.step_results) == 1
        sr = result.step_results[0]
        assert sr.step_id == "fan-step"
        assert len(sr.sub_task_results) == 2

    @pytest.mark.asyncio
    async def test_files_merged_into_parent(self, result: TaskResult) -> None:
        assert result.worktree_path is not None
        wt = Path(result.worktree_path)
        assert (wt / "schema.py").is_file()
        assert (wt / "routes.py").is_file()

    @pytest.mark.asyncio
    async def test_parent_commit_exists(self, result: TaskResult) -> None:
        assert result.worktree_path is not None
        log = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=result.worktree_path,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "fan-out gather" in log.stdout

    @pytest.mark.asyncio
    async def test_sub_task_worktrees_cleaned_up(self, result: TaskResult, git_repo: Path) -> None:
        """Sub-task worktrees should be removed after completion."""
        worktrees_dir = git_repo / ".forge-worktrees"
        # Parent worktree should exist, but sub-task worktrees should not
        if worktrees_dir.exists():
            sub_dirs = [d.name for d in worktrees_dir.iterdir() if d.is_dir()]
            # No directory with ".sub." in the name should remain
            assert not any(".sub." in d for d in sub_dirs)


# ---------------------------------------------------------------------------
# Tests — fan-out child failure
# ---------------------------------------------------------------------------


class TestEndToEndFanOutChildFailure:
    """One sub-task produces invalid code → fan-out step fails."""

    @pytest.fixture
    async def result(self, env: WorkflowEnvironment, git_repo: Path) -> TaskResult:
        _reset_e2e_fanout_state(
            responses=[
                LLMResponse(
                    files=[FileOutput(file_path="schema.py", content=_SUBTASK1_PYTHON)],
                    explanation="Created schema.",
                ),
                LLMResponse(
                    files=[FileOutput(file_path="routes.py", content=_SUBTASK2_INVALID_PYTHON)],
                    explanation="Created broken routes.",
                ),
            ]
        )
        return await _run_fanout_e2e(env, git_repo, max_sub_task_attempts=1)

    @pytest.mark.asyncio
    async def test_returns_failure(self, result: TaskResult) -> None:
        assert result.status == TransitionSignal.FAILURE_TERMINAL

    @pytest.mark.asyncio
    async def test_error_populated(self, result: TaskResult) -> None:
        assert result.error is not None
        assert "fan-out failed" in result.error
