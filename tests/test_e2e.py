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
    commit_changes_activity,
    create_worktree_activity,
    evaluate_transition,
    remove_worktree_activity,
    validate_output,
    write_output,
)
from forge.models import (
    AssembledContext,
    FileOutput,
    ForgeTaskInput,
    LLMCallResult,
    LLMResponse,
    TaskDefinition,
    TaskResult,
    TransitionSignal,
)
from forge.workflows import FORGE_TASK_QUEUE, ForgeTaskWorkflow

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
