"""Single-step ``ForgeTaskWorkflow`` scenarios (``plan=False``).

Migrated from ``tests/test_workflows.py`` in T5.5: same scenarios, same
assertions, scripted per test through :class:`ScenarioState` instead of the
module-level globals the old sections shared. This file is the exemplar the
rest of the migration follows — see
``development-plans/tasks/T5.5-harness-migration-guide.md``.

Like every scenario in the old file, these run on the **batch** lane:
``ForgeTaskInput.sync_mode`` defaults to ``False``, so generation goes through
submit → status → fetch → parse. That is what they proved before the migration
and what they prove now.
"""

from typing import TYPE_CHECKING

from forge.models import (
    FileOutput,
    ForgeTaskInput,
    LLMResponse,
    TaskDefinition,
    TransitionSignal,
    ValidationResult,
)
from tests.support.workflow_harness import ScenarioState, run_task

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment

TASK = TaskDefinition(
    task_id="test-task",
    description="Write a hello module.",
    target_files=["hello.py"],
)

FORGE_INPUT = ForgeTaskInput(
    task=TASK,
    repo_root="/tmp/repo",
    max_attempts=2,
    max_exploration_rounds=0,
)

SUCCESS = TransitionSignal.SUCCESS.value
RETRYABLE = TransitionSignal.FAILURE_RETRYABLE.value
TERMINAL = TransitionSignal.FAILURE_TERMINAL.value


class TestSuccessPath:
    async def test_returns_success_status(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(transitions={"test-task": [SUCCESS]})
        result = await run_task(env, FORGE_INPUT, state)
        assert result.status == TransitionSignal.SUCCESS

    async def test_commits_with_success(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(transitions={"test-task": [SUCCESS]})
        await run_task(env, FORGE_INPUT, state)
        assert "commit:success" in state.call_log

    async def test_output_files_collected(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(transitions={"test-task": [SUCCESS]})
        result = await run_task(env, FORGE_INPUT, state)
        assert result.output_files == {"hello.py": "print('hello')\n"}

    async def test_worktree_metadata(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(transitions={"test-task": [SUCCESS]})
        result = await run_task(env, FORGE_INPUT, state)
        assert result.worktree_path == "/tmp/repo/.forge-worktrees/test-task"
        assert result.worktree_branch == "forge/test-task"

    async def test_validation_results_populated(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(transitions={"test-task": [SUCCESS]})
        result = await run_task(env, FORGE_INPUT, state)
        assert len(result.validation_results) == 1
        assert result.validation_results[0].passed is True


class TestRetryOnValidationFailure:
    async def test_retry_then_success(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(transitions={"test-task": [RETRYABLE, SUCCESS]})
        result = await run_task(env, FORGE_INPUT, state)
        assert result.status == TransitionSignal.SUCCESS

    async def test_worktree_removed_after_retry(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(transitions={"test-task": [RETRYABLE, SUCCESS]})
        await run_task(env, FORGE_INPUT, state)
        assert state.called("remove_worktree")

    async def test_creates_fresh_worktree_for_second_attempt(
        self, env: "WorkflowEnvironment"
    ) -> None:
        state = ScenarioState(transitions={"test-task": [RETRYABLE, SUCCESS]})
        await run_task(env, FORGE_INPUT, state)
        assert state.count("create_worktree") == 2


class TestTerminalFailure:
    async def test_terminal_failure_status(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(transitions={"test-task": [TERMINAL]})
        result = await run_task(env, FORGE_INPUT, state)
        assert result.status == TransitionSignal.FAILURE_TERMINAL

    async def test_commits_with_failure(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(transitions={"test-task": [TERMINAL]})
        await run_task(env, FORGE_INPUT, state)
        assert "commit:failure" in state.call_log

    async def test_error_populated(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(transitions={"test-task": [TERMINAL]})
        result = await run_task(env, FORGE_INPUT, state)
        # A terminal transition now means validation actually failed (the inlined
        # determine_transition derives the signal from the validation result), so
        # the error carries the joined failing summary and failure_kind is set.
        assert result.error == "ruff_lint failed"
        assert result.failure_kind == "validation"

    async def test_both_attempts_fail(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(transitions={"test-task": [RETRYABLE, TERMINAL]})
        result = await run_task(env, FORGE_INPUT, state)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert state.called("remove_worktree")
        assert "commit:failure" in state.call_log

    async def test_worktree_metadata_on_failure(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(transitions={"test-task": [TERMINAL]})
        result = await run_task(env, FORGE_INPUT, state)
        assert result.worktree_path is not None
        assert result.worktree_branch is not None


# ---------------------------------------------------------------------------
# Phase 8: error-aware retry — prior_errors threaded into the next attempt
# ---------------------------------------------------------------------------

_P8_RESPONSE = LLMResponse(
    files=[FileOutput(file_path="hello.py", content="print('hello')\n")],
    explanation="output",
)


class TestSingleStepErrorAwareRetry:
    """Phase 8: prior_errors are passed through single-step retry loop."""

    async def test_first_attempt_has_no_prior_errors(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            transitions={"test-task": [SUCCESS]},
            llm_responses={"test-task": _P8_RESPONSE},
        )
        await run_task(
            env,
            ForgeTaskInput(
                task=TASK,
                repo_root="/tmp/repo",
                max_attempts=2,
                max_exploration_rounds=0,
            ),
            state,
            workflow_id="test-p8-first-attempt",
        )
        assert len(state.context_inputs) == 1
        first = state.context_inputs[0]
        assert first.prior_errors == []
        assert first.attempt == 1

    async def test_retry_passes_prior_errors(self, env: "WorkflowEnvironment") -> None:
        lint_errors = [
            ValidationResult(
                check_name="ruff_lint",
                passed=False,
                summary="ruff_lint failed",
                details="hello.py:1:1: F401 unused import",
            )
        ]
        state = ScenarioState(
            transitions={"test-task": [RETRYABLE, SUCCESS]},
            validations={
                "test-task": [
                    lint_errors,
                    [ValidationResult(check_name="ruff_lint", passed=True, summary="passed")],
                ]
            },
            llm_responses={"test-task": _P8_RESPONSE},
        )
        result = await run_task(
            env,
            ForgeTaskInput(
                task=TASK,
                repo_root="/tmp/repo",
                max_attempts=2,
                max_exploration_rounds=0,
            ),
            state,
            workflow_id="test-p8-retry-errors",
        )
        assert result.status == TransitionSignal.SUCCESS
        assert len(state.context_inputs) == 2

        # First attempt: no prior errors
        first = state.context_inputs[0]
        assert first.prior_errors == []
        assert first.attempt == 1

        # Second attempt: prior errors from first attempt
        second = state.context_inputs[1]
        assert len(second.prior_errors) == 1
        assert second.prior_errors[0].check_name == "ruff_lint"
        assert second.attempt == 2
        assert second.max_attempts == 2
