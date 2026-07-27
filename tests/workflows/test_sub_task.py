"""``ForgeSubTaskWorkflow`` scenarios (standalone sub-task execution).

Migrated from ``tests/test_workflows.py`` in T5.5 (batch 2): same scenarios,
same assertions, scripted per test through :class:`ScenarioState` instead of
the module-level globals the old sections shared. See
``development-plans/tasks/T5.5-harness-migration-guide.md``.

Like every scenario in the old file, these run on the **batch** lane:
``sync_mode`` is never set here, so generation goes through submit -> status ->
fetch -> parse. That is what they proved before the migration and what they
prove now.
"""

from typing import TYPE_CHECKING

import pytest

from forge.models import (
    FileOutput,
    LLMResponse,
    SubTask,
    SubTaskInput,
    TransitionSignal,
    ValidationResult,
)
from tests.support.workflow_harness import ScenarioState, run_sub_task

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment

SUCCESS = TransitionSignal.SUCCESS.value
RETRYABLE = TransitionSignal.FAILURE_RETRYABLE.value
TERMINAL = TransitionSignal.FAILURE_TERMINAL.value

# The compound sub-task id: the *validate identity* (ValidateOutputInput.task_id)
# and the *call identity* the assembled context stands for (call_key) for every
# scenario in this file's first class — parent "parent-task", sub-task "st1".
_ST1_KEY = "parent-task.sub.st1"

# The old counter-based parse handler alternated schema.py/routes.py by an
# odd/even call count (the D3 defect this migration removes). Every scenario
# in TestSubTaskWorkflow makes its scripted LLM response its *first* call
# under a freshly reset counter, so the old handler always produced this
# payload — reproduced here byte-identical, now keyed instead of counted.
_ST1_RESPONSE = LLMResponse(
    files=[FileOutput(file_path="schema.py", content="# schema\n")],
    explanation="Sub-task output.",
)


# ---------------------------------------------------------------------------
# Tests -- sub-task workflow
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

    async def test_success(self, env: "WorkflowEnvironment", sub_task_input: SubTaskInput) -> None:
        state = ScenarioState(
            transitions={_ST1_KEY: [SUCCESS]},
            llm_responses={_ST1_KEY: _ST1_RESPONSE},
        )
        result = await run_sub_task(env, sub_task_input, state)
        assert result.status == TransitionSignal.SUCCESS
        assert result.sub_task_id == "st1"
        assert "schema.py" in result.output_files
        assert result.digest == "Sub-task output."

    async def test_worktree_created_and_removed(
        self, env: "WorkflowEnvironment", sub_task_input: SubTaskInput
    ) -> None:
        state = ScenarioState(
            transitions={_ST1_KEY: [SUCCESS]},
            llm_responses={_ST1_KEY: _ST1_RESPONSE},
        )
        await run_sub_task(env, sub_task_input, state)
        assert state.called("create_worktree")
        assert state.called("remove_worktree")

    async def test_retry_then_success(
        self, env: "WorkflowEnvironment", sub_task_input: SubTaskInput
    ) -> None:
        state = ScenarioState(
            transitions={_ST1_KEY: [RETRYABLE, SUCCESS]},
            llm_responses={_ST1_KEY: _ST1_RESPONSE},
        )
        result = await run_sub_task(env, sub_task_input, state)
        assert result.status == TransitionSignal.SUCCESS
        # Should have created worktree twice (remove after retry, create again)
        assert state.count("create_worktree") == 2

    async def test_terminal_failure(
        self, env: "WorkflowEnvironment", sub_task_input: SubTaskInput
    ) -> None:
        state = ScenarioState(
            transitions={_ST1_KEY: [TERMINAL]},
            llm_responses={_ST1_KEY: _ST1_RESPONSE},
        )
        result = await run_sub_task(env, sub_task_input, state)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.sub_task_id == "st1"
        # Worktree should still be removed on failure
        assert state.called("remove_worktree")


# ---------------------------------------------------------------------------
# Phase 8: Sub-task error-aware retry -- prior_errors threaded through retry
# ---------------------------------------------------------------------------

# The old fixed (non-counter) handler for this section: schema.py content,
# a lowercase "sub-task output" explanation, distinct from _ST1_RESPONSE above.
_P8_ST_RESPONSE = LLMResponse(
    files=[FileOutput(file_path="schema.py", content="# schema\n")],
    explanation="sub-task output",
)


class TestSubTaskErrorAwareRetry:
    """Phase 8: prior_errors are passed through sub-task retry loop."""

    async def test_subtask_retry_passes_prior_errors(self, env: "WorkflowEnvironment") -> None:
        test_errors = [
            ValidationResult(
                check_name="tests",
                passed=False,
                summary="tests failed",
                details="FAILED test_schema.py::test_parse - AssertionError",
            )
        ]
        state = ScenarioState(
            transitions={_ST1_KEY: [RETRYABLE, SUCCESS]},
            validations={
                _ST1_KEY: [
                    test_errors,
                    [ValidationResult(check_name="tests", passed=True, summary="passed")],
                ]
            },
            llm_responses={_ST1_KEY: _P8_ST_RESPONSE},
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
        result = await run_sub_task(env, st_input, state, workflow_id="test-p8-subtask-retry")
        assert result.status == TransitionSignal.SUCCESS
        assert len(state.sub_task_context_inputs) == 2

        # First attempt: no prior errors
        first = state.sub_task_context_inputs[0]
        assert first.prior_errors == []
        assert first.attempt == 1

        # Second attempt: errors from first
        second = state.sub_task_context_inputs[1]
        assert len(second.prior_errors) == 1
        assert second.prior_errors[0].check_name == "tests"
        assert second.attempt == 2
        assert second.max_attempts == 2
