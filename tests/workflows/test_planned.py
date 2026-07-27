"""Planned ``ForgeTaskWorkflow`` scenarios (``plan=True``, no sub-tasks).

Migrated from ``tests/test_workflows.py`` in T5.5: same scenarios, same
assertions, scripted per test through :class:`ScenarioState` instead of the
module-level globals the old sections shared (``_PLAN_CALL_LOG``,
``_PLAN_TRANSITION_SEQUENCE``, ``_PLAN_LLM_CALL_COUNT``, and the Phase 8
``_P8_STEP_*`` siblings). Per the T5.5 harness migration guide, this follows
``tests/workflows/test_single_step.py``'s exemplar pattern class for class.

The old parse handler picked between the models.py and api.py responses with
an odd/even call counter (``_PLAN_LLM_CALL_COUNT % 2``) — exactly the
consumption-order defect D3 forbids. Here the two step responses are keyed by
*call identity* (the step id), so which file a step writes no longer depends
on how many LLM calls happened to run before it.

Like every scenario in the old file, these run on the **batch** lane:
``ForgeTaskInput.sync_mode`` defaults to ``False``, so generation goes through
submit -> status -> fetch -> parse.
"""

from types import MappingProxyType
from typing import TYPE_CHECKING

from forge.models import (
    FileOutput,
    ForgeTaskInput,
    LLMResponse,
    Plan,
    PlanStep,
    SubTask,
    TaskDefinition,
    TransitionSignal,
    ValidationResult,
)
from tests.support.workflow_harness import ScenarioState, run_task

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment

PLANNED_TASK = TaskDefinition(
    task_id="planned-task",
    description="Build a REST API with models and routes.",
)

PLANNED_INPUT = ForgeTaskInput(
    task=PLANNED_TASK,
    repo_root="/tmp/repo",
    plan=True,
    max_step_attempts=2,
    max_exploration_rounds=0,
)

PLAN = Plan(
    task_id="planned-task",
    steps=[
        PlanStep(step_id="step-1", description="Create models.", target_files=["models.py"]),
        PlanStep(step_id="step-2", description="Create API.", target_files=["api.py"]),
    ],
    explanation="Split into models and API layers.",
)

# Keyed by call identity (the step id) — replaces the old odd/even call-counter
# dispatch (_PLAN_LLM_CALL_COUNT % 2) that picked models.py vs api.py by
# arrival order, the same defect class D3 forbids. The explanation string is
# fixed rather than interpolating a call count as the old handler did
# (f"LLM call #{n}"); nothing in the migrated assertions reads the
# explanation text.
MODELS_RESPONSE = LLMResponse(
    files=[FileOutput(file_path="models.py", content="class Model: pass\n")],
    explanation="Created models module.",
)
API_RESPONSE = LLMResponse(
    files=[FileOutput(file_path="api.py", content="def endpoint(): pass\n")],
    explanation="Created API module.",
)
STEP_RESPONSES = MappingProxyType({"step-1": MODELS_RESPONSE, "step-2": API_RESPONSE})

SUCCESS = TransitionSignal.SUCCESS.value
RETRYABLE = TransitionSignal.FAILURE_RETRYABLE.value
TERMINAL = TransitionSignal.FAILURE_TERMINAL.value


class TestPlannedWorkflowSuccess:
    """Two-step plan, both steps succeed."""

    async def test_returns_success(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=PLAN,
            transitions={"planned-task": [SUCCESS, SUCCESS]},
            llm_responses=STEP_RESPONSES,
        )
        result = await run_task(env, PLANNED_INPUT, state)
        assert result.status == TransitionSignal.SUCCESS

    async def test_plan_populated(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=PLAN,
            transitions={"planned-task": [SUCCESS, SUCCESS]},
            llm_responses=STEP_RESPONSES,
        )
        result = await run_task(env, PLANNED_INPUT, state)
        assert result.plan is not None
        assert len(result.plan.steps) == 2

    async def test_step_results_populated(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=PLAN,
            transitions={"planned-task": [SUCCESS, SUCCESS]},
            llm_responses=STEP_RESPONSES,
        )
        result = await run_task(env, PLANNED_INPUT, state)
        assert len(result.step_results) == 2
        assert all(sr.status == TransitionSignal.SUCCESS for sr in result.step_results)
        assert result.step_results[0].step_id == "step-1"
        assert result.step_results[1].step_id == "step-2"

    async def test_two_commits(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=PLAN,
            transitions={"planned-task": [SUCCESS, SUCCESS]},
            llm_responses=STEP_RESPONSES,
        )
        await run_task(env, PLANNED_INPUT, state)
        assert len(state.entries("commit")) == 2

    async def test_worktree_created_once(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=PLAN,
            transitions={"planned-task": [SUCCESS, SUCCESS]},
            llm_responses=STEP_RESPONSES,
        )
        await run_task(env, PLANNED_INPUT, state)
        assert state.count("create_worktree") == 1

    async def test_output_files_accumulated(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=PLAN,
            transitions={"planned-task": [SUCCESS, SUCCESS]},
            llm_responses=STEP_RESPONSES,
        )
        result = await run_task(env, PLANNED_INPUT, state)
        assert "models.py" in result.output_files
        assert "api.py" in result.output_files

    async def test_step_commit_shas(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=PLAN,
            transitions={"planned-task": [SUCCESS, SUCCESS]},
            llm_responses=STEP_RESPONSES,
        )
        result = await run_task(env, PLANNED_INPUT, state)
        for sr in result.step_results:
            assert sr.commit_sha is not None
            assert len(sr.commit_sha) == 40


class TestPlannedWorkflowStepRetry:
    """Step 1 succeeds, step 2 fails then succeeds on retry."""

    async def test_retry_then_success(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=PLAN,
            transitions={"planned-task": [SUCCESS, RETRYABLE, SUCCESS]},
            llm_responses=STEP_RESPONSES,
        )
        result = await run_task(env, PLANNED_INPUT, state)
        assert result.status == TransitionSignal.SUCCESS

    async def test_reset_worktree_on_retry(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=PLAN,
            transitions={"planned-task": [SUCCESS, RETRYABLE, SUCCESS]},
            llm_responses=STEP_RESPONSES,
        )
        await run_task(env, PLANNED_INPUT, state)
        assert state.called("reset_worktree")

    async def test_two_step_results_on_retry(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=PLAN,
            transitions={"planned-task": [SUCCESS, RETRYABLE, SUCCESS]},
            llm_responses=STEP_RESPONSES,
        )
        result = await run_task(env, PLANNED_INPUT, state)
        assert len(result.step_results) == 2
        assert result.step_results[0].status == TransitionSignal.SUCCESS
        assert result.step_results[1].status == TransitionSignal.SUCCESS


class TestPlannedWorkflowStepFailure:
    """Step 1 succeeds, step 2 fails terminally."""

    async def test_terminal_failure(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=PLAN,
            transitions={"planned-task": [SUCCESS, TERMINAL]},
            llm_responses=STEP_RESPONSES,
        )
        result = await run_task(env, PLANNED_INPUT, state)
        assert result.status == TransitionSignal.FAILURE_TERMINAL

    async def test_step_results_show_failure(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=PLAN,
            transitions={"planned-task": [SUCCESS, TERMINAL]},
            llm_responses=STEP_RESPONSES,
        )
        result = await run_task(env, PLANNED_INPUT, state)
        assert len(result.step_results) == 2
        assert result.step_results[0].status == TransitionSignal.SUCCESS
        assert result.step_results[1].status == TransitionSignal.FAILURE_TERMINAL

    async def test_step1_commit_preserved(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=PLAN,
            transitions={"planned-task": [SUCCESS, TERMINAL]},
            llm_responses=STEP_RESPONSES,
        )
        result = await run_task(env, PLANNED_INPUT, state)
        assert result.step_results[0].commit_sha is not None

    async def test_error_references_step(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=PLAN,
            transitions={"planned-task": [SUCCESS, TERMINAL]},
            llm_responses=STEP_RESPONSES,
        )
        result = await run_task(env, PLANNED_INPUT, state)
        assert result.error is not None
        assert "step-2" in result.error

    async def test_plan_in_result(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=PLAN,
            transitions={"planned-task": [SUCCESS, TERMINAL]},
            llm_responses=STEP_RESPONSES,
        )
        result = await run_task(env, PLANNED_INPUT, state)
        assert result.plan is not None


class TestPlannedBackwardCompat:
    """Existing Phase 2 plans (no sub_tasks) still work."""

    async def test_no_sub_tasks_works(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=PLAN,
            transitions={"planned-task": [SUCCESS, SUCCESS]},
            llm_responses=STEP_RESPONSES,
        )
        result = await run_task(env, PLANNED_INPUT, state)
        assert result.status == TransitionSignal.SUCCESS
        assert len(result.step_results) == 2


# ---------------------------------------------------------------------------
# Phase 8: planned step error-aware retry — prior_errors threaded into retry
# ---------------------------------------------------------------------------

P8_STEP_TASK = TaskDefinition(task_id="p8-step-task", description="Build.")

P8_STEP_INPUT = ForgeTaskInput(
    task=P8_STEP_TASK,
    repo_root="/tmp/repo",
    plan=True,
    max_step_attempts=2,
    max_exploration_rounds=0,
)

P8_STEP_PLAN = Plan(
    task_id="p8-step-task",
    steps=[PlanStep(step_id="step-1", description="Create.", target_files=["a.py"])],
    explanation="One step.",
)

P8_STEP_RESPONSE = LLMResponse(
    files=[FileOutput(file_path="a.py", content="# code\n")],
    explanation="step output",
)


class TestPlannedStepErrorAwareRetry:
    """Phase 8: prior_errors are passed through planned step retry loop."""

    async def test_step_retry_passes_prior_errors(self, env: "WorkflowEnvironment") -> None:
        lint_errors = [
            ValidationResult(
                check_name="ruff_format",
                passed=False,
                summary="ruff_format failed",
                details="a.py:10:1: formatting error",
            )
        ]
        state = ScenarioState(
            plan=P8_STEP_PLAN,
            transitions={"p8-step-task": [RETRYABLE, SUCCESS]},
            validations={
                "p8-step-task": [
                    lint_errors,
                    [ValidationResult(check_name="ruff_format", passed=True, summary="passed")],
                ]
            },
            llm_responses={"step-1": P8_STEP_RESPONSE},
        )
        result = await run_task(env, P8_STEP_INPUT, state, workflow_id="test-p8-step-retry")
        assert result.status == TransitionSignal.SUCCESS
        assert len(state.step_context_inputs) == 2

        # First attempt: no prior errors
        first = state.step_context_inputs[0]
        assert first.prior_errors == []
        assert first.attempt == 1

        # Second attempt: errors from first
        second = state.step_context_inputs[1]
        assert len(second.prior_errors) == 1
        assert second.prior_errors[0].check_name == "ruff_format"
        assert second.attempt == 2


# ---------------------------------------------------------------------------
# Mixed plan: sequential step → fan-out step → sequential step
# ---------------------------------------------------------------------------

MIXED_TASK = TaskDefinition(
    task_id="fanout-task",
    description="Build schema and routes in parallel.",
)

MIXED_INPUT = ForgeTaskInput(
    task=MIXED_TASK,
    repo_root="/tmp/repo",
    plan=True,
    max_sub_task_attempts=2,
    max_exploration_rounds=0,
)

MIXED_PLAN = Plan(
    task_id="fanout-task",
    steps=[
        PlanStep(step_id="seq-1", description="Create models.", target_files=["models.py"]),
        PlanStep(
            step_id="fan-step",
            description="Fan-out step.",
            target_files=[],
            sub_tasks=[
                SubTask(
                    sub_task_id="st1", description="Create schema.", target_files=["schema.py"]
                ),
                SubTask(
                    sub_task_id="st2", description="Create routes.", target_files=["routes.py"]
                ),
            ],
        ),
        PlanStep(step_id="seq-2", description="Create tests.", target_files=["tests.py"]),
    ],
    explanation="Mixed plan.",
)


def _mixed_response(path: str) -> LLMResponse:
    return LLMResponse(
        files=[FileOutput(file_path=path, content=f"# {path.removesuffix('.py')}\n")],
        explanation=f"Created {path}.",
    )


# Keyed by call identity: step id for the two sequential steps, compound
# sub-task id for the two fan-out children. The old section drove all four from
# one 1..4 call counter, so which file each call produced depended on the order
# the two parallel children happened to run in.
MIXED_RESPONSES = MappingProxyType(
    {
        "seq-1": _mixed_response("models.py"),
        "fanout-task.sub.st1": _mixed_response("schema.py"),
        "fanout-task.sub.st2": _mixed_response("routes.py"),
        "seq-2": _mixed_response("tests.py"),
    }
)


class TestMixedPlan:
    """Sequential step → fan-out step → sequential step."""

    async def test_mixed_plan_succeeds(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(plan=MIXED_PLAN, llm_responses=MIXED_RESPONSES)
        result = await run_task(env, MIXED_INPUT, state, workflow_id="test-mixed-plan")
        assert result.status == TransitionSignal.SUCCESS
        assert len(result.step_results) == 3
        assert result.step_results[0].step_id == "seq-1"
        assert result.step_results[1].step_id == "fan-step"
        assert result.step_results[2].step_id == "seq-2"
