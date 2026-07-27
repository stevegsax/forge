"""Sanity-check scenarios for planned ``ForgeTaskWorkflow`` runs.

Migrated from ``tests/test_workflows.py`` in T5.5 (batch 2): same scenarios,
same assertions, scripted per test through :class:`ScenarioState` instead of
the module-level globals the old sections shared. See
``development-plans/tasks/T5.5-harness-migration-guide.md``.

Like every scenario in the old file, these run on the **batch** lane:
``sync_mode`` is never set here, so generation and the sanity-check arm both
go through submit -> status -> fetch -> parse. That is what they proved
before the migration and what they prove now — note in particular that the
sanity-check call itself is a ``parse_llm_response:SanityCheckResponse``
call-log entry on this lane, not a ``call_sanity_check`` entry (see the
per-test comments below where an old assertion is rewritten for that reason).
"""

from typing import TYPE_CHECKING

from forge.models import (
    ForgeTaskInput,
    Plan,
    PlanStep,
    SanityCheckCallResult,
    SanityCheckResponse,
    SanityCheckVerdict,
    TaskDefinition,
    TransitionSignal,
)
from forge.presets import THINKING_MAX_TOKENS
from tests.support.workflow_harness import ScenarioState, run_task

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment

SUCCESS = TransitionSignal.SUCCESS.value

SC_TASK = TaskDefinition(
    task_id="sc-task",
    description="Build a full API.",
)

# The default four-step plan every planner call returns unless a test
# constructs its own (three-step abort/revise scenarios, two-step skip
# scenario) — moved from the old file's reassignable module global _SC_PLAN
# to a frozen constant (D8): nothing here mutates it.
SC_PLAN = Plan(
    task_id="sc-task",
    steps=[
        PlanStep(step_id="step-1", description="Create models.", target_files=["models.py"]),
        PlanStep(step_id="step-2", description="Create API.", target_files=["api.py"]),
        PlanStep(step_id="step-3", description="Add tests.", target_files=["test_api.py"]),
        PlanStep(step_id="step-4", description="Add docs.", target_files=["docs.py"]),
    ],
    explanation="Four-step plan.",
)

DEFAULT_SC_INPUT = ForgeTaskInput(
    task=SC_TASK,
    repo_root="/tmp/repo",
    plan=True,
    max_step_attempts=2,
    max_exploration_rounds=0,
    sanity_check_interval=2,
)


# ---------------------------------------------------------------------------
# Tests -- sanity check continue
# ---------------------------------------------------------------------------


class TestSanityCheckContinue:
    """interval=2, 4 steps, sanity check fires after step 2, returns 'continue'."""

    async def test_all_steps_complete(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=SC_PLAN,
            transitions={"sc-task": [SUCCESS] * 4},
        )
        result = await run_task(env, DEFAULT_SC_INPUT, state)
        assert result.status == TransitionSignal.SUCCESS
        assert len(result.step_results) == 4

    async def test_sanity_check_count(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=SC_PLAN,
            transitions={"sc-task": [SUCCESS] * 4},
        )
        result = await run_task(env, DEFAULT_SC_INPUT, state)
        # Fires after step 2 (2 % 2 == 0, not last step)
        # Does NOT fire after step 4 (last step)
        assert result.sanity_check_count == 1

    async def test_sanity_check_activities_called(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=SC_PLAN,
            transitions={"sc-task": [SUCCESS] * 4},
        )
        await run_task(env, DEFAULT_SC_INPUT, state)
        assert state.called("assemble_sanity_check_context")
        # Batch lane: the sanity arm is a submit/parse pair, so the old
        # parse-handler-written "call_sanity_check" log entry is now this
        # parse_llm_response entry instead.
        assert "parse_llm_response:SanityCheckResponse" in state.call_log
        # Sanity-check is thinking-enabled, so its batch submit must carry the
        # explicit adaptive-thinking cap, not the generic
        # batch_submit_and_wait default (4096).
        assert state.submits_by_type["SanityCheckResponse"].max_tokens == THINKING_MAX_TOKENS


# ---------------------------------------------------------------------------
# Tests -- sanity check abort
# ---------------------------------------------------------------------------


class TestSanityCheckAbort:
    """interval=1, 3 steps, sanity check fires after step 1, returns 'abort'."""

    async def test_abort_returns_failure(self, env: "WorkflowEnvironment") -> None:
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
        state = ScenarioState(
            plan=three_step_plan,
            transitions={"sc-task": [SUCCESS]},
            sanity_responses={"sc-task": [abort_response]},
        )
        input = ForgeTaskInput(
            task=SC_TASK,
            repo_root="/tmp/repo",
            plan=True,
            max_exploration_rounds=0,
            sanity_check_interval=1,
        )
        result = await run_task(env, input, state)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert "Sanity check aborted" in (result.error or "")

    async def test_abort_only_one_step_result(self, env: "WorkflowEnvironment") -> None:
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
        state = ScenarioState(
            plan=three_step_plan,
            transitions={"sc-task": [SUCCESS]},
            sanity_responses={"sc-task": [abort_response]},
        )
        input = ForgeTaskInput(
            task=SC_TASK,
            repo_root="/tmp/repo",
            plan=True,
            max_exploration_rounds=0,
            sanity_check_interval=1,
        )
        result = await run_task(env, input, state)
        assert len(result.step_results) == 1
        assert result.sanity_check_count == 1


# ---------------------------------------------------------------------------
# Tests -- sanity check revise
# ---------------------------------------------------------------------------


class TestSanityCheckRevise:
    """interval=1, 3 steps, sanity check fires after step 1, returns 'revise' with 1 step."""

    async def test_revise_replaces_remaining_steps(self, env: "WorkflowEnvironment") -> None:
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
        state = ScenarioState(
            plan=three_step_plan,
            # step 1 succeeds, then sanity check revises.
            # revised-1 succeeds (no more sanity check since it's the last step).
            transitions={"sc-task": [SUCCESS, SUCCESS]},
            sanity_responses={"sc-task": [revise_response]},
        )
        input = ForgeTaskInput(
            task=SC_TASK,
            repo_root="/tmp/repo",
            plan=True,
            max_exploration_rounds=0,
            sanity_check_interval=1,
        )
        result = await run_task(env, input, state)
        assert result.status == TransitionSignal.SUCCESS
        assert len(result.step_results) == 2
        assert result.step_results[0].step_id == "s1"
        assert result.step_results[1].step_id == "revised-1"

    async def test_revise_updates_plan_in_result(self, env: "WorkflowEnvironment") -> None:
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
        state = ScenarioState(
            plan=three_step_plan,
            transitions={"sc-task": [SUCCESS, SUCCESS]},
            sanity_responses={"sc-task": [revise_response]},
        )
        input = ForgeTaskInput(
            task=SC_TASK,
            repo_root="/tmp/repo",
            plan=True,
            max_exploration_rounds=0,
            sanity_check_interval=1,
        )
        result = await run_task(env, input, state)
        assert result.plan is not None
        # Plan should have 2 steps: original s1 + revised-1
        assert len(result.plan.steps) == 2
        assert result.plan.steps[1].step_id == "revised-1"


# ---------------------------------------------------------------------------
# Tests -- sanity check disabled
# ---------------------------------------------------------------------------


class TestSanityCheckDisabled:
    """interval=0 (default), verify no sanity check activities called."""

    async def test_no_sanity_check_when_disabled(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan=SC_PLAN,
            transitions={"sc-task": [SUCCESS] * 4},
        )
        input = ForgeTaskInput(
            task=SC_TASK,
            repo_root="/tmp/repo",
            plan=True,
            max_exploration_rounds=0,
            sanity_check_interval=0,
        )
        result = await run_task(env, input, state)
        assert result.status == TransitionSignal.SUCCESS
        assert result.sanity_check_count == 0
        assert not state.called("assemble_sanity_check_context")
        # Batch lane: the old parse-handler-written "call_sanity_check" entry
        # is a parse_llm_response entry here; rewritten for the same reason as
        # TestSanityCheckContinue.test_sanity_check_activities_called above.
        assert "parse_llm_response:SanityCheckResponse" not in state.call_log


# ---------------------------------------------------------------------------
# Tests -- sanity check skips last step
# ---------------------------------------------------------------------------


class TestSanityCheckSkipsLastStep:
    """interval=1, 2 steps, verify sanity check fires after step 1 but not after step 2."""

    async def test_fires_after_first_not_last(self, env: "WorkflowEnvironment") -> None:
        two_step_plan = Plan(
            task_id="sc-task",
            steps=[
                PlanStep(step_id="s1", description="Step 1.", target_files=["a.py"]),
                PlanStep(step_id="s2", description="Step 2.", target_files=["b.py"]),
            ],
            explanation="Two steps.",
        )
        state = ScenarioState(
            plan=two_step_plan,
            transitions={"sc-task": [SUCCESS, SUCCESS]},
        )
        input = ForgeTaskInput(
            task=SC_TASK,
            repo_root="/tmp/repo",
            plan=True,
            max_exploration_rounds=0,
            sanity_check_interval=1,
        )
        result = await run_task(env, input, state)
        assert result.status == TransitionSignal.SUCCESS
        assert result.sanity_check_count == 1
        # Only one sanity check call, not two. Batch lane: this counts
        # parse_llm_response:SanityCheckResponse entries, the replacement for
        # the old parse-handler-written call_sanity_check entries (see the
        # module docstring).
        assert state.call_log.count("parse_llm_response:SanityCheckResponse") == 1
