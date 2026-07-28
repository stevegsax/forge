"""The plan preflight gate and the REVISE cap (T5.6).

Behavioral scenarios for the two places a structurally broken plan is now
stopped: at plan acceptance (``blocks.dispatch.dispatch_planner``, which every
planner call on both lanes goes through), and at the sanity check's REVISE
splice. The pure decision surface — which findings each check produces, and what
``splice_revision`` does with a cap — is unit-tested without Temporal in
``tests/test_plan_checks.py``; these tests prove the workflow acts on it.

Written in the T5.5 harness: one ``ScenarioState`` per test, scripting keyed by
identity, ``sync_mode=True`` (no scenario here exercises the batch transport —
the gate is lane-independent by construction, since it sits above the lane fork).
"""

from typing import TYPE_CHECKING

from forge.models import (
    MAX_PLAN_STEPS,
    ForgeTaskInput,
    Plan,
    PlanStep,
    SanityCheckCallResult,
    SanityCheckResponse,
    SanityCheckVerdict,
    SubTask,
    TaskDefinition,
    TransitionSignal,
)
from tests.support.workflow_harness import ScenarioState, run_task

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment

SUCCESS = TransitionSignal.SUCCESS.value

TASK = TaskDefinition(task_id="preflight-task", description="Build an API.")

PLANNED_INPUT = ForgeTaskInput(
    task=TASK,
    repo_root="/tmp/repo",
    plan=True,
    max_exploration_rounds=0,
    sync_mode=True,
)

GOOD_PLAN = Plan(
    task_id="preflight-task",
    steps=[
        PlanStep(step_id="s1", description="Create models.", target_files=["models.py"]),
        PlanStep(step_id="s2", description="Create API.", target_files=["api.py"]),
    ],
    explanation="Two clean steps.",
)

# Two steps sharing an id: the second would re-run the first's identity.
DUPLICATE_ID_PLAN = Plan(
    task_id="preflight-task",
    steps=[
        PlanStep(step_id="s1", description="Create models.", target_files=["models.py"]),
        PlanStep(step_id="s1", description="Create API.", target_files=["api.py"]),
    ],
    explanation="Duplicate step ids.",
)

# Two fan-out children claiming one file — the defect that used to buy an LLM
# conflict-resolution call at merge time.
OVERLAPPING_PLAN = Plan(
    task_id="preflight-task",
    steps=[
        PlanStep(
            step_id="fan-step",
            description="Two children, one file.",
            target_files=[],
            sub_tasks=[
                SubTask(sub_task_id="st1", description="a", target_files=["shared.py"]),
                SubTask(sub_task_id="st2", description="b", target_files=["shared.py"]),
            ],
        )
    ],
    explanation="Overlapping fan-out targets.",
)

# A violation two levels down: invisible to the pre-T5.6 one-level checks.
NESTED_VIOLATION_PLAN = Plan(
    task_id="preflight-task",
    steps=[
        PlanStep(
            step_id="fan-step",
            description="A nesting child and a leaf.",
            target_files=[],
            sub_tasks=[
                SubTask(
                    sub_task_id="st1",
                    description="Nested node.",
                    target_files=[],
                    sub_tasks=[
                        SubTask(sub_task_id="gc1", description="a", target_files=["gc.py"]),
                        SubTask(sub_task_id="gc1", description="b", target_files=["gc2.py"]),
                    ],
                ),
                SubTask(sub_task_id="st2", description="c", target_files=["st2.py"]),
            ],
        )
    ],
    explanation="Duplicate ids among grandchildren.",
)


class TestPreflightRetryArm:
    """A rejected plan is re-planned with the violations in the context."""

    async def test_rejected_then_accepted(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan_sequence={"preflight-task": [DUPLICATE_ID_PLAN, GOOD_PLAN]},
            transitions={"preflight-task": [SUCCESS, SUCCESS]},
        )
        result = await run_task(env, PLANNED_INPUT, state, workflow_id="test-preflight-retry")

        assert result.status == TransitionSignal.SUCCESS
        # Two planner calls: the rejected plan never reached a step.
        assert state.count("call_planner") == 2
        assert result.plan is not None
        assert [s.step_id for s in result.plan.steps] == ["s1", "s2"]
        # Both attempts are on the interactions ledger — a rejected plan still cost money.
        planner_rows = [req for req in state.persisted if getattr(req, "role", "") == "planner"]
        assert len(planner_rows) == 2

    async def test_retry_context_carries_the_violations(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(
            plan_sequence={"preflight-task": [OVERLAPPING_PLAN, GOOD_PLAN]},
            transitions={"preflight-task": [SUCCESS, SUCCESS]},
        )
        await run_task(env, PLANNED_INPUT, state, workflow_id="test-preflight-retry-context")

        planner_rows = [req for req in state.persisted if getattr(req, "role", "") == "planner"]
        first, second = planner_rows
        assert "structural validation" not in first.user_prompt
        # The escalating context names the rule and the specific offender.
        assert "attempt 2 of 3" in second.user_prompt
        assert "overlapping_sub_task_targets: fan-step: shared.py" in second.user_prompt


class TestPreflightHalt:
    """Three structurally invalid plans halt the run cleanly (Principle 5)."""

    async def test_three_strikes_is_a_terminal_result(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(plan=DUPLICATE_ID_PLAN)
        result = await run_task(env, PLANNED_INPUT, state, workflow_id="test-preflight-halt")

        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.failure_kind == "plan_preflight"
        assert "after 3 planner attempts" in (result.error or "")
        assert "duplicate_step_ids: s1" in (result.error or "")
        assert state.count("call_planner") == 3
        # No step ran, and the halt is a returned result rather than a crash:
        # the run is recorded and its worktree is left for inspection.
        assert not state.called("assemble_step_context")
        assert result.worktree_path == "/tmp/repo/.forge-worktrees/preflight-task"
        assert [type(req).__name__ for req in state.persisted].count("PersistRun") == 1

    async def test_halt_reports_the_planner_spend(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(plan=DUPLICATE_ID_PLAN)
        result = await run_task(env, PLANNED_INPUT, state, workflow_id="test-preflight-halt-spend")

        assert result.planner_stats is not None
        assert result.planner_stats.input_tokens == 300
        assert result.llm_totals is not None
        assert result.llm_totals.call_count == 1

    async def test_nested_violation_is_caught(self, env: "WorkflowEnvironment") -> None:
        """Two grandchildren sharing an id — three levels of plan, one gate."""
        state = ScenarioState(plan=NESTED_VIOLATION_PLAN)
        result = await run_task(env, PLANNED_INPUT, state, workflow_id="test-preflight-nested")

        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.failure_kind == "plan_preflight"
        assert "duplicate_sub_task_ids: fan-step/st1/gc1" in (result.error or "")
        # The fan-out never started: no child was ever assembled.
        assert not state.called("assemble_sub_task_context")


class TestReviseCap:
    """An over-cap or invalid REVISE splice terminates instead of hanging."""

    @staticmethod
    def _revise(steps: list[PlanStep]) -> SanityCheckCallResult:
        return SanityCheckCallResult(
            task_id="preflight-task",
            response=SanityCheckResponse(
                verdict=SanityCheckVerdict.REVISE,
                explanation="Rework the rest.",
                revised_steps=steps,
            ),
            model_name="mock-reasoning",
            input_tokens=200,
            output_tokens=100,
            latency_ms=300.0,
        )

    async def test_over_cap_splice_terminates_cleanly(self, env: "WorkflowEnvironment") -> None:
        """The splice would build a Plan of 26 steps.

        Constructing it raises a pydantic ``ValidationError`` *inside workflow
        code*, which Temporal retries as a workflow task forever — the run hangs
        rather than fails. The cap catches the case before construction.
        """
        oversized = [
            PlanStep(step_id=f"r{i}", description="d", target_files=[f"r{i}.py"])
            for i in range(MAX_PLAN_STEPS)
        ]
        state = ScenarioState(
            plan=GOOD_PLAN,
            transitions={"preflight-task": [SUCCESS, SUCCESS]},
            sanity_responses={"preflight-task": [self._revise(oversized)]},
        )
        input = PLANNED_INPUT.model_copy(update={"sanity_check_interval": 1})
        result = await run_task(env, input, state, workflow_id="test-revise-over-cap")

        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.failure_kind == "plan_revision"
        assert "exceed the step cap" in (result.error or "")
        # The first step ran and is reported; the revised tail never started.
        assert len(result.step_results) == 1
        assert result.sanity_check_count == 1

    async def test_structurally_invalid_revision_terminates(
        self, env: "WorkflowEnvironment"
    ) -> None:
        """A revised step reusing a completed step's id would re-run its identity."""
        state = ScenarioState(
            plan=GOOD_PLAN,
            transitions={"preflight-task": [SUCCESS, SUCCESS]},
            sanity_responses={
                "preflight-task": [
                    self._revise(
                        [PlanStep(step_id="s1", description="Again.", target_files=["again.py"])]
                    )
                ]
            },
        )
        input = PLANNED_INPUT.model_copy(update={"sanity_check_interval": 1})
        result = await run_task(env, input, state, workflow_id="test-revise-invalid")

        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.failure_kind == "plan_revision"
        assert "duplicate_step_ids: s1" in (result.error or "")

    async def test_a_valid_revision_still_splices(self, env: "WorkflowEnvironment") -> None:
        """The cap is a guard, not a behavior change for ordinary revisions."""
        state = ScenarioState(
            plan=GOOD_PLAN,
            transitions={"preflight-task": [SUCCESS, SUCCESS]},
            sanity_responses={
                "preflight-task": [
                    self._revise(
                        [
                            PlanStep(
                                step_id="revised-1",
                                description="Revised.",
                                target_files=["revised.py"],
                            )
                        ]
                    )
                ]
            },
        )
        input = PLANNED_INPUT.model_copy(update={"sanity_check_interval": 1})
        result = await run_task(env, input, state, workflow_id="test-revise-valid")

        assert result.status == TransitionSignal.SUCCESS
        assert result.plan is not None
        assert [s.step_id for s in result.plan.steps] == ["s1", "revised-1"]
