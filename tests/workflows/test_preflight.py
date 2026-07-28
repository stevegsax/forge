"""The plan preflight gate and the REVISE cap (T5.6).

Behavioral scenarios for the two places a structurally broken plan is now
stopped: at plan acceptance (``blocks.dispatch.dispatch_planner``, which every
planner call on both lanes goes through), and at the sanity check's REVISE
splice. The pure decision surface — which findings each check produces, and what
``splice_revision`` does with a cap — is unit-tested without Temporal in
``tests/test_plan_checks.py``; these tests prove the workflow acts on it.

Written in the T5.5 harness: one ``ScenarioState`` per test, scripting keyed by
identity. Most scenarios run ``sync_mode=True``, where a planner attempt is one
activity; :class:`TestPreflightOnTheBatchLane` runs the same two verdicts with
``sync_mode=False``, where each attempt is a whole submit → poll → fetch → parse
cycle. The gate sits above the lane fork, so lane-independence is a claim about
the code's shape — those tests are what make it an observed fact, including that
a retry mints a *fresh* batch request rather than re-reading the rejected one.
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

# The same planned run on the batch transport: every planner attempt becomes a
# submit -> poll -> fetch -> parse cycle instead of one activity call.
BATCH_PLANNED_INPUT = PLANNED_INPUT.model_copy(update={"sync_mode": False})

GOOD_PLAN = Plan(
    task_id="preflight-task",
    steps=[
        PlanStep(step_id="s1", description="Create models.", target_files=["models.py"]),
        PlanStep(step_id="s2", description="Create API.", target_files=["api.py"]),
    ],
    explanation="Two clean steps.",
)

# The cheapest acceptable plan: one step, whose target is what the harness's
# default generation response writes.
ONE_STEP_PLAN = Plan(
    task_id="preflight-task",
    steps=[PlanStep(step_id="s1", description="Create hello.", target_files=["hello.py"])],
    explanation="One clean step.",
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
        """All three attempts, not just the last: the run paid for all three."""
        state = ScenarioState(plan=DUPLICATE_ID_PLAN)
        result = await run_task(env, PLANNED_INPUT, state, workflow_id="test-preflight-halt-spend")

        assert state.count("call_planner") == 3
        assert result.planner_stats is not None
        # Each mocked planner call is 300 input / 150 output tokens.
        assert result.planner_stats.input_tokens == 900
        assert result.planner_stats.output_tokens == 450
        assert result.llm_totals is not None
        assert result.llm_totals.call_count == 3
        assert result.llm_totals.input_tokens == 900

    async def test_halt_error_names_every_attempt(self, env: "WorkflowEnvironment") -> None:
        """The wording is the whole halt, not a snapshot of its last attempt."""
        state = ScenarioState(
            plan_sequence={
                "preflight-task": [DUPLICATE_ID_PLAN, OVERLAPPING_PLAN, DUPLICATE_ID_PLAN]
            }
        )
        result = await run_task(env, PLANNED_INPUT, state, workflow_id="test-preflight-halt-error")

        error = result.error or ""
        assert "after 3 planner attempts" in error
        assert "attempt 1: duplicate_step_ids: s1" in error
        assert "attempt 2: overlapping_sub_task_targets: fan-step: shared.py" in error
        assert "attempt 3: duplicate_step_ids: s1" in error

    async def test_nested_violation_is_caught(self, env: "WorkflowEnvironment") -> None:
        """Two grandchildren sharing an id — three levels of plan, one gate."""
        state = ScenarioState(plan=NESTED_VIOLATION_PLAN)
        result = await run_task(env, PLANNED_INPUT, state, workflow_id="test-preflight-nested")

        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.failure_kind == "plan_preflight"
        assert "duplicate_sub_task_ids: fan-step/st1/gc1" in (result.error or "")
        # The fan-out never started: no child was ever assembled.
        assert not state.called("assemble_sub_task_context")


class TestPreflightOnTheBatchLane:
    """The same two verdicts with ``sync_mode=False`` — one batch cycle per attempt.

    On this lane a rejected plan is expensive in a way the sync lane hides: each
    attempt is its own provider round trip, and the retry must mint a *new*
    request rather than re-reading the rejected one's stored bytes. Batch-lane
    identity is recovered from the ``request_id`` the way the real transport
    does, so the parse mock hands each attempt its own scripted plan.
    """

    @staticmethod
    def _plan_submits(state: ScenarioState) -> list[str]:
        """The request_id of every planner submit, in order."""
        return [s.request_id for s in state.submits if s.output_type_name == "Plan"]

    async def test_rejected_then_accepted(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(plan_sequence={"preflight-task": [DUPLICATE_ID_PLAN, ONE_STEP_PLAN]})
        result = await run_task(
            env, BATCH_PLANNED_INPUT, state, workflow_id="test-preflight-batch-retry"
        )

        assert result.status == TransitionSignal.SUCCESS
        assert result.plan is not None
        assert [s.step_id for s in result.plan.steps] == ["s1"]
        # Two full planner cycles, each with its own request — a retry is a new
        # submission, not a re-read of the rejected one.
        request_ids = self._plan_submits(state)
        assert len(request_ids) == 2
        assert len(set(request_ids)) == 2
        assert state.call_log.count("parse_llm_response:Plan") == 2
        # ...then the step's own batch cycle. The sync lane was never touched.
        assert state.call_log.count("submit_batch_request:LLMResponse") == 1
        assert not state.called("call_planner")
        assert not state.called("call_llm")
        # D97 boundary: the run recovered, so it reports its *surviving* calls —
        # the accepted planner call (300) and the generation call — exactly as a
        # step that succeeds on its second attempt reports one generation.
        assert result.planner_stats is not None
        assert result.planner_stats.input_tokens == 300
        assert result.llm_totals is not None
        assert result.llm_totals.call_count == 2

    async def test_three_strikes_halts(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(plan=DUPLICATE_ID_PLAN)
        result = await run_task(
            env, BATCH_PLANNED_INPUT, state, workflow_id="test-preflight-batch-halt"
        )

        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.failure_kind == "plan_preflight"
        assert "after 3 planner attempts" in (result.error or "")
        # Three distinct planner batch cycles and no step at all.
        assert len(set(self._plan_submits(state))) == 3
        assert state.call_log.count("parse_llm_response:Plan") == 3
        assert not state.called("assemble_step_context")
        assert "submit_batch_request:LLMResponse" not in state.call_log
        # All three attempts are the reported spend (300 input tokens each).
        assert result.planner_stats is not None
        assert result.planner_stats.input_tokens == 900
        assert result.llm_totals is not None
        assert result.llm_totals.call_count == 3
        # The halt is a returned result, so the run is on the ledger like any other.
        runs = [req for req in state.persisted if req.kind == "run"]
        assert len(runs) == 1
        assert runs[0].task_result.failure_kind == "plan_preflight"


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
