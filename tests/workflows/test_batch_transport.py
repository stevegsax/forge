"""Batch-transport scenarios (T4.1/D88): submit → poll → fetch → parse.

Migrated from ``tests/test_workflows.py`` in T5.5. These are the scenarios that
exercise the transport itself rather than merely running on it: the thinking
fallback and ``max_tokens`` caps carried on a submit, the three ways a batch
wait fails (25h ceiling, provider-terminal status, error-bearing fetch), and the
derived child execution timeout that lets a fan-out child survive a slow batch
turnaround.

The old sections swapped whole activity lists to change transport behavior
(``_BATCH_TIMEOUT_ACTIVITIES``, ``_BATCH_FETCH_ERROR_ACTIVITIES``,
``_BATCH_FAILED_STATUS_ACTIVITIES``). Those variants are now scenario data on
:class:`ScenarioState` — ``in_progress_polls``, ``batch_state``,
``fetch_error`` — over the one canonical mock set, and each waiter's poll
counter is keyed by its own ``batch_id`` so two concurrent waits never
interleave.
"""

from typing import TYPE_CHECKING

from forge.models import (
    FileOutput,
    ForgeTaskInput,
    LLMResponse,
    ParsedLLMResponse,
    Plan,
    PlanStep,
    SubTask,
    SubTaskInput,
    TaskDefinition,
    ThinkingPolicy,
    TransitionSignal,
)
from forge.presets import THINKING_MAX_TOKENS
from tests.support.workflow_harness import ScenarioState, run_sub_task, run_task

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment

SUCCESS = TransitionSignal.SUCCESS.value

# A batch that never ends: far more in_progress polls than the 25h ceiling
# admits at any poll interval, so the waiter runs to its ceiling.
STALL_POLLS = 10_000

# The thinking policy the planner submit must carry. ``_plan_task`` overwrites
# whatever ``assemble_planner_context`` returned with ``ForgeTaskInput.thinking``,
# so this is the task-level policy (which is also ``ThinkingPolicy()``'s default)
# reaching the submit — not a value threaded from the assemble activity.
PLANNER_THINKING = ThinkingPolicy(enabled=True, effort="high")


class TestBatchSingleStep:
    """Single-step workflow with sync_mode=False uses batch path."""

    async def test_batch_generation_success(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(transitions={"batch-test": [SUCCESS]})

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

        # The timer loop polls batch_status (mocked "ended") then fetches the
        # result — no signal needed.
        result = await run_task(env, input, state, workflow_id="test-batch-single")

        assert result.status == TransitionSignal.SUCCESS
        assert state.called("submit_batch_request")
        assert "parse_llm_response:LLMResponse" in state.call_log
        # Verify sync path was NOT called
        assert not state.called("call_llm")
        assert result.output_files == {"hello.py": "print('hello')\n"}
        # The generation arm omits `thinking`; the shared fallback in
        # batch_submit_and_wait must resolve it to disabled — not to
        # ThinkingPolicy()'s own enabled=True default (D94) and not to the
        # task-level ForgeTaskInput.thinking (enabled=True here).
        assert len(state.submits) == 1
        assert state.submits[0].thinking == ThinkingPolicy(enabled=False)
        # Generation is thinking-disabled, so its cap stays the untouched
        # batch_submit_and_wait default — not the thinking-enabled bump.
        assert state.submits[0].max_tokens == 4096

    async def test_batch_generation_persists_tokens_and_stop_reason(
        self, env: "WorkflowEnvironment"
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
        state = ScenarioState(
            transitions={"batch-token-test": [SUCCESS]},
            parsed_responses={"LLMResponse": distinctive_parsed},
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

        result = await run_task(env, input, state, workflow_id="test-batch-token-check")

        assert result.status == TransitionSignal.SUCCESS
        interactions = [r for r in state.persisted if r.kind == "interaction" and r.role == "llm"]
        assert len(interactions) == 1
        row = interactions[0]
        assert row.input_tokens == 777
        assert row.output_tokens == 888
        assert row.cache_creation_input_tokens == 13
        assert row.cache_read_input_tokens == 17
        assert row.stop_reason == "end_turn"

    async def test_batch_fetch_error_records_failure(self, env: "WorkflowEnvironment") -> None:
        """An error-bearing fetch (T4.1 fast failure) ends in a graceful
        FAILURE_TERMINAL run row + cleaned worktree, not a raw workflow crash (T1.6b)."""
        state = ScenarioState(fetch_error="Batch expired")

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

        # The batch ends but its fetch carries an error; the waiter raises a
        # non-retryable ApplicationError that run()'s failure-symmetry handler
        # catches instead of letting it crash the workflow.
        result = await run_task(env, input, state, workflow_id="test-batch-error")

        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert "ApplicationError" in (result.error or "")
        assert "Batch expired" in (result.error or "")
        # Worktree was cleaned — no orphan left behind.
        assert state.called("remove_worktree")
        # Exactly one FAILURE_TERMINAL run row was persisted (same PersistRun the
        # success path uses, keyed on (workflow_id, run_id)).
        runs = [r for r in state.persisted if r.kind == "run"]
        assert len(runs) == 1
        assert runs[0].task_result.status == TransitionSignal.FAILURE_TERMINAL


class TestBatchWaitFailure:
    """A batch wait that times out or errors leaves a run row and no orphan (T1.6b)."""

    async def test_wait_timeout_records_failure_and_cleans_worktree(
        self, env: "WorkflowEnvironment"
    ) -> None:
        """The batch never ends, so the poll loop runs to the 25h ceiling and gives
        up with a non-retryable ApplicationError; run() records a terminal row and
        removes the worktree instead of crashing out with an orphan."""
        state = ScenarioState(in_progress_polls={"": STALL_POLLS})

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

        # batch_status returns in_progress forever, so the poll loop runs to
        # the 25h ceiling (the time-skipping env fast-forwards every sleep) and
        # gives up.
        result = await run_task(env, input, state, workflow_id="test-batch-wait-timeout")

        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert "ApplicationError" in (result.error or "")
        # Worktree was cleaned — no orphan.
        assert state.called("remove_worktree")
        # Exactly one FAILURE_TERMINAL run row was persisted (same PersistRun the
        # success path uses).
        runs = [r for r in state.persisted if r.kind == "run"]
        assert len(runs) == 1
        assert runs[0].task_result.status == TransitionSignal.FAILURE_TERMINAL
        # The waiter recorded a terminal MISSING outcome before giving up.
        outcomes = [
            r for r in state.persisted if r.kind == "batch_outcome" and r.status == "missing"
        ]
        assert len(outcomes) == 1

    async def test_provider_terminal_status_records_failure_and_cleans_worktree(
        self, env: "WorkflowEnvironment"
    ) -> None:
        """A provider-terminal batch status (FAILED) records a terminal FAILED
        outcome and raises a non-retryable ApplicationError; run() records a
        terminal row and removes the worktree instead of crashing (T1.6b)."""
        state = ScenarioState(batch_state="failed")

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

        result = await run_task(env, input, state, workflow_id="test-batch-provider-fail")

        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert "ApplicationError" in (result.error or "")
        assert state.called("remove_worktree")
        runs = [r for r in state.persisted if r.kind == "run"]
        assert len(runs) == 1
        assert runs[0].task_result.status == TransitionSignal.FAILURE_TERMINAL
        # The waiter recorded a terminal FAILED outcome before raising.
        outcomes = [
            r for r in state.persisted if r.kind == "batch_outcome" and r.status == "failed"
        ]
        assert len(outcomes) == 1

    async def test_subtask_wait_timeout_returns_terminal_and_cleans_worktree(
        self, env: "WorkflowEnvironment"
    ) -> None:
        """A sub-task batch wait timing out returns a FAILURE_TERMINAL SubTaskResult
        (so the parent records the run row) and removes its own worktree (T1.6b)."""
        state = ScenarioState(in_progress_polls={"": STALL_POLLS})

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

        result = await run_sub_task(env, input, state, workflow_id="test-subtask-wait-timeout")

        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.sub_task_id == "st1"
        assert "ApplicationError" in (result.error or "")
        # The sub-task's own compound-id worktree was removed — no orphan.
        assert "remove_worktree:parent-task.sub.st1" in state.call_log


class TestBatchPlanned:
    """Planned workflow with sync_mode=False uses batch path for planner + generation."""

    async def test_batch_planner_and_generation(self, env: "WorkflowEnvironment") -> None:
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
        state = ScenarioState(
            transitions={"batch-plan-task": [SUCCESS]},
            parsed_responses={"Plan": plan_parsed, "LLMResponse": gen_parsed},
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

        # Two sequential batch calls (planner then generation); each polls
        # batch_status (mocked "ended") and fetches — no signals.
        result = await run_task(env, input, state, workflow_id="test-batch-planned")

        assert result.status == TransitionSignal.SUCCESS
        assert "submit_batch_request:Plan" in state.call_log
        assert "submit_batch_request:LLMResponse" in state.call_log
        assert "parse_llm_response:Plan" in state.call_log
        assert "parse_llm_response:LLMResponse" in state.call_log
        assert result.plan is not None
        assert len(result.step_results) == 1
        # Shared thinking fallback (blocks/transport.py): the planner call
        # passes an explicit thinking policy through unchanged...
        assert state.submits_by_type["Plan"].thinking == PLANNER_THINKING
        # ...while the generation call omits `thinking` entirely and must
        # land disabled via the shared fallback, not enabled-by-default.
        assert state.submits_by_type["LLMResponse"].thinking == ThinkingPolicy(enabled=False)
        # Planner is thinking-enabled, so it carries the explicit
        # adaptive-thinking cap; generation stays thinking-disabled and keeps
        # the untouched default.
        assert state.submits_by_type["Plan"].max_tokens == THINKING_MAX_TOKENS
        assert state.submits_by_type["LLMResponse"].max_tokens == 4096


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
# ForgeSubTaskWorkflow child, in batch mode. The mode-aware _child_timeout sizes
# the child from its permitted batch-wait budget, so:
#   (a) a child survives a multi-poll (>20 min) batch turnaround the old fixed
#       15-20 min ceiling would have killed; and
#   (b) a child whose batch never ends still hits its own 25h wait ceiling, cleans its
#       worktree, and returns FAILURE_TERMINAL, so the live parent records a run row
#       (T1.6b failure symmetry extended to a spawned child).
#
# The planner batch always ends immediately, so the workflow reaches the fan-out step;
# only the child's generation batch ("LLMResponse") is delayed/stalled — the scenario
# stalls a waiter by its own output type, and the poll counters are keyed by batch_id.

CHILD_RESPONSE = LLMResponse(
    files=[FileOutput(file_path="a.py", content="# child\n")],
    explanation="Child output.",
)


def _st3c_plan(task_id: str) -> Plan:
    return Plan(
        task_id=task_id,
        steps=[
            PlanStep(
                step_id="fan-step",
                description="Single-child fan-out step.",
                target_files=[],
                sub_tasks=[
                    SubTask(sub_task_id="st1", description="Do the thing.", target_files=["a.py"]),
                ],
            ),
        ],
        explanation="One fan-out step.",
    )


def _st3c_input(task_id: str) -> ForgeTaskInput:
    """A planned, single-fan-out-step task in batch mode, with the child execution
    timeout derived by the mode-aware _child_timeout — no override on the parent."""
    return ForgeTaskInput(
        task=TaskDefinition(task_id=task_id, description="Build a thing."),
        repo_root="/tmp/repo",
        plan=True,
        max_exploration_rounds=0,
        sync_mode=False,  # batch mode
    )


class TestBatchFanOutChildTimeoutDerivation:
    """The child execution timeout is derived from its batch-wait budget (T4.1 ST3c)."""

    async def test_child_survives_slow_batch_turnaround(self, env: "WorkflowEnvironment") -> None:
        """AC (a): the child's generation batch stays in_progress for 3 polls (>20 min
        of workflow time at the 600s default) before ending. The old fixed 15-20-min
        _child_timeout would have killed the child; the derived batch budget lets it
        finish, so the fan-out step and the workflow succeed."""
        state = ScenarioState(
            plan=_st3c_plan("st3c-slow"),
            llm_responses={"": CHILD_RESPONSE},
            in_progress_polls={"LLMResponse": 3},
        )

        result = await run_task(env, _st3c_input("st3c-slow"), state)

        assert result.status == TransitionSignal.SUCCESS
        # The single fan-out step succeeded.
        assert len(result.step_results) == 1
        assert result.step_results[0].status == TransitionSignal.SUCCESS
        assert result.step_results[0].sub_task_results[0].status == TransitionSignal.SUCCESS
        # The slow path really engaged: the child's generation batch was polled at
        # least 4 times (3 in_progress + 1 ended) — >20 min at the 600s poll floor.
        assert max(state.status_polls.values()) >= 4

    async def test_child_ceiling_expiry_cleans_worktree_and_records_run(
        self, env: "WorkflowEnvironment"
    ) -> None:
        """AC (b): the child's generation batch never ends, so the child hits its own
        25h wait ceiling, cleans its compound-id worktree, and returns FAILURE_TERMINAL
        to the parent; the live parent then records a terminal run row (T1.6b symmetry
        extended to a spawned child)."""
        state = ScenarioState(
            plan=_st3c_plan("st3c-stall"),
            llm_responses={"": CHILD_RESPONSE},
            in_progress_polls={"LLMResponse": STALL_POLLS},
        )

        result = await run_task(env, _st3c_input("st3c-stall"), state)

        # 1) The parent workflow ended terminally...
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        # 2) ...because the child returned FAILURE_TERMINAL from the fan-out step.
        assert len(result.step_results) == 1
        assert result.step_results[0].status == TransitionSignal.FAILURE_TERMINAL
        child_results = result.step_results[0].sub_task_results
        assert len(child_results) == 1
        assert child_results[0].status == TransitionSignal.FAILURE_TERMINAL
        # 3) The child cleaned its own compound-id worktree — no orphan.
        assert "remove_worktree:st3c-stall.sub.st1" in state.call_log
        # 4) The live parent persisted exactly one FAILURE_TERMINAL run row.
        runs = [r for r in state.persisted if r.kind == "run"]
        assert len(runs) == 1
        assert runs[0].task_result.status == TransitionSignal.FAILURE_TERMINAL
        # The child recorded a terminal MISSING batch outcome before giving up.
        missing = [
            r for r in state.persisted if r.kind == "batch_outcome" and r.status == "missing"
        ]
        assert len(missing) == 1
