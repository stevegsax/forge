"""Crash recovery: a worker killed mid-task, restarted, finishes the run (T5.5).

The README's durability claim is that a forge task survives losing the process
executing it — Temporal keeps the workflow's state in its own event history, so
a restarted worker picks the run up where it stopped rather than starting over
or losing it. Nothing in the suite demonstrated that until this scenario.

The kill is real, not simulated: the first worker's ``assemble_context`` blocks
forever, so when its ``Worker`` context exits, that activity is cancelled out
from under the workflow exactly as a ``kill -9`` would strand it. A second
worker then starts against the same task queue and the same still-running
workflow, re-executes the stranded activity, and drives the run to SUCCESS —
without repeating the work that was already recorded (the worktree is created
once across both workers).

Both workers run with ``max_cached_workflows=0``, which turns off sticky
execution. With the cache on, Temporal routes a workflow's later tasks back to
the *same* worker's private sticky queue, and a dead worker's sticky queue only
drains after its schedule-to-start timeout — a delay the time-skipping server
does not fast-forward, so the test would sit there waiting in real time.
Disabling the cache puts every workflow task on the shared queue, which is
where a real restarted worker picks the run up once that timeout has passed.

Sync lane (D6): the transport is not what is under test here.
"""

import asyncio
from typing import TYPE_CHECKING

from sax_platform.contracts.constants import FORGE_TASK_QUEUE
from temporalio import activity
from temporalio.worker import Worker

from forge.models import (
    AssembleContextInput,
    AssembledContext,
    ForgeTaskInput,
    TaskDefinition,
    TransitionSignal,
)
from forge.workflows import ForgeTaskWorkflow
from tests.support.workflow_harness import ScenarioState, build_activities

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment

CRASH_TASK = ForgeTaskInput(
    task=TaskDefinition(
        task_id="crash-task",
        description="Survive a worker restart.",
        target_files=["hello.py"],
    ),
    repo_root="/tmp/repo",
    max_attempts=1,
    max_exploration_rounds=0,
    sync_mode=True,
)


async def _wait_for(predicate: "object", *, timeout: float = 30.0) -> None:
    """Poll a callable until it is true (the mocks run in this process)."""
    assert callable(predicate)
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            msg = "timed out waiting for the workflow to reach the blocking activity"
            raise AssertionError(msg)
        await asyncio.sleep(0.01)


class TestCrashRecovery:
    async def test_restarted_worker_resumes_and_completes(self, env: "WorkflowEnvironment") -> None:
        state = ScenarioState(transitions={"crash-task": [TransitionSignal.SUCCESS.value]})
        blocked = asyncio.Event()

        @activity.defn(name="assemble_context")
        async def blocking_assemble_context(input: AssembleContextInput) -> AssembledContext:
            """Never returns: the worker is shut down while this is in flight."""
            state.call_log.append("assemble_context:blocked")
            blocked.set()
            await asyncio.Event().wait()  # cancelled by the worker shutdown
            raise AssertionError("unreachable")  # pragma: no cover

        doomed = build_activities(state, replace={"assemble_context": blocking_assemble_context})

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeTaskWorkflow],
            activities=doomed,
            max_cached_workflows=0,
        ):
            handle = await env.client.start_workflow(
                ForgeTaskWorkflow.run,
                CRASH_TASK,
                id="test-crash-recovery",
                task_queue=FORGE_TASK_QUEUE,
            )
            await _wait_for(blocked.is_set)
            # The kill is not vacuous: the run is mid-flight, not finished.
            assert (await handle.describe()).status.name == "RUNNING"
        # The worker is gone; its in-flight activity died with it.

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeTaskWorkflow],
            activities=build_activities(state),
            max_cached_workflows=0,
        ):
            result = await handle.result()

        assert result.status == TransitionSignal.SUCCESS
        assert result.output_files == {"hello.py": "print('hello')\n"}
        # The restarted worker re-ran the stranded activity...
        assert state.count("assemble_context") == 2
        assert "assemble_context:blocked" in state.call_log
        # ...but not the work Temporal had already recorded: the worktree the
        # first worker created is the one the run finished in.
        assert state.count("create_worktree") == 1
        assert "commit:success" in state.call_log
