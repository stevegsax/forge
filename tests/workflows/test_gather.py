"""The shared gather block: per-child isolation and owned-worktree cleanup (T5.3).

Moved verbatim from ``tests/test_workflows.py`` in T5.5 — these scenarios were
already written in the target pattern (mocks built per test, closing over the
test's own recorder), so the migration only relocated them and their factory.
"""

import pytest
from sax_platform.contracts.constants import FORGE_TASK_QUEUE
from temporalio.client import WorkflowFailureError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from forge.models import (
    ForgeTaskInput,
    Plan,
    PlanStep,
    SubTask,
    SubTaskInput,
    TaskDefinition,
    TransitionSignal,
)
from forge.persist_models import PersistRequest, PersistRun
from forge.workflows import ForgeSubTaskWorkflow, ForgeTaskWorkflow
from tests.support.block_mocks import gather_activities

TWO_CHILD_PLAN = Plan(
    task_id="isolation-task",
    steps=[
        PlanStep(
            step_id="fan-step",
            description="Two-child fan-out step.",
            target_files=[],
            sub_tasks=[
                SubTask(sub_task_id="st1", description="Produce st1.", target_files=["st1.py"]),
                SubTask(sub_task_id="st2", description="Produce st2.", target_files=["st2.py"]),
            ],
        )
    ],
    explanation="One fan-out step, two sub-tasks.",
)


class TestFanOutChildCrashIsolation:
    """A child that *raises* becomes a failed SubTaskResult (T5.3).

    Before the shared gather, both gathers bare-awaited their children, so one
    crashed child propagated a ChildWorkflowError out of run(): no TaskResult,
    no run record, and every in-flight sibling terminated with the parent.
    """

    @pytest.mark.asyncio
    async def test_crashed_child_becomes_a_failed_result(self, env: WorkflowEnvironment) -> None:
        calls: list[str] = []
        persisted: list[PersistRequest] = []
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeTaskWorkflow, ForgeSubTaskWorkflow],
            activities=gather_activities(
                calls, persisted, crash_sub_task="st2", plan=TWO_CHILD_PLAN
            ),
        ):
            result = await env.client.execute_workflow(
                ForgeTaskWorkflow.run,
                ForgeTaskInput(
                    task=TaskDefinition(task_id="isolation-task", description="d"),
                    repo_root="/tmp/repo",
                    plan=True,
                    max_sub_task_attempts=1,
                    max_exploration_rounds=0,
                    sync_mode=True,
                ),
                id="test-gather-child-crash",
                task_queue=FORGE_TASK_QUEUE,
            )

        # The parent returned a TaskResult instead of crashing out.
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.failure_kind == "step_failed"

        children = {r.sub_task_id: r for r in result.step_results[0].sub_task_results}
        assert set(children) == {"st1", "st2"}
        # The sibling ran to completion rather than being terminated mid-flight.
        assert children["st1"].status == TransitionSignal.SUCCESS
        assert "call_llm:st1" in calls
        # The crashed child is a normal failed result carrying its own kind.
        assert children["st2"].status == TransitionSignal.FAILURE_TERMINAL
        assert children["st2"].failure_kind == "child_crashed"
        assert children["st2"].error.startswith("Child workflow failed: ")
        # ...and the gather's own failure names it.
        assert "st2" in result.step_results[0].error

    @pytest.mark.asyncio
    async def test_run_record_is_still_written(self, env: WorkflowEnvironment) -> None:
        """The headline consequence: a crashed child no longer costs the run its row."""
        calls: list[str] = []
        persisted: list[PersistRequest] = []
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeTaskWorkflow, ForgeSubTaskWorkflow],
            activities=gather_activities(
                calls, persisted, crash_sub_task="st2", plan=TWO_CHILD_PLAN
            ),
        ):
            await env.client.execute_workflow(
                ForgeTaskWorkflow.run,
                ForgeTaskInput(
                    task=TaskDefinition(task_id="isolation-task", description="d"),
                    repo_root="/tmp/repo",
                    plan=True,
                    max_sub_task_attempts=1,
                    max_exploration_rounds=0,
                    sync_mode=True,
                ),
                id="test-gather-child-crash-run-row",
                task_queue=FORGE_TASK_QUEUE,
            )

        runs = [req for req in persisted if isinstance(req, PersistRun)]
        assert len(runs) == 1
        assert runs[0].task_result.status == TransitionSignal.FAILURE_TERMINAL


class TestNestedGatherWorktreeCleanup:
    """An owned gather removes its worktree — and its branch — on every exit."""

    @staticmethod
    def _nested_input() -> SubTaskInput:
        return SubTaskInput(
            parent_task_id="parent-task",
            parent_description="Build an API.",
            sub_task=SubTask(
                sub_task_id="st1",
                description="Nested node.",
                target_files=[],
                sub_tasks=[
                    SubTask(sub_task_id="gc1", description="Produce gc1.", target_files=["gc1.py"]),
                    SubTask(sub_task_id="gc2", description="Produce gc2.", target_files=["gc2.py"]),
                ],
            ),
            repo_root="/tmp/repo",
            parent_branch="forge/parent-task",
            max_attempts=1,
            depth=0,
            max_depth=2,
            sync_mode=True,
        )

    @pytest.mark.asyncio
    async def test_mid_gather_exception_cleans_worktree_and_branch(
        self, env: WorkflowEnvironment
    ) -> None:
        """The leak T5.2 left: the nested gather created its worktree with no
        exception wrap, so any raise between creation and a result-path removal
        left the worktree *and* its forge/<id> branch behind — and the next run
        of that id then failed on ``worktree add``."""
        calls: list[str] = []
        persisted: list[PersistRequest] = []
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeSubTaskWorkflow],
            activities=gather_activities(
                calls, persisted, raise_in="detect_file_conflicts_activity"
            ),
        ):
            with pytest.raises(WorkflowFailureError):
                await env.client.execute_workflow(
                    ForgeSubTaskWorkflow.run,
                    self._nested_input(),
                    id="test-nested-gather-cleanup",
                    task_queue=FORGE_TASK_QUEUE,
                )

        # force=True is what makes the activity delete the branch too
        # (tests/test_git.py pins the real branch deletion).
        assert "remove_worktree:parent-task.sub.st1:force=True" in calls
        # The failure happened after the children gathered, not before.
        assert "detect_file_conflicts" in calls

    @pytest.mark.asyncio
    async def test_success_removes_the_owned_worktree_exactly_once(
        self, env: WorkflowEnvironment
    ) -> None:
        calls: list[str] = []
        persisted: list[PersistRequest] = []
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeSubTaskWorkflow],
            activities=gather_activities(calls, persisted),
        ):
            result = await env.client.execute_workflow(
                ForgeSubTaskWorkflow.run,
                self._nested_input(),
                id="test-nested-gather-success",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.status == TransitionSignal.SUCCESS
        assert sorted(result.output_files) == ["gc1.py", "gc2.py"]
        nested_removals = [c for c in calls if c.startswith("remove_worktree:parent-task.sub.st1:")]
        assert len(nested_removals) == 1
        # A nested node never commits (D16) — its output travels home instead.
        assert not [c for c in calls if c.startswith("commit:")]


class TestGatherDuplicateSubTaskIds:
    """Colliding sub-task ids fail the gather before any child is started.

    Two children with the same id would share a compound id — one worktree, one
    child-workflow id — so the second would silently reset the first's work.

    Driven through ``ForgeSubTaskWorkflow`` since T5.6: a *planner-produced*
    duplicate no longer reaches the gather at all, because the preflight gate
    rejects the plan and halts the run (see ``test_preflight.py``). This gather
    guard is still the backstop for the sub-task input path, which no planner
    output passes through — and it is the reason a duplicate can never cost a
    silently overwritten worktree.
    """

    @pytest.mark.asyncio
    async def test_duplicate_ids_fail_the_step(self, env: WorkflowEnvironment) -> None:
        calls: list[str] = []
        persisted: list[PersistRequest] = []
        dup_input = SubTaskInput(
            parent_task_id="dup-task",
            parent_description="Build an API.",
            sub_task=SubTask(
                sub_task_id="st1",
                description="Nested node with colliding children.",
                target_files=[],
                sub_tasks=[
                    SubTask(sub_task_id="gc1", description="a", target_files=["a.py"]),
                    SubTask(sub_task_id="gc1", description="b", target_files=["b.py"]),
                ],
            ),
            repo_root="/tmp/repo",
            parent_branch="forge/dup-task",
            max_attempts=1,
            depth=0,
            max_depth=2,
            sync_mode=True,
        )
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeTaskWorkflow, ForgeSubTaskWorkflow],
            activities=gather_activities(calls, persisted),
        ):
            result = await env.client.execute_workflow(
                ForgeSubTaskWorkflow.run,
                dup_input,
                id="test-gather-duplicate-ids",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.failure_kind == "duplicate_sub_task_ids"
        assert result.error == "Duplicate nested sub-task IDs detected"
        # No child ran: nothing was assembled, nothing was merged.
        assert not [c for c in calls if c.startswith("assemble_sub_task_context:")]
        assert "detect_file_conflicts" not in calls
