"""Step-block worktree cleanup and per-attempt exploration (T5.2).

Moved verbatim from ``tests/test_workflows.py`` in T5.5 — these scenarios were
already written in the target pattern (mocks built per test, closing over the
test's own recorder), so the migration only relocated them; the factory they
share now lives in ``tests/support/block_mocks.py`` because
``tests/workflows/test_interactions.py`` uses it too.
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
from forge.workflows import ForgeSubTaskWorkflow, ForgeTaskWorkflow
from tests.support.block_mocks import step_block_activities


class TestStepBlockCleanup:
    """A mid-step exception leaves no worktree and no branch behind (T5.2).

    ``remove_worktree_activity`` is asserted with ``force=True``, which is what
    makes the activity delete the ``forge/<task_id>`` branch as well; that the
    branch really disappears is pinned against real git in
    ``tests/test_git.py::TestRemoveWorktree``.
    """

    @pytest.mark.asyncio
    async def test_single_step_cleans_up(self, env: WorkflowEnvironment) -> None:
        calls: list[str] = []
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeTaskWorkflow],
            activities=step_block_activities(calls, raise_in="write_output"),
        ):
            with pytest.raises(WorkflowFailureError):
                await env.client.execute_workflow(
                    ForgeTaskWorkflow.run,
                    ForgeTaskInput(
                        task=TaskDefinition(task_id="boom-single", description="d"),
                        repo_root="/tmp/repo",
                        max_attempts=2,
                        max_exploration_rounds=0,
                        sync_mode=True,
                    ),
                    id="test-cleanup-single",
                    task_queue=FORGE_TASK_QUEUE,
                )
        assert "remove_worktree:boom-single:force=True" in calls
        # It failed inside the first attempt, so no second attempt ran.
        assert calls.count("write_output") == 1

    @pytest.mark.asyncio
    async def test_planned_step_cleans_up_the_borrowed_worktree(
        self, env: WorkflowEnvironment
    ) -> None:
        """The plan's worktree belongs to _run_planned, so its wrap does the cleanup."""
        calls: list[str] = []
        plan = Plan(
            task_id="boom-planned",
            steps=[PlanStep(step_id="step-1", description="d", target_files=["hello.py"])],
            explanation="one step",
        )
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeTaskWorkflow],
            activities=step_block_activities(calls, raise_in="write_output", plan=plan),
        ):
            with pytest.raises(WorkflowFailureError):
                await env.client.execute_workflow(
                    ForgeTaskWorkflow.run,
                    ForgeTaskInput(
                        task=TaskDefinition(task_id="boom-planned", description="d"),
                        repo_root="/tmp/repo",
                        plan=True,
                        max_step_attempts=2,
                        max_exploration_rounds=0,
                        sync_mode=True,
                    ),
                    id="test-cleanup-planned",
                    task_queue=FORGE_TASK_QUEUE,
                )
        assert "remove_worktree:boom-planned:force=True" in calls

    @pytest.mark.asyncio
    async def test_sub_task_cleans_up(self, env: WorkflowEnvironment) -> None:
        calls: list[str] = []
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeSubTaskWorkflow],
            activities=step_block_activities(calls, raise_in="write_output"),
        ):
            with pytest.raises(WorkflowFailureError):
                await env.client.execute_workflow(
                    ForgeSubTaskWorkflow.run,
                    SubTaskInput(
                        parent_task_id="boom-parent",
                        parent_description="d",
                        sub_task=SubTask(
                            sub_task_id="st1", description="d", target_files=["hello.py"]
                        ),
                        repo_root="/tmp/repo",
                        parent_branch="forge/boom-parent",
                        max_attempts=2,
                        sync_mode=True,
                    ),
                    id="test-cleanup-subtask",
                    task_queue=FORGE_TASK_QUEUE,
                )
        assert "remove_worktree:boom-parent.sub.st1:force=True" in calls


class TestExplorationPerAttempt:
    @pytest.mark.asyncio
    async def test_each_attempt_explores_its_own_worktree(self, env: WorkflowEnvironment) -> None:
        """Exploration is a hook *inside* the attempt loop, so it sees that
        attempt's worktree — not attempt 1's, which retry has already removed."""
        calls: list[str] = []
        paths = ["/tmp/repo/.forge-worktrees/explore-1", "/tmp/repo/.forge-worktrees/explore-2"]
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeTaskWorkflow],
            activities=step_block_activities(
                calls, worktree_paths=paths, fail_first_validation=True
            ),
        ):
            result = await env.client.execute_workflow(
                ForgeTaskWorkflow.run,
                ForgeTaskInput(
                    task=TaskDefinition(task_id="explore-task", description="d"),
                    repo_root="/tmp/repo",
                    max_attempts=2,
                    max_exploration_rounds=1,
                    sync_mode=True,
                ),
                id="test-exploration-per-attempt",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.status == TransitionSignal.SUCCESS
        # Ordering: exploration runs between assemble and the generation call,
        # once per attempt, against that attempt's own worktree.
        assert calls == [
            f"create_worktree:{paths[0]}",
            f"assemble_context:{paths[0]}",
            f"call_exploration_llm:{paths[0]}",
            f"fulfill_context_requests:{paths[0]}",
            f"call_llm:{paths[0]}",
            "write_output",
            "validate_output",
            "remove_worktree:explore-task:force=True",
            f"create_worktree:{paths[1]}",
            f"assemble_context:{paths[1]}",
            f"call_exploration_llm:{paths[1]}",
            f"fulfill_context_requests:{paths[1]}",
            f"call_llm:{paths[1]}",
            "write_output",
            "validate_output",
            "commit:success",
        ]
        # The explored context reached the generation call, not just the loop.
        assert result.llm_stats is not None
