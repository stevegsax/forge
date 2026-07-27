"""Interaction records: the exploration arm, and uniformity across all five arms (T5.3).

Moved verbatim from ``tests/test_workflows.py`` in T5.5 — these scenarios were
already written in the target pattern (mocks built per test, closing over the
test's own recorder), so the migration only relocated them and their factories.
"""

from typing import TYPE_CHECKING

import pytest
from sax_platform.contracts.constants import FORGE_TASK_QUEUE
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from forge.models import ForgeTaskInput, TaskDefinition, TransitionSignal
from forge.workflows import ForgeSubTaskWorkflow, ForgeTaskWorkflow
from tests.support.block_mocks import (
    exploration_batch_activities,
    five_arm_activities,
    interactions,
    step_block_activities,
)

if TYPE_CHECKING:
    from forge.persist_models import PersistRequest


class TestExplorationInteractionRecord:
    @pytest.mark.asyncio
    async def test_sync_lane_persists_role_and_tokens(self, env: WorkflowEnvironment) -> None:
        """The sync activity's envelope carries the prompts and spend into the row."""
        calls: list[str] = []
        persisted: list[PersistRequest] = []
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeTaskWorkflow],
            activities=step_block_activities(calls, persisted=persisted),
        ):
            result = await env.client.execute_workflow(
                ForgeTaskWorkflow.run,
                ForgeTaskInput(
                    task=TaskDefinition(task_id="explore-persist", description="d"),
                    repo_root="/tmp/repo",
                    max_attempts=1,
                    max_exploration_rounds=1,
                    sync_mode=True,
                ),
                id="test-exploration-persist-sync",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.status == TransitionSignal.SUCCESS
        rows = interactions(persisted)
        # Every arm this run touched produced exactly one row.
        assert [row.role for row in rows] == ["exploration", "llm"]
        explore_row = rows[0]
        assert explore_row.model_name == "mock-explorer"
        assert (explore_row.input_tokens, explore_row.output_tokens) == (41, 17)
        assert (explore_row.cache_creation_input_tokens, explore_row.cache_read_input_tokens) == (
            3,
            5,
        )
        assert explore_row.stop_reason == "end_turn"
        assert explore_row.latency_ms == 90.0
        # The prompts the activity assembled internally travelled home with it.
        assert explore_row.system_prompt == "exploration system"
        assert explore_row.user_prompt == "exploration user"
        # A second exploration round would collide on the idempotency key if the
        # per-role occurrence counter were shared with the generation arm.
        assert explore_row.idempotency_key.endswith(":exploration:0")

    @pytest.mark.asyncio
    async def test_batch_lane_persists_the_parsed_tokens(self, env: WorkflowEnvironment) -> None:
        """The batch lane assembles its own context, then persists the parsed spend."""
        calls: list[str] = []
        persisted: list[PersistRequest] = []
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeTaskWorkflow],
            activities=exploration_batch_activities(calls, persisted),
        ):
            result = await env.client.execute_workflow(
                ForgeTaskWorkflow.run,
                ForgeTaskInput(
                    task=TaskDefinition(task_id="explore-batch", description="d"),
                    repo_root="/tmp/repo",
                    max_attempts=1,
                    max_exploration_rounds=1,
                    sync_mode=False,
                ),
                id="test-exploration-persist-batch",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.status == TransitionSignal.SUCCESS
        # The exploration arm's batch lane assembles its context first, then runs
        # the same transport as every other arm.
        assert calls == [
            "assemble_context",
            "assemble_exploration_context",
            "submit_batch_request:ExplorationResponse",
            "parse_llm_response:ExplorationResponse",
            "submit_batch_request:LLMResponse",
            "parse_llm_response:LLMResponse",
        ]
        rows = interactions(persisted)
        assert [row.role for row in rows] == ["exploration", "llm"]
        explore_row = rows[0]
        assert explore_row.model_name == "mock-explorer"
        assert (explore_row.input_tokens, explore_row.output_tokens) == (123, 45)
        assert (explore_row.cache_creation_input_tokens, explore_row.cache_read_input_tokens) == (
            7,
            9,
        )
        # Prompts come from the assembled context on this lane.
        assert explore_row.system_prompt == "explore system"
        assert explore_row.user_prompt == "explore user"


class TestInteractionRecordsForEveryArm:
    """Every dispatch arm writes an interaction record with its token counts.

    The T5.3 acceptance criterion. Exploration was the arm that historically
    wrote nothing at all; the other four each had their own copy of the persist,
    which is what made "all of them, uniformly" impossible to state until the
    dispatch block owned the shape once.
    """

    @pytest.mark.asyncio
    async def test_all_five_arms_persist_with_tokens(self, env: WorkflowEnvironment) -> None:
        calls: list[str] = []
        persisted: list[PersistRequest] = []
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[ForgeTaskWorkflow, ForgeSubTaskWorkflow],
            activities=five_arm_activities(calls, persisted),
        ):
            result = await env.client.execute_workflow(
                ForgeTaskWorkflow.run,
                ForgeTaskInput(
                    task=TaskDefinition(task_id="five-arm-task", description="d"),
                    repo_root="/tmp/repo",
                    plan=True,
                    max_exploration_rounds=1,
                    sanity_check_interval=1,
                    max_sub_task_attempts=1,
                    resolve_conflicts=True,
                    sync_mode=True,
                ),
                id="test-five-arm-interactions",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.status == TransitionSignal.SUCCESS
        # Every arm actually ran (a missing arm would make the role check vacuous).
        assert "call_exploration_llm" in calls
        assert "call_planner" in calls
        assert "call_conflict_resolution" in calls
        assert "call_sanity_check" in calls
        assert [c for c in calls if c.startswith("call_llm:")]

        rows = interactions(persisted)
        assert {row.role for row in rows} == {
            "exploration",
            "planner",
            "llm",
            "conflict_resolution",
            "sanity_check",
        }
        for row in rows:
            # Spend is what T7.4's budget enforcement will read; a row with zeroed
            # tokens is indistinguishable from a call that never happened.
            assert row.input_tokens > 0, row.role
            assert row.output_tokens > 0, row.role
            assert row.latency_ms > 0, row.role
            assert row.model_name, row.role
            # ...and the prompts that produced it, for every arm.
            assert row.system_prompt, row.role
            assert row.user_prompt, row.role
        # Keys are unique per (role, occurrence) — three generation calls here
        # (two children plus the second step) must not collide.
        keys = [row.idempotency_key for row in rows]
        assert len(set(keys)) == len(keys)
        assert sum(1 for row in rows if row.role == "llm") == 3
