"""Temporal worker entry point for Forge.

Connects to the Temporal server, registers all activities and workflows,
and runs the worker until interrupted.
"""

from __future__ import annotations

import os

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from forge.activities import (
    assemble_context,
    assemble_planner_context,
    assemble_step_context,
    assemble_sub_task_context,
    call_llm,
    call_planner,
    commit_changes_activity,
    create_worktree_activity,
    evaluate_transition,
    remove_worktree_activity,
    reset_worktree_activity,
    validate_output,
    write_files,
    write_output,
)
from forge.workflows import FORGE_TASK_QUEUE, ForgeSubTaskWorkflow, ForgeTaskWorkflow

DEFAULT_TEMPORAL_ADDRESS = "localhost:7233"


async def run_worker(address: str | None = None) -> None:
    """Connect to Temporal and run the Forge worker."""
    if address is None:
        address = os.environ.get("FORGE_TEMPORAL_ADDRESS", DEFAULT_TEMPORAL_ADDRESS)

    client = await Client.connect(
        address,
        data_converter=pydantic_data_converter,
    )

    worker = Worker(
        client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=[ForgeTaskWorkflow, ForgeSubTaskWorkflow],
        activities=[
            assemble_context,
            assemble_planner_context,
            assemble_step_context,
            assemble_sub_task_context,
            call_llm,
            call_planner,
            commit_changes_activity,
            create_worktree_activity,
            evaluate_transition,
            remove_worktree_activity,
            reset_worktree_activity,
            validate_output,
            write_files,
            write_output,
        ],
    )

    await worker.run()
