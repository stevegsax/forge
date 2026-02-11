"""Temporal worker entry point for Forge.

Connects to the Temporal server, registers all activities and workflows,
and runs the worker until interrupted.
"""

from __future__ import annotations

import logging
import os

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from forge.activities import (
    assemble_conflict_resolution_context,
    assemble_context,
    assemble_planner_context,
    assemble_sanity_check_context,
    assemble_step_context,
    assemble_sub_task_context,
    call_conflict_resolution,
    call_exploration_llm,
    call_extraction_llm,
    call_llm,
    call_planner,
    call_sanity_check,
    commit_changes_activity,
    create_worktree_activity,
    evaluate_transition,
    fetch_extraction_input,
    fulfill_context_requests,
    remove_worktree_activity,
    reset_worktree_activity,
    save_extraction_results,
    validate_output,
    write_files,
    write_output,
)
from forge.extraction_workflow import ForgeExtractionWorkflow
from forge.workflows import FORGE_TASK_QUEUE, ForgeSubTaskWorkflow, ForgeTaskWorkflow

DEFAULT_TEMPORAL_ADDRESS = "localhost:7233"

logger = logging.getLogger(__name__)


def _init_store() -> None:
    """Run database migrations on startup (best-effort)."""
    try:
        from forge.store import get_db_path, run_migrations

        db_path = get_db_path()
        if db_path is None:
            logger.info("Observability store disabled (FORGE_DB_PATH is empty)")
            return

        run_migrations(db_path)
        logger.info("Database migrations complete: %s", db_path)
    except Exception:
        logger.warning("Failed to run database migrations", exc_info=True)


async def run_worker(address: str | None = None) -> None:
    """Connect to Temporal and run the Forge worker."""
    from forge.tracing import init_tracing, shutdown_tracing

    if address is None:
        address = os.environ.get("FORGE_TEMPORAL_ADDRESS", DEFAULT_TEMPORAL_ADDRESS)

    _init_store()
    init_tracing()

    client = await Client.connect(
        address,
        data_converter=pydantic_data_converter,
    )

    worker = Worker(
        client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=[ForgeTaskWorkflow, ForgeSubTaskWorkflow, ForgeExtractionWorkflow],
        activities=[
            assemble_conflict_resolution_context,
            assemble_context,
            assemble_planner_context,
            assemble_sanity_check_context,
            assemble_step_context,
            assemble_sub_task_context,
            call_conflict_resolution,
            call_exploration_llm,
            call_extraction_llm,
            call_llm,
            call_planner,
            call_sanity_check,
            commit_changes_activity,
            create_worktree_activity,
            evaluate_transition,
            fetch_extraction_input,
            fulfill_context_requests,
            remove_worktree_activity,
            reset_worktree_activity,
            save_extraction_results,
            validate_output,
            write_files,
            write_output,
        ],
    )

    try:
        await worker.run()
    finally:
        shutdown_tracing()
