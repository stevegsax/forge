"""Temporal worker entry point for Forge.

Connects to the Temporal server, registers all activities and workflows,
and runs the worker until interrupted.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleIntervalSpec,
    ScheduleSpec,
    ScheduleState,
    ScheduleUpdate,
    ScheduleUpdateInput,
)
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.service import RPCError
from temporalio.worker import Worker

from forge.activities import (
    assemble_conflict_resolution_context,
    assemble_context,
    assemble_exploration_context,
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
    discover_capabilities,
    evaluate_transition,
    execute_mcp_tool,
    execute_skill,
    fetch_extraction_input,
    fulfill_context_requests,
    parse_llm_response,
    poll_batch_results,
    remove_worktree_activity,
    reset_worktree_activity,
    save_extraction_results,
    submit_batch_request,
    validate_output,
    write_files,
    write_output,
)
from forge.activities.batch_poll import set_temporal_client
from forge.batch_poller_workflow import BatchPollerWorkflow
from forge.extraction_workflow import ForgeExtractionWorkflow
from forge.models import BatchPollerInput, ExtractionWorkflowInput
from forge.workflows import (
    FORGE_TASK_QUEUE,
    ForgeActionWorkflow,
    ForgeSubTaskWorkflow,
    ForgeTaskWorkflow,
)

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


async def _ensure_schedule(
    client: Client,
    schedule_id: str,
    workflow_name: str,
    workflow_arg: object,
    interval: timedelta,
) -> None:
    """Create or update a Temporal schedule (idempotent).

    On first run, creates the schedule. On subsequent runs, updates the interval
    if it has changed. Handles the "already exists" case gracefully.
    """
    schedule = Schedule(
        action=ScheduleActionStartWorkflow(
            workflow_name,
            workflow_arg,
            id=f"{schedule_id}-run",
            task_queue=FORGE_TASK_QUEUE,
        ),
        spec=ScheduleSpec(
            intervals=[ScheduleIntervalSpec(every=interval)],
        ),
        state=ScheduleState(
            note=f"Forge schedule: {schedule_id}",
        ),
    )

    try:
        await client.create_schedule(schedule_id, schedule)
        logger.info("Created schedule %s (interval=%s)", schedule_id, interval)
    except RPCError as e:
        if "already exists" in str(e).lower():
            # Update the existing schedule with the new interval
            handle = client.get_schedule_handle(schedule_id)

            async def _updater(input: ScheduleUpdateInput) -> ScheduleUpdate:
                input.description.schedule.spec = schedule.spec
                return ScheduleUpdate(schedule=input.description.schedule)

            await handle.update(_updater)
            logger.info("Updated schedule %s (interval=%s)", schedule_id, interval)
        else:
            raise


async def run_worker(
    address: str | None = None,
    *,
    batch_poll_interval: int = 60,
    extraction_interval: int = 14400,
) -> None:
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

    # Inject Temporal client for poll activity signal delivery
    set_temporal_client(client)

    # Create/update schedules (best-effort)
    try:
        await _ensure_schedule(
            client,
            schedule_id="forge-batch-poller",
            workflow_name="BatchPollerWorkflow",
            workflow_arg=BatchPollerInput(),
            interval=timedelta(seconds=batch_poll_interval),
        )
    except Exception:
        logger.warning("Failed to create batch poller schedule", exc_info=True)

    try:
        await _ensure_schedule(
            client,
            schedule_id="forge-extraction-schedule",
            workflow_name="ForgeExtractionWorkflow",
            workflow_arg=ExtractionWorkflowInput(),
            interval=timedelta(seconds=extraction_interval),
        )
    except Exception:
        logger.warning("Failed to create extraction schedule", exc_info=True)

    worker = Worker(
        client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=[
            ForgeTaskWorkflow,
            ForgeSubTaskWorkflow,
            ForgeActionWorkflow,
            ForgeExtractionWorkflow,
            BatchPollerWorkflow,
        ],
        activities=[
            assemble_conflict_resolution_context,
            assemble_context,
            assemble_exploration_context,
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
            discover_capabilities,
            evaluate_transition,
            execute_mcp_tool,
            execute_skill,
            fetch_extraction_input,
            fulfill_context_requests,
            parse_llm_response,
            poll_batch_results,
            remove_worktree_activity,
            reset_worktree_activity,
            save_extraction_results,
            submit_batch_request,
            validate_output,
            write_files,
            write_output,
        ],
    )

    try:
        await worker.run()
    finally:
        shutdown_tracing()
