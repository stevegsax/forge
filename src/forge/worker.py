"""Temporal worker entry point for Forge.

Connects to the Temporal server, registers all activities and workflows,
and runs the worker until interrupted.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from sax_platform.contracts.constants import FORGE_TASK_QUEUE
from sax_platform.temporal.worker import run_worker as _run_platform_worker
from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleAlreadyRunningError,
    ScheduleIntervalSpec,
    ScheduleOverlapPolicy,
    SchedulePolicy,
    ScheduleSpec,
    ScheduleState,
    ScheduleUpdate,
    ScheduleUpdateInput,
)

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
    detect_file_conflicts_activity,
    evaluate_transition,
    export_single_playbook,
    fetch_existing_playbooks,
    fetch_extraction_input,
    fetch_playbook_ids,
    fulfill_context_requests,
    parse_llm_response,
    persist_to_store,
    poll_batch_results,
    remove_worktree_activity,
    reset_worktree_activity,
    review_manual_playbook,
    save_extraction_results,
    submit_batch_blob,
    submit_batch_request,
    validate_output,
    validate_playbook_entry,
    write_files,
    write_output,
)
from forge.activities.batch_poll import set_temporal_client
from forge.batch_poller_workflow import BatchPollerWorkflow
from forge.export_playbook_workflow import ExportPlaybookWorkflow

try:
    from forge.activities.ingestion import prepare_transcript
    from forge.ingestion_workflow import BatchIngestionWorkflow, TranscriptIngestionWorkflow

    _INGESTION_AVAILABLE = True
except ImportError:
    _INGESTION_AVAILABLE = False
    prepare_transcript = None  # type: ignore[assignment]
    BatchIngestionWorkflow = None  # type: ignore[assignment,misc]
    TranscriptIngestionWorkflow = None  # type: ignore[assignment,misc]

from forge.manual_playbook_workflow import ManualPlaybookWorkflow
from forge.models import BatchPollerInput
from forge.temporal_client import connect_temporal
from forge.workflows import ForgeSubTaskWorkflow, ForgeTaskWorkflow

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_TEMPORAL_ADDRESS = "localhost:7233"

# INTERIM (Phase 4 deletes the schedules): execution-timeout backstop for the
# scheduled batch-poller run. A run that exceeds its timeout is terminated by
# Temporal so the overlap=SKIP schedule can fire the next run — guarding against a
# single wedged run starving every later cycle. The poller activity's
# start_to_close is 5 min, so the schedule gets a wider 10-minute budget.
_POLLER_EXECUTION_TIMEOUT = timedelta(minutes=10)

logger = logging.getLogger(__name__)


def _init_store() -> None:
    """Run database migrations on startup.

    The store is mandatory: ``FORGE_DB_URL`` must be set (a ``sqlite:///`` URL
    for dev/tests, a ``postgresql+psycopg2://`` URL for production). An unset
    URL or an unreachable database raises, and the worker refuses to start.
    """
    from sqlalchemy.engine import make_url

    from forge.store import get_store_url, run_migrations

    url = get_store_url()
    run_migrations(url)
    logger.info(
        "Database migrations complete: %s",
        make_url(url).render_as_string(hide_password=True),
    )


async def _ensure_schedule(
    client: Client,
    schedule_id: str,
    workflow_name: str,
    workflow_arg: object,
    interval: timedelta,
    execution_timeout: timedelta = _POLLER_EXECUTION_TIMEOUT,
) -> None:
    """Create or update a Temporal schedule (idempotent).

    On first run, creates the schedule. On subsequent runs, updates the spec,
    action, and policy so an existing schedule picks up config changes. Handles
    the "already exists" case gracefully.

    ``execution_timeout`` caps each run and ``overlap=SKIP`` skips a new run while
    one is in flight — together the interim guard against a wedged run starving
    every later cycle (see T1.3; Phase 4 deletes these schedules).
    """
    schedule = Schedule(
        action=ScheduleActionStartWorkflow(
            workflow_name,
            workflow_arg,
            id=f"{schedule_id}-run",
            task_queue=FORGE_TASK_QUEUE,
            execution_timeout=execution_timeout,
        ),
        spec=ScheduleSpec(
            intervals=[ScheduleIntervalSpec(every=interval)],
        ),
        policy=SchedulePolicy(overlap=ScheduleOverlapPolicy.SKIP),
        state=ScheduleState(
            note=f"Forge schedule: {schedule_id}",
        ),
    )

    try:
        await client.create_schedule(schedule_id, schedule)
        logger.info("Created schedule %s (interval=%s)", schedule_id, interval)
    except ScheduleAlreadyRunningError:
        # Reconcile the existing schedule: apply the current spec, action (with
        # the execution-timeout backstop), and policy (overlap=SKIP) so a running
        # schedule created before T1.3 picks up the wedge guard.
        handle = client.get_schedule_handle(schedule_id)

        async def _updater(input: ScheduleUpdateInput) -> ScheduleUpdate:
            input.description.schedule.spec = schedule.spec
            input.description.schedule.action = schedule.action
            input.description.schedule.policy = schedule.policy
            return ScheduleUpdate(schedule=input.description.schedule)

        await handle.update(_updater)
        logger.info("Updated schedule %s (interval=%s)", schedule_id, interval)


async def run_worker(
    address: str | None = None,
    *,
    batch_poll_interval: int = 600,
    identity: str | None = None,
) -> None:
    """Connect to Temporal and run the Forge worker."""
    from forge.tracing import init_tracing, shutdown_tracing

    if address is None:
        address = os.environ.get("FORGE_TEMPORAL_ADDRESS", DEFAULT_TEMPORAL_ADDRESS)

    from forge.logging_config import silence_noisy_loggers

    _init_store()
    init_tracing()
    silence_noisy_loggers()

    client = await connect_temporal(address, identity=identity)

    # Inject Temporal client for poll activity signal delivery
    set_temporal_client(client)

    # Create/update schedules — if these fail, the worker is useless
    await _ensure_schedule(
        client,
        schedule_id="forge-batch-poller",
        workflow_name="BatchPollerWorkflow",
        workflow_arg=BatchPollerInput(),
        interval=timedelta(seconds=batch_poll_interval),
        execution_timeout=_POLLER_EXECUTION_TIMEOUT,
    )

    workflows: list[type] = [
        ForgeTaskWorkflow,
        ForgeSubTaskWorkflow,
        ExportPlaybookWorkflow,
        ManualPlaybookWorkflow,
        BatchPollerWorkflow,
    ]
    if _INGESTION_AVAILABLE:
        assert TranscriptIngestionWorkflow is not None
        assert BatchIngestionWorkflow is not None
        workflows.extend([TranscriptIngestionWorkflow, BatchIngestionWorkflow])
    else:
        logger.warning("pbook not installed — ingestion workflows skipped at worker registration")

    activities: list[Callable[..., Any]] = [
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
        detect_file_conflicts_activity,
        evaluate_transition,
        export_single_playbook,
        fetch_existing_playbooks,
        fetch_extraction_input,
        fetch_playbook_ids,
        fulfill_context_requests,
        parse_llm_response,
        persist_to_store,
        poll_batch_results,
        remove_worktree_activity,
        reset_worktree_activity,
        review_manual_playbook,
        save_extraction_results,
        submit_batch_blob,
        submit_batch_request,
        validate_output,
        validate_playbook_entry,
        write_files,
        write_output,
    ]
    if _INGESTION_AVAILABLE:
        assert prepare_transcript is not None
        activities.append(prepare_transcript)

    # Worker construction and the signal-handled graceful-drain loop now live in
    # sax_platform.temporal.worker.run_worker (T3.4, ST7) — forge keeps only its
    # own setup (store, output types, tracing, schedules) and the activity/workflow
    # registration lists. graceful_shutdown_timeout is explicit here (it matches
    # the platform default) so the 5-minute drain — long enough to never cancel an
    # in-flight LLM call, unlike the prior hardcoded 30s — stays visible in forge.
    try:
        await _run_platform_worker(
            client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=workflows,
            activities=activities,
            graceful_shutdown_timeout=timedelta(minutes=5),
        )
    finally:
        shutdown_tracing()
