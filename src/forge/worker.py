"""Temporal worker entry point for Forge.

Connects to the Temporal server, registers all activities and workflows,
and runs the worker until interrupted.
"""

from __future__ import annotations

import logging
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
    BatchActivities,
    ContextActivities,
    LlmActivities,
    StoreActivities,
    assemble_conflict_resolution_context,
    assemble_exploration_context,
    assemble_planner_context,
    assemble_sanity_check_context,
    commit_changes_activity,
    create_worktree_activity,
    detect_file_conflicts_activity,
    evaluate_transition,
    remove_worktree_activity,
    reset_worktree_activity,
    validate_output,
    validate_playbook_entry,
    write_files,
    write_output,
)
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

# INTERIM (Phase 4 deletes the schedules): execution-timeout backstop for the
# scheduled batch-poller run. A run that exceeds its timeout is terminated by
# Temporal so the overlap=SKIP schedule can fire the next run — guarding against a
# single wedged run starving every later cycle. The poller activity's
# start_to_close is 5 min, so the schedule gets a wider 10-minute budget.
_POLLER_EXECUTION_TIMEOUT = timedelta(minutes=10)

logger = logging.getLogger(__name__)


def _init_store(url: str) -> None:
    """Run database migrations against the configured store *url* on startup.

    The URL comes from ``ForgeSettings().db.url`` (fail-fast: an unset
    ``FORGE_DB_URL`` raises when the settings are built, before this is reached).
    An unreachable database raises here, and the worker refuses to start.
    """
    from sqlalchemy.engine import make_url

    from forge.store import run_migrations

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
    """Connect to Temporal and run the Forge worker.

    The single composition root: ``ForgeSettings()`` reads the environment once
    (fail-fast on an unset ``FORGE_DB_URL``); the store engine, the shared
    AnthropicLLM/AsyncAnthropic client, the optional S3 blob store, and the
    optional MistralOcr client are each built exactly once and injected into the
    four activity classes. No activity reads config or builds a client itself.
    """
    from sax_platform.contracts.s3_blobs import S3Blobs
    from sax_platform.db import get_store_engine
    from sax_platform.llm import AnthropicLLM, make_client

    from forge.logging_config import silence_noisy_loggers
    from forge.output_types import OUTPUT_TYPES
    from forge.settings import ForgeSettings
    from forge.tracing import init_tracing, shutdown_tracing

    settings = ForgeSettings()

    _init_store(settings.db.url)
    init_tracing(settings.tracing.exporter)
    silence_noisy_loggers()

    resolved_address = settings.temporal.address if address is None else address
    client = await connect_temporal(resolved_address, identity=identity, settings=settings.temporal)

    # Build the process-wide dependencies ONCE. One AsyncAnthropic SDK client is
    # shared by the sync lane (AnthropicLLM) and the batch lane (BatchActivities);
    # one store engine (bounded Postgres pool) serves every store activity. The
    # blob store and MistralOcr client are built only when configured.
    sdk_client = make_client()
    llm = AnthropicLLM(sdk_client)
    engine = get_store_engine(settings.db.url)
    blobs = S3Blobs(settings.blob.bucket, settings.blob.prefix) if settings.blob.bucket else None
    mistral = None
    if settings.llm.mistral_api_key:
        from sax_platform.ocr import MistralOcr, make_mistral_client

        mistral = MistralOcr(make_mistral_client(settings.llm.mistral_api_key))

    store_activities = StoreActivities(engine)
    context_activities = ContextActivities(engine)
    llm_activities = LlmActivities(llm)
    batch_activities = BatchActivities(
        client=sdk_client,
        output_types=OUTPUT_TYPES,
        engine=engine,
        blob_store=blobs,
        temporal_client=client,
        mistral_ocr=mistral,
    )

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
        # Free-function activities (no dependency to inject).
        assemble_conflict_resolution_context,
        assemble_exploration_context,
        assemble_planner_context,
        assemble_sanity_check_context,
        commit_changes_activity,
        create_worktree_activity,
        detect_file_conflicts_activity,
        evaluate_transition,
        remove_worktree_activity,
        reset_worktree_activity,
        validate_output,
        validate_playbook_entry,
        write_files,
        write_output,
        # Composition-root class bound methods (dependencies injected).
        store_activities.fetch_extraction_input,
        store_activities.save_extraction_results,
        store_activities.persist_to_store,
        store_activities.fetch_existing_playbooks,
        store_activities.fetch_playbook_ids,
        store_activities.export_single_playbook,
        context_activities.assemble_context,
        context_activities.assemble_step_context,
        context_activities.assemble_sub_task_context,
        context_activities.fulfill_context_requests,
        llm_activities.call_llm,
        llm_activities.call_planner,
        llm_activities.call_exploration_llm,
        llm_activities.call_sanity_check,
        llm_activities.call_conflict_resolution,
        llm_activities.call_extraction_llm,
        llm_activities.review_manual_playbook,
        batch_activities.submit_batch_request,
        batch_activities.submit_batch_blob,
        batch_activities.poll_batch_results,
        batch_activities.parse_llm_response,
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
