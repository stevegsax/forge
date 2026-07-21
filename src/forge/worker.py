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
from forge.temporal_client import connect_temporal
from forge.workflows import ForgeSubTaskWorkflow, ForgeTaskWorkflow

if TYPE_CHECKING:
    from collections.abc import Callable

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


async def run_worker(
    address: str | None = None,
    *,
    identity: str | None = None,
) -> None:
    """Connect to Temporal and run the Forge worker.

    The single composition root: ``ForgeSettings()`` reads the environment once
    (fail-fast on an unset ``FORGE_DB_URL``); the store engine, the shared
    AnthropicLLM/AsyncAnthropic client, and the optional S3 blob store are each
    built exactly once and injected into the four activity classes. Forge submits
    anthropic only (T4.2 ST3) — no MistralOcr client. No activity reads config or
    builds a client itself.
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
    # blob store is built only when configured. Forge submits anthropic only
    # (T4.2 ST3) — no MistralOcr client here.
    sdk_client = make_client()
    llm = AnthropicLLM(sdk_client)
    engine = get_store_engine(settings.db.url)
    blobs = S3Blobs(settings.blob.bucket, settings.blob.prefix) if settings.blob.bucket else None

    store_activities = StoreActivities(engine)
    context_activities = ContextActivities(engine)
    llm_activities = LlmActivities(llm)
    batch_activities = BatchActivities(
        client=sdk_client,
        output_types=OUTPUT_TYPES,
        engine=engine,
        blob_store=blobs,
    )

    workflows: list[type] = [
        ForgeTaskWorkflow,
        ForgeSubTaskWorkflow,
        ExportPlaybookWorkflow,
        ManualPlaybookWorkflow,
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
        batch_activities.parse_llm_response,
        batch_activities.batch_status,
        batch_activities.fetch_batch_result,
    ]
    if _INGESTION_AVAILABLE:
        assert prepare_transcript is not None
        activities.append(prepare_transcript)

    # Worker construction and the signal-handled graceful-drain loop now live in
    # sax_platform.temporal.worker.run_worker (T3.4, ST7) — forge keeps only its
    # own setup (store, output types, tracing) and the activity/workflow
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
