"""Temporal worker for the playbook service.

Runs on a separate task queue from Forge.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from datetime import timedelta

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import (
    SandboxedWorkflowRunner,
    SandboxRestrictions,
)

from pbook.activities.cli_ops import (
    add_entry_activity,
    approve_entry_activity,
    check_duplicate_activity,
    filter_already_ingested_activity,
    get_entry_activity,
    get_session_text_activity,
    list_entries_activity,
    list_sessions_activity,
    list_sources_activity,
    list_tags_activity,
    prune_activity,
    record_feedback_activity,
    record_started_sessions_activity,
    reject_entry_activity,
    review_queue_activity,
    update_entry_activity,
)

# Import activities
from pbook.activities.export import export_single_entry, fetch_entry_ids
from pbook.activities.extraction import (
    record_ingested_session,
    record_ingested_session_error,
    save_extracted_entries,
)
from pbook.activities.maintenance import (
    cluster_similar_entries,
    fetch_all_entries_for_maintenance,
    prune_entries,
    save_consolidated_entry,
)
from pbook.activities.retrieval import (
    compute_similarities_by_id,
    fetch_candidates,
    record_retrieval_event,
    score_and_pack,
)
from pbook.activities.review import (
    fetch_existing_entries,
    find_duplicates,
    validate_entry,
)
from pbook.workflow_steps import (
    llm_chat,
    llm_embed,
    register_output_type,
)

# Import workflows
from pbook.workflows.cli_ops import (
    AddEntryWorkflow,
    ApproveEntryWorkflow,
    CheckDuplicateWorkflow,
    FilterAlreadyIngestedWorkflow,
    GetEntryWorkflow,
    GetSessionTextWorkflow,
    ListEntriesWorkflow,
    ListSessionsWorkflow,
    ListSourcesWorkflow,
    ListTagsWorkflow,
    PruneWorkflow,
    RecordFeedbackWorkflow,
    RecordStartedSessionsWorkflow,
    RejectEntryWorkflow,
    ReviewQueueWorkflow,
    UpdateEntryWorkflow,
)
from pbook.workflows.export import ExportWorkflow
from pbook.workflows.extraction import ExtractionWorkflow
from pbook.workflows.maintenance import MaintenanceWorkflow
from pbook.workflows.manual_entry import ManualEntryWorkflow
from pbook.workflows.retrieval import RetrievalWorkflow

logger = logging.getLogger(__name__)

PBOOK_TASK_QUEUE = "pbook-task-queue"


def _register_llm_provider() -> None:
    """Register the default LLM provider (Anthropic) for extraction and review."""
    from sax_llm import get_provider

    from pbook.llm import set_provider

    # Default to a reasoning model for complex extraction and consolidation
    provider = get_provider("anthropic:claude-3-5-sonnet-20241022")
    set_provider(provider)
    logger.info("Registered LLM provider: anthropic:claude-3-5-sonnet-20241022")


def _register_output_types() -> None:
    """Register pbook's structured-output classes with the local registry.

    The generic ``llm_chat`` activity resolves output types by name at
    activity-time; without this registration the activity raises KeyError.
    """
    from pbook.llm import ConsolidationResult, ExtractionResult, ReviewResult

    register_output_type("ExtractionResult", ExtractionResult)
    register_output_type("ReviewResult", ReviewResult)
    register_output_type("ConsolidationResult", ConsolidationResult)
    logger.info("Registered output types: ExtractionResult, ReviewResult, ConsolidationResult")


def _migrate_if_configured() -> None:
    """Run Alembic migrations to head once at startup, if a store is configured.

    This is the worker's single migration point — activities no longer
    migrate per call. If ``PBOOK_DATABASE_URL`` is unset the store is
    disabled and migration is skipped.
    """
    from pbook.store import get_database_url, run_migrations

    url = get_database_url()
    if url is None:
        logger.warning("PBOOK_DATABASE_URL not set — skipping migrations (store disabled)")
        return
    run_migrations(url)
    logger.info("Database migrations applied (head)")


async def _run_until_shutdown(worker: Worker, stop: asyncio.Event) -> None:
    """Run ``worker`` until it exits on its own or ``stop`` requests a drain.

    Races ``worker.run()`` against ``stop.wait()``. If ``stop`` fires first (a
    signal handler set it), this requests a graceful shutdown and then waits
    for ``run()`` to return — ``Worker.shutdown()`` and ``Worker.run()`` both
    block until the same underlying drain completes, so the second await
    resolves right away. If ``run()`` finishes first — clean exit or a crash —
    the stop waiter is cancelled and ``run_task`` is awaited so any exception
    propagates to the caller unchanged.
    """
    run_task: asyncio.Task[None] = asyncio.create_task(worker.run())
    stop_task: asyncio.Task[bool] = asyncio.create_task(stop.wait())
    done, _pending = await asyncio.wait({run_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
    if run_task in done:
        stop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stop_task
        await run_task  # re-raise a crash, if any
        return
    await worker.shutdown()
    await run_task


def _request_shutdown(sig: signal.Signals, stop: asyncio.Event) -> None:
    """Signal handler: log receipt and flag the run loop to begin a graceful drain."""
    logger.info("received %s — draining (graceful_shutdown_timeout=30s)", sig.name)
    stop.set()


async def run_worker(address: str = "localhost:7233") -> None:
    """Connect to Temporal and run the pbook worker."""
    from pbook.log_config import setup_logging

    setup_logging(console=True)
    _register_llm_provider()
    _register_output_types()
    _migrate_if_configured()
    logger.info("Connecting to Temporal at %s", address)

    # Use the official Pydantic v2 data converter end-to-end so Pydantic
    # models (RetrievalInput, RetrievalResult, PlaybookEntry, etc.) round-trip
    # cleanly without the legacy-converter UserWarning at every workflow
    # submission.
    client = await Client.connect(address, data_converter=pydantic_data_converter)

    # Pass pydantic through the workflow sandbox. The sandbox is created
    # fresh per workflow run; without passthrough, pydantic_core gets
    # imported lazily inside the workflow body and triggers
    # "imported after initial workflow load" warnings, plus we'd pay the
    # re-import cost on every replay. pydantic + pydantic_core are
    # deterministic and safe to share across runs.
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(
            "pydantic",
            "pydantic_core",
        ),
    )

    worker = Worker(
        client,
        task_queue=PBOOK_TASK_QUEUE,
        workflow_runner=runner,
        workflows=[
            # Retrieval / extraction / manual / maintenance / export
            RetrievalWorkflow,
            ExportWorkflow,
            ExtractionWorkflow,
            ManualEntryWorkflow,
            MaintenanceWorkflow,
            # CLI-op workflows (every direct-DB CLI command except `migrate`)
            GetEntryWorkflow,
            ListEntriesWorkflow,
            ListSourcesWorkflow,
            ListTagsWorkflow,
            ReviewQueueWorkflow,
            ListSessionsWorkflow,
            GetSessionTextWorkflow,
            CheckDuplicateWorkflow,
            AddEntryWorkflow,
            ApproveEntryWorkflow,
            RejectEntryWorkflow,
            UpdateEntryWorkflow,
            RecordFeedbackWorkflow,
            PruneWorkflow,
            FilterAlreadyIngestedWorkflow,
            RecordStartedSessionsWorkflow,
        ],
        activities=[
            fetch_candidates,
            compute_similarities_by_id,
            score_and_pack,
            record_retrieval_event,
            fetch_entry_ids,
            export_single_entry,
            save_extracted_entries,
            validate_entry,
            fetch_existing_entries,
            find_duplicates,
            fetch_all_entries_for_maintenance,
            cluster_similar_entries,
            prune_entries,
            save_consolidated_entry,
            record_ingested_session,
            record_ingested_session_error,
            # Generic LLM workflow steps used by all workflows in this worker.
            llm_chat,
            llm_embed,
            # CLI-op activities — one per direct-DB command.
            get_entry_activity,
            list_entries_activity,
            list_sources_activity,
            list_tags_activity,
            review_queue_activity,
            list_sessions_activity,
            get_session_text_activity,
            check_duplicate_activity,
            add_entry_activity,
            approve_entry_activity,
            reject_entry_activity,
            update_entry_activity,
            record_feedback_activity,
            prune_activity,
            filter_already_ingested_activity,
            record_started_sessions_activity,
        ],
        graceful_shutdown_timeout=timedelta(seconds=30),
    )

    logger.info("pbook worker starting on queue %s", PBOOK_TASK_QUEUE)

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    handled_signals = (signal.SIGTERM, signal.SIGINT)
    for sig in handled_signals:
        loop.add_signal_handler(sig, _request_shutdown, sig, stop)

    try:
        await _run_until_shutdown(worker, stop)
        logger.info("worker exited cleanly")
    finally:
        for sig in handled_signals:
            loop.remove_signal_handler(sig)
