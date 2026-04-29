"""Temporal worker for the playbook service.

Runs on a separate task queue from Forge.
"""

from __future__ import annotations

import asyncio
import logging
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


async def run_worker(address: str = "localhost:7233") -> None:
    """Connect to Temporal and run the pbook worker."""
    from pbook.log_config import setup_logging

    setup_logging(console=True)
    _register_llm_provider()
    _register_output_types()
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
            "pydantic", "pydantic_core",
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

    try:
        await worker.run()
    except asyncio.CancelledError:
        logger.info("pbook worker shutting down")
