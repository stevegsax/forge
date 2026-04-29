"""Temporal worker for the playbook service.

Runs on a separate task queue from Forge.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from temporalio.client import Client
from temporalio.converter import DataConverter
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import (
    SandboxedWorkflowRunner,
    SandboxRestrictions,
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
from pbook.workflows.export import ExportWorkflow
from pbook.workflows.extraction import ExtractionWorkflow
from pbook.workflows.maintenance import MaintenanceWorkflow
from pbook.workflows.manual_entry import ManualEntryWorkflow
from pbook.workflows.retrieval import RetrievalWorkflow

logger = logging.getLogger(__name__)

PBOOK_TASK_QUEUE = "pbook-task-queue"


def _get_data_converter() -> DataConverter:
    """Get a DataConverter that supports Pydantic models natively (Phase 4)."""
    from temporalio.contrib.pydantic import PydanticPayloadConverter

    return DataConverter(
        payload_converter_class=PydanticPayloadConverter,
    )


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

    # Use custom DataConverter for Phase 4 improvements
    client = await Client.connect(address, data_converter=_get_data_converter())

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
            RetrievalWorkflow,
            ExportWorkflow,
            ExtractionWorkflow,
            ManualEntryWorkflow,
            MaintenanceWorkflow,
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
        ],
        graceful_shutdown_timeout=timedelta(seconds=30),
    )

    logger.info("pbook worker starting on queue %s", PBOOK_TASK_QUEUE)

    try:
        await worker.run()
    except asyncio.CancelledError:
        logger.info("pbook worker shutting down")
