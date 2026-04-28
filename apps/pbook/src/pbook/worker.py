"""Temporal worker for the playbook service.

Runs on a separate task queue from Forge.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.converter import DataConverter

# Import activities
from pbook.activities.export import export_single_entry, fetch_entry_ids
from pbook.activities.extraction import (
    compute_embedding,
    extract_from_experience,
    record_ingested_session,
    record_ingested_session_error,
    save_extracted_entries,
)
from pbook.activities.maintenance import (
    consolidate_entries_llm,
    fetch_all_entries_for_maintenance,
    prune_entries,
)
from pbook.activities.retrieval import fetch_candidates, record_retrieval_event
from pbook.activities.review import (
    fetch_existing_entries,
    find_duplicates,
    review_entry,
    validate_entry,
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


async def run_worker(address: str = "localhost:7233") -> None:
    """Connect to Temporal and run the pbook worker."""
    from pbook.log_config import setup_logging

    setup_logging(console=True)
    _register_llm_provider()
    logger.info("Connecting to Temporal at %s", address)
    
    # Use custom DataConverter for Phase 4 improvements
    client = await Client.connect(address, data_converter=_get_data_converter())

    worker = Worker(
        client,
        task_queue=PBOOK_TASK_QUEUE,
        workflows=[
            RetrievalWorkflow,
            ExportWorkflow,
            ExtractionWorkflow,
            ManualEntryWorkflow,
            MaintenanceWorkflow,
        ],
        activities=[
            fetch_candidates,
            record_retrieval_event,
            fetch_entry_ids,
            export_single_entry,
            extract_from_experience,
            save_extracted_entries,
            validate_entry,
            fetch_existing_entries,
            review_entry,
            compute_embedding,
            find_duplicates,
            fetch_all_entries_for_maintenance,
            prune_entries,
            consolidate_entries_llm,
            record_ingested_session,
            record_ingested_session_error,
        ],
        graceful_shutdown_timeout=timedelta(seconds=30),
    )

    logger.info("pbook worker starting on queue %s", PBOOK_TASK_QUEUE)

    try:
        await worker.run()
    except asyncio.CancelledError:
        logger.info("pbook worker shutting down")
