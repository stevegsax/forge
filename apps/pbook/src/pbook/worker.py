"""Temporal worker for the playbook service.

Runs on a separate task queue from Forge.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from temporalio.client import Client
from temporalio.worker import Worker

from pbook.activities.export import export_single_entry, fetch_entry_ids
from pbook.activities.extraction import extract_from_experience, save_extracted_entries
from pbook.activities.retrieval import fetch_candidates
from pbook.activities.review import fetch_existing_entries, review_entry, validate_entry
from pbook.workflows.export import ExportWorkflow
from pbook.workflows.extraction import ExtractionWorkflow
from pbook.workflows.manual_entry import ManualEntryWorkflow
from pbook.workflows.retrieval import RetrievalWorkflow

logger = logging.getLogger(__name__)

PBOOK_TASK_QUEUE = "pbook-task-queue"


def _register_llm_provider() -> None:
    """Register the default LLM provider (Anthropic) for extraction and review."""
    from sax_llm import get_provider

    from pbook.llm import set_provider

    provider = get_provider("anthropic:claude-haiku-4-5-20251001")
    set_provider(provider)
    logger.info("Registered LLM provider: anthropic:claude-haiku-4-5-20251001")


async def run_worker(address: str = "localhost:7233") -> None:
    """Connect to Temporal and run the pbook worker."""
    from pbook.log_config import setup_logging

    setup_logging(console=True)
    _register_llm_provider()
    logger.info("Connecting to Temporal at %s", address)
    client = await Client.connect(address)

    worker = Worker(
        client,
        task_queue=PBOOK_TASK_QUEUE,
        workflows=[
            RetrievalWorkflow,
            ExportWorkflow,
            ExtractionWorkflow,
            ManualEntryWorkflow,
        ],
        activities=[
            fetch_candidates,
            fetch_entry_ids,
            export_single_entry,
            extract_from_experience,
            save_extracted_entries,
            validate_entry,
            fetch_existing_entries,
            review_entry,
        ],
        graceful_shutdown_timeout=timedelta(seconds=30),
    )

    logger.info("pbook worker starting on queue %s", PBOOK_TASK_QUEUE)

    try:
        await worker.run()
    except asyncio.CancelledError:
        logger.info("pbook worker shutting down")
