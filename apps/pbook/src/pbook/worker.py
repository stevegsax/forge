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
from pbook.activities.retrieval import fetch_candidates
from pbook.workflows.export import ExportWorkflow
from pbook.workflows.retrieval import RetrievalWorkflow

logger = logging.getLogger(__name__)

PBOOK_TASK_QUEUE = "pbook-task-queue"


async def run_worker(address: str = "localhost:7233") -> None:
    """Connect to Temporal and run the pbook worker."""
    client = await Client.connect(address)

    worker = Worker(
        client,
        task_queue=PBOOK_TASK_QUEUE,
        workflows=[
            RetrievalWorkflow,
            ExportWorkflow,
        ],
        activities=[
            fetch_candidates,
            fetch_entry_ids,
            export_single_entry,
        ],
        graceful_shutdown_timeout=timedelta(seconds=30),
    )

    logger.info("pbook worker starting on queue %s", PBOOK_TASK_QUEUE)

    try:
        await worker.run()
    except asyncio.CancelledError:
        logger.info("pbook worker shutting down")
