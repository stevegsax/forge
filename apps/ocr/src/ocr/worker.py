"""Temporal worker entry point for the OCR app.

Runs on ``ocr-task-queue`` (same namespace + database as the platform). Registers
the OCR workflows and activities. The platform's batch poller (a separate worker on
``forge-task-queue``) signals OcrStoreWorkflow when a batch completes.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from forge_contracts.constants import OCR_TASK_QUEUE
from forge_contracts.temporal import connect_temporal
from temporalio.worker import Worker

from ocr.activities import (
    build_ocr_request_blob,
    check_ocr_duplicate,
    clear_ocr_removal_mark,
    delete_file_content_blob,
    export_ocr_document,
    list_ocr_jobs,
    mark_ocr_for_removal,
    read_and_store_file_content,
    reassemble_ocr_chunks,
    split_file_into_chunks,
    store_ocr_result,
    upsert_ocr_status,
)
from ocr.workflow_export import OcrExportWorkflow
from ocr.workflow_gather import OcrGatherWorkflow
from ocr.workflow_list_jobs import OcrListJobsWorkflow
from ocr.workflow_mark_removal import (
    OcrClearRemovalMarkWorkflow,
    OcrMarkForRemovalWorkflow,
)
from ocr.workflow_store import OcrStoreWorkflow
from ocr.workflow_submit import OcrSubmitWorkflow

DEFAULT_TEMPORAL_ADDRESS = "localhost:7233"

logger = logging.getLogger(__name__)


def _init_store() -> None:
    """Run the OCR Alembic chain on startup. ``FORGE_DB_URL`` must be set."""
    from sqlalchemy.engine import make_url

    from ocr.store import get_store_url, run_migrations

    url = get_store_url()
    run_migrations(url)
    logger.info(
        "OCR migrations complete: %s",
        make_url(url).render_as_string(hide_password=True),
    )


def workflows() -> list[type]:
    """The OCR workflow classes registered on the worker."""
    return [
        OcrSubmitWorkflow,
        OcrStoreWorkflow,
        OcrGatherWorkflow,
        OcrExportWorkflow,
        OcrListJobsWorkflow,
        OcrMarkForRemovalWorkflow,
        OcrClearRemovalMarkWorkflow,
    ]


def activities() -> list:
    """The OCR activity functions registered on the worker."""
    return [
        read_and_store_file_content,
        split_file_into_chunks,
        build_ocr_request_blob,
        delete_file_content_blob,
        store_ocr_result,
        upsert_ocr_status,
        reassemble_ocr_chunks,
        export_ocr_document,
        check_ocr_duplicate,
        mark_ocr_for_removal,
        clear_ocr_removal_mark,
        list_ocr_jobs,
    ]


async def run_worker(address: str | None = None, *, identity: str | None = None) -> None:
    """Connect to Temporal and run the OCR worker until interrupted."""
    import os

    if address is None:
        address = os.environ.get("FORGE_TEMPORAL_ADDRESS", DEFAULT_TEMPORAL_ADDRESS)

    _init_store()
    client = await connect_temporal(address, identity=identity)
    worker = Worker(
        client,
        task_queue=OCR_TASK_QUEUE,
        workflows=workflows(),
        activities=activities(),
        graceful_shutdown_timeout=timedelta(seconds=30),
    )
    await worker.run()
