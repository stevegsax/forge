"""Temporal worker entry point for the OCR app.

Runs on ``ocr-task-queue`` (same namespace + database as the platform). Registers
the OCR workflows and activities. The platform's batch poller (a separate worker on
``forge-task-queue``) signals OcrStoreWorkflow when a batch completes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sax_platform.contracts.constants import OCR_TASK_QUEUE
from sax_platform.temporal.client import connect_temporal
from sax_platform.temporal.worker import run_worker as _run_worker

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

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
from ocr.deps import set_mistral_ocr
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


def _init_mistral_ocr() -> None:
    """Construct and register the Mistral OCR capability (D88 / T3.3).

    Local import: unlike the rest of this module's top-level imports,
    ``sax_platform.ocr`` pulls in ``mistralai`` eagerly at module level (see
    its module docstring), so it is kept out of worker.py's top-level import
    graph rather than risk that dependency reaching workflow-sandbox-sensitive
    paths — mirroring ``_init_store``'s local import of ``ocr.store``/
    ``sqlalchemy`` above.
    """
    from sax_platform.ocr import MistralOcr, make_mistral_client

    set_mistral_ocr(MistralOcr(make_mistral_client()))


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


def activities() -> list[Callable[..., Any]]:
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
    """Connect to Temporal and run the OCR worker until interrupted.

    Owns app-specific setup (migrations, Mistral OCR DI, connecting); the
    ``Worker`` construction plus the signal-handled graceful-drain loop is
    ``sax_platform.temporal.worker.run_worker`` (shared across the platform
    and its consumer apps). ``graceful_shutdown_timeout`` is set explicitly to
    5 minutes to match the shared default rather than leaving it implicit.
    """
    import os
    from datetime import timedelta

    if address is None:
        address = os.environ.get("FORGE_TEMPORAL_ADDRESS", DEFAULT_TEMPORAL_ADDRESS)

    _init_store()
    _init_mistral_ocr()
    client = await connect_temporal(address, identity=identity)
    await _run_worker(
        client,
        task_queue=OCR_TASK_QUEUE,
        workflows=workflows(),
        activities=activities(),
        graceful_shutdown_timeout=timedelta(minutes=5),
    )
