"""Temporal worker entry point for the OCR app — the composition root (T3.6).

Runs on ``ocr-task-queue`` (same namespace + database as the platform). Builds
the per-process settings, store engine, blob client, and Mistral OCR capability
ONCE at startup, injects all three into ``OcrStoreActivities``, and registers its
bound methods. Each OcrStoreWorkflow polls its own Mistral batch on a timer
(T4.2) and records the terminal outcome on the platform ``batch_jobs`` ledger
cross-queue — no signals.

This module is where OCR first gained logging: nothing configured a handler
before T3.6, so worker output went nowhere. ``setup_logging("ocr", console=True)``
attaches a rotating file handler (under ``$XDG_STATE_HOME/ocr`` or
``FORGE_LOG_DIR``) plus a stderr console handler, and ``silence_noisy_loggers``
quiets the Mistral SDK's OTel JSON-parse warnings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sax_platform.contracts.constants import OCR_TASK_QUEUE
from sax_platform.contracts.s3_blobs import S3Blobs
from sax_platform.db import get_store_engine
from sax_platform.logging import setup_logging, silence_noisy_loggers
from sax_platform.temporal.client import connect_temporal
from sax_platform.temporal.worker import run_worker as _run_worker

from ocr.activities import OcrStoreActivities
from ocr.settings import OcrSettings
from ocr.workflow_export import OcrExportWorkflow
from ocr.workflow_list_jobs import OcrListJobsWorkflow
from ocr.workflow_mark_removal import (
    OcrClearRemovalMarkWorkflow,
    OcrMarkForRemovalWorkflow,
)
from ocr.workflow_store import OcrStoreWorkflow
from ocr.workflow_submit import OcrSubmitWorkflow

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from sax_platform.ocr import MistralOcr

logger = logging.getLogger(__name__)


def _init_store(url: str) -> None:
    """Run the OCR Alembic chain against *url* on startup."""
    from sqlalchemy.engine import make_url

    from ocr.store import run_migrations

    run_migrations(url)
    logger.info(
        "OCR migrations complete: %s",
        make_url(url).render_as_string(hide_password=True),
    )


def _build_mistral_ocr(api_key: str | None) -> MistralOcr | None:
    """Construct the Mistral OCR capability, or ``None`` when no key is set.

    OCR now polls its own Mistral batches (T4.2, D88): ``OcrStoreActivities``
    submits, polls status, and fetches results through this capability, so the
    worker requires it and injects it (``run_worker`` fails fast on a missing
    key). This builder stays total — it returns ``None`` for an empty key and the
    composition root turns that into the startup error.

    Local import: ``sax_platform.ocr``'s runtime touch points pull in ``mistralai``
    lazily, but the client construction here does, so it is kept out of worker.py's
    top-level import graph (mirroring ``_init_store``'s local import).
    """
    if not api_key:
        return None
    from sax_platform.ocr import MistralOcr, make_mistral_client

    return MistralOcr(make_mistral_client(api_key))


def workflows() -> list[type]:
    """The OCR workflow classes registered on the worker."""
    return [
        OcrSubmitWorkflow,
        OcrStoreWorkflow,
        OcrExportWorkflow,
        OcrListJobsWorkflow,
        OcrMarkForRemovalWorkflow,
        OcrClearRemovalMarkWorkflow,
    ]


def activity_methods(store: OcrStoreActivities) -> list[Callable[..., Any]]:
    """The OCR activities registered on the worker: ``store``'s bound methods.

    One list, shared by the worker and the consumer-side e2e test, so the
    registered set stays defined in exactly one place.
    """
    return [
        store.read_and_store_file_content,
        store.split_file_into_chunks,
        store.build_ocr_request_blob,
        store.submit_ocr_batch,
        store.ocr_batch_status,
        store.fetch_and_store_ocr_result,
        store.delete_file_content_blob,
        store.upsert_ocr_status,
        store.reassemble_ocr_chunks,
        store.export_ocr_document,
        store.check_ocr_duplicate,
        store.mark_ocr_for_removal,
        store.clear_ocr_removal_mark,
        store.list_ocr_jobs,
    ]


async def run_worker(address: str | None = None, *, identity: str | None = None) -> None:
    """Connect to Temporal and run the OCR worker until interrupted.

    The composition root: reads settings once (fail-fast on a missing
    ``FORGE_DB_URL`` or ``MISTRAL_API_KEY``), runs migrations, configures logging,
    builds the store engine + blob client + Mistral capability ONCE, injects all
    three into ``OcrStoreActivities``, then hands the bound methods and workflows
    to the shared ``sax_platform.temporal.worker.run_worker`` (Worker construction
    + graceful SIGINT/SIGTERM drain). ``address`` overrides the settings-resolved
    Temporal address (used by the CLI's ``--temporal-address`` option);
    ``graceful_shutdown_timeout`` is 5 minutes to match the shared default.
    """
    from datetime import timedelta

    settings = OcrSettings()

    _init_store(settings.db.url)

    setup_logging("ocr", console=True)
    silence_noisy_loggers()

    engine = get_store_engine(settings.db.url)
    # OCR requires S3: build the client unconditionally so an unset bucket fails
    # fast here (S3Blobs raises on an empty bucket) rather than at first use.
    blobs = S3Blobs(settings.blob.bucket or "", settings.blob.prefix)
    # OCR now polls its own Mistral batches (T4.2): the capability is required.
    # Fail fast at startup on a missing key rather than at the first submit/poll.
    mistral = _build_mistral_ocr(settings.llm.mistral_api_key)
    if mistral is None:
        msg = "OCR worker requires MISTRAL_API_KEY (it submits and polls its own Mistral batches)."
        raise ValueError(msg)
    logger.info("OCR worker: store engine + blobs + Mistral OCR ready")

    store = OcrStoreActivities(engine, blobs, mistral)

    client = await connect_temporal(
        address or settings.temporal.address,
        identity=identity,
        settings=settings.temporal,
    )
    await _run_worker(
        client,
        task_queue=OCR_TASK_QUEUE,
        workflows=workflows(),
        activities=activity_methods(store),
        graceful_shutdown_timeout=timedelta(minutes=5),
    )
