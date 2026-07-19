"""Temporal worker for the playbook service.

Runs on a separate task queue from Forge. This module is pbook's worker-side
composition root (T3.6): it constructs frozen settings once, builds the engine
/ LLM provider / embedder once, wires them into the class-based activities
(:mod:`pbook.roots`), and runs the shared platform worker scaffold.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from openai import AsyncOpenAI
from sax_platform.config import TemporalSettings
from sax_platform.embeddings import OpenAIEmbeddings
from sax_platform.llm import AnthropicLLM, make_client
from sax_platform.logging import setup_logging
from sax_platform.temporal import connect_temporal
from sax_platform.temporal.worker import run_worker as run_platform_worker

from pbook.activities.cli_ops import get_session_text_activity
from pbook.activities.review import validate_entry
from pbook.roots import EmbeddingActivities, LlmActivities, StoreActivities
from pbook.settings import PbookDbSettings, PbookSettings
from pbook.store import build_engine, run_migrations

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

_WORKFLOWS: list[type] = [
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
]


def _migrate_if_configured(db: PbookDbSettings) -> None:
    """Run Alembic migrations to head once at startup, if a store is configured.

    This is the worker's single migration point — activities no longer
    migrate per call. If ``PBOOK_DATABASE_URL`` is unset the store is
    disabled and migration is skipped.
    """
    if not db.url:
        logger.warning("PBOOK_DATABASE_URL not set — skipping migrations (store disabled)")
        return
    run_migrations(db.url)
    logger.info("Database migrations applied (head)")


async def run_worker(address: str = "localhost:7233") -> None:
    """Connect to Temporal and run the pbook worker (composition root)."""
    settings = PbookSettings()
    setup_logging("pbook", log_path=settings.log_path, console=True)

    engine = build_engine(settings.db)
    _migrate_if_configured(settings.db)

    # Build the injected dependencies ONCE. ``make_client`` reads
    # ANTHROPIC_API_KEY lazily, so no key is required at construction; the
    # per-call model comes from the tier-resolved ``LLMChatInput.model``. The
    # embedder is None when no OPENAI_API_KEY is configured — llm_embed then
    # fails fast and non-retryably rather than hanging.
    provider = AnthropicLLM(make_client())
    embedder = (
        OpenAIEmbeddings(AsyncOpenAI(api_key=settings.openai_api_key))
        if settings.openai_api_key
        else None
    )

    store = StoreActivities(engine)
    llm = LlmActivities(provider)
    embeddings = EmbeddingActivities(embedder)

    activities = [
        *store.all_activities(),
        llm.llm_chat,
        embeddings.llm_embed,
        # No-dependency activities stay bare free functions.
        validate_entry,
        get_session_text_activity,
    ]

    logger.info("Connecting to Temporal at %s", address)
    client = await connect_temporal(address, settings=TemporalSettings())

    logger.info("pbook worker starting on queue %s", PBOOK_TASK_QUEUE)
    await run_platform_worker(
        client,
        task_queue=PBOOK_TASK_QUEUE,
        workflows=_WORKFLOWS,
        activities=activities,
        graceful_shutdown_timeout=timedelta(minutes=5),
    )
