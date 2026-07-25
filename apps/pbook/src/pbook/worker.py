"""Temporal worker for the playbook service.

Runs on a separate task queue from Forge. This module is pbook's worker-side
composition root (T3.6): it constructs frozen settings once, builds the engine
/ LLM provider / embedder once, wires them into the class-based activities
(:mod:`pbook.roots`), and runs the shared platform worker scaffold.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta

from openai import AsyncOpenAI
from sax_platform.config import (
    TemporalSettings,
    require_namespace_coherence,
    resolve_forge_env,
)
from sax_platform.embeddings import OpenAIEmbeddings
from sax_platform.llm import AnthropicLLM, make_client
from sax_platform.logging import setup_logging
from sax_platform.temporal import connect_temporal, stamped_worker_identity
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


async def run_worker(address: str = "localhost:7233", *, identity: str | None = None) -> None:
    """Connect to Temporal and run the pbook worker (composition root).

    The environment guard runs FIRST, before settings/store setup: an unset or
    invalid ``FORGE_ENV`` raises :class:`ForgeEnvError` (its message is complete
    and actionable) so the worker can never reach a database without an
    explicitly declared environment.

    ``identity`` is the base worker identity (pbook's CLI supplies none, so the
    default is the SDK's ``{pid}@{hostname}``); it is stamped with the launch-time
    git version before connecting — see :mod:`sax_platform.temporal.identity`.
    """
    env = resolve_forge_env(os.environ)
    settings = PbookSettings()
    temporal_settings = TemporalSettings()

    # Enforce env/namespace coherence BEFORE store setup: a dev/test worker must
    # never poll production's namespace (or vice versa). An incoherent pairing
    # raises ForgeEnvError here so the worker fails fast, before it touches a DB.
    require_namespace_coherence(env, temporal_settings.namespace)

    setup_logging("pbook", log_path=settings.log_path, console=True)
    logger.info("pbook worker starting: env=%s", env)

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
    # Stamp the launch-time git version onto the identity (see
    # sax_platform.temporal.identity): the worker binds its code at import while the
    # tree it was exec'd from moves on, so `temporal task-queue describe
    # --task-queue pbook-task-queue` is where "which code is this worker running?"
    # gets answered. An undiscoverable version leaves the identity unchanged.
    client = await connect_temporal(
        address,
        identity=stamped_worker_identity(identity),
        namespace=temporal_settings.namespace,
        settings=temporal_settings,
    )

    logger.info("pbook worker starting on queue %s", PBOOK_TASK_QUEUE)
    await run_platform_worker(
        client,
        task_queue=PBOOK_TASK_QUEUE,
        workflows=_WORKFLOWS,
        activities=activities,
        graceful_shutdown_timeout=timedelta(minutes=5),
    )
