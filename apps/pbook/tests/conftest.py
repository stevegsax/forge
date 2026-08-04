"""Shared test fixtures for pbook.

The store targets PostgreSQL + pgvector, so the suite runs against a real
Postgres. Resolution order for the test database:

1. ``PBOOK_TEST_DATABASE_URL`` — an external Postgres (must have the
   ``vector`` extension available). CI sets it, because a GitHub runner
   has no shared stack and provisions a service container instead.
2. Otherwise ``sax_platform.testing.PBOOK_TRUST_TEST_DB_URL`` — the
   ``pbook_test`` database on the shared sax-datastores **dev** stack,
   reached as the credential-free ``pbook_test`` role through that
   stack's pg_hba ``trust`` row. Nothing is provisioned and nothing is
   torn down: the sax-datastores rationale §22 forbids an agent session
   self-provisioning a container, which is what this suite used to do.

The suite needs Postgres unconditionally, so a stack that is not running
fails it loudly with ``UnreachableTestDatabaseError`` naming both fixes.
That is deliberate — the alternative, skipping, reads as a pass.

Per-test isolation is by TRUNCATE (RESTART IDENTITY) of the pbk_ tables
between tests, so ids restart at 1 the way the SQLite-era tests expected.

The session Temporal environment fixture is imported from
``sax_platform.testing`` (D93) and re-exported here as ``env`` — the name
the workflow tests request. It carries ``pydantic_data_converter``, matching
production (the worker connects with it too).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa
from sax_platform.testing import (
    PBOOK_TEST_DB_URL_ENV,
    PBOOK_TRUST_TEST_DB_URL,
    require_reachable_test_database,
    resolve_test_database_url,
    temporal_env,
)

from pbook.settings import PbookDbSettings
from pbook.store import EMBEDDING_DIM, SCHEMA, build_engine, run_migrations

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy import Engine

# Re-export the shared session Temporal fixture under the name the suite uses.
# It is the SAME fixture object; request only ``env`` (not ``temporal_env``)
# within a session so the time-skipping server starts once.
env = temporal_env


def make_embedding(*coords: float) -> list[float]:
    """Build a full-width (1536-dim) test vector from leading coords.

    The ``Vector(1536)`` columns reject mismatched dimensions, so tests
    place their meaningful values in the first positions and zero-pad the
    rest. Cosine relationships between such vectors are preserved (the
    padding contributes nothing), so orthogonal/parallel test cases hold.
    """
    vec = [0.0] * EMBEDDING_DIM
    for i, value in enumerate(coords):
        vec[i] = float(value)
    return vec


def encode_test_embedding(*coords: float) -> str:
    """Base64-encode a full-width test vector for activity-boundary inputs."""
    from pbook.embeddings import encode_embedding

    return encode_embedding(make_embedding(*coords))


_PBK_TABLES = (
    "pbk_entry_tags",
    "pbk_entry_sources",
    "pbk_ingested_sessions",
    "pbk_entries",
)

# The single session-scoped store engine (one engine per process, the T3.6
# invariant). Populated by the ``_store_engine`` fixture and read by the
# module-level ``setup_db`` helper, which test bodies call directly (not via
# fixture injection). Disposed at session end by the fixture's finalizer.
_SESSION_ENGINE: Engine | None = None


@pytest.fixture(scope="session")
def _pg_url() -> str:
    """Provide a migrated Postgres URL for the whole test session.

    Nothing is provisioned and nothing is torn down — the database already
    exists (CI's service container, or the shared dev stack's ``pbook_test``).
    The reachability check runs first so a down stack fails here, by name,
    rather than as an opaque Alembic connection error.
    """
    url = resolve_test_database_url(
        os.environ,
        env_var=PBOOK_TEST_DB_URL_ENV,
        default=PBOOK_TRUST_TEST_DB_URL,
    )
    require_reachable_test_database(url, env_var=PBOOK_TEST_DB_URL_ENV)
    run_migrations(url)
    return url


@pytest.fixture(scope="session")
def _store_engine(_pg_url: str) -> Iterator[Engine]:
    """The one store engine for the session, disposed at session end.

    Also published to the module-level ``_SESSION_ENGINE`` so ``setup_db``
    (called directly from test bodies) hands back the same engine rather than
    building a fresh pool per test.
    """
    global _SESSION_ENGINE
    engine = build_engine(PbookDbSettings(url=_pg_url))
    assert engine is not None
    _SESSION_ENGINE = engine
    try:
        yield engine
    finally:
        engine.dispose()
        _SESSION_ENGINE = None


@pytest.fixture(autouse=True)
def _forge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declare ``FORGE_ENV=test`` (and a test Temporal address) for the guard.

    Every pbook CLI command and worker startup now resolves ``FORGE_ENV`` and
    refuses to run without it, and every connect additionally enforces
    derives its Temporal target from that env. The namespace needs no fixture — it
    is derived as ``forge-test`` — but ``test`` has no canonical server address
    (its server is an ephemeral per-job container), so one must be declared. This
    one central fixture satisfies both — ``FORGE_ENV=test`` plus a
    ``FORGE_TEMPORAL_ADDRESS`` — for the whole suite; the guard's and
    coherence's own tests override them with ``monkeypatch.delenv``/``setenv``.
    """
    monkeypatch.setenv("FORGE_ENV", "test")
    monkeypatch.setenv("FORGE_TEMPORAL_ADDRESS", "127.0.0.1:7233")


@pytest.fixture(autouse=True)
def _isolate_db(_pg_url: str, _store_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the store env at the test DB and truncate tables before each test."""
    monkeypatch.setenv("PBOOK_DATABASE_URL", _pg_url)
    qualified = ", ".join(f"{SCHEMA}.{t}" for t in _PBK_TABLES)
    with _store_engine.begin() as conn:
        conn.execute(sa.text(f"TRUNCATE {qualified} RESTART IDENTITY CASCADE"))


@pytest.fixture(autouse=True)
def _bypass_cli_workflows(monkeypatch: pytest.MonkeyPatch):
    """Make ``pbook.cli._execute_workflow`` dispatch to the activity in-process.

    Production CLI commands submit workflows to a Temporal server. Tests
    don't want to spin one up; the activity-level path runs the same DB
    code, so behavior is faithful. This fixture replaces the workflow
    submission with a direct call to the matching
    :class:`~pbook.roots.StoreActivities` bound method, constructing the
    class from the current ``PBOOK_DATABASE_URL`` at call time (so the
    store-disabled tests, which blank the env var, still see ``None``).

    Workflows whose ``.run`` isn't in the map (e.g. RetrievalWorkflow,
    which has multiple activities) fall through to the real
    ``_execute_workflow`` and tests that need them spin up a real
    ``WorkflowEnvironment``.
    """
    import asyncio

    from pbook import cli
    from pbook.roots import StoreActivities
    from pbook.store import build_engine
    from pbook.workflows import cli_ops as workflows

    # workflow ``.run`` → StoreActivities method name.
    mapping = {
        workflows.GetEntryWorkflow.run: "get_entry_activity",
        workflows.ListEntriesWorkflow.run: "list_entries_activity",
        workflows.ListSourcesWorkflow.run: "list_sources_activity",
        workflows.ListTagsWorkflow.run: "list_tags_activity",
        workflows.ReviewQueueWorkflow.run: "review_queue_activity",
        workflows.ListSessionsWorkflow.run: "list_sessions_activity",
        workflows.GetSessionTextWorkflow.run: "get_session_text_activity",
        workflows.CheckDuplicateWorkflow.run: "check_duplicate_activity",
        workflows.AddEntryWorkflow.run: "add_entry_activity",
        workflows.ApproveEntryWorkflow.run: "approve_entry_activity",
        workflows.RejectEntryWorkflow.run: "reject_entry_activity",
        workflows.UpdateEntryWorkflow.run: "update_entry_activity",
        workflows.RecordFeedbackWorkflow.run: "record_feedback_activity",
        workflows.PruneWorkflow.run: "prune_activity",
        workflows.FilterAlreadyIngestedWorkflow.run: "filter_already_ingested_activity",
        workflows.RecordStartedSessionsWorkflow.run: "record_started_sessions_activity",
    }

    real_execute = cli._execute_workflow

    def bypassed(workflow_fn, arg, *, id_prefix="pbook", temporal_address=""):
        method_name = mapping.get(workflow_fn)
        if method_name is None:
            return real_execute(
                workflow_fn,
                arg,
                id_prefix=id_prefix,
                temporal_address=temporal_address,
            )
        # GetSessionTextWorkflow maps to a no-dependency free function; the
        # others are engine-bound StoreActivities methods.
        if method_name == "get_session_text_activity":
            from pbook.activities.cli_ops import get_session_text_activity

            payload = arg.model_dump() if hasattr(arg, "model_dump") else arg
            return asyncio.run(get_session_text_activity(payload))

        engine = build_engine(PbookDbSettings())
        store = StoreActivities(engine)
        method = getattr(store, method_name)
        payload = arg.model_dump() if hasattr(arg, "model_dump") else arg
        try:
            return asyncio.run(method(payload))
        finally:
            if engine is not None:
                engine.dispose()

    monkeypatch.setattr(cli, "_execute_workflow", bypassed)


def setup_db(_tmp_path=None):
    """Return ``(engine, url)`` for the configured test database.

    Hands back the session-scoped engine (one per process); migrations have
    already run for the session and the per-test TRUNCATE fixture provides
    isolation. The ``_tmp_path`` argument is accepted and ignored for
    compatibility with the previous SQLite-era signature.
    """
    assert _SESSION_ENGINE is not None, "the _store_engine fixture must be active (via _isolate_db)"
    url = os.environ["PBOOK_DATABASE_URL"]
    return _SESSION_ENGINE, url
