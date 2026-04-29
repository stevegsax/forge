"""Shared test fixtures for pbook."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa

from pbook.store import get_engine, run_migrations

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point PBOOK_DB_PATH to a temporary directory for every test."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("PBOOK_DB_PATH", str(db_path))


@pytest.fixture(autouse=True)
def _dispose_store_engines(monkeypatch: pytest.MonkeyPatch):
    """Dispose SQLAlchemy engines created via pbook.store.get_engine after each test."""
    original_create_engine = sa.create_engine
    created_engines = []

    def tracking_create_engine(*args, **kwargs):
        engine = original_create_engine(*args, **kwargs)
        created_engines.append(engine)
        return engine

    monkeypatch.setattr(sa, "create_engine", tracking_create_engine)

    yield

    for engine in created_engines:
        engine.dispose()


@pytest.fixture(autouse=True)
def _bypass_cli_workflows(monkeypatch: pytest.MonkeyPatch):
    """Make ``pbook.cli._execute_workflow`` dispatch to the activity in-process.

    Production CLI commands submit workflows to a Temporal server. Tests
    don't want to spin one up; the activity-level path runs the same DB
    code, so behavior is faithful. This fixture replaces the workflow
    submission with a direct activity call keyed off the workflow's
    ``.run`` method.

    Workflows whose ``.run`` isn't in the map (e.g. RetrievalWorkflow,
    which has multiple activities) fall through to the real
    ``_execute_workflow`` and tests that need them spin up a real
    ``WorkflowEnvironment``.
    """
    import asyncio

    from pbook import cli
    from pbook.activities import cli_ops as activities
    from pbook.workflows import cli_ops as workflows

    mapping = {
        workflows.GetEntryWorkflow.run: activities.get_entry_activity,
        workflows.ListEntriesWorkflow.run: activities.list_entries_activity,
        workflows.ListSourcesWorkflow.run: activities.list_sources_activity,
        workflows.ListTagsWorkflow.run: activities.list_tags_activity,
        workflows.ReviewQueueWorkflow.run: activities.review_queue_activity,
        workflows.ListSessionsWorkflow.run: activities.list_sessions_activity,
        workflows.GetSessionTextWorkflow.run: activities.get_session_text_activity,
        workflows.CheckDuplicateWorkflow.run: activities.check_duplicate_activity,
        workflows.AddEntryWorkflow.run: activities.add_entry_activity,
        workflows.ApproveEntryWorkflow.run: activities.approve_entry_activity,
        workflows.RejectEntryWorkflow.run: activities.reject_entry_activity,
        workflows.UpdateEntryWorkflow.run: activities.update_entry_activity,
        workflows.RecordFeedbackWorkflow.run: activities.record_feedback_activity,
        workflows.PruneWorkflow.run: activities.prune_activity,
        workflows.FilterAlreadyIngestedWorkflow.run: activities.filter_already_ingested_activity,
        workflows.RecordStartedSessionsWorkflow.run: activities.record_started_sessions_activity,
    }

    real_execute = cli._execute_workflow

    def bypassed(workflow_fn, arg, *, id_prefix="pbook", temporal_address=""):
        activity_fn = mapping.get(workflow_fn)
        if activity_fn is None:
            return real_execute(
                workflow_fn, arg,
                id_prefix=id_prefix, temporal_address=temporal_address,
            )
        payload = arg.model_dump() if hasattr(arg, "model_dump") else arg
        return asyncio.run(activity_fn(payload))

    monkeypatch.setattr(cli, "_execute_workflow", bypassed)


def setup_db(tmp_path: Path):
    """Create a test database and return (engine, db_path)."""
    db_path = tmp_path / "test.db"
    run_migrations(db_path)
    engine = get_engine(db_path)
    return engine, db_path
