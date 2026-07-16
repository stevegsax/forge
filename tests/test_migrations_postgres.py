"""Validate Alembic migrations against a real Postgres via testcontainers.

SQLite tests can't catch Postgres-specific DDL/SQL issues (e.g. the
``batch_alter_table`` migrations 008/014/015, or the ``ON CONFLICT DO NOTHING``
path used by ``insert_or_ignore``). These tests spin up a throwaway Postgres in
Docker and exercise both.

Opt-in: marked ``postgres`` (excluded from the default run via ``addopts``) and
requires Docker. Run with::

    uv run pytest -m postgres
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.postgres

_EXPECTED_TABLES = {
    "interactions",
    "runs",
    "batch_jobs",
    "playbooks",
}


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    """A migrated-able Postgres URL backed by a throwaway container.

    Skips (rather than fails) when Docker or the image is unavailable, so the
    opt-in suite degrades gracefully on machines without Docker.
    """
    pytest.importorskip("testcontainers.postgres")
    from testcontainers.postgres import PostgresContainer

    try:
        container = PostgresContainer("postgres:16-alpine", driver="psycopg2")
        container.start()
    except Exception as exc:
        pytest.skip(f"Postgres testcontainer unavailable: {exc}")

    try:
        yield container.get_connection_url()
    finally:
        container.stop()


def test_migrations_apply_cleanly_on_postgres(postgres_url: str) -> None:
    import sqlalchemy as sa

    from forge.store import run_migrations

    run_migrations(postgres_url)

    engine = sa.create_engine(postgres_url)
    try:
        insp = sa.inspect(engine)
        assert set(insp.get_table_names()) >= _EXPECTED_TABLES

        # The platform chain owns its own version table and no OCR tables.
        assert "alembic_version_forge" in insp.get_table_names()
        assert "ocr_results" not in insp.get_table_names()

        # batch_jobs is generic — no OCR domain columns.
        batch_cols = {c["name"] for c in insp.get_columns("batch_jobs")}
        assert "file_path" not in batch_cols
        assert "document_id" not in batch_cols

        # idempotency_key present on interactions + playbooks.
        for table in ("interactions", "playbooks"):
            cols = {c["name"] for c in insp.get_columns(table)}
            assert "idempotency_key" in cols, (table, cols)

        # runs is rekeyed on (workflow_id, run_id) (T1.6a).
        runs_cols = {c["name"] for c in insp.get_columns("runs")}
        assert "run_id" in runs_cols, runs_cols
        runs_uniques = {tuple(uc["column_names"]) for uc in insp.get_unique_constraints("runs")}
        assert ("workflow_id", "run_id") in runs_uniques
        assert ("workflow_id",) not in runs_uniques
    finally:
        engine.dispose()


def test_migrations_rerun_is_noop_on_postgres(postgres_url: str) -> None:
    # Upgrading to head again must be a clean no-op (stable migration ordering).
    from forge.store import run_migrations

    run_migrations(postgres_url)
    run_migrations(postgres_url)


def test_concurrent_migrations_serialize_on_postgres(postgres_url: str) -> None:
    """Simultaneous migrators must serialize, not race Alembic's DDL.

    Reproduces the launchd first-boot failure: both workers migrate at
    startup, both saw the schema behind head, and the loser died on
    ``DuplicateColumn``. ``run_migrations`` now takes a session-level
    ``pg_advisory_lock``, so the second caller blocks, then no-ops.

    Runs against a fresh database inside the module's container — the shared
    ``postgres_url`` database is already at head by the time this test runs,
    which would mask the race.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import sqlalchemy as sa

    from forge.store import run_migrations

    admin = sa.create_engine(postgres_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(sa.text("CREATE DATABASE concurrent_migrations"))
    finally:
        admin.dispose()
    fresh_url = (
        sa.engine.make_url(postgres_url)
        .set(database="concurrent_migrations")
        .render_as_string(hide_password=False)
    )

    barrier = threading.Barrier(2)

    def migrate() -> None:
        barrier.wait()
        run_migrations(fresh_url)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(migrate) for _ in range(2)]
        for future in futures:
            future.result()  # raises if a migrator raced and died

    engine = sa.create_engine(fresh_url)
    try:
        assert set(sa.inspect(engine).get_table_names()) >= _EXPECTED_TABLES
    finally:
        engine.dispose()


def test_insert_or_ignore_idempotent_on_postgres(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the postgresql ON CONFLICT DO NOTHING path + pooled engine."""
    from forge.models import TaskResult, TransitionSignal
    from forge.store import get_store_engine, run_migrations, save_run

    run_migrations(postgres_url)
    monkeypatch.setenv("FORGE_DB_URL", postgres_url)

    engine = get_store_engine()  # postgres branch: pool_pre_ping + bounded pool
    try:
        assert engine.dialect.name == "postgresql"
        task_result = TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        assert save_run(engine, task_result, "wf-pg-idem", "run-1") is True
        assert save_run(engine, task_result, "wf-pg-idem", "run-1") is False
    finally:
        engine.dispose()


def test_runs_rekey_records_reruns_on_postgres(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The (workflow_id, run_id) rekey lets a reused workflow_id record reruns.

    Exercises the real Postgres ON CONFLICT arbiter on the composite unique
    ``uq_runs_workflow_id_run_id`` (T1.6a).
    """
    import sqlalchemy as sa

    from forge.models import TaskResult, TransitionSignal
    from forge.store import get_store_engine, list_recent_runs, run_migrations, save_run

    run_migrations(postgres_url)
    monkeypatch.setenv("FORGE_DB_URL", postgres_url)

    engine = get_store_engine()
    try:
        # The rekey lives on the runs table as a composite unique constraint.
        uniques = {
            tuple(uc["column_names"]) for uc in sa.inspect(engine).get_unique_constraints("runs")
        }
        assert ("workflow_id", "run_id") in uniques

        task_result = TaskResult(task_id="rk", status=TransitionSignal.SUCCESS)
        assert save_run(engine, task_result, "forge-task-rk", "run-A") is True
        assert save_run(engine, task_result, "forge-task-rk", "run-A") is False
        assert save_run(engine, task_result, "forge-task-rk", "run-B") is True

        run_ids = {
            r["run_id"] for r in list_recent_runs(engine) if r["workflow_id"] == "forge-task-rk"
        }
        assert run_ids == {"run-A", "run-B"}
    finally:
        engine.dispose()


def test_get_playbooks_by_tags_on_postgres(postgres_url: str) -> None:
    """Tag-filtered playbook queries must run on Postgres (context assembly path).

    ``get_playbooks_by_tags`` / ``get_playbook_ids`` unnest ``tags_json`` to match
    tags. The SQLite ``json_each()`` table function does not exist on Postgres, so
    this exercises the real prod path (``activities/context.py`` injects playbooks
    into every task context).
    """
    import sqlalchemy as sa

    from forge.store import (
        Playbook,
        get_playbook_ids,
        get_playbooks_by_tags,
        run_migrations,
    )

    run_migrations(postgres_url)
    engine = sa.create_engine(postgres_url)
    try:
        row = {
            "idempotency_key": "pg-tags-1",
            "title": "Use uv for deps",
            "content": "Always use uv sync.",
            "tags_json": '["python", "tooling"]',
            "source_task_id": "task-1",
            "source_workflow_id": "wf-1",
            "extraction_workflow_id": "ewf-1",
        }
        with engine.begin() as conn:
            conn.execute(sa.insert(Playbook.__table__).values(**row))

        matched = get_playbooks_by_tags(engine, ["python"], limit=5)
        assert [m["title"] for m in matched] == ["Use uv for deps"]

        ids = get_playbook_ids(engine, tags=["tooling"], limit=5)
        assert len(ids) == 1
    finally:
        engine.dispose()
