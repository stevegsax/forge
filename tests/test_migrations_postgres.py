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
    "ocr_results",
    "file_content_blobs",
    "ocr_images",
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

        # Phase B: OCR blobs reference S3 — s3_key present, inline data dropped.
        for table in ("file_content_blobs", "ocr_images"):
            cols = {c["name"] for c in insp.get_columns(table)}
            assert "s3_key" in cols, (table, cols)
            assert "data" not in cols, (table, cols)

        # Phase C: idempotency_key added to interactions + playbooks.
        for table in ("interactions", "playbooks"):
            cols = {c["name"] for c in insp.get_columns(table)}
            assert "idempotency_key" in cols, (table, cols)
    finally:
        engine.dispose()


def test_migrations_rerun_is_noop_on_postgres(postgres_url: str) -> None:
    # Upgrading to head again must be a clean no-op (stable migration ordering).
    from forge.store import run_migrations

    run_migrations(postgres_url)
    run_migrations(postgres_url)


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
        assert save_run(engine, task_result, "wf-pg-idem") is True
        assert save_run(engine, task_result, "wf-pg-idem") is False
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
