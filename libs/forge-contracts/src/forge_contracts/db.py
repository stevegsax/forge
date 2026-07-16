"""Generic store engine + idempotent-insert helpers (no table/ORM knowledge).

Shared by the platform and consumer apps: both connect to the same database via
``FORGE_DB_URL`` with identical pooling / SQLite-WAL config, and use the same
idempotent ``insert_or_ignore``. Each repo keeps its OWN ``run_migrations``
(pointing at its own Alembic chain) — only the connection + insert primitives are
shared here.

NOTE: the ``FORGE_DB_URL`` env name is retained from the pre-split layout.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.engine import make_url

if TYPE_CHECKING:
    from sqlalchemy import Engine, Table


class StoreConfigError(RuntimeError):
    """The store is misconfigured (e.g. ``FORGE_DB_URL`` unset)."""


def get_store_url() -> str:
    """Return the configured store URL from ``FORGE_DB_URL``.

    The store is mandatory infrastructure with no implicit default and no
    runtime failover. A ``sqlite:///<path>`` URL is the dev/test configuration;
    a ``postgresql+psycopg2://...`` URL is production. Unset or empty raises.
    """
    url = os.environ.get("FORGE_DB_URL")
    if not url:
        raise StoreConfigError(
            "FORGE_DB_URL is not set. Set it to a 'sqlite:///<path>' URL for "
            "development and tests, or a 'postgresql+psycopg2://...' URL for "
            "production."
        )
    return url


def ensure_sqlite_parent(url: str) -> None:
    """Create the parent directory for a file-based SQLite URL."""
    database = make_url(url).database
    if database and database != ":memory:":
        Path(database).parent.mkdir(parents=True, exist_ok=True)


def get_store_engine() -> Engine:
    """Build the store engine from ``FORGE_DB_URL``.

    SQLite URLs get WAL journaling (and the parent directory is created);
    Postgres URLs get connection pre-ping and a small bounded pool to respect
    the managed-database connection caps. Connection errors are not caught here
    — they propagate (no runtime failover).
    """
    url = get_store_url()
    if make_url(url).get_backend_name() == "sqlite":
        ensure_sqlite_parent(url)
        engine = sa.create_engine(url)

        @sa.event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection: object, _connection_record: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[union-attr]
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

        return engine

    return sa.create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def insert_or_ignore(
    engine: Engine,
    table: Table,
    values: dict,
    *,
    index_elements: list[str],
) -> bool:
    """Idempotent insert via the dialect's ``ON CONFLICT DO NOTHING``.

    Returns ``True`` if a new row was written, ``False`` if an existing row on
    ``index_elements`` absorbed the insert (a no-op). This makes every write safe
    to re-apply on a Temporal retry: a duplicate never raises, and the caller can
    tell whether it was the first writer.
    """
    dialect = engine.dialect.name
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as _dialect_insert
    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as _dialect_insert
    else:  # pragma: no cover - only sqlite/postgres are supported stores
        msg = f"insert_or_ignore is unsupported on dialect {dialect!r}"
        raise StoreConfigError(msg)

    stmt = (
        _dialect_insert(table)
        .values(**values)
        .on_conflict_do_nothing(index_elements=index_elements)
    )
    with engine.begin() as conn:
        result = conn.execute(stmt)
    return bool(result.rowcount)
