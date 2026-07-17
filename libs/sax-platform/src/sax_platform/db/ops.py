"""Store-URL resolution and idempotent-insert helpers (no table/ORM knowledge).

Ported verbatim from ``forge_contracts.db`` (T3.4, ST3): both the platform and
its consumer apps connect to the same database via ``FORGE_DB_URL`` and use the
same idempotent ``insert_or_ignore``. Engine construction (with pooler
detection and SQLite hygiene) lives in :mod:`sax_platform.db.engine`; each
consumer keeps its OWN ``run_migrations`` call site (pointing at its own
Alembic chain) via :mod:`sax_platform.db.migrations`.

NOTE: the ``FORGE_DB_URL`` env name is retained from the pre-split layout.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.engine import make_url

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy import Engine, Table
    from sqlalchemy.dialects.postgresql import Insert as PostgresInsert
    from sqlalchemy.dialects.sqlite import Insert as SQLiteInsert


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


def insert_or_ignore(
    engine: Engine,
    table: Table,
    values: Mapping[str, object],
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
    # sqlite's and postgres's `insert()` return dialect-specific `Insert` types
    # (different `on_conflict_do_nothing` signatures under the stubs), so each
    # branch builds and types its own statement rather than funneling both
    # through one shared, incompatibly-typed callable.
    stmt: SQLiteInsert | PostgresInsert
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        stmt = (
            sqlite_insert(table)
            .values(**values)
            .on_conflict_do_nothing(index_elements=index_elements)
        )
    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as postgres_insert

        stmt = (
            postgres_insert(table)
            .values(**values)
            .on_conflict_do_nothing(index_elements=index_elements)
        )
    else:  # pragma: no cover - only sqlite/postgres are supported stores
        msg = f"insert_or_ignore is unsupported on dialect {dialect!r}"
        raise StoreConfigError(msg)

    with engine.begin() as conn:
        result = conn.execute(stmt)
    return bool(result.rowcount)
