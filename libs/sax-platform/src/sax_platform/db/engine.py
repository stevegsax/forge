"""Store engine factory: SQLite WAL/busy-timeout hygiene, Postgres pooling.

``get_store_engine`` ports ``forge_contracts.db.get_store_engine`` verbatim for
the ``FORGE_DB_URL``-read and pooled-Postgres paths, then adds two things
neither forge-contracts nor any single consumer had on its own (T3.4, ST3):

* Supabase transaction-pooler detection (generalized from pbook's
  ``_is_pooler``), so ``psycopg``'s server-side prepared statements — which
  the pooler's PgBouncer breaks — are disabled automatically rather than
  per-consumer.
* A SQLite ``busy_timeout`` pragma alongside the existing WAL journal mode, so
  a writer waits out a brief lock instead of raising ``database is locked``
  immediately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.engine import make_url

from sax_platform.db.ops import ensure_sqlite_parent

if TYPE_CHECKING:
    from sqlalchemy import Engine
    from sqlalchemy.engine.interfaces import DBAPIConnection
    from sqlalchemy.pool import ConnectionPoolEntry

# SQLite busy_timeout is in milliseconds. A few seconds is enough for a brief
# writer-vs-writer collision to clear without the caller seeing "database is
# locked" on what would otherwise resolve itself.
_SQLITE_BUSY_TIMEOUT_MS = 5_000


def is_pooler_url(url: str, *, pooler_override: bool = False) -> bool:
    """True when ``url`` targets Supabase's transaction-mode connection pooler.

    Pure: the caller (``get_store_engine``, the imperative shell) passes the
    override in rather than this predicate reading any environment itself.
    Detected by host (``pooler.supabase.com`), the pooler's default port
    (``6543``), or the explicit override — the transaction-mode pooler
    (PgBouncer) breaks server-side prepared statements, so psycopg must be told
    to skip them.
    """
    if pooler_override:
        return True
    return "pooler.supabase.com" in url or ":6543" in url


def get_store_engine(url: str, *, pooler_override: bool = False) -> Engine:
    """Build the store engine for ``url``.

    ``url`` and ``pooler_override`` are required, explicit config: the caller
    (a composition root, via :class:`~sax_platform.config.DbSettings`) resolves
    both. This factory reads no environment itself.

    SQLite URLs get WAL journaling plus a busy-timeout pragma (and the parent
    directory is created); Postgres URLs get connection pre-ping, a small
    bounded pool to respect the managed-database connection caps, and —
    when the URL looks like Supabase's transaction pooler — prepared
    statements disabled via ``connect_args``. Connection errors are not caught
    here — they propagate (no runtime failover).
    """
    if make_url(url).get_backend_name() == "sqlite":
        ensure_sqlite_parent(url)
        engine = sa.create_engine(url)

        @sa.event.listens_for(engine, "connect")
        def _set_sqlite_pragma(
            dbapi_connection: DBAPIConnection, _connection_record: ConnectionPoolEntry
        ) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
            cursor.close()

        return engine

    connect_args: dict[str, object] = {}
    if is_pooler_url(url, pooler_override=pooler_override):
        connect_args["prepare_threshold"] = None

    return sa.create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        connect_args=connect_args,
    )
