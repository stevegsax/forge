"""Shared Alembic migration runner with a per-chain Postgres advisory lock.

Ports ``forge.store.run_migrations`` (T3.4, ST3), generalized so every
consumer with its own Alembic chain — forge, ocr, and future members alike —
can call one shared runner instead of hand-rolling it. forge's original had a
single hardcoded lock key (the ASCII bytes ``b"forge-mg"``); here the key is
derived from the caller's ``version_table``, so each chain gets its own lock
and chains never contend on each other's key. This also gives ocr an advisory
lock it did not previously have (``ocr.store.run_migrations`` upgrades with no
locking at all today).

The lock is acquired by *polling* ``pg_try_advisory_lock`` from the client,
never by the blocking ``pg_advisory_lock`` — see
:func:`_acquire_advisory_lock` for the mechanism. That is a correctness
requirement, not a style choice: a migrator waiting inside Postgres
deadlocks against a concurrent ``CREATE INDEX CONCURRENTLY``.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

import sqlalchemy as sa
from sqlalchemy.engine import make_url

from sax_platform.db.ops import ensure_sqlite_parent

if TYPE_CHECKING:
    from collections.abc import Callable

#: Client-side wait between ``pg_try_advisory_lock`` attempts. Only spins
#: while another migrator is actually running the chain, so a sub-second
#: interval costs nothing and keeps worker startup responsive.
_LOCK_POLL_INTERVAL_SECONDS: Final = 0.5


def advisory_lock_key(version_table: str) -> int:
    """Deterministic ``pg_advisory_lock`` key for one Alembic chain.

    Pure function of ``version_table``: distinct chains (distinct version
    tables) get distinct keys, and the same chain always derives the same
    key, so concurrent migrators against the same chain serialize on it while
    unrelated chains never contend. ``pg_advisory_lock`` takes a signed
    bigint, so the key is taken from a SHA-256 digest of the table name,
    interpreted as a signed 64-bit integer.
    """
    digest = hashlib.sha256(version_table.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def run_migrations(url: str, *, version_table: str, script_location: str) -> None:
    """Run one Alembic chain to head against ``url`` (SQLite or Postgres).

    ``script_location`` is the directory holding that chain's ``env.py``,
    ``versions/``, and ``alembic.ini``; ``version_table`` identifies the chain
    for the advisory-lock key (see ``advisory_lock_key``) — the chain's own
    ``env.py`` is still responsible for passing ``version_table`` to
    ``context.configure`` (see :mod:`sax_platform.db.alembic`).

    On Postgres, a session-level advisory lock serializes concurrent callers:
    the second migrator waits (polling ``pg_try_advisory_lock`` from the
    client — see :func:`_acquire_advisory_lock`) until the first finishes,
    then finds the schema at head and no-ops. SQLite callers are
    single-process dev/test paths; no lock is taken.
    """
    from alembic import command
    from alembic.config import Config

    script_dir = Path(script_location)
    ini_path = script_dir / "alembic.ini"

    cfg = Config(str(ini_path))
    cfg.set_main_option("script_location", str(script_dir))

    backend = make_url(url).get_backend_name()
    if backend == "sqlite":
        ensure_sqlite_parent(url)
    # Alembic's Config is backed by configparser with interpolation enabled, so a
    # bare '%' (e.g. URL-encoded password chars %23, %21, %40) is parsed as an
    # interpolation token and raises. Escape as '%%'; env.py's
    # get_main_option() reverses this on read, so the engine still receives
    # the original URL.
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))

    if backend != "postgresql":
        command.upgrade(cfg, "head")
        return

    _upgrade_with_advisory_lock(cfg, url, version_table, command.upgrade)


def _acquire_advisory_lock(conn: sa.Connection, key: int) -> None:
    """Wait until this session holds advisory lock ``key``, polling from the client.

    Every attempt is a single fast statement — ``pg_try_advisory_lock``
    returns immediately with true/false — and the wait between attempts is a
    CLIENT-side :func:`time.sleep`.

    **Do not "simplify" this back to a server-side wait** (the blocking
    ``pg_advisory_lock``, or a ``pg_sleep`` retry): a statement that waits
    inside Postgres holds its snapshot open for the whole wait, and
    ``CREATE INDEX CONCURRENTLY`` — which forge's chain uses, and which the
    schema-change process makes the default for indexes on existing tables —
    waits for exactly those snapshots to drain before it can finish. Two
    migrators then wait on each other: the loser blocks on the advisory lock
    while holding a snapshot, and the winner's concurrent index build waits
    on that snapshot. Neither wait is an ordinary lock cycle, so Postgres's
    deadlock detector never breaks it and both hang forever. Keeping the
    waiting on the client side is what makes a concurrent index build safe to
    run under this lock.

    There is deliberately no overall timeout: the previous behaviour was to
    block indefinitely, and this preserves it. The loop only spins while
    another migrator is genuinely running the chain.
    """
    while True:
        acquired = bool(
            conn.execute(sa.text("SELECT pg_try_advisory_lock(:key)"), {"key": key}).scalar()
        )
        if acquired:
            return
        time.sleep(_LOCK_POLL_INTERVAL_SECONDS)


def _upgrade_with_advisory_lock(
    cfg: object, url: str, version_table: str, upgrade: Callable[..., None]
) -> None:
    """Hold a Postgres session-level advisory lock across ``upgrade(cfg, "head")``.

    The lock lives on its own connection and is held across Alembic's whole
    run (Alembic builds its own engine from ``cfg``). Closing the connection
    releases a session-level advisory lock, so the explicit unlock is belt
    and braces for the happy path.

    The lock connection runs in ``AUTOCOMMIT`` for the same reason the wait is
    client-side (see :func:`_acquire_advisory_lock`): it must never sit inside
    an open transaction while Alembic's own connection runs DDL, so that a
    concurrent index build has no snapshot of ours to wait on. Advisory locks
    taken by ``pg_try_advisory_lock`` are session-level and outlive any
    transaction, so autocommit does not shorten the lock's life.
    """
    key = advisory_lock_key(version_table)
    lock_engine = sa.create_engine(url, poolclass=sa.pool.NullPool, isolation_level="AUTOCOMMIT")
    try:
        with lock_engine.connect() as conn:
            _acquire_advisory_lock(conn, key)
            try:
                upgrade(cfg, "head")
            finally:
                conn.execute(sa.text("SELECT pg_advisory_unlock(:key)"), {"key": key})
    finally:
        lock_engine.dispose()
