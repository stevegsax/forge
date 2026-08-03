"""Store engine, idempotent-insert, and shared migration-runner helpers.

Absorbed from ``forge_contracts.db`` (T3.4, ST3) and generalized: Postgres
transaction-pooler detection (``is_pooler_url``, generalized from pbook's
``_is_pooler``), SQLite ``busy_timeout`` hygiene alongside the existing WAL
pragma, a shared migration runner with a per-chain Postgres advisory lock
(``advisory_lock_key``, generalizing forge's single hardcoded lock constant),
and a hardened Alembic ``context.configure`` helper for consumer ``env.py``
files.

Unlike ``sax_platform.contracts`` and ``sax_platform.llm``, this subpackage is
never imported inside a Temporal workflow sandbox (DB access is an activity
concern), so exports here are eager rather than lazy via PEP 562.
"""

from __future__ import annotations

from sax_platform.db.alembic import configure_migration_context
from sax_platform.db.engine import get_store_engine, is_pooler_url
from sax_platform.db.migrations import advisory_lock_key, run_migrations
from sax_platform.db.ops import (
    StoreConfigError,
    ensure_sqlite_parent,
    insert_or_ignore,
)
from sax_platform.db.verify import (
    SchemaVerdict,
    SchemaVersionError,
    classify_schema_version,
    describe_verdict,
    verify_schema_version,
)

__all__ = [
    "SchemaVerdict",
    "SchemaVersionError",
    "StoreConfigError",
    "advisory_lock_key",
    "classify_schema_version",
    "configure_migration_context",
    "describe_verdict",
    "ensure_sqlite_parent",
    "get_store_engine",
    "insert_or_ignore",
    "is_pooler_url",
    "run_migrations",
    "verify_schema_version",
]
