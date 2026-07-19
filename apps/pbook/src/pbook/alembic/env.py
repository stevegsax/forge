"""Alembic migration environment for the pbook store (PostgreSQL only)."""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import context

from pbook.store import SCHEMA, VERSION_TABLE, Base, normalize_url

target_metadata = Base.metadata


def _resolve_url() -> str:
    """Connection URL for the migration run.

    Resolution order:

    1. ``cfg.attributes["pbook_url"]`` — how :func:`pbook.store.run_migrations`
       (the worker/CLI in-process path) passes the URL, avoiding both
       ConfigParser %-interpolation on a percent-encoded password and any
       ``os.environ`` write.
    2. ``sqlalchemy.url`` from alembic.ini.
    3. ``PBOOK_DATABASE_URL`` env var — kept ONLY for the operator running the
       standalone ``alembic`` CLI directly (not via ``pbook migrate``), which
       has no way to populate ``cfg.attributes``.
    """
    url = context.config.attributes.get("pbook_url")
    if not url:
        url = context.config.get_main_option("sqlalchemy.url")
    if not url:
        url = os.environ.get("PBOOK_DATABASE_URL")
    if not url:
        msg = (
            "No database URL: set cfg.attributes['pbook_url'], sqlalchemy.url, "
            "or PBOOK_DATABASE_URL."
        )
        raise RuntimeError(msg)
    return normalize_url(url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL, no DB connection)."""
    context.configure(
        url=_resolve_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=VERSION_TABLE,
        version_table_schema=SCHEMA,
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = sa.create_engine(_resolve_url())
    with connectable.connect() as connection:
        # The schema and pgvector must exist before Alembic touches the
        # (schema-qualified) version table or the baseline's vector columns.
        connection.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        # Prefer Supabase's conventional `extensions` schema for pgvector so
        # it doesn't clutter the API-exposed `public` schema; fall back to
        # the default for plain Postgres (e.g. the pgvector test container,
        # which has no `extensions` schema).
        has_extensions_schema = connection.execute(
            sa.text("SELECT 1 FROM pg_namespace WHERE nspname = 'extensions'"),
        ).first()
        if has_extensions_schema:
            connection.execute(
                sa.text("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions"),
            )
        else:
            connection.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Ensure the vector type/operators resolve during the baseline DDL
        # regardless of which schema holds the extension (a missing schema
        # in the search_path is silently ignored).
        connection.execute(sa.text("SET search_path TO public, extensions"))
        connection.commit()
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table=VERSION_TABLE,
            version_table_schema=SCHEMA,
            include_schemas=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
