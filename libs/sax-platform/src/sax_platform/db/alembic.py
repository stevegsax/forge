"""Shared ``context.configure(...)`` helper for per-chain Alembic ``env.py`` files.

forge's and ocr's ``env.py`` each hand-roll their own offline/online
``context.configure`` calls today, with no autogenerate hardening
(``compare_type`` / ``compare_server_default`` are both Alembic defaults of
``False``, so type and server-default drift silently fails to autogenerate)
and no SQLite-safe ``ALTER TABLE`` strategy. ``configure_migration_context``
centralizes both fixes so every chain gets them uniformly (T3.4, ST3). A
later sub-task rewires forge's and ocr's ``env.py`` to call this instead of
``context.configure`` directly — this module stays generic, with no knowledge
of either app's tables or version-table name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def configure_migration_context(
    context: Any,
    *,
    target_metadata: Any,
    version_table: str,
    version_table_schema: str | None = None,
    include_object: Callable[..., bool] | None = None,
    include_schemas: bool = False,
    url: str | None = None,
    connection: Any | None = None,
) -> None:
    """Configure an Alembic migration ``context`` with the platform's hardened defaults.

    Exactly one of ``url`` (offline mode — emits literal-bound SQL, no live
    connection) or ``connection`` (online mode — a live DBAPI connection) must
    be supplied; that choice selects which of Alembic's two mutually
    exclusive ``context.configure`` call shapes is used.

    Every call gets:

    * ``render_as_batch=True`` — SQLite lacks most ``ALTER TABLE`` variants,
      so Alembic rebuilds the table under a batch operation; a no-op on
      Postgres, which supports ``ALTER TABLE`` natively.
    * ``compare_type=True`` and ``compare_server_default=True`` — Alembic's
      autogenerate defaults leave both off, so column-type and
      server-default drift between models and the live schema silently
      fails to generate a migration.

    ``version_table_schema``, ``include_object``, and ``include_schemas`` are
    passed through only when the caller supplies them, so chains that don't
    need multi-schema support see no behavior change.
    """
    if (url is None) == (connection is None):
        msg = "configure_migration_context requires exactly one of url or connection"
        raise ValueError(msg)

    kwargs: dict[str, Any] = {
        "target_metadata": target_metadata,
        "version_table": version_table,
        "render_as_batch": True,
        "compare_type": True,
        "compare_server_default": True,
        "include_schemas": include_schemas,
    }
    if version_table_schema is not None:
        kwargs["version_table_schema"] = version_table_schema
    if include_object is not None:
        kwargs["include_object"] = include_object

    if url is not None:
        context.configure(
            url=url,
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
            **kwargs,
        )
    else:
        context.configure(connection=connection, **kwargs)
