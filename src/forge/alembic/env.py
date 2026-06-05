"""Alembic migration environment for the Forge platform store.

Forge and its consumer apps (e.g. OCR) share one database but keep isolated
Alembic chains: a distinct ``version_table`` per app, plus an ``include_object``
filter so neither app's autogenerate touches the other's tables. Forge manages
only the tables in its own ``Base.metadata``.
"""

from __future__ import annotations

from alembic import context

from forge.store import Base

target_metadata = Base.metadata

VERSION_TABLE = "alembic_version_forge"


def _include_object(
    obj: object, name: str | None, type_: str, reflected: bool, compare_to: object
) -> bool:
    """Manage only tables Forge owns; ignore consumer-app tables in the shared DB."""
    return not (type_ == "table" and name not in target_metadata.tables)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = context.config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=VERSION_TABLE,
        include_object=_include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    from sqlalchemy import create_engine

    url = context.config.get_main_option("sqlalchemy.url")
    connectable = create_engine(url)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table=VERSION_TABLE,
            include_object=_include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
