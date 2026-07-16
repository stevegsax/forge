"""Alembic migration environment for the OCR app store.

OCR shares the platform's database but owns an isolated Alembic chain: a distinct
``version_table`` plus an ``include_object`` filter so OCR's autogenerate never
touches the platform's (or anyone else's) tables. OCR manages only the tables in
its own ``Base.metadata``.
"""

from __future__ import annotations

from alembic import context

from ocr.store import Base

target_metadata = Base.metadata

VERSION_TABLE = "alembic_version_ocr"


def _include_object(
    obj: object, name: str | None, type_: str, reflected: bool, compare_to: object
) -> bool:
    """Manage only tables OCR owns; ignore platform tables in the shared DB."""
    return not (type_ == "table" and name not in target_metadata.tables)


def run_migrations_offline() -> None:
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
