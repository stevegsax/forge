"""Tests for the shared ``configure_migration_context`` helper.

``context`` is a mock standing in for Alembic's ``alembic.context`` proxy
object — no real Alembic environment is spun up, *except* in
``TestRenderAsBatchEndToEnd`` below, which drives a real (temp-directory,
temp-SQLite-file) Alembic project end to end to prove ``render_as_batch``
actually does something (T3.4, ST10a / AC #5) — self-contained, no forge/ocr
imports.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from sax_platform.db.alembic import configure_migration_context

if TYPE_CHECKING:
    from alembic.script.base import Script


class TestOfflineMode:
    def test_hardened_defaults_are_passed(self) -> None:
        context = mock.MagicMock()
        target_metadata = mock.sentinel.metadata

        configure_migration_context(
            context,
            target_metadata=target_metadata,
            version_table="alembic_version_forge",
            url="sqlite:///forge.db",
        )

        context.configure.assert_called_once()
        _, kwargs = context.configure.call_args
        assert kwargs["render_as_batch"] is True
        assert kwargs["compare_type"] is True
        assert kwargs["compare_server_default"] is True

    def test_offline_shape_uses_url_and_literal_binds(self) -> None:
        context = mock.MagicMock()

        configure_migration_context(
            context,
            target_metadata=mock.sentinel.metadata,
            version_table="alembic_version_forge",
            url="sqlite:///forge.db",
        )

        _, kwargs = context.configure.call_args
        assert kwargs["url"] == "sqlite:///forge.db"
        assert kwargs["literal_binds"] is True
        assert kwargs["dialect_opts"] == {"paramstyle": "named"}
        assert "connection" not in kwargs

    def test_version_table_and_target_metadata_pass_through(self) -> None:
        context = mock.MagicMock()
        target_metadata = mock.sentinel.metadata

        configure_migration_context(
            context,
            target_metadata=target_metadata,
            version_table="alembic_version_ocr",
            url="sqlite:///ocr.db",
        )

        _, kwargs = context.configure.call_args
        assert kwargs["target_metadata"] is target_metadata
        assert kwargs["version_table"] == "alembic_version_ocr"


class TestOnlineMode:
    def test_hardened_defaults_are_passed(self) -> None:
        context = mock.MagicMock()
        connection = mock.sentinel.connection

        configure_migration_context(
            context,
            target_metadata=mock.sentinel.metadata,
            version_table="alembic_version_forge",
            connection=connection,
        )

        context.configure.assert_called_once()
        _, kwargs = context.configure.call_args
        assert kwargs["render_as_batch"] is True
        assert kwargs["compare_type"] is True
        assert kwargs["compare_server_default"] is True

    def test_online_shape_uses_connection_with_no_literal_binds(self) -> None:
        context = mock.MagicMock()
        connection = mock.sentinel.connection

        configure_migration_context(
            context,
            target_metadata=mock.sentinel.metadata,
            version_table="alembic_version_forge",
            connection=connection,
        )

        _, kwargs = context.configure.call_args
        assert kwargs["connection"] is connection
        assert "url" not in kwargs
        assert "literal_binds" not in kwargs


class TestOptionalPassthrough:
    def test_include_object_and_schemas_pass_through_when_given(self) -> None:
        context = mock.MagicMock()

        def include_object(*_args: object) -> bool:
            return True

        configure_migration_context(
            context,
            target_metadata=mock.sentinel.metadata,
            version_table="alembic_version_forge",
            version_table_schema="forge",
            include_object=include_object,
            include_schemas=True,
            url="sqlite:///forge.db",
        )

        _, kwargs = context.configure.call_args
        assert kwargs["include_object"] is include_object
        assert kwargs["include_schemas"] is True
        assert kwargs["version_table_schema"] == "forge"

    def test_omitted_optional_args_use_conservative_defaults(self) -> None:
        context = mock.MagicMock()

        configure_migration_context(
            context,
            target_metadata=mock.sentinel.metadata,
            version_table="alembic_version_forge",
            url="sqlite:///forge.db",
        )

        _, kwargs = context.configure.call_args
        assert kwargs["include_schemas"] is False
        assert "include_object" not in kwargs
        assert "version_table_schema" not in kwargs


class TestArgumentValidation:
    def test_neither_url_nor_connection_raises(self) -> None:
        context = mock.MagicMock()
        with pytest.raises(ValueError, match="exactly one"):
            configure_migration_context(
                context,
                target_metadata=mock.sentinel.metadata,
                version_table="alembic_version_forge",
            )

    def test_both_url_and_connection_raises(self) -> None:
        context = mock.MagicMock()
        with pytest.raises(ValueError, match="exactly one"):
            configure_migration_context(
                context,
                target_metadata=mock.sentinel.metadata,
                version_table="alembic_version_forge",
                url="sqlite:///forge.db",
                connection=mock.sentinel.connection,
            )


# ---------------------------------------------------------------------------
# AC #5: render_as_batch end to end on SQLite
# ---------------------------------------------------------------------------
#
# Everything above mocks ``alembic.context`` and only checks which kwargs
# ``configure_migration_context`` passes through. ``render_as_batch`` is an
# *autogenerate-rendering* flag: it does not change how a hand-written
# migration executes, only whether ``alembic revision --autogenerate``
# *writes* diffs wrapped in ``op.batch_alter_table(...)``. So the only way to
# prove it actually does something is to run a real, tiny Alembic project
# through both halves: autogenerate a migration against a live SQLite table,
# then apply it — SQLite has no ``ALTER TABLE ... ALTER COLUMN``, so this only
# succeeds if the generated migration is batch-mode (table rebuild + copy).
#
# A minimal ``widgets(id, name)`` table stands in for a real chain: revision
# 0001 (hand-written) creates it with ``name`` nullable; the temp project's
# ``env.py`` target metadata declares ``name`` as ``nullable=False`` through
# ``configure_migration_context`` (with no override — so ``render_as_batch``
# is on, same as every real chain). Autogenerating against the live (still
# nullable) table then must detect the diff and, if ``render_as_batch`` is
# doing its job, emit it as a batch op.

_ENV_PY = '''\
"""Minimal env.py for the AC5 fixture project -- exercises the real helper."""
from __future__ import annotations

import sqlalchemy as sa
from alembic import context

from sax_platform.db import configure_migration_context

metadata = sa.MetaData()
sa.Table(
    "widgets",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("name", sa.String(), nullable=False),
)


def run_migrations_offline() -> None:
    url = context.config.get_main_option("sqlalchemy.url")
    configure_migration_context(
        context,
        target_metadata=metadata,
        version_table="alembic_version_ac5_fixture",
        url=url,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = sa.create_engine(context.config.get_main_option("sqlalchemy.url"))
    with connectable.connect() as connection:
        configure_migration_context(
            context,
            target_metadata=metadata,
            version_table="alembic_version_ac5_fixture",
            connection=connection,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''

_REV1_CREATE_WIDGETS = '''\
"""create widgets, name nullable (the pre-ALTER baseline state)"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "widgets",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("widgets")
'''


def _make_fixture_project(tmp_path: Path) -> tuple[Config, str]:
    """Scaffold a throwaway Alembic project wired through the real helper.

    Returns the ``Config`` (with ``sqlalchemy.url`` and ``script_location``
    already set) and the SQLite URL. ``command.init`` writes the standard
    Alembic project skeleton (``script.py.mako``, ``README``, a starter
    ``env.py``); the starter ``env.py`` and a hand-written revision 0001 are
    then overwritten/added so the project is entirely ours.
    """
    script_dir = tmp_path / "alembic"
    db_path = tmp_path / "ac5.db"
    url = f"sqlite:///{db_path}"

    # Config.stdout swallows alembic's "Creating directory .../ done" chatter
    # so it doesn't spam the pytest run.
    cfg = Config(str(tmp_path / "alembic.ini"), stdout=io.StringIO())
    cfg.set_main_option("script_location", str(script_dir))
    cfg.set_main_option("sqlalchemy.url", url)

    command.init(cfg, str(script_dir))
    (script_dir / "env.py").write_text(_ENV_PY)
    (script_dir / "versions" / "0001_create_widgets.py").write_text(_REV1_CREATE_WIDGETS)

    return cfg, url


def _as_single_script(result: Script | list[Script | None] | None) -> Script:
    """Normalize ``command.revision``'s return type to one ``Script``.

    ``command.revision`` returns a list only when branch labels create
    multiple heads at once; this fixture project never does that, so a
    non-list, non-``None`` result is always what actually comes back --
    the assert documents that assumption rather than silently mis-indexing.
    """
    assert not isinstance(result, list), f"expected a single Script, got a list: {result}"
    assert result is not None
    return result


class TestRenderAsBatchEndToEnd:
    def test_autogenerated_alter_batches_and_applies_on_sqlite(self, tmp_path: Path) -> None:
        cfg, url = _make_fixture_project(tmp_path)
        engine = sa.create_engine(url)
        try:
            # Baseline: name is nullable, and already has data -- the batch
            # rebuild must carry this row through unharmed.
            command.upgrade(cfg, "head")
            with engine.begin() as conn:
                conn.execute(sa.text("INSERT INTO widgets (id, name) VALUES (1, 'gizmo')"))

            # Autogenerate against the live (nullable) table vs. the env.py
            # target metadata (not nullable): a diff exists, and rendering it
            # is exactly what render_as_batch=True controls.
            script = _as_single_script(
                command.revision(cfg, autogenerate=True, message="widgets name not null")
            )
            assert script.path is not None, "autogenerate did not write a migration file"
            generated_source = Path(script.path).read_text()
            assert "batch_alter_table" in generated_source, (
                "expected the autogenerated diff to be rendered as a batch op "
                "(render_as_batch=True from configure_migration_context) -- got:\n"
                f"{generated_source}"
            )

            # Applying it is the real proof: plain SQLite has no ALTER
            # COLUMN, so this only succeeds because batch mode rebuilds the
            # table.
            command.upgrade(cfg, "head")

            inspector = sa.inspect(engine)
            columns = {c["name"]: c for c in inspector.get_columns("widgets")}
            assert columns["name"]["nullable"] is False

            # The pre-existing row survived the rebuild-and-copy.
            with engine.connect() as conn:
                row = conn.execute(sa.text("SELECT name FROM widgets WHERE id = 1")).one()
            assert row.name == "gizmo"

            # And the constraint is for real, not just DDL cosmetics.
            with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
                conn.execute(sa.text("INSERT INTO widgets (id, name) VALUES (2, NULL)"))
        finally:
            engine.dispose()
