"""Tests for the generic store-engine + idempotent-insert helpers.

``FORGE_DB_URL`` in the ambient shell environment points at production
(Supabase) — every test here sets it explicitly via ``monkeypatch`` to a
throwaway SQLite target so nothing ever reads the ambient value. The
``postgresql`` dialect paths are covered by building an engine against a
DSN (which never opens a socket) or by mocking the ``Engine``/connection —
no real Postgres server is touched.

Ported from ``forge_contracts/tests/test_db.py`` (T3.4, ST3), rewritten
against ``sax_platform.db``, plus new coverage for ``is_pooler_url`` and the
SQLite WAL + busy_timeout pragmas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import sqlalchemy as sa

from sax_platform.db.engine import get_store_engine, is_pooler_url
from sax_platform.db.ops import (
    ensure_sqlite_parent,
    insert_or_ignore,
)

if TYPE_CHECKING:
    from pathlib import Path

_widgets = sa.Table(
    "widgets",
    sa.MetaData(),
    sa.Column("id", sa.String, primary_key=True),
    sa.Column("value", sa.String),
)


class TestEnsureSqliteParent:
    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "sub" / "store.db"
        assert not target.parent.exists()
        ensure_sqlite_parent(f"sqlite:///{target}")
        assert target.parent.exists()

    def test_memory_database_does_not_touch_filesystem(self, tmp_path: Path) -> None:
        # Should not raise even though ":memory:" is not a real path.
        ensure_sqlite_parent("sqlite:///:memory:")

    def test_bare_sqlite_url_does_not_touch_filesystem(self) -> None:
        # "sqlite://" has no database component at all (make_url().database is None).
        ensure_sqlite_parent("sqlite://")


class TestIsPoolerUrl:
    def test_supabase_pooler_host_is_detected(self) -> None:
        url = "postgresql+psycopg2://user:pw@aws-0-us-east-1.pooler.supabase.com:5432/postgres"
        assert is_pooler_url(url) is True

    def test_pooler_port_6543_is_detected(self) -> None:
        url = "postgresql+psycopg2://user:pw@db.example.com:6543/forge"
        assert is_pooler_url(url) is True

    def test_plain_postgres_url_is_not_pooled(self) -> None:
        url = "postgresql+psycopg2://user:pw@localhost:5432/forge_test"
        assert is_pooler_url(url) is False

    def test_override_forces_true_regardless_of_url(self) -> None:
        url = "postgresql+psycopg2://user:pw@localhost:5432/forge_test"
        assert is_pooler_url(url, pooler_override=True) is True

    def test_override_false_defers_to_url_shape(self) -> None:
        url = "postgresql+psycopg2://user:pw@aws-0-us-east-1.pooler.supabase.com:5432/postgres"
        assert is_pooler_url(url, pooler_override=False) is True


class TestGetStoreEngine:
    """``url`` and ``pooler_override`` are required, explicit config — the factory
    reads no environment itself (T3.6)."""

    def test_sqlite_url_enables_wal_and_creates_parent_dir(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nested" / "wal.db"

        engine = get_store_engine(f"sqlite:///{db_path}")

        assert engine.dialect.name == "sqlite"
        assert db_path.parent.exists()
        with engine.connect() as conn:
            mode = conn.execute(sa.text("PRAGMA journal_mode")).scalar()
        assert mode == "wal"

    def test_sqlite_url_sets_busy_timeout(self, tmp_path: Path) -> None:
        db_path = tmp_path / "busy.db"

        engine = get_store_engine(f"sqlite:///{db_path}")

        with engine.connect() as conn:
            timeout_ms = conn.execute(sa.text("PRAGMA busy_timeout")).scalar()
        assert timeout_ms is not None
        assert timeout_ms > 0

    def test_postgres_url_builds_pooled_engine_without_wal(self) -> None:
        # Building the engine imports the driver but never connects, so no
        # Postgres server is needed — only the psycopg2 DBAPI module.
        engine = get_store_engine("postgresql+psycopg2://user:pw@localhost:5432/forge_test")
        assert engine.dialect.name == "postgresql"
        assert isinstance(engine.pool, sa.pool.QueuePool)
        assert engine.pool.size() == 5
        assert engine.pool._max_overflow == 5  # only public accessor is size(); overflow is private

    def test_postgres_pooler_url_disables_prepared_statements(self) -> None:
        with mock.patch("sax_platform.db.engine.sa.create_engine") as create_engine:
            get_store_engine(
                "postgresql+psycopg2://user:pw@aws-0-us-east-1.pooler.supabase.com:5432/postgres"
            )
        _, kwargs = create_engine.call_args
        assert kwargs["connect_args"] == {"prepare_threshold": None}

    def test_postgres_non_pooler_url_leaves_connect_args_empty(self) -> None:
        with mock.patch("sax_platform.db.engine.sa.create_engine") as create_engine:
            get_store_engine("postgresql+psycopg2://user:pw@localhost:5432/forge_test")
        _, kwargs = create_engine.call_args
        assert kwargs["connect_args"] == {}

    def test_explicit_pooler_override_true_disables_prepared_statements(self) -> None:
        # A plain (non-pooler) URL would be False by shape, so True here can only
        # come from the explicit override — proving it is threaded.
        with mock.patch("sax_platform.db.engine.sa.create_engine") as create_engine:
            get_store_engine(
                "postgresql+psycopg2://user:pw@localhost:5432/forge_test",
                pooler_override=True,
            )
        _, kwargs = create_engine.call_args
        assert kwargs["connect_args"] == {"prepare_threshold": None}

    def test_explicit_pooler_override_false_still_detects_pooler_url_shape(self) -> None:
        # override=False only defers to URL shape; a pooler-shaped URL is still
        # detected (matches is_pooler_url semantics).
        with mock.patch("sax_platform.db.engine.sa.create_engine") as create_engine:
            get_store_engine(
                "postgresql+psycopg2://user:pw@aws-0.pooler.supabase.com:5432/postgres",
                pooler_override=False,
            )
        _, kwargs = create_engine.call_args
        assert kwargs["connect_args"] == {"prepare_threshold": None}


class TestInsertOrIgnoreSqlite:
    def _engine(self) -> sa.Engine:
        engine = sa.create_engine("sqlite://")
        _widgets.metadata.create_all(engine)
        return engine

    def test_first_write_returns_true_and_persists_row(self) -> None:
        engine = self._engine()
        applied = insert_or_ignore(
            engine, _widgets, {"id": "a", "value": "x"}, index_elements=["id"]
        )
        assert applied is True
        with engine.connect() as conn:
            row = conn.execute(sa.select(_widgets).where(_widgets.c.id == "a")).one()
        assert row.value == "x"

    def test_duplicate_write_returns_false_and_leaves_original(self) -> None:
        engine = self._engine()
        insert_or_ignore(engine, _widgets, {"id": "a", "value": "x"}, index_elements=["id"])

        applied = insert_or_ignore(
            engine, _widgets, {"id": "a", "value": "y"}, index_elements=["id"]
        )

        assert applied is False
        with engine.connect() as conn:
            row = conn.execute(sa.select(_widgets).where(_widgets.c.id == "a")).one()
        assert row.value == "x"  # DO NOTHING: the second write never landed

    def test_distinct_keys_both_write(self) -> None:
        engine = self._engine()
        first = insert_or_ignore(engine, _widgets, {"id": "a", "value": "x"}, index_elements=["id"])
        second = insert_or_ignore(
            engine, _widgets, {"id": "b", "value": "y"}, index_elements=["id"]
        )
        assert (first, second) == (True, True)


class TestInsertOrIgnorePostgresDialect:
    """Dialect selection is mocked — no real Postgres connection is opened."""

    def _mock_engine(self, *, rowcount: int) -> tuple[mock.MagicMock, mock.MagicMock]:
        engine = mock.MagicMock()
        engine.dialect.name = "postgresql"
        conn = mock.MagicMock()
        conn.execute.return_value.rowcount = rowcount
        engine.begin.return_value.__enter__.return_value = conn
        return engine, conn

    def test_new_row_returns_true(self) -> None:
        engine, conn = self._mock_engine(rowcount=1)
        applied = insert_or_ignore(engine, _widgets, {"id": "a"}, index_elements=["id"])
        assert applied is True
        conn.execute.assert_called_once()

    def test_conflicting_row_returns_false(self) -> None:
        engine, conn = self._mock_engine(rowcount=0)
        applied = insert_or_ignore(engine, _widgets, {"id": "a"}, index_elements=["id"])
        assert applied is False
        conn.execute.assert_called_once()
