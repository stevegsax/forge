"""Tests for the shared migration runner and its advisory-lock key.

``advisory_lock_key`` is pure and gets full coverage here. ``run_migrations``
is exercised only for the SQLite no-lock path, with ``alembic.command.upgrade``
mocked out so no real migration chain needs to exist on disk. Real cross-DB
migration application (SQLite and Postgres both actually applying a chain to
head, including the advisory lock serializing two concurrent Postgres
migrators) is covered later, in ST10 and the ``-m postgres`` CI job — this
file intentionally does not stand up Postgres.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest

from sax_platform.db.migrations import (
    _LOCK_POLL_INTERVAL_SECONDS,
    advisory_lock_key,
    run_migrations,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestAdvisoryLockKey:
    def test_deterministic_for_same_version_table(self) -> None:
        assert advisory_lock_key("alembic_version_forge") == advisory_lock_key(
            "alembic_version_forge"
        )

    def test_distinct_for_distinct_version_tables(self) -> None:
        assert advisory_lock_key("alembic_version_forge") != advisory_lock_key(
            "alembic_version_ocr"
        )

    def test_forge_and_pbook_and_ocr_chains_are_pairwise_distinct(self) -> None:
        keys = {
            advisory_lock_key(name)
            for name in ("alembic_version_forge", "alembic_version_ocr", "pbk_alembic_version")
        }
        assert len(keys) == 3

    def test_returns_a_signed_bigint_in_postgres_range(self) -> None:
        key = advisory_lock_key("alembic_version_forge")
        assert isinstance(key, int)
        assert -(2**63) <= key < 2**63

    def test_empty_string_does_not_raise(self) -> None:
        # Degenerate input, but the function should still return a valid int
        # rather than raising — callers passing an empty version_table is a
        # caller bug to catch elsewhere, not something this pure function
        # should special-case.
        assert isinstance(advisory_lock_key(""), int)


class TestRunMigrationsSqlitePath:
    def test_sqlite_path_upgrades_without_creating_a_lock_engine(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        upgrade_calls: list[tuple[object, str]] = []

        def fake_upgrade(cfg: object, revision: str) -> None:
            upgrade_calls.append((cfg, revision))

        monkeypatch.setattr("alembic.command.upgrade", fake_upgrade)

        with mock.patch("sax_platform.db.migrations.sa.create_engine") as create_engine:
            db_path = tmp_path / "nested" / "x.db"
            run_migrations(
                f"sqlite:///{db_path}",
                version_table="alembic_version_test",
                script_location=str(tmp_path),
            )

        assert len(upgrade_calls) == 1
        assert upgrade_calls[0][1] == "head"
        # No lock engine (NullPool) is ever built on the SQLite path.
        create_engine.assert_not_called()

    def test_sqlite_path_creates_parent_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("alembic.command.upgrade", lambda cfg, rev: None)

        db_path = tmp_path / "deep" / "nested" / "x.db"
        assert not db_path.parent.exists()

        run_migrations(
            f"sqlite:///{db_path}",
            version_table="alembic_version_test",
            script_location=str(tmp_path),
        )

        assert db_path.parent.exists()


class TestRunMigrationsPostgresPath:
    """The Postgres branch is exercised with a mocked lock engine/connection.

    No real Postgres server is contacted: ``sa.create_engine`` and the
    connection it returns are both mocks, so this only proves the lock is
    acquired/released around ``command.upgrade`` and keyed as expected — not
    that a real migration chain applies (that's ST10 / the postgres marker).
    """

    def test_acquires_and_releases_lock_keyed_on_version_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        upgrade_calls: list[tuple[object, str]] = []
        monkeypatch.setattr(
            "alembic.command.upgrade",
            lambda cfg, rev: upgrade_calls.append((cfg, rev)),
        )

        conn = mock.MagicMock()
        lock_engine = mock.MagicMock()
        lock_engine.connect.return_value.__enter__.return_value = conn

        expected_key = advisory_lock_key("alembic_version_test")

        with mock.patch(
            "sax_platform.db.migrations.sa.create_engine", return_value=lock_engine
        ) as create_engine:
            run_migrations(
                "postgresql+psycopg2://user:pw@localhost:5432/forge_test",
                version_table="alembic_version_test",
                script_location="/nonexistent/alembic/dir",
            )

        create_engine.assert_called_once()
        assert len(upgrade_calls) == 1

        executed_keys = [call.args[1]["key"] for call in conn.execute.call_args_list]
        assert executed_keys == [expected_key, expected_key]
        lock_engine.dispose.assert_called_once()

    def test_lock_connection_is_autocommit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The lock connection must not sit in an open transaction during the upgrade.

        An open transaction on the lock connection is a snapshot a concurrent
        ``CREATE INDEX CONCURRENTLY`` would wait on — the same hazard as a
        server-side wait.
        """
        monkeypatch.setattr("alembic.command.upgrade", lambda cfg, rev: None)

        lock_engine = mock.MagicMock()

        with mock.patch(
            "sax_platform.db.migrations.sa.create_engine", return_value=lock_engine
        ) as create_engine:
            run_migrations(
                "postgresql+psycopg2://user:pw@localhost:5432/forge_test",
                version_table="alembic_version_test",
                script_location="/nonexistent/alembic/dir",
            )

        assert create_engine.call_args.kwargs["isolation_level"] == "AUTOCOMMIT"

    def test_polls_try_lock_and_sleeps_client_side_until_acquired(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A contended lock is waited for by polling, with the sleep in the client.

        A server-side wait (blocking ``pg_advisory_lock``/``pg_sleep``) holds a
        snapshot for its whole duration and deadlocks against a concurrent
        index build, so the mechanism itself is pinned here: every statement is
        a ``pg_try_advisory_lock``, and the waiting happens in ``time.sleep``.
        """
        monkeypatch.setattr("alembic.command.upgrade", lambda cfg, rev: None)

        sleeps: list[float] = []
        monkeypatch.setattr("sax_platform.db.migrations.time.sleep", sleeps.append)

        # Two refusals, then the lock is granted; the unlock's result is unused.
        results = [False, False, True, None]
        conn = mock.MagicMock()
        conn.execute.side_effect = [mock.MagicMock(**{"scalar.return_value": r}) for r in results]
        lock_engine = mock.MagicMock()
        lock_engine.connect.return_value.__enter__.return_value = conn

        with mock.patch("sax_platform.db.migrations.sa.create_engine", return_value=lock_engine):
            run_migrations(
                "postgresql+psycopg2://user:pw@localhost:5432/forge_test",
                version_table="alembic_version_test",
                script_location="/nonexistent/alembic/dir",
            )

        statements = [str(call.args[0]) for call in conn.execute.call_args_list]
        assert statements[:3] == ["SELECT pg_try_advisory_lock(:key)"] * 3
        assert statements[3] == "SELECT pg_advisory_unlock(:key)"
        # One client-side sleep per refusal, and none after the grant.
        assert sleeps == [_LOCK_POLL_INTERVAL_SECONDS] * 2
        assert all("pg_advisory_lock(" not in stmt for stmt in statements[:3])
        assert all("pg_sleep" not in stmt for stmt in statements)

    def test_unlocks_even_when_upgrade_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(cfg: object, rev: str) -> None:
            raise RuntimeError("migration failed")

        monkeypatch.setattr("alembic.command.upgrade", boom)

        conn = mock.MagicMock()
        lock_engine = mock.MagicMock()
        lock_engine.connect.return_value.__enter__.return_value = conn

        with (
            mock.patch("sax_platform.db.migrations.sa.create_engine", return_value=lock_engine),
            pytest.raises(RuntimeError, match="migration failed"),
        ):
            run_migrations(
                "postgresql+psycopg2://user:pw@localhost:5432/forge_test",
                version_table="alembic_version_test",
                script_location="/nonexistent/alembic/dir",
            )

        # pg_try_advisory_lock and pg_advisory_unlock both still ran.
        assert conn.execute.call_count == 2
        lock_engine.dispose.assert_called_once()
