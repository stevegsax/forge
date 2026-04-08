"""Shared test fixtures for pbook."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa

from pbook.store import get_engine, run_migrations

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point PBOOK_DB_PATH to a temporary directory for every test."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("PBOOK_DB_PATH", str(db_path))


@pytest.fixture(autouse=True)
def _dispose_store_engines(monkeypatch: pytest.MonkeyPatch):
    """Dispose SQLAlchemy engines created via pbook.store.get_engine after each test."""
    original_create_engine = sa.create_engine
    created_engines = []

    def tracking_create_engine(*args, **kwargs):
        engine = original_create_engine(*args, **kwargs)
        created_engines.append(engine)
        return engine

    monkeypatch.setattr(sa, "create_engine", tracking_create_engine)

    yield

    for engine in created_engines:
        engine.dispose()


def setup_db(tmp_path: Path):
    """Create a test database and return (engine, db_path)."""
    db_path = tmp_path / "test.db"
    run_migrations(db_path)
    engine = get_engine(db_path)
    return engine, db_path
