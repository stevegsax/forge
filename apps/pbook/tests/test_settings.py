"""Tests for pbook.settings (the frozen composition-root config)."""

from __future__ import annotations

import pytest

from pbook.settings import PbookDbSettings, PbookSettings


class TestPbookDbSettings:
    def test_defaults_disable_store(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("PBOOK_DATABASE_URL", raising=False)
        monkeypatch.delenv("PBOOK_DB_POOLER", raising=False)
        s = PbookDbSettings()
        assert s.url is None
        assert s.pooler is False

    def test_reads_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PBOOK_DATABASE_URL", "postgresql://u:p@h/db")
        monkeypatch.setenv("PBOOK_DB_POOLER", "1")
        s = PbookDbSettings()
        assert s.url == "postgresql://u:p@h/db"
        assert s.pooler is True

    def test_constructed_by_field_name(self):
        s = PbookDbSettings(url="postgresql://u@h/db", pooler=True)
        assert s.url == "postgresql://u@h/db"
        assert s.pooler is True

    def test_frozen(self):
        s = PbookDbSettings()
        with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError on frozen
            s.url = "x"  # type: ignore[misc]


class TestPbookSettings:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch):
        for var in ("PBOOK_DATABASE_URL", "PBOOK_DB_POOLER", "PBOOK_LOG_PATH", "OPENAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        s = PbookSettings()
        assert isinstance(s.db, PbookDbSettings)
        assert s.db.url is None
        assert s.log_path is None
        assert s.openai_api_key is None

    def test_composes_db_group_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PBOOK_DATABASE_URL", "postgresql://u@h/db")
        monkeypatch.setenv("PBOOK_LOG_PATH", "/tmp/pbook.log")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        s = PbookSettings()
        assert s.db.url == "postgresql://u@h/db"
        assert s.log_path == "/tmp/pbook.log"
        assert s.openai_api_key == "sk-test"

    def test_frozen(self):
        s = PbookSettings()
        with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError on frozen
            s.log_path = "x"  # type: ignore[misc]
