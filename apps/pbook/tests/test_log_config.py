"""Tests for pbook.log_config."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pbook.log_config import get_log_path, setup_logging

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


# ---------------------------------------------------------------------------
# get_log_path
# ---------------------------------------------------------------------------


class TestGetLogPath:
    def test_default_path(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("PBOOK_LOG_PATH", raising=False)
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        path = get_log_path()
        assert path is not None
        assert path.name == "pbook.log"
        assert "pbook" in str(path)

    def test_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        log_file = tmp_path / "custom.log"
        monkeypatch.setenv("PBOOK_LOG_PATH", str(log_file))
        assert get_log_path() == log_file

    def test_empty_env_disables(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PBOOK_LOG_PATH", "")
        assert get_log_path() is None

    def test_xdg_state_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("PBOOK_LOG_PATH", raising=False)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        path = get_log_path()
        assert path is not None
        assert path == tmp_path / "pbook" / "pbook.log"


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_creates_log_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        log_file = tmp_path / "test.log"
        monkeypatch.setenv("PBOOK_LOG_PATH", str(log_file))

        setup_logging(console=False)

        test_logger = logging.getLogger("pbook.test_setup")
        test_logger.info("test message")

        assert log_file.exists()
        content = log_file.read_text()
        assert "test message" in content

    def test_console_only(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PBOOK_LOG_PATH", "")

        setup_logging(console=True)

        pbook_logger = logging.getLogger("pbook")
        assert len(pbook_logger.handlers) == 1
        assert isinstance(pbook_logger.handlers[0], logging.StreamHandler)

    def test_no_handlers_when_all_disabled(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PBOOK_LOG_PATH", "")

        setup_logging(console=False)

        pbook_logger = logging.getLogger("pbook")
        assert len(pbook_logger.handlers) == 0

    def test_debug_level(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        log_file = tmp_path / "debug.log"
        monkeypatch.setenv("PBOOK_LOG_PATH", str(log_file))

        setup_logging(level=logging.DEBUG, console=False)

        test_logger = logging.getLogger("pbook.test_debug")
        test_logger.debug("debug message")

        content = log_file.read_text()
        assert "debug message" in content

    def test_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        log_file = tmp_path / "idem.log"
        monkeypatch.setenv("PBOOK_LOG_PATH", str(log_file))

        setup_logging(console=True)
        setup_logging(console=True)

        pbook_logger = logging.getLogger("pbook")
        # Should have exactly 2 handlers (file + console), not 4
        assert len(pbook_logger.handlers) == 2
