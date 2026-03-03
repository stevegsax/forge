"""Tests for forge.logging_config — file-based worker logging (D85)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from forge.logging_config import (
    DEFAULT_BACKUP_COUNT,
    DEFAULT_MAX_BYTES,
    build_file_handler_config,
    configure_file_handler,
    get_log_dir,
)

if TYPE_CHECKING:
    import pytest


# ---------------------------------------------------------------------------
# get_log_dir
# ---------------------------------------------------------------------------


class TestGetLogDir:
    """Tests for XDG-aware log directory resolution."""

    def test_default_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FORGE_LOG_DIR", raising=False)
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        result = get_log_dir()
        assert result == Path.home() / ".local" / "state" / "forge"

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("FORGE_LOG_DIR", str(tmp_path / "custom"))
        result = get_log_dir()
        assert result == tmp_path / "custom"

    def test_empty_string_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_LOG_DIR", "")
        assert get_log_dir() is None

    def test_xdg_state_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("FORGE_LOG_DIR", raising=False)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
        result = get_log_dir()
        assert result == tmp_path / "xdg" / "forge"

    def test_explicit_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Explicit override takes precedence over env vars."""
        monkeypatch.setenv("FORGE_LOG_DIR", "/should/be/ignored")
        result = get_log_dir(log_dir_override=tmp_path / "explicit")
        assert result == tmp_path / "explicit"

    def test_explicit_empty_string_disables(self) -> None:
        assert get_log_dir(log_dir_override="") is None


# ---------------------------------------------------------------------------
# build_file_handler_config
# ---------------------------------------------------------------------------


class TestBuildFileHandlerConfig:
    """Tests for the pure handler config builder."""

    def test_default_config(self, tmp_path: Path) -> None:
        config = build_file_handler_config(tmp_path)
        assert config == {
            "filename": str(tmp_path / "forge.log"),
            "maxBytes": DEFAULT_MAX_BYTES,
            "backupCount": DEFAULT_BACKUP_COUNT,
        }

    def test_custom_log_name(self, tmp_path: Path) -> None:
        config = build_file_handler_config(tmp_path, log_name="worker")
        assert config["filename"] == str(tmp_path / "worker.log")

    def test_custom_rotation(self, tmp_path: Path) -> None:
        config = build_file_handler_config(tmp_path, max_bytes=1024, backup_count=2)
        assert config["maxBytes"] == 1024
        assert config["backupCount"] == 2


# ---------------------------------------------------------------------------
# configure_file_handler
# ---------------------------------------------------------------------------


class TestConfigureFileHandler:
    """Tests for the imperative handler setup."""

    def test_creates_handler(self, tmp_path: Path) -> None:
        root = logging.getLogger()
        handler = configure_file_handler(log_dir_override=tmp_path)
        try:
            assert handler is not None
            assert handler in root.handlers
            assert handler.level == logging.DEBUG
            assert (tmp_path / "forge.log").exists()
        finally:
            if handler is not None:
                root.removeHandler(handler)
                handler.close()

    def test_disabled_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_LOG_DIR", "")
        handler = configure_file_handler()
        assert handler is None

    def test_worker_log_name(self, tmp_path: Path) -> None:
        root = logging.getLogger()
        handler = configure_file_handler(log_name="worker", log_dir_override=tmp_path)
        try:
            assert handler is not None
            assert (tmp_path / "worker.log").exists()
        finally:
            if handler is not None:
                root.removeHandler(handler)
                handler.close()

    def test_creates_directory(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "nested" / "logs"
        root = logging.getLogger()
        handler = configure_file_handler(log_dir_override=log_dir)
        try:
            assert handler is not None
            assert log_dir.is_dir()
        finally:
            if handler is not None:
                root.removeHandler(handler)
                handler.close()

    def test_override_parameter(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """log_dir_override parameter takes precedence over env."""
        monkeypatch.setenv("FORGE_LOG_DIR", "/should/be/ignored")
        root = logging.getLogger()
        handler = configure_file_handler(log_dir_override=tmp_path)
        try:
            assert handler is not None
            assert (tmp_path / "forge.log").exists()
        finally:
            if handler is not None:
                root.removeHandler(handler)
                handler.close()
