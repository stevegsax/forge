"""Tests for forge.logging_config — file-based worker logging (D85)."""

from __future__ import annotations

import logging
from pathlib import Path

from forge.logging_config import (
    DEFAULT_BACKUP_COUNT,
    DEFAULT_MAX_BYTES,
    build_file_handler_config,
    configure_file_handler,
    get_log_dir,
)

# ---------------------------------------------------------------------------
# get_log_dir
# ---------------------------------------------------------------------------


class TestGetLogDir:
    """Tests for log directory resolution from explicit values (no env reads).

    ``get_log_dir`` no longer reads ``FORGE_LOG_DIR``/``XDG_STATE_HOME`` itself
    (T3.6): those reads moved to ``sax_platform.config.LogSettings`` and the
    composition root passes the values in as arguments.
    """

    def test_default_path(self) -> None:
        result = get_log_dir()
        assert result == Path.home() / ".local" / "state" / "forge"

    def test_override_param(self, tmp_path: Path) -> None:
        result = get_log_dir(str(tmp_path / "custom"))
        assert result == tmp_path / "custom"

    def test_empty_string_disables(self) -> None:
        assert get_log_dir("") is None

    def test_xdg_state_home_param(self, tmp_path: Path) -> None:
        result = get_log_dir(xdg_state_home=str(tmp_path / "xdg"))
        assert result == tmp_path / "xdg" / "forge"

    def test_explicit_override_wins_over_xdg(self, tmp_path: Path) -> None:
        """A supplied log-dir override takes precedence over the XDG value."""
        result = get_log_dir(
            log_dir_override=tmp_path / "explicit",
            xdg_state_home="/should/be/ignored",
        )
        assert result == tmp_path / "explicit"

    def test_explicit_empty_string_disables_even_with_xdg(self, tmp_path: Path) -> None:
        assert get_log_dir(log_dir_override="", xdg_state_home=str(tmp_path)) is None


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

    def test_disabled_returns_none(self) -> None:
        handler = configure_file_handler(log_dir_override="")
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

    def test_override_wins_over_xdg(self, tmp_path: Path) -> None:
        """log_dir_override takes precedence over the xdg_state_home argument."""
        root = logging.getLogger()
        handler = configure_file_handler(
            log_dir_override=tmp_path, xdg_state_home="/should/be/ignored"
        )
        try:
            assert handler is not None
            assert (tmp_path / "forge.log").exists()
        finally:
            if handler is not None:
                root.removeHandler(handler)
                handler.close()
