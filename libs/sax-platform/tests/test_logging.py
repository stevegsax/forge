"""Tests for sax_platform.logging — the unified forge/pbook logging setup.

Pure helpers (`get_log_dir`, `resolve_log_path`, `build_file_handler_config`)
are tested directly, with env vars monkeypatched and no filesystem I/O
performed. The imperative shell (`setup_logging`) is tested against
`tmp_path`, writing real files. Every test that calls `setup_logging` uses a
throwaway, uniquely-named logger (`app_name`) and tears its handlers down
afterward (`cleanup_logger`), so tests never leak handler state into each
other or into the ambient `pytest` root logger.
"""

import logging
import logging.handlers
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from sax_platform.logging import (
    DEFAULT_BACKUP_COUNT,
    DEFAULT_MAX_BYTES,
    build_file_handler_config,
    get_log_dir,
    resolve_log_path,
    setup_logging,
    silence_noisy_loggers,
)

CleanupLogger = Callable[[str], str]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app_name() -> str:
    """A unique per-test logger name, so `setup_logging` calls never collide
    with each other or with any real app logger."""
    return f"testapp-{uuid.uuid4().hex}"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip env vars that would otherwise leak the real ambient environment
    (FORGE_LOG_DIR, XDG_STATE_HOME, etc.) into path-resolution tests."""
    for key in ("XDG_STATE_HOME", "FORGE_LOG_DIR", "PBOOK_LOG_DIR", "PBOOK_LOG_PATH"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def cleanup_logger() -> Iterator[CleanupLogger]:
    """Track logger names configured during a test and clear their handlers
    on teardown, so `setup_logging`'s effects never outlive the test."""
    names: list[str] = []

    def _track(name: str) -> str:
        names.append(name)
        return name

    yield _track

    for name in names:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            handler.close()
        logger.handlers.clear()


# ---------------------------------------------------------------------------
# get_log_dir — pure
# ---------------------------------------------------------------------------


class TestGetLogDir:
    def test_explicit_override_wins(self) -> None:
        assert get_log_dir("forge", log_dir_override="/explicit/dir") == Path("/explicit/dir")

    def test_explicit_override_empty_string_disables(self) -> None:
        assert get_log_dir("forge", log_dir_override="") is None

    def test_app_specific_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_LOG_DIR", "/env/forge/logs")
        assert get_log_dir("forge") == Path("/env/forge/logs")

    def test_env_var_is_derived_from_app_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PBOOK_LOG_DIR", "/env/pbook/logs")
        assert get_log_dir("pbook") == Path("/env/pbook/logs")
        # A different app's env var must not leak across.
        monkeypatch.delenv("PBOOK_LOG_DIR")
        assert get_log_dir("pbook") != Path("/env/pbook/logs")

    def test_env_var_empty_string_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_LOG_DIR", "")
        assert get_log_dir("forge") is None

    def test_override_beats_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_LOG_DIR", "/env/dir")
        assert get_log_dir("forge", log_dir_override="/override/dir") == Path("/override/dir")

    def test_xdg_state_home_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_STATE_HOME", "/xdg/state")
        assert get_log_dir("forge") == Path("/xdg/state/forge")

    def test_xdg_state_home_is_per_app(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_STATE_HOME", "/xdg/state")
        assert get_log_dir("pbook") == Path("/xdg/state/pbook")

    def test_home_fallback_when_nothing_set(self) -> None:
        assert get_log_dir("forge") == Path.home() / ".local" / "state" / "forge"


# ---------------------------------------------------------------------------
# resolve_log_path — pure
# ---------------------------------------------------------------------------


class TestResolveLogPath:
    def test_explicit_log_path_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_LOG_DIR", "/env/dir")
        assert resolve_log_path("forge", log_path="/explicit/file.log") == Path(
            "/explicit/file.log"
        )

    def test_empty_log_path_disables(self) -> None:
        assert resolve_log_path("forge", log_path="") is None

    def test_falls_back_to_log_dir_plus_app_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_LOG_DIR", "/env/dir")
        assert resolve_log_path("forge") == Path("/env/dir/forge.log")

    def test_disabled_log_dir_propagates_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_LOG_DIR", "")
        assert resolve_log_path("forge") is None

    def test_log_dir_override_param_is_forwarded(self) -> None:
        assert resolve_log_path("forge", log_dir_override="/override/dir") == Path(
            "/override/dir/forge.log"
        )


# ---------------------------------------------------------------------------
# build_file_handler_config — pure, no I/O
# ---------------------------------------------------------------------------


class TestBuildFileHandlerConfig:
    def test_default_shape(self, tmp_path: Path) -> None:
        log_path = tmp_path / "does-not-exist" / "app.log"
        config = build_file_handler_config(log_path)
        assert config == {
            "filename": str(log_path),
            "maxBytes": DEFAULT_MAX_BYTES,
            "backupCount": DEFAULT_BACKUP_COUNT,
        }

    def test_custom_rotation(self, tmp_path: Path) -> None:
        log_path = tmp_path / "app.log"
        config = build_file_handler_config(log_path, max_bytes=1024, backup_count=2)
        assert config["maxBytes"] == 1024
        assert config["backupCount"] == 2

    def test_performs_no_io(self, tmp_path: Path) -> None:
        """A pure function must not create the file or its parent dir."""
        log_path = tmp_path / "untouched" / "app.log"
        build_file_handler_config(log_path)
        assert not log_path.parent.exists()
        assert not log_path.exists()


# ---------------------------------------------------------------------------
# setup_logging — imperative shell
# ---------------------------------------------------------------------------


class TestSetupLoggingFileHandler:
    def test_writes_to_log_file(
        self, tmp_path: Path, app_name: str, cleanup_logger: CleanupLogger
    ) -> None:
        log_path = tmp_path / "app.log"
        cleanup_logger(app_name)

        setup_logging(app_name, log_path=str(log_path))
        logging.getLogger(app_name).info("hello from test")

        assert log_path.exists()
        assert "hello from test" in log_path.read_text()

    def test_creates_parent_directories(
        self, tmp_path: Path, app_name: str, cleanup_logger: CleanupLogger
    ) -> None:
        log_path = tmp_path / "nested" / "dir" / "app.log"
        cleanup_logger(app_name)

        setup_logging(app_name, log_path=str(log_path))

        assert log_path.parent.is_dir()

    def test_empty_log_path_disables_file_logging(
        self, app_name: str, cleanup_logger: CleanupLogger
    ) -> None:
        cleanup_logger(app_name)

        setup_logging(app_name, log_path="")

        handlers = logging.getLogger(app_name).handlers
        assert not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in handlers)

    def test_sets_logger_level(
        self, tmp_path: Path, app_name: str, cleanup_logger: CleanupLogger
    ) -> None:
        log_path = tmp_path / "app.log"
        cleanup_logger(app_name)

        setup_logging(app_name, log_path=str(log_path), level=logging.WARNING)

        assert logging.getLogger(app_name).level == logging.WARNING


class TestSetupLoggingIdempotent:
    def test_calling_twice_does_not_multiply_handlers(
        self, tmp_path: Path, app_name: str, cleanup_logger: CleanupLogger
    ) -> None:
        log_path = tmp_path / "app.log"
        cleanup_logger(app_name)

        setup_logging(app_name, log_path=str(log_path), console=True)
        first_count = len(logging.getLogger(app_name).handlers)
        setup_logging(app_name, log_path=str(log_path), console=True)
        second_count = len(logging.getLogger(app_name).handlers)

        assert first_count == 2  # file + console
        assert second_count == first_count

    def test_repeated_calls_still_log_correctly(
        self, tmp_path: Path, app_name: str, cleanup_logger: CleanupLogger
    ) -> None:
        log_path = tmp_path / "app.log"
        cleanup_logger(app_name)

        setup_logging(app_name, log_path=str(log_path))
        setup_logging(app_name, log_path=str(log_path))
        logging.getLogger(app_name).info("only once per line")

        contents = log_path.read_text()
        assert contents.count("only once per line") == 1


class TestSetupLoggingConsoleHandler:
    def test_console_false_by_default(
        self, tmp_path: Path, app_name: str, cleanup_logger: CleanupLogger
    ) -> None:
        log_path = tmp_path / "app.log"
        cleanup_logger(app_name)

        setup_logging(app_name, log_path=str(log_path))

        handlers = logging.getLogger(app_name).handlers
        assert not any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.RotatingFileHandler)
            for h in handlers
        )

    def test_console_true_adds_stream_handler(
        self, tmp_path: Path, app_name: str, cleanup_logger: CleanupLogger
    ) -> None:
        log_path = tmp_path / "app.log"
        cleanup_logger(app_name)

        setup_logging(app_name, log_path=str(log_path), console=True)

        handlers = logging.getLogger(app_name).handlers
        assert any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.RotatingFileHandler)
            for h in handlers
        )

    def test_console_only_when_file_logging_disabled(
        self, app_name: str, cleanup_logger: CleanupLogger
    ) -> None:
        cleanup_logger(app_name)

        setup_logging(app_name, log_path="", console=True)

        handlers = logging.getLogger(app_name).handlers
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.StreamHandler)


# ---------------------------------------------------------------------------
# silence_noisy_loggers
# ---------------------------------------------------------------------------


class TestSilenceNoisyLoggers:
    def test_silences_mistralai_otel_logger(self) -> None:
        target = logging.getLogger("mistralai.extra.observability.otel")
        target.setLevel(logging.DEBUG)

        silence_noisy_loggers()

        assert target.level == logging.ERROR
