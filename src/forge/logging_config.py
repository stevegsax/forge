"""File-based logging configuration (D85).

Follows Function Core / Imperative Shell:
- Pure functions: get_log_dir, build_file_handler_config
- Imperative shell: configure_file_handler
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Rotation defaults
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5
FILE_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
FILE_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def get_log_dir(log_dir_override: str | Path | None = None) -> Path | None:
    """Resolve the log directory.

    Resolution order:

    1. *log_dir_override* parameter (explicit caller override).
    2. ``FORGE_LOG_DIR`` environment variable.
    3. ``$XDG_STATE_HOME/forge/``
    4. ``~/.local/state/forge/``

    Returns ``None`` if *log_dir_override* or ``FORGE_LOG_DIR`` is an empty
    string (disables file logging).
    """
    if log_dir_override is not None:
        if str(log_dir_override) == "":
            return None
        return Path(log_dir_override)

    env_value = os.environ.get("FORGE_LOG_DIR")
    if env_value is not None:
        if env_value == "":
            return None
        return Path(env_value)

    xdg_state = os.environ.get("XDG_STATE_HOME")
    if xdg_state:
        return Path(xdg_state) / "forge"

    return Path.home() / ".local" / "state" / "forge"


def build_file_handler_config(
    log_dir: Path,
    log_name: str = "forge",
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> dict[str, Any]:
    """Return kwargs suitable for constructing a ``RotatingFileHandler``.

    This is a pure function — it performs no I/O.
    """
    return {
        "filename": str(log_dir / f"{log_name}.log"),
        "maxBytes": max_bytes,
        "backupCount": backup_count,
    }


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def silence_noisy_loggers() -> None:
    """Suppress known noisy third-party loggers.

    The Mistral SDK's OTel hook tries to JSON-parse multipart file upload
    bodies, which fails with a warning on every file API call.  This is a
    Mistral SDK bug (doesn't guard against non-JSON request bodies) and the
    warnings are harmless.
    """
    logging.getLogger("mistralai.extra.observability.otel").setLevel(logging.ERROR)


def configure_file_handler(
    log_name: str = "forge",
    log_dir_override: str | Path | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> RotatingFileHandler | None:
    """Create a ``RotatingFileHandler`` and attach it to the root logger.

    Best-effort: returns ``None`` on failure (directory creation errors,
    permission issues, etc.) so callers never need to handle exceptions.
    """
    try:
        log_dir = get_log_dir(log_dir_override)
        if log_dir is None:
            return None

        log_dir.mkdir(parents=True, exist_ok=True)

        config = build_file_handler_config(
            log_dir=log_dir,
            log_name=log_name,
            max_bytes=max_bytes,
            backup_count=backup_count,
        )
        handler = RotatingFileHandler(**config)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(FILE_LOG_FORMAT, datefmt=FILE_LOG_DATEFMT))

        logging.getLogger().addHandler(handler)
        logger.debug("File logging enabled: %s", config["filename"])
        return handler
    except Exception:
        logger.debug("Failed to configure file logging", exc_info=True)
        return None
