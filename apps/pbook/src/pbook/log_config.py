"""Logging configuration for the playbook service.

Log file path follows XDG Base Directory Specification:

1. ``$PBOOK_LOG_PATH`` environment variable (empty string disables file logging)
2. ``$XDG_STATE_HOME/pbook/pbook.log``
3. ``~/.local/state/pbook/pbook.log``

All pbook modules use ``logging.getLogger(__name__)`` — this module
configures the root ``pbook`` logger so all child loggers inherit the
configuration.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_log_path() -> Path | None:
    """Resolve the log file path.

    Returns ``None`` if ``PBOOK_LOG_PATH`` is set to an empty string
    (disables file logging).
    """
    env_value = os.environ.get("PBOOK_LOG_PATH")
    if env_value is not None:
        if env_value == "":
            return None
        return Path(env_value)

    xdg_state = os.environ.get("XDG_STATE_HOME")
    if xdg_state:
        return Path(xdg_state) / "pbook" / "pbook.log"

    return Path.home() / ".local" / "state" / "pbook" / "pbook.log"


def setup_logging(
    *,
    level: int = logging.INFO,
    console: bool = True,
) -> None:
    """Configure the ``pbook`` logger with file and optional console handlers.

    - File handler: RotatingFileHandler (5 MB max, 3 backups)
    - Console handler: StreamHandler to stderr (if *console* is ``True``)

    Safe to call multiple times — clears existing handlers first.
    """
    logger = logging.getLogger("pbook")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path = get_log_path()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
