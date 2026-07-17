"""File-based logging configuration, unified across forge and pbook (T3.4/ST6).

Ported and merged from `forge.logging_config` (`get_log_dir`,
`build_file_handler_config`, `configure_file_handler`, `silence_noisy_loggers`)
and `pbook.log_config` (`setup_logging`, `get_log_path`). Both apps configured
a *named* logger (the root logger for forge, the `"pbook"` logger for pbook)
with a `RotatingFileHandler` under an XDG-state-home-derived directory, plus
an optional console `StreamHandler`; they differed only in rotation size,
default console behavior, and env-var naming (`FORGE_LOG_DIR` — a directory —
vs `PBOOK_LOG_PATH` — a full file path).

This module unifies on forge's `{APP}_LOG_DIR` convention, generalized to any
`app_name`, and forge's 10 MB x 5 rotation defaults. `setup_logging` configures
`logging.getLogger(app_name)` (not the root logger) so multiple apps sharing a
process never clobber each other's handlers — closer to pbook's shape than
forge's, and safe for forge to adopt unchanged (forge's app_name is "forge").

Follows Function Core / Imperative Shell:
- Pure functions: `get_log_dir`, `resolve_log_path`, `build_file_handler_config`
- Imperative shell: `setup_logging`, `silence_noisy_loggers`

This module only provides the mechanism — adopting it in forge or pbook is a
later sub-task; neither app is modified here.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_BACKUP_COUNT",
    "DEFAULT_MAX_BYTES",
    "LOG_DATEFMT",
    "LOG_FORMAT",
    "build_file_handler_config",
    "get_log_dir",
    "resolve_log_path",
    "setup_logging",
    "silence_noisy_loggers",
]

# Rotation defaults (forge's; pbook used 5 MB x 3).
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def get_log_dir(app_name: str, log_dir_override: str | Path | None = None) -> Path | None:
    """Resolve the log directory for *app_name*.

    Resolution order:

    1. *log_dir_override* parameter (explicit caller override).
    2. ``{APP_NAME}_LOG_DIR`` environment variable (e.g. ``FORGE_LOG_DIR``
       for ``app_name="forge"``).
    3. ``$XDG_STATE_HOME/{app_name}/``
    4. ``~/.local/state/{app_name}/``

    Returns ``None`` if *log_dir_override* or the env var is an empty string
    (disables file logging).
    """
    if log_dir_override is not None:
        if str(log_dir_override) == "":
            return None
        return Path(log_dir_override)

    env_value = os.environ.get(f"{app_name.upper()}_LOG_DIR")
    if env_value is not None:
        if env_value == "":
            return None
        return Path(env_value)

    xdg_state = os.environ.get("XDG_STATE_HOME")
    if xdg_state:
        return Path(xdg_state) / app_name

    return Path.home() / ".local" / "state" / app_name


def resolve_log_path(
    app_name: str,
    log_path: str | Path | None = None,
    log_dir_override: str | Path | None = None,
) -> Path | None:
    """Resolve the final log *file* path for *app_name*.

    Resolution order:

    1. *log_path* parameter (an explicit full file path; empty string
       disables file logging — mirrors pbook's ``PBOOK_LOG_PATH`` override).
    2. ``get_log_dir(app_name, log_dir_override) / f"{app_name}.log"``.

    Returns ``None`` when file logging is disabled at either level.
    """
    if log_path is not None:
        if str(log_path) == "":
            return None
        return Path(log_path)

    log_dir = get_log_dir(app_name, log_dir_override)
    if log_dir is None:
        return None
    return log_dir / f"{app_name}.log"


def build_file_handler_config(
    log_path: Path,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> dict[str, Any]:
    """Return kwargs suitable for constructing a ``RotatingFileHandler``.

    This is a pure function — it performs no I/O.
    """
    return {
        "filename": str(log_path),
        "maxBytes": max_bytes,
        "backupCount": backup_count,
    }


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def silence_noisy_loggers() -> None:
    """Suppress known noisy third-party loggers.

    The Mistral SDK's OTel hook tries to JSON-parse multipart file upload
    bodies, which fails with a warning on every file API call. This is a
    Mistral SDK bug (doesn't guard against non-JSON request bodies) and the
    warnings are harmless. Ported from `forge.logging_config`.
    """
    logging.getLogger("mistralai.extra.observability.otel").setLevel(logging.ERROR)


def setup_logging(
    app_name: str,
    *,
    log_path: str | Path | None = None,
    console: bool = False,
    level: int = logging.INFO,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> None:
    """Configure the named ``app_name`` logger with file and optional console handlers.

    - File handler: ``RotatingFileHandler`` (10 MB max, 5 backups by default;
      overridable via *max_bytes*/*backup_count*). Skipped entirely when the
      resolved log path is disabled (see `resolve_log_path`) — best-effort,
      but unlike forge's `configure_file_handler` this does not swallow
      unexpected errors (e.g. permission failures) since callers now choose
      *log_path* explicitly rather than relying on silent fallback.
    - Console handler: ``StreamHandler`` to stderr, only when *console* is
      ``True`` (pbook defaulted this on; forge had no console handler at
      all — this unification defaults it off and leaves the choice explicit).

    Safe to call multiple times — clears the named logger's existing handlers
    first, so repeated calls never multiply handlers (ported from pbook's
    `setup_logging`).
    """
    logger = logging.getLogger(app_name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)

    resolved_path = resolve_log_path(app_name, log_path)
    if resolved_path is not None:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        config = build_file_handler_config(
            log_path=resolved_path,
            max_bytes=max_bytes,
            backup_count=backup_count,
        )
        file_handler = RotatingFileHandler(**config)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
