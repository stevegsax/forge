"""Best-effort message logging for API request/response inspection.

Writes full JSON payloads to a ``messages/`` directory inside the worktree,
excluded from git commits via a local ``.gitignore``.

Design follows the D42 best-effort pattern: logging failures are silently
swallowed so they never disrupt the workflow.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def write_message_log(worktree_path: str, prefix: str, content: str) -> None:
    """Best-effort write of a message log file. Never raises (D42).

    Args:
        worktree_path: Root of the worktree. If empty, does nothing.
        prefix: File name prefix (e.g. "request", "response").
        content: JSON string to write.
    """
    if not worktree_path:
        return

    try:
        messages_dir = Path(worktree_path) / "messages"
        messages_dir.mkdir(parents=True, exist_ok=True)

        # Ensure messages/ is git-ignored
        gitignore = messages_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n")

        timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"{prefix}-{timestamp}.json"
        (messages_dir / filename).write_text(content)
    except Exception:
        logger.debug("Failed to write message log", exc_info=True)
