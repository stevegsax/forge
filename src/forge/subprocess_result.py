"""Shared subprocess result dataclass.

Used by both validation (ruff, tests) and git operations.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class SubprocessResult:
    """Result of a subprocess invocation. Internal transport only."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0
