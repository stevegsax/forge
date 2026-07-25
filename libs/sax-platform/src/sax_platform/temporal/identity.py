"""Stamp the launch-time git version into a Temporal worker's identity.

Why this exists
---------------
A Python worker binds its code at import: the modules loaded when the process
started are the modules it will run until it is restarted. The working tree does
not stand still — the launchd/tmux workers ``exec uv run`` straight out of the
live repo (D99), so the tree can be several commits (or a half-finished edit)
ahead of any given running process. Nothing in Temporal records which code a
poller is actually executing: a static identity like ``desktop-forge-worker-1``
names the slot, not the build.

Stamping the git version captured *once at startup* into the identity makes the
server the authority::

    temporal task-queue describe --task-queue forge-task-queue

lists each poller's identity, so the answer to "which code is this worker
running?" is a query rather than a guess.

Shape of the stamped identity
-----------------------------
``<base>@<version>``, where *base* is the caller's identity (forge's
``FORGE_WORKER_IDENTITY``) or, when it has none, the same ``{pid}@{hostname}``
string the SDK would have defaulted to — stamping must never cost the
process-identifying half of the answer::

    desktop-forge-worker-1@bb64d88
    desktop-forge-worker-1@bb64d88-dirty
    12345@desktop@bb64d88

The ``-dirty`` suffix is load-bearing rather than cosmetic: because the worker
execs the live tree, a launch from a modified tree means the commit alone does
not describe the loaded code, and the suffix says so.

Failure policy
--------------
Version discovery is best-effort and total: a non-repo working directory, a
missing ``git``, a timeout, or a non-zero exit all yield ``None``, and ``None``
propagates as "do not stamp" — the identity is left exactly as the caller passed
it (``None`` staying ``None`` so the SDK default applies). Version stamping must
never be the reason a worker fails to start, and a version that cannot be
verified is never invented.

Structure
---------
Functional core / imperative shell: :func:`compose_identity` is the pure
composition rule (fully table-testable, no clock/host/subprocess), while
:func:`code_version` and :func:`stamped_worker_identity` are the shell that runs
``git`` and reads the pid/hostname. This module is worker-startup code — it is
deliberately *not* part of ``sax_platform.temporal``'s eager import set, since
``subprocess`` and ``socket`` have no business in a workflow sandbox's import
graph.
"""

import os
import socket
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Final

__all__ = [
    "code_version",
    "compose_identity",
    "stamped_worker_identity",
]

# Generous for a local `git rev-parse` / `git status`, short enough that a wedged
# git cannot stall worker startup: on timeout the version is simply unknown.
_GIT_TIMEOUT_SECONDS: Final = 5.0


# ---------------------------------------------------------------------------
# Functional core
# ---------------------------------------------------------------------------


def compose_identity(base: str | None, version: str | None, fallback_base: str) -> str | None:
    """Combine a worker identity with a code version.

    Args:
        base: The identity the caller asked for, or ``None`` to accept the
            SDK default.
        version: The discovered code version, or ``None`` when unknown.
        fallback_base: The identity to stamp onto when *base* is ``None`` —
            callers pass the SDK's own ``{pid}@{hostname}`` default so the
            stamped identity still names the process.

    Returns:
        ``None`` when there is nothing to say (no base, no version) — the caller
        passes that straight through and the SDK applies its default. An unknown
        *version* returns *base* unchanged: a version that was not verified is
        never stamped.
    """
    if version is None:
        return base
    return f"{base or fallback_base}@{version}"


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def _run_git(args: Sequence[str], cwd: Path) -> str | None:
    """Run ``git <args>`` in *cwd*, returning stripped stdout or ``None`` on any failure.

    ``None`` covers every way this can go wrong — ``git`` not on ``PATH`` or
    *cwd* gone (``OSError``), the call exceeding its timeout
    (``subprocess.SubprocessError``), and a non-zero exit (not a repository, no
    commits yet). Nothing propagates: the caller's contract is that an unknown
    version is a normal outcome, not an error.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def code_version(cwd: Path | None = None) -> str | None:
    """Return the short commit of the tree at *cwd*, suffixed ``-dirty`` if modified.

    Args:
        cwd: Directory to interrogate. Defaults to the process working
            directory, which for every worker is the repo root it was exec'd
            from.

    Returns:
        e.g. ``"bb64d88"`` or ``"bb64d88-dirty"``, or ``None`` when the version
        cannot be established. ``None`` is also returned when the commit is
        known but the dirty check failed: reporting a bare commit in that case
        would assert a clean tree that was never verified, which is precisely
        the claim the ``-dirty`` suffix exists to avoid.

    Never raises.
    """
    root = Path.cwd() if cwd is None else cwd
    commit = _run_git(("rev-parse", "--short", "HEAD"), root)
    if not commit:
        return None
    status = _run_git(("status", "--porcelain"), root)
    if status is None:
        return None
    return f"{commit}-dirty" if status else commit


def stamped_worker_identity(base: str | None = None) -> str | None:
    """Return *base* stamped with the launch-time code version (see module docstring).

    The one call each worker's composition root makes, immediately before it
    connects: ``identity = stamped_worker_identity(identity)``. When the version
    is unknown the argument is returned unchanged, so a worker whose repo cannot
    be interrogated still starts with exactly the identity it would have had.

    ``{pid}@{hostname}`` mirrors the Temporal SDK's own default identity, and is
    used as the base only when the caller supplied none.
    """
    return compose_identity(base, code_version(), f"{os.getpid()}@{socket.gethostname()}")
