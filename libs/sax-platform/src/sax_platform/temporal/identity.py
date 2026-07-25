"""The launch-time code version: stamp it into the worker identity, and gate prod on it.

Why this exists
---------------
A Python worker binds its code at import: the modules loaded when the process
started are the modules it will run until it is restarted. The working tree does
not stand still — a worker ``exec``s ``uv run`` straight out of the checkout it
was launched from (D99), so that checkout can be several commits (or a
half-finished edit) ahead of any given running process. Nothing in Temporal
records which code a poller is actually executing: a static identity like
``prod-forge-worker-1`` names the lane and the slot, not the build.

Two consumers, one fact. :func:`stamped_worker_identity` publishes the version
so the running code is *queryable*; :func:`require_clean_prod_code` refuses to
start a production worker whose version is dirty or unknown, so production code
is *committed* (D103). Both rest on :func:`code_version`.

Stamping the git version captured *once at startup* into the identity makes the
server the authority::

    temporal task-queue describe --task-queue forge-task-queue

lists each poller's identity, so the answer to "which code is this worker
running?" is a query rather than a guess.

Shape of the stamped identity
-----------------------------
``<base>@<version>``, where *base* is the caller's identity
(``FORGE_WORKER_IDENTITY`` / ``--worker-identity``, which every supervised
worker sets to its lane: ``prod-forge-worker-1``, ``dev-ocr-worker``) or, when
it has none, the same ``{pid}@{hostname}`` string the SDK would have defaulted
to — stamping must never cost the process-identifying half of the answer::

    prod-forge-worker-1@bb64d88
    prod-forge-worker-1@bb64d88-dirty
    dev-ocr-worker@bb64d88
    12345@buchla@bb64d88

The ``-dirty`` suffix is load-bearing rather than cosmetic: because the worker
execs the live tree, a launch from a modified tree means the commit alone does
not describe the loaded code, and the suffix says so.

Failure policy
--------------
Version discovery is best-effort and total: a non-repo working directory, a
missing ``git``, a timeout, or a non-zero exit all yield ``None``. The two
consumers then diverge deliberately, because the cost of being wrong differs:

* Stamping treats ``None`` as "do not stamp" — the identity is left exactly as
  the caller passed it. Version stamping must never be the reason a worker fails
  to start, and a version that cannot be verified is never invented.
* The production guard treats ``None`` as a refusal. On prod, "I cannot prove
  which commit this is" and "this is not a commit" are the same answer, and the
  safe response to both is not to start.

Structure
---------
Functional core / imperative shell: :func:`compose_identity` and
:func:`clean_prod_violation` are the pure rules (fully table-testable, no
clock/host/subprocess), while :func:`code_version`,
:func:`stamped_worker_identity`, and :func:`require_clean_prod_code` are the
shell that runs ``git``, reads the pid/hostname, and exits the process. This
module is worker-startup code — it is deliberately *not* part of
``sax_platform.temporal``'s eager import set, since ``subprocess`` and
``socket`` have no business in a workflow sandbox's import graph.
"""

import os
import socket
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from sax_platform.config import ForgeEnv

__all__ = [
    "EXIT_CONFIG_ERROR",
    "clean_prod_violation",
    "code_version",
    "compose_identity",
    "require_clean_prod_code",
    "stamped_worker_identity",
]

# Generous for a local `git rev-parse` / `git status`, short enough that a wedged
# git cannot stall worker startup: on timeout the version is simply unknown.
_GIT_TIMEOUT_SECONDS: Final = 5.0

#: Suffix :func:`code_version` appends when the checkout has uncommitted changes.
DIRTY_SUFFIX: Final = "-dirty"

#: ``EX_CONFIG`` (sysexits.h) — the exit code the D102 guard family uses for
#: "this process is not configured to run", kept identical here so an operator or
#: a supervisor sees one code for every refusal-to-start.
EXIT_CONFIG_ERROR: Final = 78

#: The sanctioned production deploy path (D103); named in the refusal message.
_PROD_DEPLOY_SCRIPT: Final = "deploy/prod-deploy.sh"


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


def clean_prod_violation(env: ForgeEnv, version: str | None) -> str | None:
    """Say why *env* must not run on *version*, or ``None`` when it may (D103).

    The pure rule behind :func:`require_clean_prod_code`. Production may run
    only code that a commit fully describes, because the worker execs its
    checkout: whatever is in that tree at launch *is* the running code.

    Args:
        env: The resolved target environment. Only ``prod`` is constrained —
            dev and test are working lanes where editing the checkout under a
            worker is the point.
        version: The result of :func:`code_version` — a short commit, a
            ``-dirty``-suffixed commit, or ``None`` when it could not be
            established.

    Returns:
        A short phrase naming the violation (for the operator-facing message),
        or ``None`` when startup may proceed. ``None`` version and a dirty
        version are both violations on prod: an unverifiable checkout is not
        evidence of a clean one.
    """
    if env is not ForgeEnv.PROD:
        return None
    if version is None:
        return "the code version could not be determined (no git, not a repository, or git failed)"
    if version.endswith(DIRTY_SUFFIX):
        return f"the checkout has uncommitted changes ({version})"
    return None


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


def require_clean_prod_code(env: ForgeEnv, cwd: Path | None = None) -> None:
    """Refuse to start a production worker on uncommitted or unverifiable code (D103).

    Called by each worker's composition root immediately after it resolves
    ``FORGE_ENV`` and before it touches a database or Temporal — the same
    position, and the same exit code (78, ``EX_CONFIG``), as the D102
    environment guard, so every "not configured to run" refusal looks alike.

    The mechanism it defends: a worker ``exec``s ``uv run`` inside its checkout,
    so the tree's contents at launch *are* the running code. On a live working
    tree that made an uncommitted edit deployable by accident — it happened
    (2026-07-25: an installer run on a dirty tree put uncommitted code into
    production, caught only afterwards by the ``-dirty`` identity stamp). D103's
    answer is a pinned worktree deployed by ``deploy/prod-deploy.sh``; this guard
    is what makes the pin non-optional, since a pin nobody enforces is a
    convention.

    Args:
        env: The resolved environment. Non-prod returns immediately — dev and
            test workers are *supposed* to run edited trees.
        cwd: Checkout to interrogate; defaults to the process working directory,
            which is the checkout the worker was exec'd from.

    Raises:
        SystemExit: with code 78 on prod when the version is dirty or unknown,
            after writing the reason and the fix to stderr. stderr rather than
            the logger on purpose: this runs before the workers configure
            logging, and launchd captures stderr to the agent's log either way.
    """
    violation = clean_prod_violation(env, code_version(cwd))
    if violation is None:
        return

    checkout = Path.cwd() if cwd is None else cwd
    print(
        f"FORGE_ENV=prod, but {violation}.\n"
        f"Checkout inspected: {checkout}\n\n"
        "A worker runs `uv run` out of its checkout, so that tree's contents at "
        "launch ARE the running code: starting here would put code into "
        "production that no commit describes, and nothing could later say what "
        "ran.\n\n"
        f"Deploy production with `{_PROD_DEPLOY_SCRIPT} <ref>`, which pins the "
        "production worktree to a commit, syncs it, and restarts the workers "
        "(D103). To inspect the current state instead, run `git status` in the "
        "checkout above.\n\n"
        "Refusing to start.",
        file=sys.stderr,
    )
    raise SystemExit(EXIT_CONFIG_ERROR)
