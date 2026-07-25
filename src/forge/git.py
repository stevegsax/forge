"""Git worktree management for Forge.

Provides functions to create, remove, and manage git worktrees for task isolation.
Each task gets its own worktree branched from the base branch, enabling parallel
independent work without conflicts.

Design follows Function Core / Imperative Shell:
- Pure functions: worktree_path, branch_name, commit_message, _validate_task_id,
  _registered_worktree_paths
- Subprocess wrapper: _run_git (thin, never raises on non-zero; uses SubprocessResult)
- Imperative shell: create_worktree, remove_worktree, commit_changes, etc.

The worktree and commit seams are idempotent so they survive a Temporal activity
retry and a crashed prior run: creating recreates over leftovers, committing
recognises a commit that already landed, and removing an absent worktree is a
no-op.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from forge.subprocess_result import SubprocessResult

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

SUBPROCESS_TIMEOUT_SECONDS = 30


class ForgeGitError(Exception):
    """Base exception for all Forge git operations."""


class WorktreeCreateError(ForgeGitError):
    """Failed to create a git worktree."""


class WorktreeRemoveError(ForgeGitError):
    """Failed to remove a git worktree."""


class WorktreeResetError(ForgeGitError):
    """Failed to reset a git worktree."""


class CommitError(ForgeGitError):
    """Failed to commit changes."""


class RepoDiscoveryError(ForgeGitError):
    """Failed to discover the git repository root."""


# ---------------------------------------------------------------------------
# Task-ID validation
# ---------------------------------------------------------------------------

_TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_task_id(task_id: str) -> None:
    """Reject task IDs with unsafe characters.

    Valid task IDs start with an alphanumeric character and contain only
    alphanumerics, hyphens, underscores, and dots.

    Raises:
        ValueError: If the task_id is empty or contains unsafe characters.
    """
    if not task_id:
        msg = "task_id must not be empty"
        raise ValueError(msg)
    if not _TASK_ID_PATTERN.match(task_id):
        msg = (
            f"Invalid task_id {task_id!r}: must start with an alphanumeric character "
            "and contain only alphanumerics, hyphens, underscores, and dots."
        )
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

_WORKTREE_DIR = ".forge-worktrees"


def worktree_path(repo_root: Path, task_id: str) -> Path:
    """Compute the worktree directory path for a task.

    Returns ``<repo_root>/.forge-worktrees/<task_id>``.
    """
    _validate_task_id(task_id)
    return repo_root / _WORKTREE_DIR / task_id


def branch_name(task_id: str) -> str:
    """Compute the branch name for a task.

    Returns ``forge/<task_id>``.
    """
    _validate_task_id(task_id)
    return f"forge/{task_id}"


def commit_message(task_id: str, status: str) -> str:
    """Build a standardized commit message for a task.

    Format: ``forge(<task_id>): <status>``
    """
    _validate_task_id(task_id)
    return f"forge({task_id}): {status}"


_WORKTREE_LINE_PREFIX = "worktree "


def _registered_worktree_paths(porcelain_stdout: str) -> frozenset[str]:
    """Extract the worktree paths from ``git worktree list --porcelain`` output.

    The porcelain format opens each record with a ``worktree <path>`` line;
    every other line is an attribute of the record and is ignored here.
    """
    return frozenset(
        line[len(_WORKTREE_LINE_PREFIX) :]
        for line in porcelain_stdout.splitlines()
        if line.startswith(_WORKTREE_LINE_PREFIX)
    )


# ---------------------------------------------------------------------------
# Subprocess wrapper
# ---------------------------------------------------------------------------


def _run_git(*args: str, cwd: Path) -> SubprocessResult:
    """Execute ``git <args>`` and return the result.

    Does **not** raise on non-zero exit codes — callers decide what constitutes
    an error.
    """
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    return SubprocessResult(
        returncode=result.returncode,
        stdout=result.stdout.strip(),
        stderr=result.stderr.strip(),
    )


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def discover_repo_root(path: Path | None = None) -> Path:
    """Discover the git repository root.

    Args:
        path: Directory to start searching from. Defaults to the current
            working directory.

    Returns:
        Absolute path to the repository root.

    Raises:
        RepoDiscoveryError: If the path is not inside a git repository.
    """
    cwd = path or Path.cwd()
    result = _run_git("rev-parse", "--show-toplevel", cwd=cwd)
    if not result.ok:
        msg = f"Not a git repository (or any parent up to mount point): {cwd}"
        raise RepoDiscoveryError(msg)
    return Path(result.stdout)


def _is_registered_worktree(repo_root: Path, wt_path: Path) -> bool:
    """Report whether git currently registers *wt_path* as a worktree of the repo."""
    result = _run_git("worktree", "list", "--porcelain", cwd=repo_root)
    if not result.ok:
        return False
    return str(wt_path) in _registered_worktree_paths(result.stdout)


def _delete_branch(repo_root: Path, br_name: str, *, force: bool) -> None:
    """Delete a task branch, best effort.

    Failure is acceptable (the branch may not exist, may already have been
    deleted, or may be unmerged under a non-forced delete), so the result is
    intentionally ignored.
    """
    _run_git("branch", "-D" if force else "-d", br_name, cwd=repo_root)


def create_worktree(repo_root: Path, task_id: str, base_branch: str = "main") -> Path:
    """Create a git worktree for a task, recreating over a crashed prior run.

    Creates branch ``forge/<task_id>`` from *base_branch* and checks it out
    into ``<repo_root>/.forge-worktrees/<task_id>``.

    Leftovers from a prior run that died mid-task are cleared first: stale
    registrations are pruned, a worktree still registered at the same path is
    force-removed, and an existing branch is reset rather than rejected
    (``worktree add -B``). Worktrees are disposable, and Temporal workflow-id
    uniqueness already excludes two live runs of the same task id, so a
    leftover is always debris rather than concurrent work. A leftover
    *directory* that git does not know about is deliberately left alone and
    surfaces as a create error — this function never deletes untracked paths.

    Args:
        repo_root: Path to the repository root.
        task_id: Unique task identifier.
        base_branch: Branch to create the worktree from.

    Returns:
        Path to the created worktree directory.

    Raises:
        WorktreeCreateError: If the worktree could not be created.
    """
    wt_path = worktree_path(repo_root, task_id)
    br_name = branch_name(task_id)

    # Clear debris from a crashed prior run: prune drops registrations whose
    # directories are gone; anything still registered at this path is removed.
    _run_git("worktree", "prune", cwd=repo_root)
    if _is_registered_worktree(repo_root, wt_path):
        _run_git("worktree", "remove", "--force", str(wt_path), cwd=repo_root)

    result = _run_git("worktree", "add", "-B", br_name, str(wt_path), base_branch, cwd=repo_root)
    if not result.ok:
        msg = f"Failed to create worktree for task {task_id!r}: {result.stderr}"
        raise WorktreeCreateError(msg)

    return wt_path


def remove_worktree(repo_root: Path, task_id: str, *, force: bool = False) -> None:
    """Remove a git worktree and its associated branch.

    Removal is idempotent: when the worktree directory is already gone — a
    retried removal, or a removal that landed before the activity was retried —
    the stale registration (if any) is pruned, the branch delete still runs, and
    the call returns successfully. Genuine removal failures, such as a worktree
    with uncommitted changes and *force* unset, still raise.

    Args:
        repo_root: Path to the repository root.
        task_id: Unique task identifier.
        force: If True, remove even if the worktree has uncommitted changes.

    Raises:
        WorktreeRemoveError: If an existing worktree could not be removed.
    """
    wt_path = worktree_path(repo_root, task_id)
    br_name = branch_name(task_id)

    if not wt_path.is_dir():
        # Nothing left to remove. Prune clears a registration whose directory
        # has vanished; without the directory `worktree remove` would only fail.
        _run_git("worktree", "prune", cwd=repo_root)
        _delete_branch(repo_root, br_name, force=force)
        return

    remove_args = ["worktree", "remove", str(wt_path)]
    if force:
        remove_args.append("--force")

    result = _run_git(*remove_args, cwd=repo_root)
    if not result.ok:
        msg = f"Failed to remove worktree for task {task_id!r}: {result.stderr}"
        raise WorktreeRemoveError(msg)

    _delete_branch(repo_root, br_name, force=force)


def commit_changes(
    repo_root: Path,
    task_id: str,
    status: str,
    file_paths: list[str] | None = None,
    message: str | None = None,
) -> str:
    """Stage and commit changes in a task's worktree.

    Committing is idempotent under activity retry: when nothing is staged, the
    intended commit message is compared with HEAD's. A match means the commit
    landed on a prior attempt, so HEAD's SHA is returned instead of failing; a
    mismatch means there was genuinely nothing to commit, which is still an
    error.

    Args:
        repo_root: Path to the repository root.
        task_id: Unique task identifier.
        status: Status string included in the commit message.
        file_paths: Specific files to stage. If ``None``, stages all changes.
        message: Override the auto-generated commit message.

    Returns:
        The commit SHA — the new commit's, or HEAD's when the commit had
        already landed.

    Raises:
        CommitError: If staging or committing fails, or if there is nothing to
            commit and HEAD is not the intended commit.
    """
    wt_path = worktree_path(repo_root, task_id)
    msg_text = message if message is not None else commit_message(task_id, status)

    # Stage
    if file_paths:
        result = _run_git("add", "--", *file_paths, cwd=wt_path)
    else:
        result = _run_git("add", "-A", cwd=wt_path)

    if not result.ok:
        msg = f"Failed to stage changes for task {task_id!r}: {result.stderr}"
        raise CommitError(msg)

    # Nothing staged: either this attempt is a retry of a commit that already
    # landed (HEAD carries the intended message) or there is genuinely nothing
    # to commit.
    diff_result = _run_git("diff", "--cached", "--quiet", cwd=wt_path)
    if diff_result.ok:
        head_message = _run_git("log", "-1", "--format=%B", cwd=wt_path)
        if not head_message.ok or head_message.stdout.strip() != msg_text.strip():
            msg = f"Nothing to commit for task {task_id!r}"
            raise CommitError(msg)
        return _head_sha(wt_path, task_id)

    # Commit
    result = _run_git("commit", "-m", msg_text, cwd=wt_path)
    if not result.ok:
        msg = f"Failed to commit for task {task_id!r}: {result.stderr}"
        raise CommitError(msg)

    return _head_sha(wt_path, task_id)


def _head_sha(wt_path: Path, task_id: str) -> str:
    """Read the worktree's HEAD commit SHA.

    Raises:
        CommitError: If the SHA could not be read.
    """
    sha_result = _run_git("rev-parse", "HEAD", cwd=wt_path)
    if not sha_result.ok:
        msg = f"Failed to read commit SHA for task {task_id!r}: {sha_result.stderr}"
        raise CommitError(msg)

    return sha_result.stdout


def reset_worktree(repo_root: Path, task_id: str) -> None:
    """Reset a task's worktree to HEAD, discarding all uncommitted changes.

    Runs ``git reset --hard HEAD`` followed by ``git clean -fd`` in the
    worktree directory. Used for step-level retry in planned execution.

    Args:
        repo_root: Path to the repository root.
        task_id: Unique task identifier.

    Raises:
        WorktreeResetError: If the reset or clean operation fails.
    """
    wt_path = worktree_path(repo_root, task_id)

    if not wt_path.is_dir():
        msg = f"Worktree directory does not exist for task {task_id!r}: {wt_path}"
        raise WorktreeResetError(msg)

    result = _run_git("reset", "--hard", "HEAD", cwd=wt_path)
    if not result.ok:
        msg = f"Failed to reset worktree for task {task_id!r}: {result.stderr}"
        raise WorktreeResetError(msg)

    result = _run_git("clean", "-fd", cwd=wt_path)
    if not result.ok:
        msg = f"Failed to clean worktree for task {task_id!r}: {result.stderr}"
        raise WorktreeResetError(msg)


def worktree_exists(repo_root: Path, task_id: str) -> bool:
    """Check whether a worktree for the given task exists.

    Checks both that the directory is present on disk and that git
    recognises it as a worktree.
    """
    _validate_task_id(task_id)
    wt_path = worktree_path(repo_root, task_id)

    if not wt_path.is_dir():
        return False

    result = _run_git("worktree", "list", "--porcelain", cwd=repo_root)
    if not result.ok:
        return False

    return str(wt_path) in result.stdout


def list_worktrees(repo_root: Path) -> list[str]:
    """Return existing forge worktree task IDs.

    Args:
        repo_root: Path to the repository root.

    Returns:
        List of task IDs for existing forge worktrees.
    """
    result = _run_git("worktree", "list", "--porcelain", cwd=repo_root)
    if not result.ok:
        return []

    task_ids: list[str] = []
    worktree_prefix = str(repo_root / _WORKTREE_DIR) + "/"

    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            path = line[len("worktree ") :]
            if path.startswith(worktree_prefix):
                task_id = path[len(worktree_prefix) :]
                if task_id:
                    task_ids.append(task_id)

    return task_ids
