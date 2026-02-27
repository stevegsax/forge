"""Git worktree management for Forge.

Provides functions to create, remove, and manage git worktrees for task isolation.
Each task gets its own worktree branched from the base branch, enabling parallel
independent work without conflicts.

Design follows Function Core / Imperative Shell:
- Pure functions: worktree_path, branch_name, commit_message, _validate_task_id
- Subprocess wrapper: _run_git (thin, never raises on non-zero; uses SubprocessResult)
- Imperative shell: create_worktree, remove_worktree, commit_changes, etc.
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


def create_worktree(repo_root: Path, task_id: str, base_branch: str = "main") -> Path:
    """Create a new git worktree for a task.

    Creates branch ``forge/<task_id>`` from *base_branch* and checks it out
    into ``<repo_root>/.forge-worktrees/<task_id>``.

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

    result = _run_git("worktree", "add", "-b", br_name, str(wt_path), base_branch, cwd=repo_root)
    if not result.ok:
        msg = f"Failed to create worktree for task {task_id!r}: {result.stderr}"
        raise WorktreeCreateError(msg)

    return wt_path


def remove_worktree(repo_root: Path, task_id: str, *, force: bool = False) -> None:
    """Remove a git worktree and its associated branch.

    Args:
        repo_root: Path to the repository root.
        task_id: Unique task identifier.
        force: If True, remove even if the worktree has uncommitted changes.

    Raises:
        WorktreeRemoveError: If the worktree could not be removed.
    """
    wt_path = worktree_path(repo_root, task_id)
    br_name = branch_name(task_id)

    remove_args = ["worktree", "remove", str(wt_path)]
    if force:
        remove_args.append("--force")

    result = _run_git(*remove_args, cwd=repo_root)
    if not result.ok:
        msg = f"Failed to remove worktree for task {task_id!r}: {result.stderr}"
        raise WorktreeRemoveError(msg)

    # Clean up the branch. Failure is acceptable (branch may not exist or
    # may have already been deleted), so we intentionally ignore the result.
    delete_flag = "-D" if force else "-d"
    _run_git("branch", delete_flag, br_name, cwd=repo_root)


def commit_changes(
    repo_root: Path,
    task_id: str,
    status: str,
    file_paths: list[str] | None = None,
    message: str | None = None,
) -> str:
    """Stage and commit changes in a task's worktree.

    Args:
        repo_root: Path to the repository root.
        task_id: Unique task identifier.
        status: Status string included in the commit message.
        file_paths: Specific files to stage. If ``None``, stages all changes.
        message: Override the auto-generated commit message.

    Returns:
        The commit SHA.

    Raises:
        CommitError: If staging or committing fails.
    """
    wt_path = worktree_path(repo_root, task_id)

    # Stage
    if file_paths:
        result = _run_git("add", "--", *file_paths, cwd=wt_path)
    else:
        result = _run_git("add", "-A", cwd=wt_path)

    if not result.ok:
        msg = f"Failed to stage changes for task {task_id!r}: {result.stderr}"
        raise CommitError(msg)

    # Check if there's anything to commit
    diff_result = _run_git("diff", "--cached", "--quiet", cwd=wt_path)
    if diff_result.ok:
        msg = f"Nothing to commit for task {task_id!r}"
        raise CommitError(msg)

    # Commit
    msg_text = message if message is not None else commit_message(task_id, status)
    result = _run_git("commit", "-m", msg_text, cwd=wt_path)
    if not result.ok:
        msg = f"Failed to commit for task {task_id!r}: {result.stderr}"
        raise CommitError(msg)

    # Get the commit SHA
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
