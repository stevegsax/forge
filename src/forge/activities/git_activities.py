"""Git-related Temporal activities for Forge.

Thin wrappers around forge.git functions so that subprocess-based git
operations run as Temporal activities (outside the deterministic workflow).
"""

from __future__ import annotations

import logging
from pathlib import Path

from temporalio import activity

from forge.git import branch_name, commit_changes, create_worktree, remove_worktree, reset_worktree
from forge.models import (
    CommitChangesInput,
    CommitChangesOutput,
    CreateWorktreeInput,
    CreateWorktreeOutput,
    RemoveWorktreeInput,
    ResetWorktreeInput,
)

logger = logging.getLogger(__name__)


@activity.defn
async def create_worktree_activity(input: CreateWorktreeInput) -> CreateWorktreeOutput:
    """Create a git worktree for a task."""
    logger.info("Create worktree: task_id=%s base=%s", input.task_id, input.base_branch)
    wt_path = create_worktree(
        repo_root=Path(input.repo_root),
        task_id=input.task_id,
        base_branch=input.base_branch,
    )
    return CreateWorktreeOutput(
        worktree_path=str(wt_path),
        branch_name=branch_name(input.task_id),
    )


@activity.defn
async def remove_worktree_activity(input: RemoveWorktreeInput) -> None:
    """Remove a git worktree and its associated branch."""
    logger.info("Remove worktree: task_id=%s", input.task_id)
    remove_worktree(
        repo_root=Path(input.repo_root),
        task_id=input.task_id,
        force=input.force,
    )


@activity.defn
async def commit_changes_activity(input: CommitChangesInput) -> CommitChangesOutput:
    """Stage and commit changes in a task's worktree."""
    sha = commit_changes(
        repo_root=Path(input.repo_root),
        task_id=input.task_id,
        status=input.status,
        file_paths=input.file_paths,
        message=input.message,
    )
    logger.info("Commit: task_id=%s status=%s sha=%s", input.task_id, input.status, sha)
    return CommitChangesOutput(commit_sha=sha)


@activity.defn
async def reset_worktree_activity(input: ResetWorktreeInput) -> None:
    """Reset a worktree to HEAD, discarding uncommitted changes."""
    logger.info("Reset worktree: task_id=%s", input.task_id)
    reset_worktree(
        repo_root=Path(input.repo_root),
        task_id=input.task_id,
    )
