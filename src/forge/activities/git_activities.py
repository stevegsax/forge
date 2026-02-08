"""Git-related Temporal activities for Forge.

Thin wrappers around forge.git functions so that subprocess-based git
operations run as Temporal activities (outside the deterministic workflow).
"""

from __future__ import annotations

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


@activity.defn
async def create_worktree_activity(input: CreateWorktreeInput) -> CreateWorktreeOutput:
    """Create a git worktree for a task."""
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
    return CommitChangesOutput(commit_sha=sha)


@activity.defn
async def reset_worktree_activity(input: ResetWorktreeInput) -> None:
    """Reset a worktree to HEAD, discarding uncommitted changes."""
    reset_worktree(
        repo_root=Path(input.repo_root),
        task_id=input.task_id,
    )
