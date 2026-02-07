"""Tests for forge.activities.git_activities — git activity wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from forge.activities.git_activities import (
    commit_changes_activity,
    create_worktree_activity,
    remove_worktree_activity,
)
from forge.git import worktree_exists, worktree_path
from forge.models import (
    CommitChangesInput,
    CreateWorktreeInput,
    RemoveWorktreeInput,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# create_worktree_activity
# ---------------------------------------------------------------------------


class TestCreateWorktreeActivity:
    @pytest.mark.asyncio
    async def test_creates_worktree(self, git_repo: Path) -> None:
        result = await create_worktree_activity(
            CreateWorktreeInput(
                repo_root=str(git_repo),
                task_id="task-1",
            )
        )
        expected_path = worktree_path(git_repo, "task-1")
        assert result.worktree_path == str(expected_path)
        assert result.branch_name == "forge/task-1"
        assert expected_path.is_dir()

    @pytest.mark.asyncio
    async def test_worktree_contains_repo_files(self, git_repo: Path) -> None:
        result = await create_worktree_activity(
            CreateWorktreeInput(
                repo_root=str(git_repo),
                task_id="task-2",
            )
        )
        readme = worktree_path(git_repo, "task-2") / "README.md"
        assert readme.is_file()
        assert result.branch_name == "forge/task-2"


# ---------------------------------------------------------------------------
# remove_worktree_activity
# ---------------------------------------------------------------------------


class TestRemoveWorktreeActivity:
    @pytest.mark.asyncio
    async def test_removes_existing_worktree(self, git_repo: Path) -> None:
        await create_worktree_activity(
            CreateWorktreeInput(repo_root=str(git_repo), task_id="rm-task")
        )
        assert worktree_exists(git_repo, "rm-task")

        await remove_worktree_activity(
            RemoveWorktreeInput(repo_root=str(git_repo), task_id="rm-task")
        )
        assert not worktree_exists(git_repo, "rm-task")


# ---------------------------------------------------------------------------
# commit_changes_activity
# ---------------------------------------------------------------------------


class TestCommitChangesActivity:
    @pytest.mark.asyncio
    async def test_commits_and_returns_sha(self, git_repo: Path) -> None:
        await create_worktree_activity(
            CreateWorktreeInput(repo_root=str(git_repo), task_id="commit-task")
        )
        wt = worktree_path(git_repo, "commit-task")
        (wt / "output.py").write_text("print('hello')\n")

        result = await commit_changes_activity(
            CommitChangesInput(
                repo_root=str(git_repo),
                task_id="commit-task",
                status="success",
            )
        )
        assert len(result.commit_sha) == 40
        assert all(c in "0123456789abcdef" for c in result.commit_sha)
