"""Tests for forge.git — git worktree management."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from forge.git import (
    CommitError,
    RepoDiscoveryError,
    WorktreeCreateError,
    WorktreeRemoveError,
    _validate_task_id,
    branch_name,
    commit_changes,
    commit_message,
    create_worktree,
    discover_repo_root,
    remove_worktree,
    worktree_exists,
    worktree_path,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


class TestWorktreePath:
    def test_returns_expected_path(self, git_repo: Path) -> None:
        result = worktree_path(git_repo, "task-1")
        assert result == git_repo / ".forge-worktrees" / "task-1"

    def test_rejects_invalid_task_id(self, git_repo: Path) -> None:
        with pytest.raises(ValueError):
            worktree_path(git_repo, "")


class TestBranchName:
    def test_returns_expected_name(self) -> None:
        assert branch_name("task-1") == "forge/task-1"

    def test_rejects_invalid_task_id(self) -> None:
        with pytest.raises(ValueError):
            branch_name("")


class TestCommitMessage:
    def test_format(self) -> None:
        assert commit_message("task-1", "success") == "forge(task-1): success"


class TestValidateTaskId:
    def test_valid_ids(self) -> None:
        for task_id in ["task-1", "abc", "A.B-C_D", "0123"]:
            _validate_task_id(task_id)  # should not raise

    def test_empty(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_task_id("")

    def test_leading_dot(self) -> None:
        with pytest.raises(ValueError, match="Invalid task_id"):
            _validate_task_id(".hidden")

    def test_leading_hyphen(self) -> None:
        with pytest.raises(ValueError, match="Invalid task_id"):
            _validate_task_id("-flag")

    def test_path_separator(self) -> None:
        with pytest.raises(ValueError, match="Invalid task_id"):
            _validate_task_id("a/b")

    def test_shell_metacharacter(self) -> None:
        with pytest.raises(ValueError, match="Invalid task_id"):
            _validate_task_id("task;rm -rf /")

    def test_space(self) -> None:
        with pytest.raises(ValueError, match="Invalid task_id"):
            _validate_task_id("task 1")


# ---------------------------------------------------------------------------
# discover_repo_root
# ---------------------------------------------------------------------------


class TestDiscoverRepoRoot:
    def test_finds_root(self, git_repo: Path) -> None:
        root = discover_repo_root(git_repo)
        assert root == git_repo

    def test_finds_root_from_subdirectory(self, git_repo: Path) -> None:
        sub = git_repo / "subdir"
        sub.mkdir()
        root = discover_repo_root(sub)
        assert root == git_repo

    def test_raises_for_non_repo(self, tmp_path: Path) -> None:
        non_repo = tmp_path / "not-a-repo"
        non_repo.mkdir()
        with pytest.raises(RepoDiscoveryError):
            discover_repo_root(non_repo)


# ---------------------------------------------------------------------------
# create_worktree
# ---------------------------------------------------------------------------


class TestCreateWorktree:
    def test_creates_worktree(self, git_repo: Path) -> None:
        wt = create_worktree(git_repo, "task-1")
        assert wt.is_dir()
        assert (wt / "README.md").exists()

    def test_creates_correct_branch(self, git_repo: Path) -> None:
        create_worktree(git_repo, "task-2")

        from forge.git import _run_git

        result = _run_git("branch", "--list", "forge/task-2", cwd=git_repo)
        assert "forge/task-2" in result.stdout

    def test_raises_on_duplicate(self, git_repo: Path) -> None:
        create_worktree(git_repo, "dup-task")
        with pytest.raises(WorktreeCreateError):
            create_worktree(git_repo, "dup-task")


# ---------------------------------------------------------------------------
# remove_worktree
# ---------------------------------------------------------------------------


class TestRemoveWorktree:
    def test_removes_directory_and_branch(self, git_repo: Path) -> None:
        wt = create_worktree(git_repo, "rm-task")
        assert wt.is_dir()

        remove_worktree(git_repo, "rm-task")

        assert not wt.is_dir()
        from forge.git import _run_git

        result = _run_git("branch", "--list", "forge/rm-task", cwd=git_repo)
        assert result.stdout == ""

    def test_raises_on_nonexistent(self, git_repo: Path) -> None:
        with pytest.raises(WorktreeRemoveError):
            remove_worktree(git_repo, "ghost")

    def test_force_removes_dirty_worktree(self, git_repo: Path) -> None:
        wt = create_worktree(git_repo, "dirty-task")
        (wt / "uncommitted.txt").write_text("dirty content\n")

        remove_worktree(git_repo, "dirty-task", force=True)
        assert not wt.is_dir()

    def test_can_recreate_after_remove(self, git_repo: Path) -> None:
        create_worktree(git_repo, "cycle-task")
        remove_worktree(git_repo, "cycle-task")

        wt = create_worktree(git_repo, "cycle-task")
        assert wt.is_dir()


# ---------------------------------------------------------------------------
# commit_changes
# ---------------------------------------------------------------------------


class TestCommitChanges:
    def test_commits_new_file(self, git_repo: Path) -> None:
        wt = create_worktree(git_repo, "commit-task")
        (wt / "hello.py").write_text("print('hello')\n")

        sha = commit_changes(git_repo, "commit-task", "success")
        assert len(sha) == 40  # full SHA

    def test_correct_message_format(self, git_repo: Path) -> None:
        wt = create_worktree(git_repo, "msg-task")
        (wt / "file.txt").write_text("content\n")

        commit_changes(git_repo, "msg-task", "success")

        from forge.git import _run_git

        result = _run_git("log", "-1", "--format=%s", cwd=wt)
        assert result.stdout == "forge(msg-task): success"

    def test_selective_file_staging(self, git_repo: Path) -> None:
        wt = create_worktree(git_repo, "select-task")
        (wt / "included.txt").write_text("yes\n")
        (wt / "excluded.txt").write_text("no\n")

        commit_changes(git_repo, "select-task", "partial", file_paths=["included.txt"])

        from forge.git import _run_git

        result = _run_git("diff", "--name-only", "HEAD~1", cwd=wt)
        assert "included.txt" in result.stdout
        assert "excluded.txt" not in result.stdout

    def test_raises_on_nothing_to_commit(self, git_repo: Path) -> None:
        create_worktree(git_repo, "empty-task")
        with pytest.raises(CommitError, match="Nothing to commit"):
            commit_changes(git_repo, "empty-task", "success")


# ---------------------------------------------------------------------------
# worktree_exists
# ---------------------------------------------------------------------------


class TestWorktreeExists:
    def test_true_when_present(self, git_repo: Path) -> None:
        create_worktree(git_repo, "exists-task")
        assert worktree_exists(git_repo, "exists-task") is True

    def test_false_when_absent(self, git_repo: Path) -> None:
        assert worktree_exists(git_repo, "nope") is False

    def test_false_after_removal(self, git_repo: Path) -> None:
        create_worktree(git_repo, "gone-task")
        remove_worktree(git_repo, "gone-task")
        assert worktree_exists(git_repo, "gone-task") is False
