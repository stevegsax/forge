"""Tests for forge.git — git worktree management."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest

from forge.git import (
    CommitError,
    RepoDiscoveryError,
    WorktreeCreateError,
    WorktreeRemoveError,
    WorktreeResetError,
    _validate_task_id,
    branch_name,
    commit_changes,
    commit_message,
    create_worktree,
    discover_repo_root,
    remove_worktree,
    reset_worktree,
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

    def test_recreates_over_leftover_branch(self, git_repo: Path) -> None:
        """A crashed run that left only the branch behind does not block a rerun."""
        wt = create_worktree(git_repo, "leftover-branch")
        shutil.rmtree(wt)

        from forge.git import _run_git

        _run_git("worktree", "prune", cwd=git_repo)
        branches = _run_git("branch", "--list", "forge/leftover-branch", cwd=git_repo)
        assert "forge/leftover-branch" in branches.stdout

        recreated = create_worktree(git_repo, "leftover-branch")

        assert recreated == wt
        assert (recreated / "README.md").exists()

    def test_recreates_over_stale_registration(self, git_repo: Path) -> None:
        """A registration whose directory vanished is pruned, not fatal."""
        wt = create_worktree(git_repo, "stale-reg")
        shutil.rmtree(wt)

        recreated = create_worktree(git_repo, "stale-reg")

        assert recreated == wt
        assert (recreated / "README.md").exists()

    def test_recreates_over_crashed_leftover(self, git_repo: Path) -> None:
        """Branch, registration, and dirty directory left behind are all cleared."""
        wt = create_worktree(git_repo, "crashed-task")
        (wt / "committed.txt").write_text("from the crashed run\n")
        commit_changes(git_repo, "crashed-task", "success")
        (wt / "uncommitted.txt").write_text("dirty\n")

        recreated = create_worktree(git_repo, "crashed-task")

        assert recreated == wt
        assert (recreated / "README.md").exists()
        assert not (recreated / "committed.txt").exists()
        assert not (recreated / "uncommitted.txt").exists()

        from forge.git import _run_git

        # ``worktree add -B`` reset the branch back onto the base branch.
        assert (
            _run_git("rev-parse", "HEAD", cwd=recreated).stdout
            == _run_git("rev-parse", "main", cwd=git_repo).stdout
        )

    def test_raises_on_unregistered_leftover_directory(self, git_repo: Path) -> None:
        """A directory git does not know about is never deleted — it is an error."""
        stray = worktree_path(git_repo, "stray-dir")
        stray.mkdir(parents=True)
        (stray / "not-a-worktree.txt").write_text("who put this here\n")

        with pytest.raises(WorktreeCreateError):
            create_worktree(git_repo, "stray-dir")

        assert (stray / "not-a-worktree.txt").exists()


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

    def test_noop_on_nonexistent(self, git_repo: Path) -> None:
        """Removing a worktree that was never created is a no-op, not an error."""
        remove_worktree(git_repo, "ghost")

        assert not worktree_exists(git_repo, "ghost")

    def test_noop_on_repeated_removal(self, git_repo: Path) -> None:
        """A retried removal after one that landed succeeds (activity idempotency)."""
        create_worktree(git_repo, "retry-rm")
        remove_worktree(git_repo, "retry-rm")

        remove_worktree(git_repo, "retry-rm")

        assert not worktree_exists(git_repo, "retry-rm")

    def test_noop_on_stale_registration(self, git_repo: Path) -> None:
        """A registration left behind by a vanished directory is pruned, not fatal."""
        wt = create_worktree(git_repo, "stale-rm")
        shutil.rmtree(wt)

        remove_worktree(git_repo, "stale-rm")

        from forge.git import _run_git

        listing = _run_git("worktree", "list", "--porcelain", cwd=git_repo)
        assert str(wt) not in listing.stdout

    def test_raises_on_dirty_worktree_without_force(self, git_repo: Path) -> None:
        """A real removal failure still raises."""
        wt = create_worktree(git_repo, "dirty-nonforce")
        (wt / "uncommitted.txt").write_text("dirty content\n")

        with pytest.raises(WorktreeRemoveError):
            remove_worktree(git_repo, "dirty-nonforce")

        assert wt.is_dir()

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

    def test_custom_message_override(self, git_repo: Path) -> None:
        wt = create_worktree(git_repo, "custom-msg")
        (wt / "file.txt").write_text("content\n")

        commit_changes(git_repo, "custom-msg", "success", message="step 1: create module")

        from forge.git import _run_git

        result = _run_git("log", "-1", "--format=%s", cwd=wt)
        assert result.stdout == "step 1: create module"

    def test_retry_after_landed_commit_returns_same_sha(self, git_repo: Path) -> None:
        """A retried commit whose commit already landed returns HEAD's SHA."""
        wt = create_worktree(git_repo, "retry-task")
        (wt / "hello.py").write_text("print('hello')\n")

        first = commit_changes(git_repo, "retry-task", "success")
        second = commit_changes(git_repo, "retry-task", "success")

        assert second == first

        from forge.git import _run_git

        # No empty second commit was created.
        assert _run_git("rev-list", "--count", "HEAD", cwd=wt).stdout == "2"

    def test_retry_with_custom_message_returns_same_sha(self, git_repo: Path) -> None:
        wt = create_worktree(git_repo, "retry-custom")
        (wt / "file.txt").write_text("content\n")

        first = commit_changes(git_repo, "retry-custom", "success", message="step 1: create module")
        second = commit_changes(
            git_repo, "retry-custom", "success", message="step 1: create module"
        )

        assert second == first

    def test_raises_when_head_message_differs(self, git_repo: Path) -> None:
        """Nothing staged and a different HEAD message is a genuine empty commit."""
        wt = create_worktree(git_repo, "stale-head")
        (wt / "file.txt").write_text("content\n")
        commit_changes(git_repo, "stale-head", "success")

        with pytest.raises(CommitError, match="Nothing to commit"):
            commit_changes(git_repo, "stale-head", "failure")


# ---------------------------------------------------------------------------
# worktree_exists
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# reset_worktree
# ---------------------------------------------------------------------------


class TestResetWorktree:
    def test_discards_uncommitted_changes(self, git_repo: Path) -> None:
        wt = create_worktree(git_repo, "reset-task")
        (wt / "dirty.txt").write_text("dirty\n")

        reset_worktree(git_repo, "reset-task")

        assert not (wt / "dirty.txt").exists()

    def test_preserves_committed_changes(self, git_repo: Path) -> None:
        wt = create_worktree(git_repo, "reset-keep")
        (wt / "committed.txt").write_text("keep\n")
        commit_changes(git_repo, "reset-keep", "step1")

        # Add dirty changes then reset
        (wt / "dirty.txt").write_text("dirty\n")
        reset_worktree(git_repo, "reset-keep")

        assert (wt / "committed.txt").exists()
        assert not (wt / "dirty.txt").exists()

    def test_raises_on_nonexistent_worktree(self, git_repo: Path) -> None:
        with pytest.raises(WorktreeResetError):
            reset_worktree(git_repo, "ghost")


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
