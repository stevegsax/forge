"""Tests for worker identity version stamping (``sax_platform.temporal.identity``).

Two layers, matching the module's Functional-Core / Imperative-Shell split:

* **Composition** — ``compose_identity`` is pure, so it is a plain table: what a
  present/absent base and a present/absent version compose to, and that the
  fallback base is consulted only when the caller supplied none.
* **Discovery** — ``code_version`` and ``stamped_worker_identity`` shell out to
  ``git``, so they run against real temporary repositories (clean, modified,
  untracked) and real failure modes: a directory that is not a repository, a
  ``git`` that is not on ``PATH``, a working directory that does not exist, a
  ``git`` that exits non-zero on the dirty check, and a timeout. The contract
  under test in every failure case is the same: ``None``, never an exception —
  version stamping must not be able to stop a worker from starting.
"""

import os
import socket
import subprocess
from pathlib import Path

import pytest

from sax_platform.temporal import identity
from sax_platform.temporal.identity import (
    code_version,
    compose_identity,
    stamped_worker_identity,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path) -> str:
    """Run a real ``git`` command that is expected to succeed; return its stdout."""
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A temporary git repository with one commit on ``main``.

    Author identity is configured locally so the commit succeeds on a machine
    (or CI runner) with no global git config.
    """
    _git("init", "-b", "main", cwd=tmp_path)
    _git("config", "user.email", "test@sax.test", cwd=tmp_path)
    _git("config", "user.name", "Platform Test", cwd=tmp_path)
    (tmp_path / "tracked.txt").write_text("original\n")
    _git("add", "tracked.txt", cwd=tmp_path)
    _git("commit", "-m", "initial", cwd=tmp_path)
    return tmp_path


def _head_short_sha(repo: Path) -> str:
    return _git("rev-parse", "--short", "HEAD", cwd=repo)


@pytest.fixture
def git_only_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Replace ``git`` on ``PATH`` with a stub: ``rev-parse`` succeeds, all else fails.

    Exercises the "commit known, dirty check failed" branch through the real
    subprocess plumbing rather than by patching it out.
    """
    bin_dir = tmp_path / "stub-bin"
    bin_dir.mkdir()
    stub = bin_dir / "git"
    stub.write_text('#!/bin/sh\ncase "$1" in\n  rev-parse) echo deadbee ;;\n  *) exit 1 ;;\nesac\n')
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    return bin_dir


# ---------------------------------------------------------------------------
# Functional core — composition
# ---------------------------------------------------------------------------


class TestComposeIdentity:
    @pytest.mark.parametrize(
        ("base", "version", "expected"),
        [
            ("desktop-forge-worker-1", "bb64d88", "desktop-forge-worker-1@bb64d88"),
            ("desktop-forge-worker-1", "bb64d88-dirty", "desktop-forge-worker-1@bb64d88-dirty"),
            (None, "bb64d88", "12345@desktop@bb64d88"),
            ("desktop-forge-worker-1", None, "desktop-forge-worker-1"),
            (None, None, None),
        ],
    )
    def test_composition_table(
        self, base: str | None, version: str | None, expected: str | None
    ) -> None:
        assert compose_identity(base, version, "12345@desktop") == expected

    def test_empty_base_falls_back(self) -> None:
        # An empty string is as uninformative as None; the SDK-style default wins.
        assert compose_identity("", "bb64d88", "12345@desktop") == "12345@desktop@bb64d88"


# ---------------------------------------------------------------------------
# Imperative shell — version discovery against real repositories
# ---------------------------------------------------------------------------


class TestCodeVersion:
    def test_clean_repo_returns_short_commit(self, git_repo: Path) -> None:
        assert code_version(git_repo) == _head_short_sha(git_repo)

    def test_modified_tracked_file_appends_dirty(self, git_repo: Path) -> None:
        (git_repo / "tracked.txt").write_text("edited\n")
        assert code_version(git_repo) == f"{_head_short_sha(git_repo)}-dirty"

    def test_untracked_file_appends_dirty(self, git_repo: Path) -> None:
        # `git status --porcelain` reports untracked files, and that is deliberate:
        # a launch-time tree with extra files is not the committed tree.
        (git_repo / "scratch.py").write_text("x = 1\n")
        assert code_version(git_repo) == f"{_head_short_sha(git_repo)}-dirty"

    def test_defaults_to_process_working_directory(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(git_repo)
        assert code_version() == _head_short_sha(git_repo)

    def test_non_repository_directory_returns_none(self, tmp_path: Path) -> None:
        assert code_version(tmp_path) is None

    def test_missing_directory_returns_none(self, tmp_path: Path) -> None:
        assert code_version(tmp_path / "does-not-exist") is None

    def test_missing_git_returns_none(
        self, git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        empty_bin = tmp_path / "empty-bin"
        empty_bin.mkdir()
        monkeypatch.setenv("PATH", str(empty_bin))
        assert code_version(git_repo) is None

    def test_dirty_check_failure_returns_none(self, git_only_on_path: Path, tmp_path: Path) -> None:
        # The commit is known but its cleanliness is not: reporting a bare commit
        # would assert a clean tree nobody verified.
        assert code_version(tmp_path) is None

    def test_timeout_returns_none(self, git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _timeout(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd="git", timeout=5.0)

        monkeypatch.setattr(identity.subprocess, "run", _timeout)
        assert code_version(git_repo) is None


class TestStampedWorkerIdentity:
    def test_stamps_supplied_base(self, git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(git_repo)
        expected = f"desktop-forge-worker-1@{_head_short_sha(git_repo)}"
        assert stamped_worker_identity("desktop-forge-worker-1") == expected

    def test_stamps_sdk_default_base_when_none(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(git_repo)
        expected = f"{os.getpid()}@{socket.gethostname()}@{_head_short_sha(git_repo)}"
        assert stamped_worker_identity() == expected

    def test_dirty_tree_stamps_dirty_suffix(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (git_repo / "tracked.txt").write_text("edited\n")
        monkeypatch.chdir(git_repo)
        expected = f"w1@{_head_short_sha(git_repo)}-dirty"
        assert stamped_worker_identity("w1") == expected

    def test_unknown_version_returns_base_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert stamped_worker_identity("desktop-forge-worker-1") == "desktop-forge-worker-1"

    def test_unknown_version_and_no_base_stays_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # None must survive: the SDK then applies its own default identity.
        monkeypatch.chdir(tmp_path)
        assert stamped_worker_identity() is None


class TestPackageExport:
    def test_lazy_exports_resolve(self) -> None:
        from sax_platform import temporal

        assert temporal.stamped_worker_identity is stamped_worker_identity
        assert temporal.compose_identity is compose_identity
        assert temporal.code_version is code_version
