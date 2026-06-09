"""Tests for forge.path_safety — read-path confinement to the worktree."""

from pathlib import Path

import pytest

from forge.path_safety import resolve_within


class TestResolveWithin:
    def test_simple_relative_path_is_allowed(self, tmp_path: Path) -> None:
        target = tmp_path / "file.txt"
        target.write_text("ok")
        assert resolve_within(tmp_path, "file.txt") == target.resolve()

    def test_nested_relative_path_is_allowed(self, tmp_path: Path) -> None:
        (tmp_path / "pkg").mkdir()
        target = tmp_path / "pkg" / "mod.py"
        target.write_text("ok")
        assert resolve_within(tmp_path, "pkg/mod.py") == target.resolve()

    def test_nonexistent_path_inside_base_resolves(self, tmp_path: Path) -> None:
        # Confinement is independent of existence; callers check is_file() after.
        resolved = resolve_within(tmp_path, "does/not/exist.py")
        assert resolved is not None
        assert resolved.is_relative_to(tmp_path.resolve())

    @pytest.mark.parametrize(
        "candidate",
        [
            "../outside.txt",
            "../../etc/passwd",
            "pkg/../../outside.txt",
            "a/b/c/../../../../outside.txt",
        ],
    )
    def test_parent_traversal_is_rejected(self, tmp_path: Path, candidate: str) -> None:
        base = tmp_path / "worktree"
        base.mkdir()
        assert resolve_within(base, candidate) is None

    def test_absolute_path_is_rejected(self, tmp_path: Path) -> None:
        base = tmp_path / "worktree"
        base.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        assert resolve_within(base, str(outside)) is None

    def test_symlink_escaping_base_is_rejected(self, tmp_path: Path) -> None:
        base = tmp_path / "worktree"
        base.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        # A symlink that lives inside the worktree but targets a file outside it.
        link = base / "link.txt"
        link.symlink_to(outside)
        assert resolve_within(base, "link.txt") is None

    def test_symlink_staying_within_base_is_allowed(self, tmp_path: Path) -> None:
        base = tmp_path / "worktree"
        base.mkdir()
        inside = base / "real.txt"
        inside.write_text("ok")
        link = base / "link.txt"
        link.symlink_to(inside)
        assert resolve_within(base, "link.txt") == inside.resolve()

    def test_base_itself_is_allowed(self, tmp_path: Path) -> None:
        # is_relative_to treats base as relative to itself.
        assert resolve_within(tmp_path, ".") == tmp_path.resolve()
