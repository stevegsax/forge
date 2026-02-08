"""Shared test fixtures for Forge."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUFF_CONFIG_SRC = _REPO_ROOT / "tool-config" / "ruff.toml"


@pytest.fixture
def ruff_config(tmp_path: Path) -> Path:
    """Copy the project ruff config into tmp_path so ruff commands find it."""
    dest = tmp_path / "tool-config"
    dest.mkdir()
    shutil.copy(_RUFF_CONFIG_SRC, dest / "ruff.toml")
    return dest / "ruff.toml"


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with one initial commit.

    The repo has a committed ``README.md`` on the ``main`` branch so that
    worktrees can be created from it.

    Returns the path to the repository root.
    """
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@forge.test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Forge Test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    readme = tmp_path / "README.md"
    readme.write_text("# Test repo\n")

    # Copy ruff config so worktrees created from this repo have it.
    tool_config = tmp_path / "tool-config"
    tool_config.mkdir()
    shutil.copy(_RUFF_CONFIG_SRC, tool_config / "ruff.toml")

    subprocess.run(
        ["git", "add", "README.md", "tool-config/ruff.toml"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    return tmp_path
