"""Tests for forge.activities.context — context assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from forge.activities.context import (
    _read_context_files,
    assemble_context,
    build_system_prompt,
    build_user_prompt,
)
from forge.models import AssembleContextInput, TaskDefinition

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# build_system_prompt (pure function)
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_includes_description(self) -> None:
        task = TaskDefinition(
            task_id="t1",
            description="Create a hello world module.",
            target_files=["hello.py"],
        )
        prompt = build_system_prompt(task, {})
        assert "Create a hello world module." in prompt

    def test_includes_target_files(self) -> None:
        task = TaskDefinition(
            task_id="t1",
            description="desc",
            target_files=["src/a.py", "src/b.py"],
        )
        prompt = build_system_prompt(task, {})
        assert "- src/a.py" in prompt
        assert "- src/b.py" in prompt

    def test_includes_context_file_contents(self) -> None:
        task = TaskDefinition(
            task_id="t1",
            description="desc",
            target_files=["out.py"],
        )
        context = {"ref.py": "# reference code\nprint('hi')"}
        prompt = build_system_prompt(task, context)
        assert "### ref.py" in prompt
        assert "# reference code" in prompt

    def test_empty_context_omits_section(self) -> None:
        task = TaskDefinition(
            task_id="t1",
            description="desc",
            target_files=["out.py"],
        )
        prompt = build_system_prompt(task, {})
        assert "## Context Files" not in prompt


# ---------------------------------------------------------------------------
# build_user_prompt (pure function)
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    def test_returns_nonempty(self) -> None:
        prompt = build_user_prompt()
        assert len(prompt) > 0

    def test_is_deterministic(self) -> None:
        assert build_user_prompt() == build_user_prompt()


# ---------------------------------------------------------------------------
# _read_context_files (imperative shell)
# ---------------------------------------------------------------------------


class TestReadContextFiles:
    def test_reads_existing_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("content-a")
        result = _read_context_files(tmp_path, ["a.txt"])
        assert result == {"a.txt": "content-a"}

    def test_skips_missing_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("content-a")
        result = _read_context_files(tmp_path, ["a.txt", "missing.txt"])
        assert "a.txt" in result
        assert "missing.txt" not in result


# ---------------------------------------------------------------------------
# assemble_context (activity)
# ---------------------------------------------------------------------------


class TestAssembleContext:
    @pytest.mark.asyncio
    async def test_assembles_from_files(self, tmp_path: Path) -> None:
        (tmp_path / "ref.py").write_text("# reference")
        task = TaskDefinition(
            task_id="ctx-1",
            description="Generate code.",
            target_files=["out.py"],
            context_files=["ref.py"],
        )
        input_data = AssembleContextInput(
            task=task,
            repo_root=str(tmp_path),
            worktree_path=str(tmp_path / "wt"),
        )
        result = await assemble_context(input_data)
        assert result.task_id == "ctx-1"
        assert "# reference" in result.system_prompt
        assert len(result.user_prompt) > 0

    @pytest.mark.asyncio
    async def test_skips_missing_context_files(self, tmp_path: Path) -> None:
        task = TaskDefinition(
            task_id="ctx-2",
            description="Generate code.",
            target_files=["out.py"],
            context_files=["nonexistent.py"],
        )
        input_data = AssembleContextInput(
            task=task,
            repo_root=str(tmp_path),
            worktree_path=str(tmp_path / "wt"),
        )
        result = await assemble_context(input_data)
        assert result.task_id == "ctx-2"
        assert "## Context Files" not in result.system_prompt

    @pytest.mark.asyncio
    async def test_propagates_task_id(self, tmp_path: Path) -> None:
        task = TaskDefinition(
            task_id="my-task-id",
            description="desc",
            target_files=["f.py"],
        )
        input_data = AssembleContextInput(
            task=task,
            repo_root=str(tmp_path),
            worktree_path=str(tmp_path / "wt"),
        )
        result = await assemble_context(input_data)
        assert result.task_id == "my-task-id"
