"""Tests for forge.activities.output — output file writing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from forge.activities.output import OutputWriteError, resolve_file_paths, write_files, write_output
from forge.models import (
    FileOutput,
    LLMCallResult,
    LLMResponse,
    WriteFilesInput,
    WriteOutputInput,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# resolve_file_paths (pure function)
# ---------------------------------------------------------------------------


class TestResolveFilePaths:
    def test_resolves_relative_paths(self, tmp_path: Path) -> None:
        files = [FileOutput(file_path="src/main.py", content="code")]
        result = resolve_file_paths(str(tmp_path), files)
        assert len(result) == 1
        assert result[0][0] == tmp_path / "src" / "main.py"
        assert result[0][1] == "code"

    def test_rejects_parent_traversal(self, tmp_path: Path) -> None:
        files = [FileOutput(file_path="../escape.py", content="bad")]
        with pytest.raises(OutputWriteError, match="Path traversal rejected"):
            resolve_file_paths(str(tmp_path), files)

    def test_rejects_absolute_path(self, tmp_path: Path) -> None:
        files = [FileOutput(file_path="/etc/passwd", content="bad")]
        with pytest.raises(OutputWriteError, match="Path traversal rejected"):
            resolve_file_paths(str(tmp_path), files)

    def test_rejects_sneaky_traversal(self, tmp_path: Path) -> None:
        files = [FileOutput(file_path="src/../../escape.py", content="bad")]
        with pytest.raises(OutputWriteError, match="Path traversal rejected"):
            resolve_file_paths(str(tmp_path), files)

    def test_multiple_files(self, tmp_path: Path) -> None:
        files = [
            FileOutput(file_path="a.py", content="a"),
            FileOutput(file_path="sub/b.py", content="b"),
        ]
        result = resolve_file_paths(str(tmp_path), files)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# write_output (activity)
# ---------------------------------------------------------------------------


def _make_write_input(worktree_path: str, files: list[FileOutput]) -> WriteOutputInput:
    """Helper to construct a WriteOutputInput with minimal boilerplate."""
    return WriteOutputInput(
        llm_result=LLMCallResult(
            task_id="write-task",
            response=LLMResponse(files=files, explanation="test"),
            model_name="test-model",
            input_tokens=10,
            output_tokens=20,
            latency_ms=100.0,
        ),
        worktree_path=worktree_path,
    )


class TestWriteOutput:
    @pytest.mark.asyncio
    async def test_writes_single_file(self, tmp_path: Path) -> None:
        files = [FileOutput(file_path="hello.py", content="print('hello')")]
        input_data = _make_write_input(str(tmp_path), files)

        result = await write_output(input_data)

        assert result.task_id == "write-task"
        assert len(result.files_written) == 1
        assert (tmp_path / "hello.py").read_text() == "print('hello')"

    @pytest.mark.asyncio
    async def test_writes_multiple_files(self, tmp_path: Path) -> None:
        files = [
            FileOutput(file_path="a.py", content="# a"),
            FileOutput(file_path="b.py", content="# b"),
        ]
        input_data = _make_write_input(str(tmp_path), files)

        result = await write_output(input_data)

        assert len(result.files_written) == 2
        assert (tmp_path / "a.py").read_text() == "# a"
        assert (tmp_path / "b.py").read_text() == "# b"

    @pytest.mark.asyncio
    async def test_creates_parent_directories(self, tmp_path: Path) -> None:
        files = [FileOutput(file_path="src/pkg/mod.py", content="# nested")]
        input_data = _make_write_input(str(tmp_path), files)

        await write_output(input_data)

        assert (tmp_path / "src" / "pkg" / "mod.py").read_text() == "# nested"

    @pytest.mark.asyncio
    async def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        (tmp_path / "existing.py").write_text("old content")
        files = [FileOutput(file_path="existing.py", content="new content")]
        input_data = _make_write_input(str(tmp_path), files)

        await write_output(input_data)

        assert (tmp_path / "existing.py").read_text() == "new content"


# ---------------------------------------------------------------------------
# write_files (activity)
# ---------------------------------------------------------------------------


class TestWriteFiles:
    @pytest.mark.asyncio
    async def test_writes_dict_to_worktree(self, tmp_path: Path) -> None:
        input_data = WriteFilesInput(
            task_id="wf-task",
            worktree_path=str(tmp_path),
            files={"hello.py": "print('hello')", "sub/world.py": "print('world')"},
        )
        result = await write_files(input_data)
        assert result.task_id == "wf-task"
        assert len(result.files_written) == 2
        assert (tmp_path / "hello.py").read_text() == "print('hello')"
        assert (tmp_path / "sub" / "world.py").read_text() == "print('world')"

    @pytest.mark.asyncio
    async def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        input_data = WriteFilesInput(
            task_id="wf-task",
            worktree_path=str(tmp_path),
            files={"../escape.py": "bad"},
        )
        with pytest.raises(OutputWriteError, match="Path traversal rejected"):
            await write_files(input_data)

    @pytest.mark.asyncio
    async def test_empty_files(self, tmp_path: Path) -> None:
        input_data = WriteFilesInput(
            task_id="wf-task",
            worktree_path=str(tmp_path),
            files={},
        )
        result = await write_files(input_data)
        assert result.files_written == []
