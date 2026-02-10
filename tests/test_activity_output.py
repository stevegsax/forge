"""Tests for forge.activities.output — output file writing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from forge.activities.output import (
    EditApplicationError,
    OutputWriteError,
    apply_edits,
    resolve_edit_paths,
    resolve_file_paths,
    write_files,
    write_output,
)
from forge.models import (
    FileEdit,
    FileOutput,
    LLMCallResult,
    LLMResponse,
    SearchReplaceEdit,
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


# ---------------------------------------------------------------------------
# apply_edits (pure function)
# ---------------------------------------------------------------------------


class TestApplyEdits:
    def test_single_edit(self) -> None:
        original = "def old_func():\n    pass\n"
        edits = [SearchReplaceEdit(search="old_func", replace="new_func")]
        result = apply_edits(original, edits)
        assert result == "def new_func():\n    pass\n"

    def test_multiple_sequential_edits(self) -> None:
        original = "a = 1\nb = 2\nc = 3\n"
        edits = [
            SearchReplaceEdit(search="a = 1", replace="a = 10"),
            SearchReplaceEdit(search="b = 2", replace="b = 20"),
        ]
        result = apply_edits(original, edits)
        assert result == "a = 10\nb = 20\nc = 3\n"

    def test_empty_edits_list_returns_original(self) -> None:
        original = "unchanged"
        result = apply_edits(original, [])
        assert result == "unchanged"

    def test_error_on_empty_search_string(self) -> None:
        edits = [SearchReplaceEdit(search="", replace="x")]
        with pytest.raises(EditApplicationError, match="empty search string"):
            apply_edits("content", edits)

    def test_error_on_search_not_found(self) -> None:
        edits = [SearchReplaceEdit(search="nonexistent", replace="x")]
        with pytest.raises(EditApplicationError, match="not found"):
            apply_edits("content", edits)

    def test_error_on_ambiguous_search(self) -> None:
        original = "x = 1\nx = 2\n"
        edits = [SearchReplaceEdit(search="x = ", replace="y = ")]
        with pytest.raises(EditApplicationError, match="appears 2 times"):
            apply_edits(original, edits)

    def test_later_edit_sees_earlier_result(self) -> None:
        """Edits are applied sequentially, each sees the result of the prior."""
        original = "hello world"
        edits = [
            SearchReplaceEdit(search="hello", replace="greetings"),
            SearchReplaceEdit(search="greetings world", replace="goodbye world"),
        ]
        result = apply_edits(original, edits)
        assert result == "goodbye world"

    def test_delete_via_empty_replace(self) -> None:
        original = "line1\nDELETE_ME\nline3\n"
        edits = [SearchReplaceEdit(search="DELETE_ME\n", replace="")]
        result = apply_edits(original, edits)
        assert result == "line1\nline3\n"

    def test_multiline_search_and_replace(self) -> None:
        original = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        edits = [
            SearchReplaceEdit(
                search="def foo():\n    return 1",
                replace="def foo():\n    return 42",
            )
        ]
        result = apply_edits(original, edits)
        assert "return 42" in result
        assert "return 2" in result


# ---------------------------------------------------------------------------
# resolve_edit_paths (pure function)
# ---------------------------------------------------------------------------


class TestResolveEditPaths:
    def test_resolves_existing_file(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("original")
        edits = [FileEdit(file_path="src/main.py", edits=[])]
        result = resolve_edit_paths(str(tmp_path), edits)
        assert len(result) == 1
        assert result[0][0] == tmp_path / "src" / "main.py"

    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        edits = [FileEdit(file_path="../escape.py", edits=[])]
        with pytest.raises(OutputWriteError, match="Path traversal rejected"):
            resolve_edit_paths(str(tmp_path), edits)

    def test_rejects_nonexistent_file(self, tmp_path: Path) -> None:
        edits = [FileEdit(file_path="missing.py", edits=[])]
        with pytest.raises(OutputWriteError, match="does not exist"):
            resolve_edit_paths(str(tmp_path), edits)


# ---------------------------------------------------------------------------
# write_output with edits (activity)
# ---------------------------------------------------------------------------


def _make_write_input_with_edits(
    worktree_path: str,
    files: list[FileOutput] | None = None,
    edits: list[FileEdit] | None = None,
) -> WriteOutputInput:
    """Helper to construct a WriteOutputInput with files and/or edits."""
    return WriteOutputInput(
        llm_result=LLMCallResult(
            task_id="edit-task",
            response=LLMResponse(
                files=files or [],
                edits=edits or [],
                explanation="test edits",
            ),
            model_name="test-model",
            input_tokens=10,
            output_tokens=20,
            latency_ms=100.0,
        ),
        worktree_path=worktree_path,
    )


class TestWriteOutputWithEdits:
    @pytest.mark.asyncio
    async def test_applies_edits_to_existing_file(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("def old_func():\n    pass\n")
        edits = [
            FileEdit(
                file_path="main.py",
                edits=[SearchReplaceEdit(search="old_func", replace="new_func")],
            )
        ]
        input_data = _make_write_input_with_edits(str(tmp_path), edits=edits)
        result = await write_output(input_data)

        assert len(result.files_written) == 1
        assert (tmp_path / "main.py").read_text() == "def new_func():\n    pass\n"
        assert result.output_files["main.py"] == "def new_func():\n    pass\n"

    @pytest.mark.asyncio
    async def test_mixed_files_and_edits(self, tmp_path: Path) -> None:
        (tmp_path / "existing.py").write_text("old = True\n")
        files = [FileOutput(file_path="new.py", content="# new file\n")]
        edits = [
            FileEdit(
                file_path="existing.py",
                edits=[SearchReplaceEdit(search="old = True", replace="old = False")],
            )
        ]
        input_data = _make_write_input_with_edits(str(tmp_path), files=files, edits=edits)
        result = await write_output(input_data)

        assert len(result.files_written) == 2
        assert (tmp_path / "new.py").read_text() == "# new file\n"
        assert (tmp_path / "existing.py").read_text() == "old = False\n"
        assert result.output_files["new.py"] == "# new file\n"
        assert result.output_files["existing.py"] == "old = False\n"

    @pytest.mark.asyncio
    async def test_rejects_overlap_between_files_and_edits(self, tmp_path: Path) -> None:
        (tmp_path / "overlap.py").write_text("content")
        files = [FileOutput(file_path="overlap.py", content="new")]
        edits = [
            FileEdit(
                file_path="overlap.py",
                edits=[SearchReplaceEdit(search="content", replace="changed")],
            )
        ]
        input_data = _make_write_input_with_edits(str(tmp_path), files=files, edits=edits)
        with pytest.raises(OutputWriteError, match="both files and edits"):
            await write_output(input_data)

    @pytest.mark.asyncio
    async def test_output_files_populated_for_new_files(self, tmp_path: Path) -> None:
        files = [FileOutput(file_path="hello.py", content="print('hello')")]
        input_data = _make_write_input_with_edits(str(tmp_path), files=files)
        result = await write_output(input_data)
        assert result.output_files["hello.py"] == "print('hello')"

    @pytest.mark.asyncio
    async def test_edit_nonexistent_file_raises(self, tmp_path: Path) -> None:
        edits = [
            FileEdit(
                file_path="missing.py",
                edits=[SearchReplaceEdit(search="x", replace="y")],
            )
        ]
        input_data = _make_write_input_with_edits(str(tmp_path), edits=edits)
        with pytest.raises(OutputWriteError, match="does not exist"):
            await write_output(input_data)
