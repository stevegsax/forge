"""Tests for forge.activities.output — output file writing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from forge.activities.output import (
    EditApplicationError,
    EditApplicationResult,
    OutputWriteError,
    _exact_match,
    _fuzzy_match,
    _indentation_normalized_match,
    _reindent,
    _whitespace_normalized_match,
    apply_edits,
    apply_edits_detailed,
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
    MatchLevel,
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
# _exact_match (pure function)
# ---------------------------------------------------------------------------


class TestExactMatch:
    def test_unique_match_returns_index(self) -> None:
        assert _exact_match("hello world", "world") == 6

    def test_no_match_returns_none(self) -> None:
        assert _exact_match("hello world", "missing") is None

    def test_ambiguous_raises(self) -> None:
        with pytest.raises(EditApplicationError, match="appears 2 times"):
            _exact_match("abcabc", "abc")


# ---------------------------------------------------------------------------
# _whitespace_normalized_match (pure function)
# ---------------------------------------------------------------------------


class TestWhitespaceNormalizedMatch:
    def test_trailing_spaces_on_content_line(self) -> None:
        content = "def foo():   \n    pass\n"
        search = "def foo():\n    pass\n"
        result = _whitespace_normalized_match(content, search)
        assert result is not None
        start, end = result
        assert content[start:end] == content

    def test_trailing_spaces_on_search_line(self) -> None:
        content = "def foo():\n    pass\n"
        search = "def foo():   \n    pass\n"
        result = _whitespace_normalized_match(content, search)
        assert result is not None
        start, end = result
        assert content[start:end] == content

    def test_trailing_tabs(self) -> None:
        content = "x = 1\t\t\ny = 2\n"
        search = "x = 1\ny = 2\n"
        result = _whitespace_normalized_match(content, search)
        assert result is not None
        start, end = result
        assert content[start:end] == content

    def test_partial_match_in_larger_file(self) -> None:
        content = "line1\ndef foo():   \n    pass\nline4\n"
        search = "def foo():\n    pass\n"
        result = _whitespace_normalized_match(content, search)
        assert result is not None
        start, end = result
        assert content[start:end] == "def foo():   \n    pass\n"

    def test_no_match_returns_none(self) -> None:
        content = "def foo():\n    pass\n"
        search = "def bar():\n    pass\n"
        assert _whitespace_normalized_match(content, search) is None

    def test_ambiguous_raises(self) -> None:
        content = "x = 1   \ny = 2\nx = 1  \ny = 2\n"
        search = "x = 1\ny = 2\n"
        with pytest.raises(EditApplicationError, match="ambiguous"):
            _whitespace_normalized_match(content, search)

    def test_search_without_trailing_newline(self) -> None:
        content = "line1\ndef foo():   \n    pass\nline4\n"
        search = "def foo():\n    pass"
        result = _whitespace_normalized_match(content, search)
        assert result is not None
        start, end = result
        # Should not consume the \n after "    pass"
        assert content[start:end] == "def foo():   \n    pass"

    def test_empty_search_returns_none(self) -> None:
        assert _whitespace_normalized_match("content", "") is None

    def test_single_line_trailing_space(self) -> None:
        content = "hello   \nworld\n"
        search = "hello\n"
        result = _whitespace_normalized_match(content, search)
        assert result is not None
        start, end = result
        assert content[start:end] == "hello   \n"


# ---------------------------------------------------------------------------
# _reindent (pure function)
# ---------------------------------------------------------------------------


class TestReindent:
    def test_zero_indent_is_identity(self) -> None:
        text = "def foo():\n    pass\n"
        assert _reindent(text, 0) == text

    def test_adds_indent(self) -> None:
        text = "def foo():\n    pass\n"
        result = _reindent(text, 4)
        assert result == "    def foo():\n        pass\n"

    def test_preserves_blank_lines(self) -> None:
        text = "def foo():\n\n    pass\n"
        result = _reindent(text, 4)
        assert result == "    def foo():\n\n        pass\n"

    def test_handles_no_trailing_newline(self) -> None:
        text = "x = 1"
        result = _reindent(text, 8)
        assert result == "        x = 1"


# ---------------------------------------------------------------------------
# _indentation_normalized_match (pure function)
# ---------------------------------------------------------------------------


class TestIndentationNormalizedMatch:
    def test_search_at_wrong_indent(self) -> None:
        content = "class Foo:\n    def method(self):\n        return 42\n"
        search = "def method(self):\n    return 42\n"
        result = _indentation_normalized_match(content, search)
        assert result is not None
        start, end = result
        assert content[start:end] == "    def method(self):\n        return 42\n"

    def test_search_too_indented(self) -> None:
        content = "def foo():\n    return 42\n"
        search = "        def foo():\n            return 42\n"
        result = _indentation_normalized_match(content, search)
        assert result is not None
        start, end = result
        assert content[start:end] == "def foo():\n    return 42\n"

    def test_no_match_returns_none(self) -> None:
        content = "def foo():\n    return 42\n"
        search = "def bar():\n    return 42\n"
        assert _indentation_normalized_match(content, search) is None

    def test_skips_same_as_exact(self) -> None:
        """If re-indenting produces the original search, skip it (exact match handles it)."""
        content = "def foo():\n    pass\nfoo()\n"
        search = "def foo():\n    pass\n"
        # Exact match would find this, so indentation should also find it
        # since level 0 re-indent == dedented == original (for already-dedented search)
        # The function skips reindented == search, so if the only match is at
        # the original indentation, it returns None.
        result = _indentation_normalized_match(content, search)
        assert result is None

    def test_ambiguous_across_indent_levels_raises(self) -> None:
        # Content has the same code at two different indent levels
        content = "def foo():\n    return 1\n    def foo():\n        return 1\n"
        search = "    def foo():\n        return 1\n"
        # Dedented: "def foo():\n    return 1\n"
        # At level 0: "def foo():\n    return 1\n" — matches line 0-1
        # At level 4: "    def foo():\n        return 1\n" — matches line 2-3 (== search, skipped)
        # Only 1 match (level 0), so no ambiguity error
        result = _indentation_normalized_match(content, search)
        assert result is not None

    def test_blank_lines_in_search(self) -> None:
        content = "class Foo:\n    def method(self):\n\n        return 42\n"
        search = "def method(self):\n\n    return 42\n"
        result = _indentation_normalized_match(content, search)
        assert result is not None
        start, end = result
        assert "    def method(self):" in content[start:end]


# ---------------------------------------------------------------------------
# _fuzzy_match (pure function)
# ---------------------------------------------------------------------------


class TestFuzzyMatch:
    def test_minor_character_difference(self) -> None:
        content = "def foo():\n    return 42\n"
        search = "def foo():\n    return 43\n"  # 42 vs 43
        result = _fuzzy_match(content, search)
        assert result is not None
        start, end, score = result
        assert score > 0.6
        assert content[start:end] == content

    def test_below_threshold_returns_none(self) -> None:
        content = "def foo():\n    return 42\n"
        search = "completely different text\nanother line\n"
        assert _fuzzy_match(content, search) is None

    def test_custom_threshold(self) -> None:
        content = "def foo():\n    return 42\n"
        search = "def foo():\n    return 43\n"
        # With a very high threshold, even a close match should fail
        assert _fuzzy_match(content, search, threshold=0.999) is None

    def test_ambiguous_raises(self) -> None:
        # Two very similar blocks
        content = "def foo():\n    return 1\ndef foo():\n    return 2\n"
        search = "def foo():\n    return 3\n"
        with pytest.raises(EditApplicationError, match="ambiguous"):
            _fuzzy_match(content, search)

    def test_empty_search_returns_none(self) -> None:
        assert _fuzzy_match("content\n", "") is None

    def test_search_longer_than_content_returns_none(self) -> None:
        content = "one line\n"
        search = "line 1\nline 2\nline 3\n"
        assert _fuzzy_match(content, search) is None

    def test_unique_best_match_in_larger_file(self) -> None:
        content = "import os\n\ndef foo():\n    return 42\n\ndef bar():\n    return 99\n"
        search = "def foo():\n    return 43\n"  # minor difference
        result = _fuzzy_match(content, search)
        assert result is not None
        start, end, _score = result
        matched = content[start:end]
        assert "def foo" in matched
        assert "return 42" in matched

    def test_search_without_trailing_newline(self) -> None:
        content = "def foo():\n    return 42\nbar()\n"
        search = "def foo():\n    return 43"  # no trailing newline, minor diff
        result = _fuzzy_match(content, search)
        assert result is not None
        start, end, _score = result
        # Should not consume the \n after "return 42"
        assert not content[start:end].endswith("\n")


# ---------------------------------------------------------------------------
# apply_edits (pure function — backward compatibility)
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
        edits = [SearchReplaceEdit(search="zzz_no_match_zzz", replace="x")]
        with pytest.raises(EditApplicationError, match="not found"):
            apply_edits("def foo():\n    return 42\n", edits)

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
# apply_edits_detailed (fallback chain integration)
# ---------------------------------------------------------------------------


class TestApplyEditsDetailed:
    def test_exact_match_is_tried_first(self) -> None:
        original = "def foo():\n    pass\n"
        edits = [SearchReplaceEdit(search="def foo():", replace="def bar():")]
        result = apply_edits_detailed(original, edits)
        assert result.content == "def bar():\n    pass\n"
        assert len(result.match_results) == 1
        assert result.match_results[0].match_level == MatchLevel.EXACT
        assert result.match_results[0].similarity_score is None

    def test_whitespace_fallback(self) -> None:
        original = "def foo():   \n    pass\n"
        search = "def foo():\n    pass\n"
        edits = [SearchReplaceEdit(search=search, replace="def bar():\n    pass\n")]
        result = apply_edits_detailed(original, edits)
        assert "def bar()" in result.content
        assert result.match_results[0].match_level == MatchLevel.WHITESPACE

    def test_indentation_fallback(self) -> None:
        original = "class Foo:\n    def method(self):\n        return 42\n"
        # Search at wrong indent (0 instead of 4)
        search = "def method(self):\n    return 42\n"
        replace = "def method(self):\n    return 99\n"
        edits = [SearchReplaceEdit(search=search, replace=replace)]
        result = apply_edits_detailed(original, edits)
        assert "return 99" in result.content
        assert result.match_results[0].match_level == MatchLevel.INDENTATION

    def test_fuzzy_fallback(self) -> None:
        original = "def foo():\n    return 42\n"
        # Minor difference: single-quote vs no quotes, different value
        search = "def foo():\n    return 'forty-two'\n"
        replace = "def foo():\n    return 99\n"
        edits = [SearchReplaceEdit(search=search, replace=replace)]
        result = apply_edits_detailed(original, edits)
        assert "return 99" in result.content
        assert result.match_results[0].match_level == MatchLevel.FUZZY
        assert result.match_results[0].similarity_score is not None
        assert result.match_results[0].similarity_score > 0.6

    def test_custom_threshold(self) -> None:
        original = "def foo():\n    return 42\n"
        search = "def foo():\n    return 43\n"
        replace = "def foo():\n    return 99\n"
        edits = [SearchReplaceEdit(search=search, replace=replace)]
        # With threshold=0.999, fuzzy should fail
        with pytest.raises(EditApplicationError, match="not found"):
            apply_edits_detailed(original, edits, similarity_threshold=0.999)

    def test_multiple_edits_different_strategies(self) -> None:
        original = "a = 1\nb = 2   \nclass Foo:\n    def bar(self):\n        return 3\n"
        edits = [
            # Edit 0: exact match
            SearchReplaceEdit(search="a = 1", replace="a = 10"),
            # Edit 1: whitespace fallback (trailing spaces on "b = 2   ")
            SearchReplaceEdit(search="b = 2\n", replace="b = 20\n"),
        ]
        result = apply_edits_detailed(original, edits)
        assert "a = 10" in result.content
        assert "b = 20" in result.content
        assert result.match_results[0].match_level == MatchLevel.EXACT
        assert result.match_results[1].match_level == MatchLevel.WHITESPACE

    def test_returns_edit_application_result(self) -> None:
        original = "hello\n"
        edits = [SearchReplaceEdit(search="hello", replace="world")]
        result = apply_edits_detailed(original, edits)
        assert isinstance(result, EditApplicationResult)
        assert result.content == "world\n"

    def test_all_strategies_fail_raises(self) -> None:
        original = "def foo():\n    return 42\n"
        search = "completely unrelated text that matches nothing at all\n"
        edits = [SearchReplaceEdit(search=search, replace="replacement")]
        with pytest.raises(EditApplicationError, match="not found"):
            apply_edits_detailed(original, edits)


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
