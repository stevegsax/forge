"""Output writing activity for Forge.

Writes LLM-generated files to the worktree with path traversal protection.

Design follows Function Core / Imperative Shell:
- Pure functions: resolve_file_paths, resolve_edit_paths, apply_edits (security boundary)
- Imperative shell: _write_file, write_output
"""

from __future__ import annotations

import logging
import textwrap
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import BaseModel, Field
from temporalio import activity

from forge.models import (
    FileEdit,
    FileOutput,
    MatchLevel,
    SearchReplaceEdit,
    WriteFilesInput,
    WriteOutputInput,
    WriteResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OutputWriteError(Exception):
    """Failed to write output files."""


class EditApplicationError(Exception):
    """Failed to apply search/replace edits."""


# ---------------------------------------------------------------------------
# Edit match result models
# ---------------------------------------------------------------------------


class EditMatchResult(BaseModel):
    """Records which matching strategy succeeded for a single edit."""

    edit_index: int
    match_level: MatchLevel
    similarity_score: float | None = Field(default=None, description="Only set for fuzzy matches.")


class EditApplicationResult(BaseModel):
    """Rich result from apply_edits_detailed."""

    content: str
    match_results: list[EditMatchResult]


# ---------------------------------------------------------------------------
# Pure functions — path resolution
# ---------------------------------------------------------------------------


def resolve_file_paths(worktree_path: str, files: list[FileOutput]) -> list[tuple[Path, str]]:
    """Resolve relative file paths to absolute paths within the worktree.

    Rejects any path that would escape the worktree directory (path traversal).

    Returns:
        List of (absolute_path, content) tuples.

    Raises:
        OutputWriteError: If any resolved path is outside the worktree.
    """
    wt = Path(worktree_path).resolve()
    resolved: list[tuple[Path, str]] = []

    for file_output in files:
        target = (wt / file_output.file_path).resolve()
        if not target.is_relative_to(wt):
            msg = (
                f"Path traversal rejected: {file_output.file_path!r} "
                f"resolves to {target} which is outside {wt}"
            )
            raise OutputWriteError(msg)
        resolved.append((target, file_output.content))

    return resolved


def resolve_edit_paths(worktree_path: str, edits: list[FileEdit]) -> list[tuple[Path, FileEdit]]:
    """Resolve edit file paths to absolute paths within the worktree.

    Rejects any path that would escape the worktree directory (path traversal).
    Also verifies the target file exists.

    Returns:
        List of (absolute_path, FileEdit) tuples.

    Raises:
        OutputWriteError: If any resolved path is outside the worktree or file doesn't exist.
    """
    wt = Path(worktree_path).resolve()
    resolved: list[tuple[Path, FileEdit]] = []

    for file_edit in edits:
        target = (wt / file_edit.file_path).resolve()
        if not target.is_relative_to(wt):
            msg = (
                f"Path traversal rejected: {file_edit.file_path!r} "
                f"resolves to {target} which is outside {wt}"
            )
            raise OutputWriteError(msg)
        if not target.is_file():
            msg = f"Edit target does not exist: {file_edit.file_path!r} (resolved to {target})"
            raise OutputWriteError(msg)
        resolved.append((target, file_edit))

    return resolved


# ---------------------------------------------------------------------------
# Pure functions — matching strategies (D55 fallback chain)
# ---------------------------------------------------------------------------


def _exact_match(content: str, search: str) -> int | None:
    """Return the start index if search appears exactly once, else None.

    Raises:
        EditApplicationError: If search appears more than once (ambiguous).
    """
    count = content.count(search)
    if count == 0:
        return None
    if count > 1:
        raise EditApplicationError(f"search string appears {count} times (must be unique)")
    return content.index(search)


def _whitespace_normalized_match(content: str, search: str) -> tuple[int, int] | None:
    """Match after stripping trailing whitespace from each line.

    Returns (start, end) indices in the original content, or None if no
    unique match is found.

    Raises:
        EditApplicationError: If multiple matches found (ambiguous).
    """
    content_lines = content.splitlines(keepends=True)
    search_lines = search.splitlines(keepends=True)

    if not search_lines:
        return None

    # Strip all trailing whitespace (including newline) for comparison
    norm_content = [line.rstrip() for line in content_lines]
    norm_search = [line.rstrip() for line in search_lines]

    # Slide search window over content lines
    n = len(norm_search)
    matches: list[int] = []
    for i in range(len(norm_content) - n + 1):
        if norm_content[i : i + n] == norm_search:
            matches.append(i)

    if len(matches) == 0:
        return None
    if len(matches) > 1:
        raise EditApplicationError(
            f"whitespace-normalized search matches {len(matches)} locations (ambiguous)"
        )

    line_idx = matches[0]
    start = sum(len(line) for line in content_lines[:line_idx])
    end = start + sum(len(line) for line in content_lines[line_idx : line_idx + n])

    # If search doesn't end with \n, don't consume content's trailing \n
    if not search.endswith("\n") and content_lines[line_idx + n - 1].endswith("\n"):
        end -= 1

    return (start, end)


def _reindent(dedented: str, level: int) -> str:
    """Add ``level`` spaces of indentation to each non-empty line."""
    prefix = " " * level
    lines = dedented.splitlines(keepends=True)
    result: list[str] = []
    for line in lines:
        if line.strip():
            result.append(prefix + line)
        else:
            result.append(line)
    return "".join(result)


def _indentation_normalized_match(content: str, search: str) -> tuple[int, int] | None:
    """Match after normalizing indentation.

    Dedents the search string fully, then re-indents at each indentation
    level present in the file and checks for exact match.

    Returns (start, end) indices in the original content, or None.

    Raises:
        EditApplicationError: If multiple matches found (ambiguous).
    """
    dedented = textwrap.dedent(search)

    # Detect indentation levels present in the file
    indent_levels: set[int] = set()
    for line in content.splitlines():
        stripped = line.lstrip(" ")
        if stripped:
            indent_levels.add(len(line) - len(stripped))

    all_matches: list[tuple[int, int]] = []
    for level in sorted(indent_levels):
        reindented = _reindent(dedented, level)
        if reindented == search:
            # Would have been caught by exact match already
            continue
        count = content.count(reindented)
        if count == 1:
            pos = content.index(reindented)
            all_matches.append((pos, pos + len(reindented)))
        elif count > 1:
            raise EditApplicationError(
                f"indentation-normalized search matches {count} times "
                f"at indent level {level} (ambiguous)"
            )

    if len(all_matches) == 0:
        return None
    if len(all_matches) > 1:
        raise EditApplicationError(
            f"indentation-normalized search matches at "
            f"{len(all_matches)} different indent levels (ambiguous)"
        )

    return all_matches[0]


def _fuzzy_match(
    content: str,
    search: str,
    threshold: float = 0.6,
) -> tuple[int, int, float] | None:
    """Find the best fuzzy match above the threshold.

    Slides a window of ``len(search_lines)`` over the file's lines and
    scores each window with ``SequenceMatcher``.

    Returns (start, end, score) in the original content, or None.

    Raises:
        EditApplicationError: If two windows score within 0.05 of each other
            above the threshold (ambiguous).
    """
    content_lines = content.splitlines(keepends=True)
    search_lines = search.splitlines(keepends=True)

    if not search_lines:
        return None

    n = len(search_lines)
    if n > len(content_lines):
        return None

    best_score = 0.0
    best_idx = 0
    second_best_score = 0.0

    for i in range(len(content_lines) - n + 1):
        window = "".join(content_lines[i : i + n])
        score = SequenceMatcher(None, window, search).ratio()

        if score > best_score:
            second_best_score = best_score
            best_score = score
            best_idx = i
        elif score > second_best_score:
            second_best_score = score

    if best_score < threshold:
        return None

    # D57: uniqueness — best must be at least 0.05 ahead of second best
    if best_score - second_best_score < 0.05:
        raise EditApplicationError(
            f"fuzzy match is ambiguous: best score {best_score:.3f}, "
            f"second best {second_best_score:.3f} (gap < 0.05)"
        )

    start = sum(len(line) for line in content_lines[:best_idx])
    end = start + sum(len(line) for line in content_lines[best_idx : best_idx + n])

    # If search doesn't end with \n, don't consume content's trailing \n
    if not search.endswith("\n") and content_lines[best_idx + n - 1].endswith("\n"):
        end -= 1

    return (start, end, best_score)


# ---------------------------------------------------------------------------
# Pure functions — edit application
# ---------------------------------------------------------------------------


def apply_edits_detailed(
    original: str,
    edits: list[SearchReplaceEdit],
    *,
    similarity_threshold: float = 0.6,
) -> EditApplicationResult:
    """Apply edits with a fallback matching chain (D55).

    Tries: exact -> whitespace-normalized -> indentation-normalized -> fuzzy.
    Each level only activates if the previous found zero matches. Ambiguity
    at any level raises ``EditApplicationError``.

    Returns an ``EditApplicationResult`` with the final content and per-edit
    match metadata.
    """
    content = original
    match_results: list[EditMatchResult] = []

    for i, edit in enumerate(edits):
        if not edit.search:
            msg = f"Edit {i}: empty search string"
            raise EditApplicationError(msg)

        try:
            result = _apply_single_edit(content, edit, i, similarity_threshold)
        except EditApplicationError as exc:
            raise EditApplicationError(f"Edit {i}: {exc}") from None

        content = result[0]
        match_results.append(result[1])

    return EditApplicationResult(content=content, match_results=match_results)


def _apply_single_edit(
    content: str,
    edit: SearchReplaceEdit,
    index: int,
    similarity_threshold: float,
) -> tuple[str, EditMatchResult]:
    """Try the fallback chain for a single edit, return (new_content, match_result)."""
    # 1. Exact match
    start = _exact_match(content, edit.search)
    if start is not None:
        end = start + len(edit.search)
        new_content = content[:start] + edit.replace + content[end:]
        return new_content, EditMatchResult(edit_index=index, match_level=MatchLevel.EXACT)

    # 2. Whitespace-normalized match
    span = _whitespace_normalized_match(content, edit.search)
    if span is not None:
        start, end = span
        logger.warning("Edit %d: matched via whitespace normalization", index)
        new_content = content[:start] + edit.replace + content[end:]
        return new_content, EditMatchResult(edit_index=index, match_level=MatchLevel.WHITESPACE)

    # 3. Indentation-normalized match
    span = _indentation_normalized_match(content, edit.search)
    if span is not None:
        start, end = span
        logger.warning("Edit %d: matched via indentation normalization", index)
        new_content = content[:start] + edit.replace + content[end:]
        return new_content, EditMatchResult(edit_index=index, match_level=MatchLevel.INDENTATION)

    # 4. Fuzzy match
    fuzzy = _fuzzy_match(content, edit.search, threshold=similarity_threshold)
    if fuzzy is not None:
        start, end, score = fuzzy
        logger.warning("Edit %d: matched via fuzzy matching (score=%.3f)", index, score)
        new_content = content[:start] + edit.replace + content[end:]
        return new_content, EditMatchResult(
            edit_index=index, match_level=MatchLevel.FUZZY, similarity_score=score
        )

    # All strategies failed
    raise EditApplicationError(
        "search string not found (tried exact, whitespace, indentation, and fuzzy matching)"
    )


def apply_edits(
    original: str,
    edits: list[SearchReplaceEdit],
    *,
    similarity_threshold: float = 0.6,
) -> str:
    """Apply a sequence of search/replace edits to a string.

    Uses a fallback matching chain (exact -> whitespace-normalized ->
    indentation-normalized -> fuzzy) to recover from minor LLM output
    discrepancies. Edits are applied sequentially — later edits see the
    result of earlier ones.

    Raises:
        EditApplicationError: If a search string is empty, not found, or ambiguous.
    """
    result = apply_edits_detailed(original, edits, similarity_threshold=similarity_threshold)
    return result.content


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def _write_file(path: Path, content: str) -> None:
    """Write content to a file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@activity.defn
async def write_output(input: WriteOutputInput) -> WriteResult:
    """Write LLM-generated files to the worktree.

    Handles both new files (response.files) and edits (response.edits).
    Validates no file path appears in both lists.
    Populates output_files with final content for all written files.
    """
    response = input.llm_result.response

    # Validate no overlap between files and edits
    new_file_paths = {f.file_path for f in response.files}
    edit_file_paths = {e.file_path for e in response.edits}
    overlap = new_file_paths & edit_file_paths
    if overlap:
        msg = f"File paths appear in both files and edits: {sorted(overlap)}"
        raise OutputWriteError(msg)

    files_written: list[str] = []
    output_files: dict[str, str] = {}

    # Write new files
    resolved_new = resolve_file_paths(input.worktree_path, response.files)
    for path, content in resolved_new:
        _write_file(path, content)
        files_written.append(str(path))
        rel_path = str(path.relative_to(Path(input.worktree_path).resolve()))
        output_files[rel_path] = content

    # Apply edits to existing files
    resolved_edits = resolve_edit_paths(input.worktree_path, response.edits)
    for path, file_edit in resolved_edits:
        original = path.read_text()
        updated = apply_edits(original, file_edit.edits)
        _write_file(path, updated)
        files_written.append(str(path))
        rel_path = str(path.relative_to(Path(input.worktree_path).resolve()))
        output_files[rel_path] = updated

    return WriteResult(
        task_id=input.llm_result.task_id,
        files_written=files_written,
        output_files=output_files,
    )


def _validate_file_paths(worktree_path: str, files: dict[str, str]) -> list[tuple[Path, str]]:
    """Validate and resolve a dict of relative paths to absolute paths within the worktree.

    Reuses the same path traversal protection as resolve_file_paths.
    """
    file_outputs = [FileOutput(file_path=fp, content=c) for fp, c in files.items()]
    return resolve_file_paths(worktree_path, file_outputs)


@activity.defn
async def write_files(input: WriteFilesInput) -> WriteResult:
    """Write a dict of files to a worktree with path traversal protection."""
    resolved = _validate_file_paths(input.worktree_path, input.files)

    files_written: list[str] = []
    for path, content in resolved:
        _write_file(path, content)
        files_written.append(str(path))

    return WriteResult(
        task_id=input.task_id,
        files_written=files_written,
    )
