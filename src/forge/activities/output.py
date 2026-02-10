"""Output writing activity for Forge.

Writes LLM-generated files to the worktree with path traversal protection.

Design follows Function Core / Imperative Shell:
- Pure functions: resolve_file_paths, resolve_edit_paths, apply_edits (security boundary)
- Imperative shell: _write_file, write_output
"""

from __future__ import annotations

from pathlib import Path

from temporalio import activity

from forge.models import (
    FileEdit,
    FileOutput,
    SearchReplaceEdit,
    WriteFilesInput,
    WriteOutputInput,
    WriteResult,
)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OutputWriteError(Exception):
    """Failed to write output files."""


class EditApplicationError(Exception):
    """Failed to apply search/replace edits."""


# ---------------------------------------------------------------------------
# Pure functions
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


def apply_edits(original: str, edits: list[SearchReplaceEdit]) -> str:
    """Apply a sequence of search/replace edits to a string.

    Each search string must appear exactly once in the current content.
    Edits are applied sequentially — later edits see the result of earlier ones.

    Raises:
        EditApplicationError: If a search string is empty, not found, or ambiguous.
    """
    content = original

    for i, edit in enumerate(edits):
        if not edit.search:
            msg = f"Edit {i}: empty search string"
            raise EditApplicationError(msg)

        count = content.count(edit.search)
        if count == 0:
            msg = f"Edit {i}: search string not found in file"
            raise EditApplicationError(msg)
        if count > 1:
            msg = f"Edit {i}: search string appears {count} times (must be unique)"
            raise EditApplicationError(msg)

        content = content.replace(edit.search, edit.replace, 1)

    return content


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
