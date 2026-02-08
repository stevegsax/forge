"""Output writing activity for Forge.

Writes LLM-generated files to the worktree with path traversal protection.

Design follows Function Core / Imperative Shell:
- Pure function: resolve_file_paths (security boundary)
- Imperative shell: _write_file, write_output
"""

from __future__ import annotations

from pathlib import Path

from temporalio import activity

from forge.models import FileOutput, WriteFilesInput, WriteOutputInput, WriteResult

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OutputWriteError(Exception):
    """Failed to write output files."""


# ---------------------------------------------------------------------------
# Pure function
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


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def _write_file(path: Path, content: str) -> None:
    """Write content to a file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@activity.defn
async def write_output(input: WriteOutputInput) -> WriteResult:
    """Write LLM-generated files to the worktree."""
    resolved = resolve_file_paths(
        input.worktree_path,
        input.llm_result.response.files,
    )

    files_written: list[str] = []
    for path, content in resolved:
        _write_file(path, content)
        files_written.append(str(path))

    return WriteResult(
        task_id=input.llm_result.task_id,
        files_written=files_written,
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
