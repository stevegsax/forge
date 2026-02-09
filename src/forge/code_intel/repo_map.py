"""Compressed structural overview of the repository.

Pure functions only. Generates a repo map that fits within a token budget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from forge.code_intel.graph import RankedFile
    from forge.code_intel.parser import SymbolSummary


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RepoMap(BaseModel):
    """Compressed structural overview of the repository."""

    content: str = Field(description="The formatted map text.")
    files_with_signatures: int = Field(description="Files that got signature detail.")
    files_path_only: int = Field(description="Files listed by path only.")
    estimated_tokens: int


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Estimate token count using 4-chars-per-token heuristic."""
    return len(text) // 4


def _format_file_with_signatures(file_path: str, summary: SymbolSummary) -> str:
    """Format a file entry with its symbol signatures."""
    lines = [f"{file_path}:"]
    for symbol in summary.symbols:
        for sig_line in symbol.signature.split("\n"):
            lines.append(f"\u2502{sig_line}")
    return "\n".join(lines)


def _format_file_path_only(file_path: str) -> str:
    """Format a file entry with path only (no signatures)."""
    return f"{file_path}:"


def generate_repo_map(
    ranked_files: list[RankedFile],
    symbol_summaries: dict[str, SymbolSummary],
    token_budget: int = 2048,
) -> RepoMap:
    """Generate a repo map that fits within the token budget.

    Uses binary search to maximize the number of files with signature detail.
    Files are ordered by importance (PageRank score descending).
    """
    if not ranked_files:
        return RepoMap(
            content="",
            files_with_signatures=0,
            files_path_only=0,
            estimated_tokens=0,
        )

    file_paths = [f.file_path for f in ranked_files]

    # Binary search: find max number of files that get signature detail
    lo, hi = 0, len(file_paths)

    while lo < hi:
        mid = (lo + hi + 1) // 2
        content = _build_map_content(file_paths, symbol_summaries, sig_count=mid)
        if estimate_tokens(content) <= token_budget:
            lo = mid
        else:
            hi = mid - 1

    sig_count = lo
    content = _build_map_content(file_paths, symbol_summaries, sig_count=sig_count)

    return RepoMap(
        content=content,
        files_with_signatures=sig_count,
        files_path_only=len(file_paths) - sig_count,
        estimated_tokens=estimate_tokens(content),
    )


def _build_map_content(
    file_paths: list[str],
    symbol_summaries: dict[str, SymbolSummary],
    sig_count: int,
) -> str:
    """Build the map content with sig_count files getting signatures."""
    sections: list[str] = []

    for i, path in enumerate(file_paths):
        if i < sig_count and path in symbol_summaries:
            sections.append(_format_file_with_signatures(path, symbol_summaries[path]))
        else:
            sections.append(_format_file_path_only(path))

    return "\n".join(sections)
