"""Code intelligence package — public API and orchestration.

Imperative shell that reads files and coordinates the pure modules.
"""

from __future__ import annotations

import logging
from pathlib import Path

from forge.code_intel.budget import (
    ContextItem,
    PackedContext,
    Representation,
    build_context_items,
    pack_context,
)
from forge.code_intel.graph import (
    RankedFile,
    RankedFileSet,
    Relationship,
    build_import_graph,
    file_path_to_module,
    module_to_file_path,
    rank_files,
)
from forge.code_intel.parser import (
    ExtractedSymbol,
    SymbolKind,
    SymbolSummary,
    extract_symbols,
    format_signatures,
)
from forge.code_intel.repo_map import (
    RepoMap,
    estimate_tokens,
    generate_repo_map,
)

__all__ = [
    "ContextItem",
    "ExtractedSymbol",
    "PackedContext",
    "RankedFile",
    "RankedFileSet",
    "Relationship",
    "RepoMap",
    "Representation",
    "SymbolKind",
    "SymbolSummary",
    "discover_context",
    "estimate_tokens",
    "file_path_to_module",
    "format_signatures",
    "module_to_file_path",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers (imperative shell — file I/O)
# ---------------------------------------------------------------------------


def _read_file_contents(project_root: str, file_paths: list[str]) -> dict[str, str]:
    """Read file contents from the project root.

    Skips missing files with a warning.
    """
    root = Path(project_root)
    contents: dict[str, str] = {}
    for path in file_paths:
        full = root / path
        if full.is_file():
            contents[path] = full.read_text()
        else:
            logger.warning("File not found, skipping: %s", full)
    return contents


def _extract_symbols_for_files(
    project_root: str,
    file_paths: list[str],
    src_root: str,
) -> dict[str, str]:
    """Extract signatures for a list of files.

    Returns dict of file_path -> formatted signatures text.
    """
    root = Path(project_root)
    result: dict[str, str] = {}
    for path in file_paths:
        full = root / path
        if not full.is_file():
            continue
        source = full.read_text()
        module_name = file_path_to_module(path, src_root)
        summary = extract_symbols(source, path, module_name)
        result[path] = format_signatures(summary)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def discover_context(
    target_files: list[str],
    project_root: str,
    package_name: str,
    src_root: str = "src",
    manual_context: dict[str, str] | None = None,
    token_budget: int = 100_000,
    max_import_depth: int = 2,
    include_repo_map: bool = True,
    repo_map_tokens: int = 2048,
    include_dependencies: bool = False,
) -> PackedContext:
    """Discover and pack context automatically.

    1. Build import graph via grimp
    2. Rank files by PageRank importance
    3. Read target file contents
    4. Read direct import contents
    5. Extract symbols for transitive imports
    6. Generate repo map
    7. Build context items
    8. Pack within token budget
    """
    manual_context = manual_context or {}

    # Step 1: Build import graph
    try:
        graph = build_import_graph(package_name)
    except Exception:
        logger.warning("Failed to build import graph for %s", package_name, exc_info=True)
        # Fallback: just use manual context
        items = build_context_items(
            target_file_contents={},
            direct_import_contents={},
            transitive_summaries={},
            ranked_files=[],
            manual_context_contents=manual_context,
        )
        return pack_context(items, token_budget)

    # Step 2: Convert target files to module names and rank
    target_modules = []
    for tf in target_files:
        try:
            target_modules.append(file_path_to_module(tf, src_root))
        except (ValueError, KeyError):
            logger.warning("Cannot convert target file to module: %s", tf)

    ranked_set = rank_files(graph, target_modules, src_root, max_depth=max_import_depth)

    # Step 3: Read target file contents (existing files only)
    target_contents = _read_file_contents(project_root, target_files)

    # Step 4: Read direct import contents (only when dependencies requested)
    direct_files = [
        f.file_path for f in ranked_set.ranked_files if f.relationship == Relationship.DIRECT_IMPORT
    ]
    direct_contents = (
        _read_file_contents(project_root, direct_files) if include_dependencies else {}
    )

    # Step 5: Extract symbols for transitive imports (only when dependencies requested)
    transitive_files = [
        f.file_path
        for f in ranked_set.ranked_files
        if f.relationship == Relationship.TRANSITIVE_IMPORT
    ]
    transitive_sigs = (
        _extract_symbols_for_files(project_root, transitive_files, src_root)
        if include_dependencies
        else {}
    )

    # Step 6: Generate repo map
    repo_map: RepoMap | None = None
    repo_map_text: str | None = None
    if include_repo_map:
        # Build summaries for all ranked files
        all_summaries = {}
        root = Path(project_root)
        for rf in ranked_set.ranked_files:
            full = root / rf.file_path
            if full.is_file():
                source = full.read_text()
                summary = extract_symbols(source, rf.file_path, rf.module_name)
                all_summaries[rf.file_path] = summary

        repo_map = generate_repo_map(
            ranked_set.ranked_files,
            all_summaries,
            token_budget=repo_map_tokens,
        )
        repo_map_text = repo_map.content if repo_map.content else None

    # Step 7: Build context items
    # Also create signature fallbacks for direct imports (for graceful degradation)
    direct_sigs = (
        _extract_symbols_for_files(project_root, direct_files, src_root)
        if include_dependencies
        else {}
    )

    items = build_context_items(
        target_file_contents=target_contents,
        direct_import_contents=direct_contents,
        transitive_summaries=transitive_sigs,
        ranked_files=ranked_set.ranked_files,
        manual_context_contents=manual_context,
        repo_map_text=repo_map_text,
    )

    # Step 8: Pack
    packed = pack_context(items, token_budget, signature_fallbacks=direct_sigs)

    # Attach repo map to result
    return packed.model_copy(update={"repo_map": repo_map})
