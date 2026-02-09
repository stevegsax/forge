"""Import graph analysis and PageRank ranking.

Mix of pure functions and one imperative function (build_import_graph).
"""

from __future__ import annotations

from collections import deque
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import grimp


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Relationship(StrEnum):
    """How a file relates to the target files."""

    DIRECT_IMPORT = "direct_import"
    TRANSITIVE_IMPORT = "transitive_import"
    DOWNSTREAM = "downstream"


class RankedFile(BaseModel):
    """A file ranked by importance."""

    file_path: str = Field(description="Relative to project root.")
    module_name: str
    importance: float = Field(description="PageRank score (0.0 to 1.0).")
    distance: int = Field(description="Import distance from nearest target file.")
    relationship: Relationship


class RankedFileSet(BaseModel):
    """Result of ranking files by importance."""

    target_files: list[str]
    ranked_files: list[RankedFile] = Field(description="Sorted by importance descending.")
    graph_node_count: int = Field(description="Total files in the import graph.")


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def file_path_to_module(file_path: str, src_root: str) -> str:
    """Convert a file path to a Python module name.

    Example: file_path_to_module("src/forge/models.py", "src") -> "forge.models"
    """
    path = Path(file_path)
    rel = path.relative_to(src_root)
    parts = list(rel.parts)

    # Remove .py extension from last part
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]

    # Handle __init__.py -> package name
    if parts[-1] == "__init__":
        parts = parts[:-1]

    return ".".join(parts)


def module_to_file_path(module_name: str, src_root: str) -> str:
    """Convert a Python module name to a file path.

    Tries module/as/path.py first, falls back to module/as/path/__init__.py.
    """
    parts = module_name.split(".")
    base = Path(src_root) / Path(*parts)

    # Try as a .py file first
    py_file = base.with_suffix(".py")
    if py_file.exists():
        return str(py_file)

    # Try as a package __init__.py
    init_file = base / "__init__.py"
    if init_file.exists():
        return str(init_file)

    # Default to .py path even if it doesn't exist
    return str(py_file)


def _grimp_to_networkx(graph: grimp.ImportGraph) -> nx.DiGraph:
    """Convert a grimp ImportGraph to a NetworkX DiGraph.

    Edges point from importer to imported (dependency direction).
    """
    g = nx.DiGraph()
    modules = graph.modules
    g.add_nodes_from(modules)

    for module in modules:
        imported = graph.find_modules_directly_imported_by(module)
        for dep in imported:
            g.add_edge(module, dep)

    return g


def _compute_distances(
    graph: nx.DiGraph,
    target_modules: list[str],
    max_depth: int,
) -> dict[str, int]:
    """Compute BFS distances from target modules in the dependency graph.

    Returns a dict of module_name -> shortest distance from any target module.
    Traverses both forward (imports) and reverse (imported-by) edges.
    """
    distances: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()

    for target in target_modules:
        if target in graph:
            distances[target] = 0
            queue.append((target, 0))

    while queue:
        node, dist = queue.popleft()
        if dist >= max_depth:
            continue

        # Forward: modules this node imports
        for neighbor in graph.successors(node):
            if neighbor not in distances:
                distances[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))

        # Reverse: modules that import this node
        for neighbor in graph.predecessors(node):
            if neighbor not in distances:
                distances[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))

    return distances


def rank_files(
    graph: grimp.ImportGraph,
    target_modules: list[str],
    src_root: str,
    max_depth: int = 2,
) -> RankedFileSet:
    """Rank files by importance using personalized PageRank.

    Builds a NetworkX DiGraph from the grimp graph, runs personalized PageRank
    with seed weights on target modules, classifies relationships via BFS.
    """
    nxg = _grimp_to_networkx(graph)

    if not nxg.nodes:
        return RankedFileSet(
            target_files=[module_to_file_path(m, src_root) for m in target_modules],
            ranked_files=[],
            graph_node_count=0,
        )

    # Personalized PageRank: seed on target modules
    personalization = {}
    valid_targets = [m for m in target_modules if m in nxg]
    if valid_targets:
        for m in valid_targets:
            personalization[m] = 1.0 / len(valid_targets)
    else:
        # Uniform if no targets in graph
        for m in nxg.nodes:
            personalization[m] = 1.0 / len(nxg.nodes)

    pagerank = nx.pagerank(nxg, personalization=personalization)

    # Compute distances from targets
    distances = _compute_distances(nxg, target_modules, max_depth)

    # Classify and build ranked files
    target_set = set(target_modules)
    direct_imports: set[str] = set()
    for t in valid_targets:
        direct_imports.update(graph.find_modules_directly_imported_by(t))

    ranked: list[RankedFile] = []
    for module, score in pagerank.items():
        if module in target_set:
            continue  # Skip target files themselves

        distance = distances.get(module, max_depth + 1)
        if distance > max_depth:
            continue  # Beyond depth limit

        if module in direct_imports:
            relationship = Relationship.DIRECT_IMPORT
        elif distance <= max_depth:
            # Check if it's upstream or downstream
            is_upstream = any(
                module in (graph.find_upstream_modules(t) or set()) for t in valid_targets
            )
            relationship = (
                Relationship.TRANSITIVE_IMPORT if is_upstream else Relationship.DOWNSTREAM
            )
        else:
            relationship = Relationship.DOWNSTREAM

        ranked.append(
            RankedFile(
                file_path=module_to_file_path(module, src_root),
                module_name=module,
                importance=score,
                distance=distance,
                relationship=relationship,
            )
        )

    # Sort by importance descending
    ranked.sort(key=lambda f: f.importance, reverse=True)

    return RankedFileSet(
        target_files=[module_to_file_path(m, src_root) for m in target_modules],
        ranked_files=ranked,
        graph_node_count=len(nxg.nodes),
    )


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def build_import_graph(package_name: str) -> grimp.ImportGraph:
    """Build the import graph for a Python package.

    This is the only impure function — it calls grimp.build_graph() which
    reads the file system.
    """
    import grimp

    return grimp.build_graph(package_name)
