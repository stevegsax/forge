"""Tests for forge.code_intel.graph — import graph + PageRank."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from forge.code_intel.graph import (
    RankedFileSet,
    Relationship,
    _compute_distances,
    _grimp_to_networkx,
    file_path_to_module,
    module_to_file_path,
    rank_files,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# file_path_to_module
# ---------------------------------------------------------------------------


class TestFilePathToModule:
    def test_simple(self) -> None:
        assert file_path_to_module("src/forge/models.py", "src") == "forge.models"

    def test_init(self) -> None:
        assert file_path_to_module("src/forge/__init__.py", "src") == "forge"

    def test_nested(self) -> None:
        result = file_path_to_module("src/forge/activities/context.py", "src")
        assert result == "forge.activities.context"


# ---------------------------------------------------------------------------
# module_to_file_path
# ---------------------------------------------------------------------------


class TestModuleToFilePath:
    def test_module_file_exists(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        pkg = src / "forge"
        pkg.mkdir(parents=True)
        (pkg / "models.py").write_text("")
        result = module_to_file_path("forge.models", str(src))
        assert result.endswith("forge/models.py")

    def test_package_init_exists(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        pkg = src / "forge"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        result = module_to_file_path("forge", str(src))
        assert result.endswith("forge/__init__.py")

    def test_fallback_to_py(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        result = module_to_file_path("forge.models", str(src))
        assert result.endswith("forge/models.py")


# ---------------------------------------------------------------------------
# _grimp_to_networkx
# ---------------------------------------------------------------------------


def _make_mock_graph(
    modules: list[str],
    edges: dict[str, set[str]],
) -> MagicMock:
    """Build a mock grimp ImportGraph."""
    mock = MagicMock()
    mock.modules = set(modules)
    mock.find_modules_directly_imported_by = lambda m: edges.get(m, set())
    mock.find_upstream_modules = lambda m: set()
    return mock


class TestGrimpToNetworkx:
    def test_empty_graph(self) -> None:
        mock = _make_mock_graph([], {})
        g = _grimp_to_networkx(mock)
        assert len(g.nodes) == 0

    def test_simple_graph(self) -> None:
        mock = _make_mock_graph(
            ["a", "b", "c"],
            {"a": {"b"}, "b": {"c"}},
        )
        g = _grimp_to_networkx(mock)
        assert len(g.nodes) == 3
        assert g.has_edge("a", "b")
        assert g.has_edge("b", "c")
        assert not g.has_edge("a", "c")


# ---------------------------------------------------------------------------
# _compute_distances
# ---------------------------------------------------------------------------


class TestComputeDistances:
    def test_single_target(self) -> None:
        mock = _make_mock_graph(
            ["a", "b", "c"],
            {"a": {"b"}, "b": {"c"}},
        )
        g = _grimp_to_networkx(mock)
        distances = _compute_distances(g, ["a"], max_depth=3)
        assert distances["a"] == 0
        assert distances["b"] == 1
        assert distances["c"] == 2

    def test_depth_limit(self) -> None:
        mock = _make_mock_graph(
            ["a", "b", "c"],
            {"a": {"b"}, "b": {"c"}},
        )
        g = _grimp_to_networkx(mock)
        distances = _compute_distances(g, ["a"], max_depth=1)
        assert "a" in distances
        assert "b" in distances
        assert "c" not in distances

    def test_missing_target(self) -> None:
        mock = _make_mock_graph(["a", "b"], {"a": {"b"}})
        g = _grimp_to_networkx(mock)
        distances = _compute_distances(g, ["nonexistent"], max_depth=2)
        assert distances == {}


# ---------------------------------------------------------------------------
# rank_files
# ---------------------------------------------------------------------------


class TestRankFiles:
    def test_empty_graph(self) -> None:
        mock = _make_mock_graph([], {})
        result = rank_files(mock, ["a"], "src", max_depth=2)
        assert isinstance(result, RankedFileSet)
        assert result.ranked_files == []
        assert result.graph_node_count == 0

    def test_basic_ranking(self) -> None:
        mock = _make_mock_graph(
            ["target", "dep_a", "dep_b"],
            {"target": {"dep_a", "dep_b"}, "dep_a": {"dep_b"}},
        )
        mock.find_upstream_modules = lambda m: {
            "dep_a": {"dep_b"},
            "dep_b": set(),
            "target": {"dep_a", "dep_b"},
        }.get(m, set())

        result = rank_files(mock, ["target"], "src", max_depth=2)
        assert len(result.ranked_files) > 0
        # All results should be non-target files
        names = {f.module_name for f in result.ranked_files}
        assert "target" not in names

    def test_direct_imports_classified(self) -> None:
        mock = _make_mock_graph(
            ["target", "direct_dep"],
            {"target": {"direct_dep"}},
        )
        mock.find_upstream_modules = lambda m: set()

        result = rank_files(mock, ["target"], "src", max_depth=2)
        direct = [f for f in result.ranked_files if f.module_name == "direct_dep"]
        assert len(direct) == 1
        assert direct[0].relationship == Relationship.DIRECT_IMPORT
        assert direct[0].distance == 1

    def test_sorted_by_importance(self) -> None:
        mock = _make_mock_graph(
            ["target", "a", "b", "c"],
            {"target": {"a", "b"}, "a": {"c"}, "b": {"c"}},
        )
        mock.find_upstream_modules = lambda m: set()

        result = rank_files(mock, ["target"], "src", max_depth=2)
        scores = [f.importance for f in result.ranked_files]
        assert scores == sorted(scores, reverse=True)

    def test_graph_node_count(self) -> None:
        mock = _make_mock_graph(
            ["a", "b", "c", "d"],
            {"a": {"b"}, "c": {"d"}},
        )
        mock.find_upstream_modules = lambda m: set()
        result = rank_files(mock, ["a"], "src", max_depth=2)
        assert result.graph_node_count == 4


# ---------------------------------------------------------------------------
# Integration: build_import_graph with synthetic package
# ---------------------------------------------------------------------------


class TestBuildImportGraphIntegration:
    def test_builds_graph_for_real_package(self, tmp_path: Path) -> None:
        """Integration test with a synthetic Python package in tmp_path."""
        import sys

        # Create a small synthetic package
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "models.py").write_text("class Foo:\n    pass\n")
        (pkg / "core.py").write_text("from mypkg.models import Foo\n")

        # Add to sys.path so grimp can find it
        sys.path.insert(0, str(tmp_path))
        try:
            from forge.code_intel.graph import build_import_graph

            graph = build_import_graph("mypkg")
            modules = graph.modules
            assert "mypkg.models" in modules
            assert "mypkg.core" in modules
        finally:
            sys.path.remove(str(tmp_path))
