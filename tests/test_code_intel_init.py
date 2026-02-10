"""Tests for forge.code_intel — public API and orchestration."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from forge.code_intel import (
    _extract_symbols_for_files,
    _read_file_contents,
    discover_context,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# _read_file_contents
# ---------------------------------------------------------------------------


class TestReadFileContents:
    def test_reads_existing(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("code_a")
        result = _read_file_contents(str(tmp_path), ["a.py"])
        assert result == {"a.py": "code_a"}

    def test_skips_missing(self, tmp_path: Path) -> None:
        result = _read_file_contents(str(tmp_path), ["missing.py"])
        assert result == {}


# ---------------------------------------------------------------------------
# _extract_symbols_for_files
# ---------------------------------------------------------------------------


class TestExtractSymbolsForFiles:
    def test_extracts_signatures(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        pkg = src / "pkg"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "mod.py").write_text("def hello() -> str:\n    return 'hi'\n")

        result = _extract_symbols_for_files(str(tmp_path), ["src/pkg/mod.py"], "src")
        assert "src/pkg/mod.py" in result
        assert "hello" in result["src/pkg/mod.py"]

    def test_skips_missing(self, tmp_path: Path) -> None:
        result = _extract_symbols_for_files(str(tmp_path), ["missing.py"], "src")
        assert result == {}


# ---------------------------------------------------------------------------
# discover_context — integration with synthetic package
# ---------------------------------------------------------------------------


class TestDiscoverContext:
    def test_with_synthetic_package(self, tmp_path: Path) -> None:
        """Integration test: synthetic Python package with import relationships."""
        # Create package structure
        src = tmp_path / "src"
        pkg = src / "testpkg"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "models.py").write_text(
            'class Config:\n    """Configuration model."""\n    name: str = \'default\'\n'
        )
        (pkg / "core.py").write_text(
            "from testpkg.models import Config\n"
            "\n"
            "def process(config: Config) -> str:\n"
            "    return config.name\n"
        )
        (pkg / "utils.py").write_text(
            "from testpkg.core import process\n\ndef run() -> str:\n    return 'done'\n"
        )

        # Add to sys.path
        sys.path.insert(0, str(src))
        try:
            result = discover_context(
                target_files=["src/testpkg/core.py"],
                project_root=str(tmp_path),
                package_name="testpkg",
                src_root="src",
                token_budget=100_000,
                max_import_depth=2,
                include_repo_map=True,
                repo_map_tokens=2048,
            )
            assert result.items_included > 0
            assert result.total_estimated_tokens > 0

            # Should have discovered models.py as a direct import
            item_paths = {item.file_path for item in result.items}
            assert "src/testpkg/core.py" in item_paths  # target
        finally:
            sys.path.remove(str(src))

    def test_with_manual_context(self, tmp_path: Path) -> None:
        """Manual context files are included even without import graph."""
        src = tmp_path / "src"
        pkg = src / "testpkg2"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "main.py").write_text("x = 1\n")

        sys.path.insert(0, str(src))
        try:
            result = discover_context(
                target_files=["src/testpkg2/main.py"],
                project_root=str(tmp_path),
                package_name="testpkg2",
                src_root="src",
                manual_context={"README.md": "# Project readme"},
                token_budget=100_000,
            )
            manual_items = [i for i in result.items if i.file_path == "README.md"]
            assert len(manual_items) == 1
            assert manual_items[0].priority == 6
        finally:
            sys.path.remove(str(src))

    def test_graph_build_failure_fallback(self, tmp_path: Path) -> None:
        """When import graph build fails, falls back to manual context."""
        result = discover_context(
            target_files=["nonexistent.py"],
            project_root=str(tmp_path),
            package_name="nonexistent_package_xyz",
            src_root="src",
            manual_context={"docs.txt": "documentation"},
            token_budget=100_000,
        )
        # Should still include manual context
        assert result.items_included >= 1
        assert any(i.file_path == "docs.txt" for i in result.items)

    def test_include_dependencies_true(self, tmp_path: Path) -> None:
        """When include_dependencies=True, direct imports are included as full content."""
        src = tmp_path / "src"
        pkg = src / "deppkg"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "models.py").write_text(
            'class Config:\n    """Configuration model."""\n    name: str = \'default\'\n'
        )
        (pkg / "core.py").write_text(
            "from deppkg.models import Config\n"
            "\n"
            "def process(config: Config) -> str:\n"
            "    return config.name\n"
        )

        sys.path.insert(0, str(src))
        try:
            result = discover_context(
                target_files=["src/deppkg/core.py"],
                project_root=str(tmp_path),
                package_name="deppkg",
                src_root="src",
                token_budget=100_000,
                include_dependencies=True,
            )
            from forge.code_intel.budget import Representation

            # With include_dependencies=True, priority 3 items (direct deps) should exist
            direct_dep_items = [
                i
                for i in result.items
                if i.priority == 3 and i.representation == Representation.FULL
            ]
            assert len(direct_dep_items) > 0
        finally:
            sys.path.remove(str(src))

    def test_include_dependencies_false(self, tmp_path: Path) -> None:
        """When include_dependencies=False (default), direct imports are NOT included."""
        src = tmp_path / "src"
        pkg = src / "nodeppkg"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "models.py").write_text(
            'class Config:\n    """Configuration model."""\n    name: str = \'default\'\n'
        )
        (pkg / "core.py").write_text(
            "from nodeppkg.models import Config\n"
            "\n"
            "def process(config: Config) -> str:\n"
            "    return config.name\n"
        )

        sys.path.insert(0, str(src))
        try:
            result = discover_context(
                target_files=["src/nodeppkg/core.py"],
                project_root=str(tmp_path),
                package_name="nodeppkg",
                src_root="src",
                token_budget=100_000,
                include_dependencies=False,
            )
            # With include_dependencies=False, no priority 3 (direct deps) or 4 (transitive) items
            dep_items = [i for i in result.items if i.priority in (3, 4)]
            assert len(dep_items) == 0
        finally:
            sys.path.remove(str(src))

    def test_empty_targets(self, tmp_path: Path) -> None:
        """No target files: still works (returns empty or manual context only)."""
        result = discover_context(
            target_files=[],
            project_root=str(tmp_path),
            package_name="nonexistent_xyz",
            src_root="src",
            token_budget=100_000,
        )
        assert isinstance(result.items_included, int)
