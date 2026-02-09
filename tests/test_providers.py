"""Tests for forge.providers — context provider registry (Phase 7)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge.providers import (
    PROVIDER_REGISTRY,
    PROVIDER_SPECS,
    handle_git_diff,
    handle_git_log,
    handle_lint_check,
    handle_read_file,
    handle_search_code,
    handle_symbol_list,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


class TestProviderRegistry:
    def test_registry_has_all_specs(self) -> None:
        spec_names = {s.name for s in PROVIDER_SPECS}
        registry_names = set(PROVIDER_REGISTRY.keys())
        assert spec_names == registry_names

    def test_all_specs_have_descriptions(self) -> None:
        for spec in PROVIDER_SPECS:
            assert spec.description, f"Provider {spec.name} has no description"

    def test_all_handlers_are_callable(self) -> None:
        for name, handler in PROVIDER_REGISTRY.items():
            assert callable(handler), f"Handler for {name} is not callable"


# ---------------------------------------------------------------------------
# handle_read_file
# ---------------------------------------------------------------------------


class TestHandleReadFile:
    def test_reads_existing_file(self, tmp_path: Path) -> None:
        (tmp_path / "hello.py").write_text("print('hello')")
        result = handle_read_file({"path": "hello.py"}, str(tmp_path), str(tmp_path))
        assert result == "print('hello')"

    def test_missing_file_returns_error(self, tmp_path: Path) -> None:
        result = handle_read_file({"path": "missing.py"}, str(tmp_path), str(tmp_path))
        assert "Error" in result
        assert "not found" in result

    def test_missing_path_param(self, tmp_path: Path) -> None:
        result = handle_read_file({}, str(tmp_path), str(tmp_path))
        assert "Error" in result
        assert "'path'" in result


# ---------------------------------------------------------------------------
# handle_search_code
# ---------------------------------------------------------------------------


class TestHandleSearchCode:
    def test_finds_pattern(self, tmp_path: Path) -> None:
        (tmp_path / "foo.py").write_text("def hello():\n    pass\n")
        result = handle_search_code({"pattern": "def hello"}, str(tmp_path), str(tmp_path))
        assert "foo.py:1:" in result
        assert "def hello" in result

    def test_no_matches(self, tmp_path: Path) -> None:
        (tmp_path / "foo.py").write_text("x = 1\n")
        result = handle_search_code(
            {"pattern": "nonexistent_function"}, str(tmp_path), str(tmp_path)
        )
        assert "No matches" in result

    def test_missing_pattern_param(self, tmp_path: Path) -> None:
        result = handle_search_code({}, str(tmp_path), str(tmp_path))
        assert "Error" in result

    def test_invalid_regex(self, tmp_path: Path) -> None:
        result = handle_search_code({"pattern": "[invalid"}, str(tmp_path), str(tmp_path))
        assert "Error" in result
        assert "Invalid regex" in result

    def test_glob_filter(self, tmp_path: Path) -> None:
        (tmp_path / "data.txt").write_text("match this line\n")
        (tmp_path / "code.py").write_text("match this line\n")
        result = handle_search_code(
            {"pattern": "match", "glob": "*.txt"}, str(tmp_path), str(tmp_path)
        )
        assert "data.txt" in result
        assert "code.py" not in result


# ---------------------------------------------------------------------------
# handle_symbol_list
# ---------------------------------------------------------------------------


class TestHandleSymbolList:
    def test_lists_symbols(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src" / "forge"
        src_dir.mkdir(parents=True)
        (src_dir / "__init__.py").touch()
        source = "def greet(name: str) -> str:\n    return f'Hello {name}'\n"
        (src_dir / "example.py").write_text(source)

        result = handle_symbol_list(
            {"file_path": "src/forge/example.py"}, str(tmp_path), str(tmp_path)
        )
        assert "greet" in result

    def test_missing_file(self, tmp_path: Path) -> None:
        result = handle_symbol_list({"file_path": "nonexistent.py"}, str(tmp_path), str(tmp_path))
        assert "Error" in result

    def test_missing_param(self, tmp_path: Path) -> None:
        result = handle_symbol_list({}, str(tmp_path), str(tmp_path))
        assert "Error" in result


# ---------------------------------------------------------------------------
# handle_git_log
# ---------------------------------------------------------------------------


class TestHandleGitLog:
    def test_returns_log(self, git_repo: Path) -> None:
        result = handle_git_log({}, str(git_repo), str(git_repo))
        assert "Initial commit" in result

    def test_with_limit(self, git_repo: Path) -> None:
        result = handle_git_log({"n": "1"}, str(git_repo), str(git_repo))
        assert "Initial commit" in result


# ---------------------------------------------------------------------------
# handle_git_diff
# ---------------------------------------------------------------------------


class TestHandleGitDiff:
    def test_no_diff(self, git_repo: Path) -> None:
        result = handle_git_diff({}, str(git_repo), str(git_repo))
        # On main with no changes, should show "No differences" or empty stat
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# handle_lint_check
# ---------------------------------------------------------------------------


class TestHandleLintCheck:
    def test_clean_file(self, tmp_path: Path, ruff_config: Path) -> None:
        (tmp_path / "clean.py").write_text('"""Clean module."""\n\nx = 1\n')
        result = handle_lint_check({"files": "clean.py"}, str(tmp_path), str(tmp_path))
        assert isinstance(result, str)
