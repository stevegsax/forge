"""Tests for forge.code_intel.repo_map — compressed structural overview."""

from __future__ import annotations

from forge.code_intel.graph import RankedFile, Relationship
from forge.code_intel.parser import ExtractedSymbol, SymbolKind, SymbolSummary
from forge.code_intel.repo_map import (
    RepoMap,
    _format_file_path_only,
    _format_file_with_signatures,
    estimate_tokens,
    generate_repo_map,
)

# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty(self) -> None:
        assert estimate_tokens("") == 0

    def test_short_text(self) -> None:
        assert estimate_tokens("abcd") == 1

    def test_longer_text(self) -> None:
        assert estimate_tokens("a" * 100) == 25


# ---------------------------------------------------------------------------
# _format_file_with_signatures
# ---------------------------------------------------------------------------


class TestFormatFileWithSignatures:
    def test_with_symbols(self) -> None:
        summary = SymbolSummary(
            file_path="models.py",
            module_name="pkg.models",
            symbols=[
                ExtractedSymbol(
                    name="Foo",
                    kind=SymbolKind.CLASS,
                    signature="class Foo:",
                    docstring=None,
                    line_number=1,
                ),
            ],
        )
        result = _format_file_with_signatures("models.py", summary)
        assert "models.py:" in result
        assert "\u2502class Foo:" in result

    def test_empty_symbols(self) -> None:
        summary = SymbolSummary(file_path="empty.py", module_name="pkg.empty")
        result = _format_file_with_signatures("empty.py", summary)
        assert result == "empty.py:"


# ---------------------------------------------------------------------------
# _format_file_path_only
# ---------------------------------------------------------------------------


class TestFormatFilePathOnly:
    def test_simple(self) -> None:
        assert _format_file_path_only("src/a.py") == "src/a.py:"


# ---------------------------------------------------------------------------
# generate_repo_map
# ---------------------------------------------------------------------------


def _make_ranked_file(path: str, module: str, importance: float = 0.5) -> RankedFile:
    return RankedFile(
        file_path=path,
        module_name=module,
        importance=importance,
        distance=1,
        relationship=Relationship.DIRECT_IMPORT,
    )


def _make_summary(path: str, module: str, sig_text: str) -> SymbolSummary:
    return SymbolSummary(
        file_path=path,
        module_name=module,
        symbols=[
            ExtractedSymbol(
                name="func",
                kind=SymbolKind.FUNCTION,
                signature=sig_text,
                docstring=None,
                line_number=1,
            ),
        ],
    )


class TestGenerateRepoMap:
    def test_empty_files(self) -> None:
        result = generate_repo_map([], {}, token_budget=2048)
        assert isinstance(result, RepoMap)
        assert result.content == ""
        assert result.files_with_signatures == 0
        assert result.files_path_only == 0

    def test_all_fit_with_signatures(self) -> None:
        ranked = [
            _make_ranked_file("a.py", "pkg.a", 0.8),
            _make_ranked_file("b.py", "pkg.b", 0.5),
        ]
        summaries = {
            "a.py": _make_summary("a.py", "pkg.a", "def foo():"),
            "b.py": _make_summary("b.py", "pkg.b", "def bar():"),
        }
        result = generate_repo_map(ranked, summaries, token_budget=10000)
        assert result.files_with_signatures == 2
        assert result.files_path_only == 0
        assert "a.py:" in result.content
        assert "\u2502def foo():" in result.content
        assert "\u2502def bar():" in result.content

    def test_budget_forces_path_only(self) -> None:
        ranked = [
            _make_ranked_file("a.py", "pkg.a", 0.8),
            _make_ranked_file("b.py", "pkg.b", 0.5),
        ]
        # Very long signatures that won't fit
        summaries = {
            "a.py": _make_summary("a.py", "pkg.a", "def " + "x" * 200 + "():"),
            "b.py": _make_summary("b.py", "pkg.b", "def " + "y" * 200 + "():"),
        }
        # Tight budget: only path-only entries will fit
        result = generate_repo_map(ranked, summaries, token_budget=5)
        assert result.files_path_only > 0
        assert result.estimated_tokens <= 5

    def test_partial_signatures(self) -> None:
        """Some files get signatures, others get path-only."""
        ranked = [
            _make_ranked_file("important.py", "pkg.important", 0.9),
            _make_ranked_file("less.py", "pkg.less", 0.3),
        ]
        summaries = {
            "important.py": _make_summary("important.py", "pkg.important", "def critical():"),
            "less.py": _make_summary("less.py", "pkg.less", "def minor():"),
        }
        # Budget fits one with sigs but not both
        full_content_tokens = estimate_tokens(
            "important.py:\n\u2502def critical():\nless.py:\n\u2502def minor():"
        )
        one_sig_content_tokens = estimate_tokens("important.py:\n\u2502def critical():\nless.py:")

        budget = (full_content_tokens + one_sig_content_tokens) // 2
        result = generate_repo_map(ranked, summaries, token_budget=budget)
        assert result.files_with_signatures == 1
        assert result.files_path_only == 1

    def test_respects_importance_order(self) -> None:
        ranked = [
            _make_ranked_file("first.py", "pkg.first", 0.9),
            _make_ranked_file("second.py", "pkg.second", 0.5),
        ]
        summaries = {
            "first.py": _make_summary("first.py", "pkg.first", "def a():"),
            "second.py": _make_summary("second.py", "pkg.second", "def b():"),
        }
        result = generate_repo_map(ranked, summaries, token_budget=10000)
        # first.py should appear before second.py
        assert result.content.index("first.py") < result.content.index("second.py")
