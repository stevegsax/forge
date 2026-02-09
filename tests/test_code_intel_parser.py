"""Tests for forge.code_intel.parser — AST symbol extraction."""

from __future__ import annotations

from forge.code_intel.parser import (
    ExtractedSymbol,
    SymbolKind,
    SymbolSummary,
    extract_symbols,
    format_signatures,
)

# ---------------------------------------------------------------------------
# extract_symbols
# ---------------------------------------------------------------------------


class TestExtractSymbolsFunctions:
    def test_simple_function(self) -> None:
        source = "def greet(name: str) -> str:\n    return f'Hello, {name}'\n"
        summary = extract_symbols(source, "greet.py", "pkg.greet")
        assert len(summary.symbols) == 1
        sym = summary.symbols[0]
        assert sym.name == "greet"
        assert sym.kind == SymbolKind.FUNCTION
        assert "def greet(name: str) -> str:" in sym.signature
        assert sym.line_number == 1

    def test_async_function(self) -> None:
        source = "async def fetch(url: str) -> bytes:\n    pass\n"
        summary = extract_symbols(source, "fetch.py", "pkg.fetch")
        assert summary.symbols[0].kind == SymbolKind.ASYNC_FUNCTION
        assert "async def fetch" in summary.symbols[0].signature

    def test_function_with_docstring(self) -> None:
        source = 'def greet():\n    """Say hello."""\n    pass\n'
        summary = extract_symbols(source, "a.py", "pkg.a")
        assert summary.symbols[0].docstring == "Say hello."

    def test_function_without_docstring(self) -> None:
        source = "def greet():\n    pass\n"
        summary = extract_symbols(source, "a.py", "pkg.a")
        assert summary.symbols[0].docstring is None

    def test_function_no_return_annotation(self) -> None:
        source = "def do_stuff():\n    pass\n"
        summary = extract_symbols(source, "a.py", "pkg.a")
        assert "-> " not in summary.symbols[0].signature


class TestExtractSymbolsClasses:
    def test_simple_class(self) -> None:
        source = "class Foo:\n    pass\n"
        summary = extract_symbols(source, "a.py", "pkg.a")
        assert len(summary.symbols) == 1
        sym = summary.symbols[0]
        assert sym.name == "Foo"
        assert sym.kind == SymbolKind.CLASS

    def test_class_with_bases(self) -> None:
        source = "class Foo(Bar, Baz):\n    pass\n"
        summary = extract_symbols(source, "a.py", "pkg.a")
        assert "class Foo(Bar, Baz):" in summary.symbols[0].signature

    def test_class_with_public_methods(self) -> None:
        source = (
            "class Foo:\n"
            "    def __init__(self):\n"
            "        pass\n"
            "    def public(self) -> int:\n"
            "        return 1\n"
            "    def _private(self):\n"
            "        pass\n"
        )
        summary = extract_symbols(source, "a.py", "pkg.a")
        sig = summary.symbols[0].signature
        assert "__init__" in sig
        assert "public" in sig
        assert "_private" not in sig

    def test_class_with_docstring(self) -> None:
        source = 'class Foo:\n    """A foo class."""\n    pass\n'
        summary = extract_symbols(source, "a.py", "pkg.a")
        assert summary.symbols[0].docstring == "A foo class."


class TestExtractSymbolsConstants:
    def test_uppercase_constant(self) -> None:
        source = "MAX_RETRIES = 3\n"
        summary = extract_symbols(source, "a.py", "pkg.a")
        assert len(summary.symbols) == 1
        sym = summary.symbols[0]
        assert sym.name == "MAX_RETRIES"
        assert sym.kind == SymbolKind.CONSTANT
        assert "MAX_RETRIES = 3" in sym.signature

    def test_lowercase_not_extracted(self) -> None:
        source = "my_var = 42\n"
        summary = extract_symbols(source, "a.py", "pkg.a")
        assert len(summary.symbols) == 0

    def test_dunder_all(self) -> None:
        source = "__all__ = ['Foo', 'bar']\n"
        summary = extract_symbols(source, "a.py", "pkg.a")
        assert len(summary.symbols) == 1
        assert summary.symbols[0].name == "__all__"


class TestExtractSymbolsTypeAliases:
    def test_type_alias_annotation(self) -> None:
        source = "from typing import TypeAlias\nMyType: TypeAlias = int | str\n"
        summary = extract_symbols(source, "a.py", "pkg.a")
        type_symbols = [s for s in summary.symbols if s.kind == SymbolKind.TYPE_ALIAS]
        assert len(type_symbols) == 1
        assert type_symbols[0].name == "MyType"


class TestExtractSymbolsImports:
    def test_import_statement(self) -> None:
        source = "import os\nimport sys\n"
        summary = extract_symbols(source, "a.py", "pkg.a")
        assert "import os" in summary.imports
        assert "import sys" in summary.imports

    def test_from_import(self) -> None:
        source = "from pathlib import Path\n"
        summary = extract_symbols(source, "a.py", "pkg.a")
        assert "from pathlib import Path" in summary.imports


class TestExtractSymbolsMetadata:
    def test_file_path_and_module(self) -> None:
        summary = extract_symbols("", "src/a.py", "pkg.a")
        assert summary.file_path == "src/a.py"
        assert summary.module_name == "pkg.a"

    def test_line_count(self) -> None:
        source = "a = 1\nb = 2\nc = 3\n"
        summary = extract_symbols(source, "a.py", "pkg.a")
        assert summary.line_count == 4  # 3 lines + trailing newline

    def test_syntax_error_returns_empty_summary(self) -> None:
        source = "def broken(:\n"
        summary = extract_symbols(source, "bad.py", "pkg.bad")
        assert summary.symbols == []
        assert summary.imports == []


class TestExtractSymbolsMixed:
    def test_mixed_source(self) -> None:
        source = (
            "from __future__ import annotations\n"
            "\n"
            "MAX_SIZE = 100\n"
            "\n"
            "class Config:\n"
            '    """Configuration model."""\n'
            "    def __init__(self, size: int):\n"
            "        self.size = size\n"
            "\n"
            "def validate(config: Config) -> bool:\n"
            '    """Check validity."""\n'
            "    return config.size <= MAX_SIZE\n"
        )
        summary = extract_symbols(source, "config.py", "pkg.config")
        names = {s.name for s in summary.symbols}
        assert "MAX_SIZE" in names
        assert "Config" in names
        assert "validate" in names
        assert len(summary.imports) == 1


# ---------------------------------------------------------------------------
# format_signatures
# ---------------------------------------------------------------------------


class TestFormatSignatures:
    def test_empty_symbols(self) -> None:
        summary = SymbolSummary(file_path="empty.py", module_name="pkg.empty")
        result = format_signatures(summary)
        assert result == "empty.py:"

    def test_with_symbols(self) -> None:
        summary = SymbolSummary(
            file_path="a.py",
            module_name="pkg.a",
            symbols=[
                ExtractedSymbol(
                    name="greet",
                    kind=SymbolKind.FUNCTION,
                    signature="def greet(name: str) -> str:",
                    docstring=None,
                    line_number=1,
                ),
            ],
        )
        result = format_signatures(summary)
        assert "a.py:" in result
        assert "│def greet(name: str) -> str:" in result

    def test_multiline_class_signature(self) -> None:
        summary = SymbolSummary(
            file_path="a.py",
            module_name="pkg.a",
            symbols=[
                ExtractedSymbol(
                    name="Foo",
                    kind=SymbolKind.CLASS,
                    signature="class Foo:\n    def __init__(self):",
                    docstring=None,
                    line_number=1,
                ),
            ],
        )
        result = format_signatures(summary)
        assert "│class Foo:" in result
        assert "│    def __init__(self):" in result
