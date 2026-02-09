"""AST-based symbol extraction for Python source files.

Pure functions only — takes source strings, returns models.
No file I/O.
"""

from __future__ import annotations

import ast
from enum import StrEnum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SymbolKind(StrEnum):
    """Classification of extracted symbols."""

    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    CLASS = "class"
    TYPE_ALIAS = "type_alias"
    CONSTANT = "constant"


class ExtractedSymbol(BaseModel):
    """A single extracted symbol from a Python source file."""

    name: str
    kind: SymbolKind
    signature: str = Field(description="Reconstructed signature line(s).")
    docstring: str | None = Field(default=None, description="First line of docstring, if present.")
    line_number: int


class SymbolSummary(BaseModel):
    """Summary of all extracted symbols from a single file."""

    file_path: str
    module_name: str
    symbols: list[ExtractedSymbol] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    line_count: int = 0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_docstring(node: ast.AST) -> str | None:
    """Extract the first line of a docstring from a function or class node."""
    if not (isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and node.body):
        return None
    first = node.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        first_line = first.value.value.strip().split("\n")[0].strip()
        return first_line if first_line else None
    return None


def _is_constant(node: ast.Assign) -> bool:
    """Check if an assignment is a module-level constant (UPPER_CASE name)."""
    if len(node.targets) != 1:
        return False
    target = node.targets[0]
    return isinstance(target, ast.Name) and target.id.isupper()


def _is_type_alias(node: ast.AnnAssign | ast.Assign) -> bool:
    """Check if an assignment is a type alias (TypeAlias annotation or Name = ... pattern)."""
    if isinstance(node, ast.AnnAssign):
        # x: TypeAlias = ...
        if isinstance(node.annotation, ast.Name) and node.annotation.id == "TypeAlias":
            return True
        if isinstance(node.annotation, ast.Attribute) and node.annotation.attr == "TypeAlias":
            return True
    return False


def _format_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Format a function/async function signature line."""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    # Reconstruct args
    args = ast.unparse(node.args) if node.args else ""
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({args}){ret}:"


def _extract_function_symbol(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ExtractedSymbol:
    """Extract a function symbol."""
    kind = (
        SymbolKind.ASYNC_FUNCTION if isinstance(node, ast.AsyncFunctionDef) else SymbolKind.FUNCTION
    )
    return ExtractedSymbol(
        name=node.name,
        kind=kind,
        signature=_format_function_signature(node),
        docstring=_extract_docstring(node),
        line_number=node.lineno,
    )


def _extract_class_symbol(node: ast.ClassDef) -> ExtractedSymbol:
    """Extract a class symbol with public method signatures."""
    bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
    class_sig = f"class {node.name}({bases}):" if bases else f"class {node.name}:"

    # Extract public method signatures
    method_sigs: list[str] = []
    for item in node.body:
        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and (
            not item.name.startswith("_") or item.name == "__init__"
        ):
            method_sigs.append(f"    {_format_function_signature(item)}")

    full_sig = class_sig + "\n" + "\n".join(method_sigs) if method_sigs else class_sig

    return ExtractedSymbol(
        name=node.name,
        kind=SymbolKind.CLASS,
        signature=full_sig,
        docstring=_extract_docstring(node),
        line_number=node.lineno,
    )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def extract_symbols(source: str, file_path: str, module_name: str) -> SymbolSummary:
    """Parse source with ast.parse() and extract top-level symbols.

    Extracts functions, async functions, classes (with public method signatures),
    type aliases, constants, __all__, and imports.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return SymbolSummary(
            file_path=file_path,
            module_name=module_name,
            line_count=source.count("\n") + 1,
        )

    symbols: list[ExtractedSymbol] = []
    imports: list[str] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            symbols.append(_extract_function_symbol(node))

        elif isinstance(node, ast.ClassDef):
            symbols.append(_extract_class_symbol(node))

        elif isinstance(node, ast.AnnAssign) and _is_type_alias(node):
            symbols.append(
                ExtractedSymbol(
                    name=ast.unparse(node.target) if node.target else "",
                    kind=SymbolKind.TYPE_ALIAS,
                    signature=ast.unparse(node),
                    docstring=None,
                    line_number=node.lineno,
                )
            )

        elif isinstance(node, ast.Assign):
            # Check for __all__
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "__all__"
            ):
                symbols.append(
                    ExtractedSymbol(
                        name="__all__",
                        kind=SymbolKind.CONSTANT,
                        signature=ast.unparse(node),
                        docstring=None,
                        line_number=node.lineno,
                    )
                )
            elif _is_constant(node):
                symbols.append(
                    ExtractedSymbol(
                        name=ast.unparse(node.targets[0]),
                        kind=SymbolKind.CONSTANT,
                        signature=ast.unparse(node),
                        docstring=None,
                        line_number=node.lineno,
                    )
                )

        elif isinstance(node, ast.Import | ast.ImportFrom):
            imports.append(ast.unparse(node))

    return SymbolSummary(
        file_path=file_path,
        module_name=module_name,
        symbols=symbols,
        imports=imports,
        line_count=source.count("\n") + 1,
    )


def format_signatures(summary: SymbolSummary) -> str:
    """Compact text representation for signature-only inclusion.

    Produces a formatted block showing the file path header followed
    by each symbol's signature, prefixed with │.
    """
    if not summary.symbols:
        return f"{summary.file_path}:"

    lines = [f"{summary.file_path}:"]
    for symbol in summary.symbols:
        for sig_line in symbol.signature.split("\n"):
            lines.append(f"│{sig_line}")

    return "\n".join(lines)
