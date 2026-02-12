# Phase 13: Tree-Sitter Multi-Language Support

> **Deferred to Release 2.** This phase will be implemented once Release 1 (Phases 1–14) is stable.

## Goal

Replace Python's stdlib `ast` with tree-sitter for code analysis, enabling Forge to handle non-Python languages (TypeScript, JavaScript, Go, Rust, Java, etc.) for symbol extraction, repo maps, and error context enrichment. The `ast` module served well for Python-only analysis (D30) but cannot parse other languages.

The deliverable: Forge's code intelligence works on multi-language codebases, producing accurate symbol summaries, repo maps, and error context for any language tree-sitter supports.

## Problem Statement

Phase 4's code intelligence package (`code_intel/`) uses Python's `ast` module for three critical functions:

1. **Symbol extraction** (`parser.py`): Extracts function signatures, class definitions, type aliases, and constants from Python source files. Used for signature-only context inclusion (D35) and repo map generation (D36).
2. **Repo map generation** (`repo_map.py`): Builds a compressed structural overview showing file paths and key signatures. Depends on symbol extraction output.
3. **Error context enrichment** (Phase 8): Finds the enclosing function/class for error line numbers to provide context around lint errors.

All three are Python-only because `ast.parse()` only handles Python syntax. This limits Forge to Python projects. For multi-language codebases (common in real projects — Python backend + TypeScript frontend, or Python with Go/Rust extensions), Forge's context assembly falls back to file-path-only entries with no structural information.

tree-sitter is the industry standard for multi-language code parsing. It supports 100+ languages via grammar packages, produces concrete syntax trees (CSTs) that map directly to source positions, handles malformed code gracefully (error-tolerant parsing), and has efficient incremental re-parsing. Every major AI coding tool uses it: Aider, Claude Code, Cursor, Continue.dev, Zed, and Neovim.

## Prior Art

- **Aider**: Uses tree-sitter as the primary code analysis engine. Supports 20+ languages for repo map generation. Each language has a tag query file (`.scm`) that extracts definitions (functions, classes, methods, interfaces). Falls back to file-path-only entries for unsupported languages.
- **Claude Code**: Uses tree-sitter for code search, symbol extraction, and navigation across all supported languages.
- **Cursor**: Uses tree-sitter for incremental parsing and code structure analysis across its entire language support matrix.
- **tree-sitter-languages**: Python package bundling pre-compiled tree-sitter grammars for 20+ popular languages.

## Scope

**In scope:**

- Replace `ast`-based symbol extraction with tree-sitter-based extraction.
- Support Python, TypeScript/JavaScript, Go, Rust, Java, and C/C++ initially (the most common languages in codebases Forge would target).
- Language-specific tag query files (`.scm`) for each supported language, following Aider's proven pattern.
- Tree-sitter-based repo map generation (replacing `ast`-based repo map).
- Tree-sitter-based error context enrichment (finding enclosing scope for error lines).
- Fallback to file-path-only entries for unsupported languages (graceful degradation).
- Retain the existing `SymbolSummary` and `ExtractedSymbol` output models — the interface is stable, only the extraction backend changes.

**Out of scope (deferred):**

- Incremental re-parsing (tree-sitter supports it, but Forge analyzes files from scratch each time — no persistent parse state).
- tree-sitter-based edit generation (using CST nodes to produce more precise search/replace strings).
- Language server protocol (LSP) integration alongside tree-sitter.
- Dynamic language detection (use file extension mapping initially).

## Architecture

### Language Registry

A registry maps file extensions to tree-sitter grammars and query files:

```
LanguageConfig:
    name: str                    # e.g. "python", "typescript"
    extensions: list[str]        # e.g. [".py", ".pyi"]
    grammar: str                 # tree-sitter grammar package name
    tag_query: str               # Path to .scm tag query file
```

```
LANGUAGE_REGISTRY: dict[str, LanguageConfig]
    ".py"  -> python
    ".pyi" -> python
    ".ts"  -> typescript
    ".tsx" -> typescript
    ".js"  -> javascript
    ".jsx" -> javascript
    ".go"  -> go
    ".rs"  -> rust
    ".java" -> java
    ".c"   -> c
    ".h"   -> c
    ".cpp" -> cpp
    ".hpp" -> cpp
```

### Tag Query Files

Following Aider's pattern, each language has a `.scm` file defining tree-sitter queries that extract definition nodes. These queries use tree-sitter's S-expression query syntax to match function definitions, class definitions, method definitions, type definitions, and other structural elements.

Example Python tag query (`queries/python.scm`):

```scheme
(function_definition name: (identifier) @name.definition.function) @definition.function
(class_definition name: (identifier) @name.definition.class) @definition.class
```

Example TypeScript tag query (`queries/typescript.scm`):

```scheme
(function_declaration name: (identifier) @name.definition.function) @definition.function
(class_declaration name: (identifier) @name.definition.class) @definition.class
(interface_declaration name: (identifier) @name.definition.interface) @definition.interface
(type_alias_declaration name: (type_identifier) @name.definition.type) @definition.type
(method_definition name: (property_identifier) @name.definition.method) @definition.method
```

### Symbol Extraction with tree-sitter

The `extract_symbols` function gains a multi-language path:

```
def extract_symbols(source: str, file_path: str, module_name: str) -> SymbolSummary:
    """Extract symbols using tree-sitter (or ast fallback for Python)."""
    ext = Path(file_path).suffix
    lang_config = LANGUAGE_REGISTRY.get(ext)

    if lang_config is None:
        # Unsupported language — return empty summary
        return SymbolSummary(file_path=file_path, module_name=module_name, ...)

    parser = get_parser(lang_config.name)
    tree = parser.parse(source.encode())

    query = get_tag_query(lang_config.name)
    captures = query.captures(tree.root_node)

    symbols = _captures_to_symbols(captures, source, lang_config)
    return SymbolSummary(file_path=file_path, module_name=module_name, symbols=symbols, ...)
```

The output model (`SymbolSummary`, `ExtractedSymbol`) remains unchanged. Only the extraction backend changes.

### Signature Formatting

Tree-sitter captures provide byte ranges into the source. For signature extraction:

1. Find the definition node (function, class, etc.) from captures.
2. Extract from the definition's start byte to the end of its declaration line (before the body).
3. For classes, include method signatures by recursing into the class body.

This produces signatures in the source language's syntax:

```typescript
// TypeScript
export function processRequest(req: Request, res: Response): Promise<void>:
export interface Config:
    readonly port: number
    readonly host: string
```

```go
// Go
func (s *Server) HandleRequest(w http.ResponseWriter, r *http.Request):
type Config struct:
```

### Error Context Enrichment

Phase 8's error context enrichment (`enrich_error_context`) currently uses `ast` to find the enclosing function/class for an error line. With tree-sitter:

```
def find_enclosing_scope(source: str, line_number: int, file_path: str) -> str | None:
    """Find the enclosing function/class definition for a given line."""
    ext = Path(file_path).suffix
    lang_config = LANGUAGE_REGISTRY.get(ext)
    if lang_config is None:
        return None

    parser = get_parser(lang_config.name)
    tree = parser.parse(source.encode())

    # Walk up from the node at the target line to find the nearest
    # function/class/method definition ancestor.
    node = tree.root_node.descendant_for_point_range((line_number - 1, 0), (line_number - 1, 0))
    while node:
        if node.type in lang_config.scope_types:
            return source[node.start_byte:node.end_byte]
        node = node.parent

    return None
```

### Parser Caching

Tree-sitter parser creation has overhead. A module-level cache holds parsers per language:

```
_parser_cache: dict[str, Parser] = {}

def get_parser(language_name: str) -> Parser:
    if language_name not in _parser_cache:
        _parser_cache[language_name] = _create_parser(language_name)
    return _parser_cache[language_name]
```

Similarly, compiled queries are cached per language.

## Project Structure

New and modified files:

```
src/forge/code_intel/
├── __init__.py                 # Modified: re-export new functions
├── parser.py                   # Modified: tree-sitter backend, ast fallback
├── repo_map.py                 # Modified: multi-language support
├── languages.py                # New: language registry, parser/query caching
└── queries/                    # New: tree-sitter tag query files
    ├── python.scm
    ├── typescript.scm
    ├── javascript.scm
    ├── go.scm
    ├── rust.scm
    ├── java.scm
    ├── c.scm
    └── cpp.scm
```

## Dependencies

New runtime dependencies:

- `tree-sitter>=0.23` — Tree-sitter Python bindings. Provides the parser, query, and tree APIs.
- `tree-sitter-python` — Python grammar for tree-sitter.
- `tree-sitter-typescript` — TypeScript/JavaScript grammar.
- `tree-sitter-go` — Go grammar.
- `tree-sitter-rust` — Rust grammar.
- `tree-sitter-java` — Java grammar.
- `tree-sitter-c` — C grammar.
- `tree-sitter-cpp` — C++ grammar.

Alternatively, the `tree-sitter-languages` package bundles pre-compiled grammars for 20+ languages in a single dependency, avoiding individual grammar packages.

## Key Design Decisions

### D64: Tree-Sitter Over Language-Specific Parsers

**Decision:** Use tree-sitter as the universal parsing backend for all languages, replacing Python's `ast` for Python and providing new support for other languages.

**Rationale:** D30 deferred tree-sitter because Phase 4 targeted Python only and `ast` was simpler. Phase 13 extends to multiple languages where `ast` is not an option. Rather than maintaining two parsing backends (ast for Python, tree-sitter for everything else), a single tree-sitter backend reduces maintenance and ensures consistent behavior across languages. tree-sitter's error-tolerant parsing also handles malformed code that `ast.parse()` rejects.

### D65: Tag Query Pattern (Following Aider)

**Decision:** Use language-specific `.scm` tag query files to define which AST nodes represent definitions, following Aider's proven pattern.

**Rationale:** Aider has battle-tested tag queries for 20+ languages. The pattern separates language-specific knowledge (what constitutes a "function definition" in Go vs TypeScript) from the extraction logic (walk the tree, extract matching nodes, format signatures). Adding a new language requires only writing a `.scm` file, not modifying Python code.

### D66: Stable Output Interface

**Decision:** Retain the existing `SymbolSummary` and `ExtractedSymbol` output models. Only the extraction backend changes.

**Rationale:** The output interface is consumed by repo map generation, budget packing, and context assembly. Changing the interface would cascade into many modules. The existing models are language-agnostic — `SymbolKind` (function, class, type_alias, constant) covers the common definition types across languages. Language-specific refinements (e.g., Go interfaces, Rust traits) map to the closest existing kind.

### D67: Graceful Degradation for Unsupported Languages

**Decision:** Files with unrecognized extensions produce an empty `SymbolSummary` (path only, no symbols). No error.

**Rationale:** Multi-language codebases often include configuration files, data files, or files in niche languages. The system should not fail on encountering them. File-path-only entries in the repo map still provide structural orientation. New languages can be added incrementally by writing a tag query file.

## Implementation Order

1. Add `tree-sitter` and initial grammar packages to dependencies.
2. Create `code_intel/languages.py` with `LanguageConfig`, `LANGUAGE_REGISTRY`, parser caching, and query caching.
3. Write tag query `.scm` files for Python, TypeScript, Go, Rust, Java, C/C++.
4. Update `code_intel/parser.py` to use tree-sitter for symbol extraction, with fallback to `ast` for Python if tree-sitter is unavailable.
5. Update `code_intel/repo_map.py` to handle multi-language symbol summaries.
6. Add `find_enclosing_scope` function for Phase 8 error context enrichment.
7. Tests for each language's symbol extraction, repo map generation, and error context.

## Edge Cases

- **tree-sitter not installed:** Fallback to `ast` for Python files; empty summaries for non-Python files. A warning is logged once.
- **Malformed source code:** tree-sitter produces a partial parse tree with ERROR nodes. Symbol extraction skips error regions and extracts what it can. This is strictly better than `ast.parse()` which raises `SyntaxError`.
- **Mixed-language files:** Use the primary file extension. `.tsx` files are parsed as TypeScript (which includes JSX support). `.h` files are parsed as C (not C++); `.hpp` files are parsed as C++.
- **Very large files:** tree-sitter's incremental parsing is O(n) for the initial parse. For files over 100k characters, symbol extraction may be slow but will complete. No special handling needed.
- **Grammar version mismatch:** Grammar packages must be compatible with the installed tree-sitter version. Pin compatible versions in `pyproject.toml`.
- **Unsupported Python version patterns:** tree-sitter's Python grammar may lag behind the latest Python syntax. The `ast` fallback ensures Python files always get full support.

## Definition of Done

Phase 13 is complete when:

- Symbol extraction works for Python, TypeScript/JavaScript, Go, Rust, Java, and C/C++.
- Repo maps include signatures for all supported languages.
- Error context enrichment finds enclosing scopes in all supported languages.
- Adding a new language requires only a `.scm` tag query file (no Python code changes).
- The output interface (`SymbolSummary`, `ExtractedSymbol`) is unchanged.
- Unsupported languages degrade gracefully to path-only entries.
- All existing Python-focused tests continue to pass.
- New tests cover: multi-language symbol extraction, tag query correctness, parser caching, error-tolerant parsing of malformed code, graceful degradation.
