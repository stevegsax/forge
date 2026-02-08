# Phase 4: Intelligent Context Assembly

## Goal

Replace manual `context_files` specification with automatic context discovery and importance ranking. Add token budget management so context assembly respects model token limits and packs the most relevant information first.

The deliverable: describe a task with only `target_files` and a description, and Forge automatically discovers which existing files to include as context — import dependencies, type interfaces, project structure — ranked by importance and packed within the target model's token budget.

## Problem Statement

Phases 1-3 require the caller (or planner) to manually specify `context_files` for every task, step, and sub-task. This has three problems:

1. **Manual burden.** The caller must know which files are relevant. For non-trivial codebases, this is error-prone and tedious.
2. **All-or-nothing inclusion.** `_read_context_files` reads entire file contents verbatim. A 500-line module included for one type signature wastes context budget on 490 irrelevant lines.
3. **No budget awareness.** Context assembly has no concept of token limits. Large `context_files` lists can exceed the model's context window, causing silent truncation or API errors.

## Prior Art

The design is informed by research into how production AI coding tools solve context selection:

- **Aider's repo map**: Parses all files with tree-sitter, builds a file-level dependency graph, runs PageRank (via NetworkX) to rank symbol importance, then uses binary search to fit top-ranked symbols within a token budget. The most well-documented open-source implementation.
- **GrepRAG** (ISSTA 2026): Demonstrated that lightweight lexical retrieval with BM25 re-ranking outperforms complex graph-based methods by 7-15% in exact match, at 0.02s latency. Suggests a strong lexical baseline should complement structural analysis.
- **AST-derived graphs vs. LLM-extracted graphs** (2025 benchmark): Deterministic AST-derived dependency graphs scored 15/15 correctness vs. 6/15 for vector-only RAG and 13/15 for LLM-extracted graphs — at 2.25x cost vs. 19-46x. Validates Forge's principle: "Deterministic work should be deterministic."
- **repominify**: Signature extraction + dependency graphs achieve 78-82% token reduction on typical Python projects while preserving the information LLMs need.
- **Sourcegraph Cody**: Deprecated embeddings-based retrieval in favor of native code search (BM25 + custom signals), finding it performed better without the infrastructure overhead.
- **Anthropic context engineering guidance**: "Model accuracy degrades as token volume increases, creating diminishing marginal returns on additional context." Recommends 40-60% context utilization, progressive disclosure, and sub-agent compaction.

## Scope

**In scope:**

- Import graph analysis via `grimp`: given target files, trace Python imports to discover relevant local modules (direct and transitive).
- Symbol extraction via Python's `ast` module: extract function signatures, class signatures (with method signatures), type aliases, and constants — without implementation bodies.
- Importance ranking via PageRank: build a file-level dependency graph and rank files by how central they are to the codebase.
- Token budget management: estimate tokens and pack context in priority order using binary search, truncating gracefully.
- Repo map: a compressed structural overview of the project (file paths + top-ranked signatures) for planner and task context.
- Automatic context discovery as a new context assembly strategy that augments (not replaces) manual `context_files`.
- Python language support only (using stdlib `ast` and `grimp`).

**Out of scope (deferred to later phases):**

- LSP integration (go-to-definition, find-all-references, type inference). Research confirms the hybrid tree-sitter/ast + LSP pattern is the consensus, but LSP adds operational complexity (server lifecycle management across worktrees) that is not justified until Python-only `ast` analysis proves insufficient.
- tree-sitter. For Python-only analysis of valid committed code, the stdlib `ast` module is simpler and sufficient. tree-sitter becomes the path when adding multi-language support.
- Non-Python language support (TypeScript, etc.).
- Embedding-based semantic search. Research (Sourcegraph Cody, GrepRAG) shows well-tuned structural/lexical approaches outperform embeddings for code, without the infrastructure overhead.
- Dynamic context adjustment based on LLM feedback.
- Playbook injection (deferred knowledge extraction feature).

## Architecture

### Code Intelligence Package

A new `src/forge/code_intel/` package provides code analysis. All analysis functions are pure (Function Core); file I/O is confined to the activity shell.

```
src/forge/code_intel/
├── __init__.py          # Public API re-exports
├── parser.py            # ast-based import extraction and symbol extraction
├── graph.py             # Import graph queries (via grimp) and PageRank ranking
├── repo_map.py          # Compressed structural overview of the repository
└── budget.py            # Token estimation and priority-based context packing
```

### Import Graph Analysis (`graph.py`)

Given a set of target files and a Python package, discover import dependencies and rank files by importance.

**Import graph via grimp:**

`grimp` builds a queryable, directed graph of imports within a Python package. It handles the hard parts — import resolution, transitive dependency tracking, cycle detection — with Rust-backed performance.

```python
import grimp

graph = grimp.build_graph("forge")

# Direct dependencies of a module
graph.find_modules_directly_imported_by("forge.activities.context")

# All upstream (transitive) dependencies
graph.find_upstream_modules("forge.activities.context")

# All downstream dependents
graph.find_downstream_modules("forge.models")

# Import details with line numbers
graph.get_import_details(
    importer="forge.activities.context",
    imported="forge.models",
)
```

**Importance ranking via PageRank:**

Following Aider's approach, build a file-level graph and run PageRank to identify the most central files:

1. Use `grimp` to build the import graph for the target package.
2. Convert to a NetworkX `DiGraph` (files as nodes, import relationships as edges).
3. Run `networkx.pagerank()` with personalization — seed weights on the target files so files closer to the task rank higher.
4. The output is a ranked list of files by importance score.

**Output model:**

```
RankedFile:
    file_path: str                          # Relative to project root
    module_name: str                        # Python module name
    importance: float                       # PageRank score (0.0 to 1.0)
    distance: int                           # Import distance from nearest target file
    relationship: Relationship              # direct_import, transitive_import, downstream

Relationship: StrEnum["direct_import", "transitive_import", "downstream"]

RankedFileSet:
    target_files: list[str]
    ranked_files: list[RankedFile]          # Sorted by importance descending
    graph_node_count: int                   # Total files in the import graph
```

### Symbol Extraction (`parser.py`)

Extract structured representations of a file's public interface using Python's `ast` module.

The `ast` module is the right tool here: zero dependencies, full access to Python type annotations (parameters, return types, variable annotations), and sufficient for analyzing valid committed code. tree-sitter would add value for error-tolerant parsing and multi-language support, but neither is needed in Phase 4.

**Approach:**

1. `ast.parse()` the source file.
2. Walk the tree with `ast.NodeVisitor`.
3. For `FunctionDef` / `AsyncFunctionDef` nodes: extract name, parameters (with type annotations and defaults), return annotation, decorators, and first-line docstring. Use `ast.unparse()` to reconstruct the signature line.
4. For `ClassDef` nodes: extract name, bases, and recurse into the body for method signatures (public methods only).
5. For `AnnAssign` / `Assign` at module level: extract type aliases (`Name = ...`) and constants (`UPPER_CASE = ...`).
6. For `Import` / `ImportFrom` nodes: extract import statements for dependency tracking.

**Extractable symbols:**

| Symbol type | What's extracted | What's excluded |
|-------------|-----------------|-----------------|
| Function | `def name(params) -> return_type:` + docstring (first line) | Function body |
| Async function | `async def name(params) -> return_type:` + docstring (first line) | Function body |
| Class | `class Name(bases):` + docstring (first line) + method signatures | Method bodies, private methods (`_prefix`) |
| Type alias | `Name = ...` (the full assignment via `ast.unparse`) | — |
| Constant | `NAME = ...` (UPPER_CASE module-level assignments) | — |
| `__all__` | The export list | — |

**Output model:**

```
SymbolSummary:
    file_path: str
    module_name: str
    symbols: list[ExtractedSymbol]
    imports: list[str]                      # Import statements (for dependency context)
    line_count: int                         # Total lines in the source file

ExtractedSymbol:
    name: str
    kind: SymbolKind                        # function, async_function, class, type_alias, constant
    signature: str                          # The signature line(s), reconstructed via ast.unparse
    docstring: str | None                   # First line of docstring, if present
    line_number: int

SymbolKind: StrEnum["function", "async_function", "class", "type_alias", "constant"]
```

### Repo Map (`repo_map.py`)

A compressed structural overview of the repository, following the pattern established by Aider's repo map. The repo map gives the LLM a bird's-eye view of the codebase without consuming excessive tokens.

**What it contains:**

- File paths of all Python modules in the project.
- For top-ranked files (by PageRank importance): function and class signatures.
- For lower-ranked files: file path only (presence without detail).

**Format:**

```
src/forge/models.py:
│class TransitionSignal(StrEnum):
│class ValidationConfig(BaseModel):
│class TaskDefinition(BaseModel):
│class Plan(BaseModel):
│class TaskResult(BaseModel):

src/forge/activities/context.py:
│def build_system_prompt(task, context_file_contents) -> str:
│def build_step_system_prompt(task, step, ...) -> str:
│async def assemble_context(input) -> AssembledContext:
│async def assemble_step_context(input) -> AssembledContext:

src/forge/git.py:
src/forge/tracing.py:
src/forge/worker.py:
```

**Sizing:** The repo map is generated to fit a configurable token budget (default: 2048 tokens). Binary search selects how many files get signature detail vs. path-only entries, maximizing information density within the budget. This follows Aider's approach exactly.

**Output model:**

```
RepoMap:
    content: str                            # The formatted map text
    files_with_signatures: int              # Files that got signature detail
    files_path_only: int                    # Files listed by path only
    estimated_tokens: int
```

### Token Budget Management (`budget.py`)

Context assembly is a bin-packing problem with a priority ordering (from D14 in DESIGN.md).

**Token estimation:** Use a character-based heuristic (4 characters per token) for speed. This avoids a dependency on tokenizer libraries and is accurate enough for budget management — the goal is preventing overflow, not exact counting.

**Priority tiers:**

| Priority | Content | Representation | Notes |
|----------|---------|----------------|-------|
| 1 (highest) | Task description, definition of "done" | Verbatim | Always included |
| 2 | Target file current content (if modifying existing files) | Full content | The files being changed |
| 3 | Direct imports of target files | Full content | Immediate dependencies |
| 4 | Interface context (transitive imports, ranked by PageRank) | Signatures only | Higher-ranked files first |
| 5 | Repo map | Compressed overview | Structural orientation |
| 6 | Manually specified `context_files` (if any) | Full content | Escape hatch for non-code context |

**Packing algorithm (binary search, following Aider):**

```
pack_context(items: list[ContextItem], budget_tokens: int) -> PackedContext:
    1. Sort items by priority (ascending = highest priority first).
    2. Within each priority tier, sort by importance (PageRank score descending).
    3. Accumulate items while total estimated tokens < budget.
    4. When an item would exceed the budget:
       a. If the item can be reduced (full content -> signatures only), reduce and retry.
       b. Otherwise, skip the item and continue to the next.
    5. Use binary search to determine the maximum number of tier-4 items
       (ranked by PageRank) that fit in the remaining budget.
    6. Return the packed context with stats.
```

**Target utilization:** Aim for 50-60% of the model's context window for input context, reserving the remainder for the LLM's output and internal reasoning. Research shows model accuracy degrades with increasing context volume, so more is not always better.

**Output model:**

```
ContextItem:
    file_path: str
    content: str                            # The text to include in the prompt
    representation: Representation          # full, signatures, repo_map
    priority: int
    importance: float                       # PageRank score (for ranking within tiers)
    estimated_tokens: int

Representation: StrEnum["full", "signatures", "repo_map"]

ContextBudget:
    model_max_tokens: int                   # Total model context window
    reserved_for_output: int                # Tokens reserved for LLM response
    reserved_for_task: int                  # Tokens used by task description, instructions
    available_for_context: int              # What remains for file context

PackedContext:
    items: list[ContextItem]
    repo_map: RepoMap | None
    total_estimated_tokens: int
    budget_utilization: float               # 0.0 to 1.0
    items_included: int                     # Items that fit
    items_reduced: int                      # Items downgraded from full to signatures
    items_truncated: int                    # Items that didn't fit at all
```

### Modified Context Assembly

The four existing assembly functions gain an optional automatic context discovery path. The change is additive — manual `context_files` continue to work unchanged.

**New assembly flow (when `auto_discover=True`):**

```
1. Read manually specified context_files (existing behavior).
2. If target_files are specified:
   a. Build import graph via grimp.
   b. Run PageRank to rank files by importance.
   c. Extract symbols for transitive imports.
   d. Read current content of target files (if they exist — for modification tasks).
   e. Generate repo map.
3. Pack all context items within the token budget (binary search).
4. Build prompt from packed context.
```

**Integration point:** The assembly activities call a new pure function `discover_context(target_files, project_root, manual_context, budget)` that returns `PackedContext`. The prompt-building functions accept `PackedContext` instead of `dict[str, str]`.

### Data Models

New models added to `models.py`:

```
ContextConfig:
    auto_discover: bool = True              # Enable automatic context discovery
    token_budget: int = 100_000             # Token budget for context (targets ~50% of 200k window)
    output_reserve: int = 16_000            # Tokens reserved for LLM output
    max_import_depth: int = 2               # How deep to trace imports
    include_repo_map: bool = True           # Include compressed repo map
    repo_map_tokens: int = 2048             # Token budget for the repo map
```

Modified models:

- `TaskDefinition`: added `context: ContextConfig = Field(default_factory=ContextConfig)`.
- `ForgeTaskInput`: no change (`ContextConfig` flows through `TaskDefinition`).
- `AssembledContext`: added `context_stats: ContextStats | None = None` for observability.

```
ContextStats:
    files_discovered: int                   # Files found via import graph
    files_included_full: int                # Files included with full content
    files_included_signatures: int          # Files included with signatures only
    files_truncated: int                    # Files that didn't fit
    total_estimated_tokens: int
    budget_utilization: float               # 0.0 to 1.0
    repo_map_tokens: int                    # Tokens used by the repo map
```

### Planner Integration

The planner benefits most from automatic context discovery. When `plan=True`:

1. The planner receives the repo map and import graph analysis, giving it a structural overview of the codebase without requiring manually specified context.
2. The planner sees key interfaces (signatures of important files), enabling it to produce better `context_files` lists for individual steps.
3. Steps and sub-tasks can still specify manual `context_files` (the planner already does this). Automatic discovery supplements what the planner specifies.

The planner prompt is updated to explain that context is automatically discovered from imports, so the planner should focus `context_files` on files that are relevant but not reachable via the import graph (e.g., config files, documentation, test fixtures).

## Project Structure

New and modified files:

```
src/forge/
├── code_intel/
│   ├── __init__.py             # New: public API re-exports
│   ├── parser.py               # New: ast-based import extraction, symbol extraction
│   ├── graph.py                # New: grimp integration, PageRank ranking
│   ├── repo_map.py             # New: compressed structural overview
│   └── budget.py               # New: token estimation and priority-based packing
├── models.py                   # Modified: ContextConfig, ContextStats
└── activities/
    ├── context.py              # Modified: automatic discovery integration
    └── planner.py              # Modified: planner prompt update, repo map inclusion
```

## Dependencies

New runtime dependencies:

- `grimp>=3.5` — Python import graph analysis with Rust-backed performance. Handles import resolution, transitive dependency tracking, and cycle detection.
- `networkx>=3.0` — Graph algorithms library. Used for PageRank ranking of files by importance.

No other new dependencies. Symbol extraction uses Python's stdlib `ast` module. Token estimation uses a character-based heuristic.

## Key Design Decisions

### D30: Python `ast` Over tree-sitter for Phase 4

**Decision:** Use Python's stdlib `ast` module for code analysis. Defer tree-sitter to a future multi-language phase.

**Rationale:** Phase 4 targets Python only, and Forge analyzes valid committed code (not in-progress edits). For this use case, `ast` is simpler (zero dependencies), provides full access to Python type annotations, and has a cleaner API for extracting structured information. tree-sitter's advantages — error-tolerant parsing, incremental re-parsing, multi-language support — are not needed yet. When non-Python language support is added, tree-sitter becomes the natural choice, and the `code_intel` package's interface can accommodate both backends behind the same API.

### D31: grimp Over Custom Import Resolution

**Decision:** Use `grimp` for import graph analysis rather than building import resolution from scratch.

**Rationale:** Import resolution in Python is surprisingly complex: relative imports, namespace packages, `src/` layouts, `__init__.py` re-exports, editable installs. `grimp` handles all of these correctly with Rust-backed performance and provides a rich query API (`find_upstream_modules`, `find_downstream_modules`, `find_shortest_chain`, `get_import_details` with line numbers). Building this from scratch would be a significant effort with many edge cases. `grimp` is actively maintained (used as the engine for `import-linter`).

### D32: PageRank for Importance Ranking

**Decision:** Rank files by importance using PageRank on the import graph, following Aider's approach.

**Rationale:** Not all files in the import graph are equally important for context. A utility module imported by 30 files is more important than a leaf module imported by one. PageRank naturally surfaces "hub" files (heavily imported utilities, base classes, shared types) that provide the most context value per token. Personalized PageRank with seed weights on target files biases the ranking toward files relevant to the current task. This approach is validated by Aider's production usage and by the AST-derived graph benchmark (15/15 correctness, deterministic, cheap).

### D33: Character-Based Token Estimation

**Decision:** Estimate tokens at 4 characters per token rather than using a tokenizer library.

**Rationale:** The purpose of token budgeting is overflow prevention, not exact accounting. A 4:1 character-to-token ratio is conservative for English and code. It avoids adding a dependency on `tiktoken` (OpenAI-specific) or model-specific tokenizers. If estimation accuracy becomes a problem, a tokenizer can be substituted behind the same interface.

### D34: Automatic Discovery Augments, Does Not Replace

**Decision:** Automatic context discovery supplements manual `context_files`. If both are specified, manual files are included at priority 6 (packed if budget allows).

**Rationale:** There are context files that import graph analysis cannot discover: configuration files, documentation, test fixtures, data samples, non-Python files. Manual specification remains the escape hatch for these. The planner already produces `context_files` lists — these continue to work. Automatic discovery fills the gap when the caller or planner doesn't know what to include.

### D35: Signature Extraction as Graceful Degradation

**Decision:** When a file is too large to include in full within the token budget, fall back to including its extracted signatures instead of omitting it entirely.

**Rationale:** A file's interface (function signatures, class definitions, type annotations) is almost always more useful than nothing. This implements the "graceful truncation" principle from D14: lower-priority items are reduced before being dropped. A 500-line module's 20-line signature summary fits easily and gives the LLM enough to produce correct imports and type-compatible code. Validated by repominify's finding of 78-82% token reduction while preserving the information LLMs need.

### D36: Repo Map as Standard Context

**Decision:** Include a compressed repo map (file paths + top-ranked signatures) in every context assembly, sized to a configurable token budget (default: 2048 tokens).

**Rationale:** The repo map gives the LLM structural orientation — which modules exist, what their public interfaces are, and how they relate. This is especially valuable for the planner, which must decompose tasks across a codebase it hasn't seen. Aider, Continue.dev, Kiro, and Augment Code all include some form of repo map. The fixed token budget (with binary search sizing) ensures the map never dominates the context window.

### D37: Import Depth Limit

**Decision:** Trace imports to a configurable depth (default: 2). Direct imports get full content; deeper imports get signatures only.

**Rationale:** Import graphs in real projects can be deep and wide. Unbounded traversal would pull in the entire project. Depth 2 captures the immediate dependency neighborhood — the files that target files import, and the files those import. Beyond that, signatures provide sufficient interface information. The depth limit is configurable for tasks that need broader or narrower context.

### D38: Defer LSP to a Future Phase

**Decision:** Do not integrate LSP in Phase 4. Defer to a future phase.

**Rationale:** Research confirms the hybrid ast + LSP pattern is the consensus for production tools (Claude Code, Kiro, OpenCode all use it). However, LSP requires managing language server lifecycles across multiple git worktrees, which adds significant operational complexity. The `ast` module + `grimp` combination provides sufficient analysis for Phase 4's goals (import discovery, symbol extraction, importance ranking). LSP becomes valuable when Forge needs precise cross-file reference tracking ("find all callers of this function"), type inference beyond annotations, or diagnostic feedback loops. These are natural additions once the foundation from Phase 4 is proven.

## Implementation Order

1. `code_intel/parser.py` — `ast`-based import extraction and symbol extraction. Pure functions, tested independently with fixture files.
2. `code_intel/graph.py` — `grimp` integration for import graph queries, NetworkX PageRank ranking. Tested with Forge's own codebase as a fixture.
3. `code_intel/repo_map.py` — compressed structural overview with binary search sizing. Tested with Forge's own codebase.
4. `code_intel/budget.py` — token estimation and priority-based packing. Tested with synthetic context items.
5. Models: add `ContextConfig`, `ContextStats` to `models.py`.
6. Integration: modify `activities/context.py` to call `discover_context` when `auto_discover=True`.
7. Planner prompt: update `activities/planner.py` to include repo map and explain automatic discovery.
8. End-to-end test: submit a task with only `target_files`, verify that context is automatically discovered and the LLM produces correct output.

## CLI Usage

```bash
# Automatic context discovery (default when auto_discover=True)
forge run \
    --task-id my-task \
    --description "Add error handling to the API client" \
    --target-file src/forge/api/client.py

# Disable automatic discovery (revert to manual context_files)
forge run \
    --task-id my-task \
    --description "Write a greeting module" \
    --target-file hello.py \
    --context-file existing_module.py \
    --no-auto-discover

# Custom token budget
forge run \
    --task-id my-task \
    --description "Refactor the validation module" \
    --target-file src/forge/activities/validate.py \
    --token-budget 100000
```

New CLI options:

- `--no-auto-discover` — Disable automatic context discovery.
- `--token-budget` — Total token budget for context (default: `100000`).
- `--max-import-depth` — How deep to trace imports (default: `2`).

## Edge Cases

- **No target_files specified:** Automatic discovery is skipped; repo map is still included if `include_repo_map=True`. Manual `context_files` are used if provided. Otherwise, the prompt contains only the task description and repo map.
- **Target file doesn't exist yet:** Import graph analysis skips non-existent files. This is the common case for "create a new file" tasks. The repo map and task description provide structural context.
- **Circular imports:** Handled by `grimp` (each module appears at most once in traversal results).
- **Dynamic imports / `importlib`:** Not detected by static analysis. These are rare in well-structured code and can be handled via manual `context_files`.
- **Monorepo with multiple packages:** `grimp.build_graph()` accepts a package name. Multiple packages can be analyzed by building separate graphs. Configurable package names can be added later.
- **Binary / non-Python files in context:** `_read_context_files` continues to skip unreadable files with a warning. The repo map only includes `.py` files.
- **Very large codebases:** `grimp`'s Rust backend and PageRank's efficient convergence handle large graphs well. The repo map's binary search sizing ensures output stays within token budget regardless of codebase size.

## Definition of Done

Phase 4 is complete when:

- Given a task with `target_files` and `auto_discover=True`, Forge automatically discovers import dependencies and includes them as context, ranked by importance.
- Direct imports are included with full content; transitive imports are included with signatures only; remaining files appear in the repo map.
- Context assembly respects the token budget and truncates gracefully (signatures before omission).
- `ContextStats` in `AssembledContext` reports discovery and packing metrics.
- The repo map provides a compressed structural overview within its own token budget.
- Manual `context_files` continue to work alongside automatic discovery.
- The planner produces better plans when automatic discovery provides richer context.
- All Phase 1, 2, and 3 tests continue to pass (backward compatible).
- Unit tests cover: import extraction, symbol extraction, PageRank ranking, repo map generation, budget packing.
- Integration test: a task with only `target_files` succeeds with automatically discovered context.
