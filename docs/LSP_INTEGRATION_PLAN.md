# LSP Integration Plan for Forge Context Generator

## Executive Summary

This report investigates what it will take to add Language Server Protocol (LSP) support to Forge's context generator, how LSP integrates with the tools planned for Phase 4, when LSP is the right approach, and when alternatives are better. The analysis is grounded in Forge's existing architecture (Phases 1-4), a review of 15+ production tools and research papers, and the current state of LSP in AI coding agents as of early 2026.

**Key finding:** LSP is not a replacement for Phase 4's `ast` + `grimp` approach — it is a complementary layer that addresses specific gaps Phase 4 cannot fill. The right integration point is a future Phase 5, after Phase 4's foundation is proven. LSP should be added incrementally, starting with diagnostics (the highest-value, lowest-complexity feature), then expanding to cross-file reference tracking and type inference as needs justify the operational cost.

---

## 1. What Phase 4 Provides (and Where It Stops)

Phase 4's `code_intel` package delivers four capabilities using Python's `ast` module and `grimp`:

| Capability | Mechanism | Limitation |
|---|---|---|
| **Import discovery** | `grimp` traces Python import graph | Static analysis only; misses dynamic imports, `importlib`, plugin systems |
| **Symbol extraction** | `ast` module parses function/class signatures | Python only; no type inference beyond annotations |
| **Importance ranking** | PageRank on import graph via NetworkX | File-level granularity; cannot rank individual symbols within a file |
| **Token budgeting** | Character-based estimation with priority packing | No awareness of what the LLM actually needs for a specific edit |

These capabilities are sufficient for Phase 4's goals: automatic context discovery for Python codebases where target files are known. Decision D30 correctly identifies that `ast` is simpler and sufficient for valid committed Python code, and D38 correctly defers LSP to a future phase.

### Gaps That LSP Would Fill

1. **Cross-file reference tracking.** "Find all callers of this function" is not possible with import graph analysis alone. `grimp` tells you which modules import a module; it does not tell you which specific functions call a specific function. LSP's `textDocument/references` provides this.

2. **Type inference beyond annotations.** Python code with incomplete type annotations (common in practice) cannot be fully analyzed by `ast`. Pyright's type inference engine resolves inferred types, union narrowing, and generic specialization — information that helps the LLM produce type-correct code.

3. **Diagnostic feedback loops.** After the LLM writes code, LSP diagnostics can immediately report type errors, unresolved imports, and semantic issues — before running the full validation pipeline. This enables a tighter retry loop: generate, check diagnostics, fix, validate.

4. **Rename/refactor safety.** Knowing all references to a symbol before renaming it is essential for safe refactoring tasks. `ast` can find definitions; only LSP (or equivalent) can reliably find all usages across a codebase.

5. **Multi-language support.** When Forge expands beyond Python, LSP provides a uniform interface across languages. A Go task and a TypeScript task use the same `textDocument/definition`, `textDocument/references`, and `textDocument/diagnostics` requests — only the server binary differs.

---

## 2. How Production Tools Use LSP (Landscape Review)

### 2.1 Claude Code

Claude Code v2.0.74 (December 2025) added native LSP support for 11 languages. The architecture follows a three-layer design: LSP servers (external processes), a manager layer for lifecycle control, and analyzer layers for high-level operations. Five LSP operations are exposed: `goToDefinition`, `findReferences`, `hover`, `documentSymbol`, and `getDiagnostics`. LSP initialization happens during CLI startup, before the main conversation loop.

**Performance claim:** 50ms for finding all call sites of a function via LSP vs. 45 seconds with text search — a 900x improvement.

**Relevance to Forge:** Claude Code's architecture validates the three-layer design (servers, lifecycle manager, analyzer API). However, Claude Code runs interactively in a single workspace. Forge must manage servers across multiple parallel git worktrees, which is a fundamentally different operational model.

### 2.2 OpenCode

OpenCode integrates LSP primarily for **diagnostics feedback loops**. When the LLM makes a change, OpenCode sends `textDocument/didChange` to the server, waits for diagnostics (debounced by 150ms), and feeds them back into the LLM's context. Notably, while OpenCode's LSP client supports the full protocol, **only diagnostics are currently exposed to the AI assistant**.

**Key insight:** OpenCode's experience suggests diagnostics alone provide the highest value-to-complexity ratio. The full LSP protocol is available but not yet needed for effective agent operation.

### 2.3 Kiro

Kiro implements a **two-tier architecture**: Tree-sitter (built-in) provides out-of-the-box code intelligence for 18 languages with fuzzy symbol search and definition lookup. LSP (optional) adds enhanced precision: find references, hover documentation, rename refactoring, and diagnostics. LSP servers are spawned per-workspace and communicate via JSON-RPC over stdio.

**Relevance to Forge:** Kiro's two-tier model maps naturally to Forge's phased approach: Phase 4's `ast`-based analysis is the "built-in" tier; LSP becomes the optional precision tier. The per-workspace spawning model is directly applicable to Forge's per-worktree execution.

### 2.4 Aider

Aider uses tree-sitter exclusively for its repo map — no LSP. The team acknowledged that LSP would be better but found it "more cumbersome to deploy for a broad array of languages." Their approach: parse all files with tree-sitter, build a file-level dependency graph, run PageRank (via NetworkX), and use binary search to fit top-ranked symbols within a token budget.

**Relevance to Forge:** Phase 4 already follows Aider's approach (D32, D36). Aider's deliberate avoidance of LSP reinforces that the `ast`/tree-sitter + import graph approach is sufficient for repo map and context ranking. LSP becomes necessary only for precise reference tracking and diagnostics — capabilities Aider doesn't need because it operates interactively with a human who can provide that context.

### 2.5 Continue.dev

Continue combines tree-sitter and LSP in its autocomplete context engine using "root path context." Tree-sitter queries identify important sub-nodes (like type annotations), then LSP `goToDefinition` recursively resolves type definitions. Continue notes a key limitation: "there is little control over responsiveness" with LSP — each language server's performance varies, and this is exacerbated by per-keystroke requests.

**Relevance to Forge:** Forge operates in batch mode, not per-keystroke. LSP responsiveness is less of a concern when you're making one request per task step rather than per keystroke. Continue's caching strategy (flat cache of N most recent LSP requests with high hit rate due to AST root path stability) is worth adopting.

### 2.6 Augment Code

Augment Code's Context Engine processes up to 500,000 files by analyzing dependencies, architectural patterns, and cross-repository relationships. They rank #1 on SWE-Bench Pro. Their approach goes beyond LSP — they build a comprehensive map that includes commit history, codebase patterns, external sources, and "tribal knowledge."

**Relevance to Forge:** Augment validates that "context quality" (smart understanding of what tokens represent) matters more than "context quantity" (bigger windows). This aligns with Forge's D14 (context assembly as a packing problem) and supports the case for LSP as a precision tool, not a brute-force one.

### 2.7 Sourcegraph Cody

Cody deprecated embeddings-based retrieval in favor of native Sourcegraph Search (BM25 + custom signals). Their reasoning: embeddings required sending code to OpenAI, needed ongoing maintenance, and couldn't scale to >100K repositories. They replaced it with keyword search + adapted BM25 ranking.

**Relevance to Forge:** Reinforces that embeddings are unnecessary overhead for code retrieval (aligning with Phase 4's decision to defer embedding-based search). Structural/lexical approaches outperform embeddings for code, consistent with D38's rationale.

---

## 3. Research Findings

### 3.1 LSPRAG (ICSE 2026)

LSPRAG uses LSP for language-agnostic context retrieval in unit test generation. The framework combines LSP's `goToDefinition` and `findReferences` with AST structural information to extract concise context for focal methods. A compile-free self-repair mechanism uses LSP diagnostics to fix syntax errors in generated tests without compilation.

**Results:** Line coverage improved by up to 174% (Go), 213% (Java), and 31% (Python) compared to baselines.

**Key insight for Forge:** LSPRAG's hybrid LSP + AST approach validates the pattern Forge should follow. The compile-free diagnostic loop is directly applicable to Forge's validation step — LSP diagnostics can serve as a fast pre-check before running the full `ruff` + test pipeline.

### 3.2 LSPAI (FSE Industry 2025)

LSPAI uses four LSP providers for dependency analysis: Definition Provider (jump to declaration), Reference Provider (list all usages), Symbol Provider (file/class/function tree), and Diagnosis Provider (error detection). Applied at Tencent, it achieved 145% coverage improvement for Java, 931% for Go, and 95% for Python compared to Copilot.

**Key insight for Forge:** LSPAI demonstrates the specific LSP operations that matter most for code generation context. The four-provider pattern (definition, reference, symbol, diagnosis) maps to the four LSP capabilities Forge should prioritize.

### 3.3 GrepRAG (2026)

GrepRAG showed that lightweight lexical retrieval with identifier-weighted re-ranking outperforms complex graph-based methods by 7-15% in exact match, at 0.02s latency. Key finding: BM25 emphasizes global token overlap rather than identifiers close to the completion site, which can be problematic.

**Key insight for Forge:** Lexical search remains a strong baseline. LSP should be used for precision queries (definition, references) where lexical search fails — not as a replacement for the import graph + PageRank approach.

### 3.4 AST-Derived Graphs Benchmark (2025)

Deterministic AST-derived dependency graphs scored 15/15 correctness vs. 6/15 for vector-only RAG and 13/15 for LLM-extracted graphs — at 2.25x cost vs. 19-46x for LLM-extracted.

**Key insight for Forge:** Validates Forge's principle "deterministic work should be deterministic." Phase 4's `ast` + `grimp` approach is on the right track. LSP adds value on top of this foundation — it doesn't replace it.

### 3.5 Anthropic Context Engineering Guidance

"Model accuracy degrades as token volume increases, creating diminishing marginal returns on additional context." Recommends 40-60% context utilization, progressive disclosure, and sub-agent compaction.

**Key insight for Forge:** LSP-retrieved context should be surgically precise (specific definitions, specific references) rather than broad. The goal is higher signal-to-noise ratio, not more tokens.

---

## 4. When to Use LSP vs. Alternatives

| Scenario | Right Tool | Why |
|---|---|---|
| **Discover which files are relevant to a task** | Import graph (`grimp`) + PageRank | LSP operates on individual symbols, not file-level relevance. Import graphs give file-level structure efficiently. |
| **Extract function/class signatures for context** | `ast` module (Python) or tree-sitter (multi-language) | Parsing is faster than LSP queries, no server lifecycle to manage, works on committed code. |
| **Build a repo map** | tree-sitter / `ast` + PageRank | Repo maps need structural overview, not semantic precision. Aider, Kiro, Continue all use tree-sitter for this. |
| **Find all callers of a specific function** | LSP `textDocument/references` | Import graph tells you which modules import a module. Only LSP (or equivalent analysis) can trace function-level call sites. |
| **Resolve inferred types for a symbol** | LSP `textDocument/hover` or Pyright CLI | `ast` only sees explicitly annotated types. Pyright infers types through control flow, generics, and overloads. |
| **Get diagnostics after LLM writes code** | LSP `textDocument/diagnostics` | Faster than running full lint/type-check CLI. Provides immediate, incremental feedback. |
| **Validate type correctness of generated code** | Pyright CLI (`--outputjson`) | For batch validation, the CLI is simpler than managing an LSP server. Use LSP diagnostics only when you need incremental, per-edit feedback. |
| **Find the definition of an imported symbol** | LSP `textDocument/definition` | Useful for resolving symbols from third-party libraries where source may not be in the import graph. |
| **Rank files by importance** | PageRank on import graph | LSP has no concept of importance ranking. This is a graph-level operation, not a symbol-level one. |
| **Chunk code for context packing** | tree-sitter / `ast` | Chunking requires parsing the full file structure. LSP is a query interface, not a parsing tool. |
| **Multi-language analysis** | tree-sitter (syntax) + LSP (semantics) | tree-sitter provides the syntactic layer across all languages. LSP adds per-language semantic precision. |
| **Refactoring (rename symbol safely)** | LSP `textDocument/rename` | Requires knowing all references across all files. Only LSP can do this reliably. |

### Decision Framework

Use LSP when you need **precision about a specific symbol** — its type, its callers, its definition, its diagnostics. Use `ast`/tree-sitter/`grimp` when you need **structural understanding of the codebase** — its shape, its dependencies, its important files, its signatures.

---

## 5. Integration Architecture for Forge

### 5.1 Phased Approach

**Phase 4 (current plan, no changes):** `ast` + `grimp` + PageRank. Import graph analysis, symbol extraction, repo map, token budgeting. Python only.

**Phase 5a — LSP Diagnostics:** Add LSP diagnostic feedback to the validation pipeline. After `write_output` and before `validate_output`, query LSP diagnostics for the written files. Feed diagnostic results back as structured context if retry is needed. This is the highest-value, lowest-complexity LSP integration.

**Phase 5b — LSP Cross-Reference Queries:** Add `textDocument/references` and `textDocument/definition` to context assembly. When the planner or task needs to know "what calls this function" or "where is this type defined," LSP provides precise answers that import graph analysis cannot.

**Phase 5c — Multi-Language Support:** Replace `ast` with tree-sitter as the parsing backend (already anticipated in D30). Add LSP servers for TypeScript, Go, Rust, etc. The `code_intel` package interface accommodates both backends.

### 5.2 LSP Server Lifecycle Management

This is the hardest operational problem. Forge runs tasks in parallel across multiple git worktrees. Each worktree may need its own LSP server instance (or shared server with multi-root workspace support).

**Proposed architecture:**

```
LSPManager (singleton per Forge worker process)
├── ServerPool
│   ├── PyRight server for worktree A  (spawned on demand)
│   ├── PyRight server for worktree B  (spawned on demand)
│   └── ... (bounded pool, LRU eviction)
├── ServerConfig
│   ├── Language → server binary mapping
│   ├── Startup timeout, ready probe
│   └── Per-language initialization options
└── RequestRouter
    ├── Route requests to correct server by worktree path
    ├── Handle server crashes (restart + retry)
    └── Debounce diagnostics (150ms, following OpenCode)
```

**Key design decisions:**

1. **One server per worktree** (not multi-root). Multi-root workspace support in LSP is inconsistent across servers (see Section 6 — Challenges). A dedicated server per worktree is simpler, more predictable, and aligns with Forge's isolation principles (D6).

2. **Lazy startup.** Don't start LSP servers until a task actually needs LSP capabilities. Many tasks (especially with Phase 4's automatic discovery) won't need LSP at all.

3. **Bounded pool.** Limit the number of concurrent LSP servers. Pyright can consume significant CPU and memory during initial analysis. An LRU eviction policy removes servers for completed worktrees.

4. **Graceful degradation.** If an LSP server fails to start or crashes, fall back to Phase 4's `ast`-based analysis. LSP is an enhancement, not a requirement.

5. **Server readiness probing.** After spawning a server, wait for the `initialized` response before sending requests. Pyright's initial analysis of a project can take seconds to minutes depending on codebase size.

### 5.3 Integration with Temporal Activities

LSP operations should be **part of the context assembly activity**, not separate activities. The context assembly activity already reads files and computes the import graph. Adding LSP queries (when needed) is a natural extension.

```
assemble_context (modified):
  1. Import graph analysis (grimp)           — existing Phase 4
  2. Symbol extraction (ast)                 — existing Phase 4
  3. PageRank ranking                        — existing Phase 4
  4. LSP cross-reference queries (if needed) — new Phase 5b
  5. Token budget packing                    — existing Phase 4

validate_output (modified):
  1. Auto-fix (ruff --fix, ruff format)      — existing
  2. LSP diagnostics (fast pre-check)        — new Phase 5a
  3. Ruff lint                               — existing
  4. Ruff format check                       — existing
  5. Test execution (optional)               — existing
```

### 5.4 Data Models

New models for LSP integration:

```
LSPConfig:
    enabled: bool = False                    # Opt-in for Phase 5
    server_pool_size: int = 4                # Max concurrent LSP servers
    startup_timeout_ms: int = 30_000         # Max wait for server ready
    diagnostic_debounce_ms: int = 150        # Following OpenCode
    languages: dict[str, LSPServerConfig]    # Language → server config

LSPServerConfig:
    command: list[str]                       # e.g. ["pyright-langserver", "--stdio"]
    initialization_options: dict = {}
    workspace_config: dict = {}

LSPDiagnostic:
    file_path: str
    line: int
    column: int
    severity: DiagnosticSeverity             # error, warning, information, hint
    message: str
    source: str                              # e.g. "Pyright"

LSPReference:
    file_path: str
    line: int
    column: int
    context_line: str                        # The line of code containing the reference

LSPDefinition:
    file_path: str
    line: int
    column: int
    symbol_name: str
```

### 5.5 Cost-Benefit Analysis

| LSP Feature | Value to Forge | Implementation Cost | Priority |
|---|---|---|---|
| `getDiagnostics` | High — fast validation pre-check, tighter retry loops | Medium — requires server lifecycle management | Phase 5a |
| `findReferences` | High — enables "find all callers" for context assembly | Low (once server infra exists) | Phase 5b |
| `goToDefinition` | Medium — resolves third-party types beyond import graph | Low (once server infra exists) | Phase 5b |
| `hover` | Medium — type inference for unannotated code | Low (once server infra exists) | Phase 5b |
| `documentSymbol` | Low — `ast` module already extracts symbols | Minimal | Not needed |
| `rename` | Low — Forge generates whole files, not incremental edits | Low | Deferred |

---

## 6. Challenges and Risks

### 6.1 Server Lifecycle Across Worktrees

LSP was designed for "the host tool tightly controls the lifecycle of the language server and the language server has access to the files making up the development workspace." Forge's multi-worktree model breaks this assumption. Each worktree is an independent workspace that may exist for minutes and then be destroyed.

**Mitigation:** Treat LSP servers as ephemeral. Start on demand, destroy when the worktree is removed. The bounded server pool prevents resource exhaustion.

### 6.2 Startup Latency

Pyright's initial analysis of a Python project can take seconds to minutes. For a task that runs in 30 seconds, spending 10 seconds waiting for Pyright to index the worktree is a significant overhead.

**Mitigation:**
- Pre-warm a server on the base branch before creating worktrees. Sub-task worktrees differ from the parent by only a few files; incremental re-analysis is fast.
- Use Pyright CLI (`--outputjson`) for one-shot batch validation instead of the LSP server for simple diagnostic checks.
- Make LSP optional — tasks that don't need cross-reference tracking can skip it entirely.

### 6.3 Multi-Root Workspace Inconsistencies

LSP's multi-root workspace support is inconsistent across servers. Pyright has reported issues with multi-root mode consuming excessive CPU. Each server implementation handles workspace folders differently.

**Mitigation:** One server per worktree (not multi-root). Accept the memory cost for predictability.

### 6.4 Non-Python Language Server Availability

For Phase 5c (multi-language), each target language needs a language server installed on the Forge worker machine. Server quality and feature completeness varies widely.

**Mitigation:** Start with Python (Pyright) only. Add languages one at a time, with fallback to tree-sitter if the LSP server is unavailable or unreliable.

### 6.5 Find References Performance

Neovim users reported LSP `findReferences` taking 35+ seconds on codebases with 40K files. Forge's worktrees are typically smaller (subset of a monorepo), but this is a risk for large projects.

**Mitigation:** Set timeouts on LSP requests. If `findReferences` times out, fall back to `grimp` downstream module queries (less precise but available).

---

## 7. Alternative Considered: Pyright CLI Instead of LSP

For Python-only diagnostics, Pyright's CLI (`pyright --outputjson`) provides batch type checking without managing an LSP server. This is simpler to integrate (subprocess call, parse JSON output) and already works in Forge's validation model.

**When to use Pyright CLI instead of LSP:**
- One-shot validation after code generation (batch mode)
- CI/CD-style type checking
- When you don't need incremental, per-edit diagnostics

**When LSP is better:**
- Incremental diagnostics during multi-step tasks (check after each step without re-analyzing the entire project)
- Cross-reference queries (`findReferences`, `goToDefinition`) — not available via CLI
- Real-time feedback during retry loops

**Recommendation:** Use Pyright CLI for Phase 5a diagnostics initially (simpler integration). Migrate to LSP server when Phase 5b cross-reference queries justify the server lifecycle investment.

---

## 8. Recommended Implementation Plan

### Phase 5a: Diagnostic Feedback (Pyright CLI)

**Goal:** Add type-checking diagnostics as a fast pre-check in the validation pipeline.

**Approach:**
1. Add Pyright CLI (`pyright --outputjson`) as a new validation check in `validate_output`.
2. Parse JSON output into `LSPDiagnostic` models.
3. Feed diagnostic summaries back to the LLM on retry (concise error messages with file/line, not raw JSON).
4. Run after auto-fix, before ruff lint (catches type errors that ruff doesn't).

**Complexity:** Low. No server lifecycle. Subprocess call + JSON parsing.

**Dependency:** `pyright` or `basedpyright` installed in the worker environment.

### Phase 5b: LSP Server for Cross-Reference Queries

**Goal:** Enable precise "find all callers" and "go to definition" for context assembly.

**Approach:**
1. Implement `LSPManager` with server pool, lazy startup, and LRU eviction.
2. Add `find_references(file, symbol)` and `go_to_definition(file, symbol)` to `code_intel` API.
3. Integrate into context assembly: when a target file modifies a function, include its callers as additional context.
4. Use cclsp's symbol-based resolution pattern (query `documentSymbol` first, then resolve by name) to avoid position-sensitivity issues.

**Complexity:** Medium-high. Server lifecycle management is the main challenge.

**Dependency:** Phase 5a proven, server pool infrastructure.

### Phase 5c: Multi-Language via tree-sitter + LSP

**Goal:** Support TypeScript, Go, Rust, and other languages.

**Approach:**
1. Replace `ast` with tree-sitter as the parsing backend in `code_intel/parser.py`.
2. Add tree-sitter language grammars for target languages.
3. Add LSP server configurations for target languages.
4. The `code_intel` API remains the same — backends swap underneath.

**Complexity:** Medium. tree-sitter integration is well-documented. Per-language LSP configuration is the main effort.

**Dependency:** Phase 5b proven, tree-sitter Python bindings.

---

## 9. References

### Production Tools Reviewed

| Project | What I Learned | Source |
|---|---|---|
| **Claude Code** | Three-layer LSP architecture (servers, manager, analyzer). 900x perf improvement over text search for find-references. Supports 11 languages. | [Claude Code LSP Setup Guide](https://www.aifreeapi.com/en/posts/claude-code-lsp), [Claude Code v2.0.74 LSP Update](https://www.how2shout.com/news/claude-code-v2-0-74-lsp-language-server-protocol-update.html), [Hacker News Discussion](https://news.ycombinator.com/item?id=46355165) |
| **OpenCode** | Diagnostics-only LSP exposure is sufficient for effective agent operation. 150ms debounce for diagnostic stability. Event bus architecture for LSP integration. | [OpenCode LSP Docs](https://opencode.ai/docs/lsp/), [OpenCode Internals Deep Dive](https://cefboud.com/posts/coding-agents-internals-opencode-deepdive/), [OpenCode LSP Integration (DeepWiki)](https://deepwiki.com/tencent-source/opencode/4.1-lsp-integration) |
| **Kiro** | Two-tier model: tree-sitter built-in + LSP optional. Per-workspace LSP server spawning. 18 languages via tree-sitter, 8 via LSP. | [Kiro Code Intelligence Docs](https://kiro.dev/docs/cli/code-intelligence/), [Kiro Code Intelligence Changelog](https://kiro.dev/changelog/code-intelligence-and-knowledge-index/), [Introducing Kiro](https://kiro.dev/blog/introducing-kiro/) |
| **Aider** | Tree-sitter + PageRank without LSP. Deliberate choice to avoid LSP deployment complexity. Repo map binary search sizing. | [Aider Repo Map with tree-sitter](https://aider.chat/2023/10/22/repomap.html), [Aider Repo Map Docs](https://aider.chat/docs/repomap.html), [Aider Language Support](https://aider.chat/docs/languages.html) |
| **Continue.dev** | LSP for go-to-definition in autocomplete context ("root path context"). tree-sitter for chunking. Caching strategy for LSP requests with high hit rate. | [Continue Root Path Context](https://blog.continue.dev/root-path-context-the-secret-ingredient-in-continues-autocomplete-prompt/), [Continue Codebase Retrieval Limits](https://blog.continue.dev/accuracy-limits-of-codebase-retrieval/), [Continue Codebase Retrieval Docs](https://docs.continue.dev/features/codebase-embeddings) |
| **Augment Code** | Context Engine processes 500K files. "Context quality" over "context quantity." Ranks #1 on SWE-Bench Pro. | [Augment Context Engine](https://www.augmentcode.com/context-engine), [Augment Context Engineering Guide](https://www.augmentcode.com/guides/mastering-context-engineering-for-ai-driven-development) |
| **Sourcegraph Cody** | Deprecated embeddings for BM25 + native search. Embeddings couldn't scale to >100K repos. | [How Cody Understands Your Codebase](https://sourcegraph.com/blog/how-cody-understands-your-codebase), [Cody Embeddings Docs](https://sourcegraph.com/docs/cody/core-concepts/embeddings) |
| **cclsp** | MCP-to-LSP bridge for AI agents. Symbol-based resolution avoids position-sensitivity. Three-layer architecture (MCP, LSP management, configuration). | [cclsp GitHub](https://github.com/ktnyt/cclsp), [cclsp Architecture (DeepWiki)](https://deepwiki.com/ktnyt/cclsp), [cclsp npm](https://www.npmjs.com/package/cclsp) |
| **RepoMapper** | Standalone Aider repo map reimplementation. Tree-sitter + PageRank as MCP server. | [RepoMapper GitHub](https://github.com/pdavis68/RepoMapper) |
| **repominify** | Knowledge graph from code, 78-82% token reduction. Signature extraction preserves LLM-needed information. | [repominify GitHub](https://github.com/mikewcasale/repominify), [repominify Medium Article](https://mike-w-casale.medium.com/repominify-a-practical-tool-for-ai-assisted-coding-challenges-c6628a03e4a3) |

### Research Papers Reviewed

| Paper | What I Learned | Source |
|---|---|---|
| **LSPRAG** (ICSE 2026) | Hybrid LSP + AST approach for context retrieval. Compile-free self-repair via LSP diagnostics. Up to 213% coverage improvement. | [arXiv](https://arxiv.org/abs/2510.22210), [GitHub](https://github.com/THU-WingTecher/LSPRAG), [ICSE 2026 Listing](https://conf.researchr.org/details/icse-2026/icse-2026-research-track/147/LSPRAG-LSP-Guided-RAG-for-Language-Agnostic-Real-Time-Unit-Test-Generation) |
| **LSPAI** (FSE Industry 2025) | Four LSP providers (definition, reference, symbol, diagnosis) for dependency analysis. Applied at Tencent. 931% Go coverage improvement. | [ACM DL](https://doi.org/10.1145/3696630.3728540), [GitHub](https://github.com/THU-WingTecher/LSPAI), [PDF](http://www.wingtecher.com/themes/WingTecherResearch/assets/papers/paper_from_25/LSPAI_FSE-Industry25.pdf) |
| **GrepRAG** (2026) | Lightweight lexical retrieval + BM25 re-ranking outperforms complex graph methods. Identifier-weighted ranking critical. | [arXiv](https://arxiv.org/abs/2601.23254), [arXiv HTML](https://arxiv.org/html/2601.23254) |
| **AST-derived graphs benchmark** (2025) | Deterministic AST graphs scored 15/15 vs. 6/15 for vector RAG. 70x lower cost than LLM-extracted graphs. | Referenced in Forge Phase 4 docs (D32) |
| **Tree-sitter vs. LSP hybrid architecture** | Tree-sitter for syntax (fast, incremental), LSP for semantics (types, references). Modern editors (Zed) prove hybrid wins. | [byteiota Analysis](https://byteiota.com/tree-sitter-vs-lsp-why-hybrid-ide-architecture-wins/) |
| **Context engineering for coding agents** | Hybrid retrieval (keyword + semantic + graph) is the emerging consensus. 40-60% context utilization optimal. | [Martin Fowler](https://martinfowler.com/articles/exploring-gen-ai/context-engineering-coding-agents.html), [LangChain](https://blog.langchain.com/context-engineering-for-agents/), [LlamaIndex](https://www.llamaindex.ai/blog/context-engineering-what-it-is-and-techniques-to-consider) |

### Protocol and Infrastructure References

| Resource | What I Learned | Source |
|---|---|---|
| **LSP Specification 3.17** | Workspace concept, capabilities exchange, multi-root support. | [LSP Spec](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/) |
| **LSP4IJ lifecycle management** | Server state machine: discovery → registration → startup → operation → shutdown. | [LSP4IJ (DeepWiki)](https://deepwiki.com/redhat-developer/lsp4ij/3.2-language-server-lifecycle-management) |
| **Pyright** | CLI with `--outputjson` for batch type checking. No stable programmatic API. basedpyright provides pip-installable variant. | [Pyright GitHub](https://github.com/microsoft/pyright), [Pyright CLI Docs](https://github.com/microsoft/pyright/blob/main/docs/command-line.md), [basedpyright](https://github.com/DetachHead/basedpyright) |
| **LSP multi-workspace challenges** | Unclear semantics across editors, CPU issues with multi-root, session persistence problems. | [emacs-lsp Discussion](https://github.com/emacs-lsp/lsp-mode/discussions/3095), [biome Issue](https://github.com/biomejs/biome/issues/1573) |

---

## 10. Conclusions

1. **Phase 4's approach is sound.** The `ast` + `grimp` + PageRank combination is validated by Aider's production usage, the AST-derived graphs benchmark, and the broader industry consensus. Do not change Phase 4.

2. **LSP is the right next step after Phase 4, not a replacement for it.** Every production tool reviewed (Claude Code, Kiro, OpenCode, Continue.dev) uses LSP as a complementary layer on top of structural analysis, not instead of it.

3. **Start with diagnostics.** OpenCode's experience shows that exposing only diagnostics to the LLM provides the highest value with the lowest complexity. Use Pyright CLI initially to avoid server lifecycle overhead.

4. **Server lifecycle is the hard problem.** The multi-worktree model creates challenges that no existing LSP tool has solved because they all operate in single-workspace contexts. A bounded server pool with lazy startup and LRU eviction is the proposed solution.

5. **Cross-reference queries are the high-value LSP feature for Forge.** Import graphs tell you which modules depend on which; LSP tells you which specific functions call which specific functions. This is the gap between Phase 4 and production-quality context assembly.

6. **tree-sitter is the bridge to multi-language.** When Forge expands beyond Python, tree-sitter replaces `ast` for parsing (already anticipated in D30), and LSP servers provide per-language semantic intelligence.

7. **Embeddings are not needed.** Multiple sources (Sourcegraph Cody, GrepRAG, the AST benchmark) confirm that structural/lexical approaches outperform embeddings for code context retrieval, at lower cost and complexity.
