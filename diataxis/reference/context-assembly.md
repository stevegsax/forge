# Context Assembly Reference

Technical reference for Forge's context assembly pipeline: discovery mechanisms, priority ordering, exploration providers, token budget algorithm, data models, and CLI controls. For the system prompt section table and cache breakpoint reference, see [Prompt Construction Reference](prompt-construction.md).

For background on why the pipeline works this way, see [Context Assembly](../explanation/context-assembly.md). For practical recipes, see [How to Control Context Assembly](../howto/control-context-assembly.md).

## Context Priority Ordering

Items are packed into the token budget in priority order. Within each tier, items are sorted by PageRank importance score (descending).

| Priority | Content | Representation | Inclusion Rule |
|----------|---------|----------------|----------------|
| 1 (highest) | Task description, definition of "done" | Verbatim | Always included |
| 2 | Target file current content | Full content | Included if files exist in worktree |
| 3 | Direct imports of target files | Full content | Only with `--include-deps` |
| 4 | Transitive imports (ranked by PageRank) | Signatures only | Only with `--include-deps`; binary search determines how many fit |
| 5 | Repo map; playbooks | Compressed overview; extracted lessons | Included if budget allows |
| 6 (lowest) | Manually specified `context_files` | Full content | Included if budget allows |

When an item at priority 3 or 4 would exceed the budget, the packer attempts to reduce it (full content to signatures only) before skipping it.

## Exploration Providers

Providers are registered in `src/forge/providers.py`. Each provider implements the `ProviderHandler` protocol: `(params: dict[str, str], repo_root: str, worktree_path: str) -> str`.

| Provider | Description | Parameters | Return Format |
|----------|-------------|------------|---------------|
| `read_file` | Read full contents of a file from the worktree | `path` (required) | File content as plain text |
| `search_code` | Regex pattern search across files | `pattern` (required), `glob` (default `*.py`) | Up to 100 matches with file path, line number, and matching line |
| `symbol_list` | Extract public API of a module | `file_path` (required) | Function signatures, class definitions, constants |
| `import_graph` | Show what a module imports and what imports it | `file_path` (required) | Lists of imported and importing module paths |
| `run_tests` | Execute `pytest` and return results | `path` (optional), `marker` (optional) | Test output (30-second timeout) |
| `lint_check` | Run `ruff` linter on specified files | `files` (required, comma-separated) | Lint output |
| `git_log` | Show recent commit history | `path` (optional), `n` (optional, default 10) | Commit log |
| `git_diff` | Diff against a base branch | `base` (optional) | Unified diff output |
| `repo_map` | Generate PageRank-ranked project structure | (none) | File tree with top-ranked signatures |
| `discover_context` | Run full auto-discovery for target files (always includes dependencies) | `target_files` (required, comma-separated) | Full content for direct imports, signatures for transitive imports |
| `past_runs` | Show recent workflow run results from the observability store | `limit` (optional, default 5) | Run summaries |
| `playbooks` | Retrieve playbook entries by tag | `tags` (required, comma-separated) | Matching playbook entries |

The exploration LLM receives `ContextProviderSpec` objects describing each provider. It returns `ContextRequest` objects specifying which providers to call and with what parameters. An empty request list signals that exploration is complete.

### Request and Response Formats

A single `ContextRequest` specifying one provider call:

```json
{
    "provider": "read_file",
    "params": {"path": "src/forge/models.py"},
    "reasoning": "Need to inspect the AssembledContext model fields"
}
```

An `ExplorationResponse` with multiple requests issued in one round:

```json
{
    "requests": [
        {"provider": "read_file", "params": {"path": "src/forge/models.py"}, "reasoning": "Inspect AssembledContext fields"},
        {"provider": "symbol_list", "params": {"file_path": "src/forge/providers.py"}, "reasoning": "Check available provider names"}
    ]
}
```

An `ExplorationResponse` signalling that exploration is complete (no further context needed):

```json
{
    "requests": []
}
```

## Token Budget Algorithm

**Inputs:**

- `token_budget` (default: 100,000) -- total tokens available for context
- `output_reserve` (default: 16,000) -- tokens reserved for LLM output
- `model_max_tokens` -- model context window size
- Ranked list of context items with priority and importance scores

**Algorithm:**

1. Sort items by priority (ascending, highest priority first).
2. Within each priority tier, sort by PageRank importance score (descending).
3. Accumulate items while `total_estimated_tokens < budget`.
4. When an item would exceed the budget:

    a. If the item can be reduced (full content to signatures only), reduce and retry.

    b. Otherwise, skip the item and continue to the next.

5. For tier-4 items (transitive imports), use binary search to determine the maximum number that fit in the remaining budget.
6. Return `PackedContext` with included items and stats.

**Token estimation:** 4 characters per token (character-based heuristic). Conservative for English and code. Avoids tokenizer library dependencies.

**Target utilization:** 50-60% of the model's context window for input context, reserving the remainder for LLM output and internal reasoning.

## Data Models

All models are defined in `src/forge/models.py`.

### AssembledContext

Output of `assemble_context`, input to `call_llm`.

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Task identifier |
| `system_prompt` | `str` | The fully assembled system prompt |
| `user_prompt` | `str` | The domain-specific user instruction |
| `context_stats` | `ContextStats \| None` | Discovery and packing metrics |
| `step_id` | `str \| None` | Step identifier (planned mode) |
| `sub_task_id` | `str \| None` | Sub-task identifier (fan-out mode) |
| `model_name` | `str` | Model to use for this call |
| `log_messages` | `bool` | Whether to log API messages |
| `worktree_path` | `str` | Path to the git worktree |

### ContextStats

Observability stats from context assembly.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `files_discovered` | `int` | `0` | Files found via import graph |
| `files_included_full` | `int` | `0` | Files included with full content |
| `files_included_signatures` | `int` | `0` | Files included with signatures only |
| `files_truncated` | `int` | `0` | Files that did not fit in the budget |
| `total_estimated_tokens` | `int` | `0` | Total estimated tokens in assembled context |
| `budget_utilization` | `float` | `0.0` | Fraction of budget used (0.0 to 1.0) |
| `repo_map_tokens` | `int` | `0` | Tokens used by the repo map |

### ContextConfig

Configuration for automatic context discovery. Attached to `TaskDefinition` via the `context` field.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auto_discover` | `bool` | `True` | Enable automatic context discovery |
| `include_dependencies` | `bool` | `False` | Include direct import contents and transitive signatures upfront |
| `token_budget` | `int` | `100_000` | Token budget for context |
| `output_reserve` | `int` | `16_000` | Tokens reserved for LLM output |
| `max_import_depth` | `int` | `2` | How deep to trace imports |
| `include_repo_map` | `bool` | `True` | Include compressed repo map |

### ContextProviderSpec

Description of an available context provider, shown to the exploration LLM.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Provider identifier (e.g., `read_file`) |
| `description` | `str` | Human-readable description of what the provider does |
| `parameters` | `dict[str, str]` | Parameter name to description mapping |

### ContextRequest

A request for specific context from a provider.

| Field | Type | Description |
|-------|------|-------------|
| `provider` | `str` | Provider name to call |
| `params` | `dict[str, str]` | Parameters to pass to the provider |
| `reasoning` | `str` | Why this context is needed (for observability) |

### ExplorationResponse

Output from the exploration LLM call.

| Field | Type | Description |
|-------|------|-------------|
| `requests` | `list[ContextRequest]` | Context requests; empty list signals readiness to generate |

### ContextResult

Result of fulfilling a single context request.

| Field | Type | Description |
|-------|------|-------------|
| `provider` | `str` | Provider that produced this result |
| `content` | `str` | The returned context text |
| `estimated_tokens` | `int` | Estimated token count of the content |

## CLI Flags

Flags that control context assembly behavior on `forge run`:

| Flag | Default | Description |
|------|---------|-------------|
| `--context-file` | (none) | Include a specific file as manual context (repeatable) |
| `--include-deps` | off | Include dependency file contents in upfront context |
| `--no-auto-discover` | off | Disable automatic context discovery entirely |
| `--no-explore` | off | Disable the exploration loop |
| `--max-exploration-rounds` | `10` | Maximum number of exploration rounds (0 disables) |
| `--token-budget` | `100000` | Total token budget for context |
| `--max-import-depth` | `2` | How deep to trace imports in the dependency graph |
