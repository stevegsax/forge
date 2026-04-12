# Prompt Construction Reference

Tabular reference for the system prompt structure: the eleven sections, cache breakpoint placement, and the error injection format used on retry. For the design rationale behind this structure, see [Prompt Construction](../explanation/prompt-construction.md). For how the content of each section is discovered and ranked, see the [Context Assembly Reference](context-assembly.md).

## System prompt sections

The system prompt is assembled in cache-optimized order. Sections toward the top are the most stable; sections toward the bottom are the most volatile.

| # | Section | Source | Stability | Cache Behavior |
|---|---------|--------|-----------|----------------|
| 1 | Role statement | `DomainConfig.role_prompt` | Stable across all calls for a domain | Cached as prefix start |
| 2 | Output requirements | `DomainConfig.output_requirements` | Stable across all calls | Cached; breakpoint 1 placed after this section |
| 3 | Project instructions | `CLAUDE.md` in repo root | Stable across all calls for the same repo | Cached |
| 4 | Repository structure | `code_intel/repo_map.py` (PageRank-ranked) | Stable within a repo (changes on file add/remove) | Cached |
| 5 | Playbooks | Observability store (`playbooks` table) | Stable within a task | Cached |
| 6 | Task description + target files | `TaskDefinition.description`, `TaskDefinition.target_files` | Stable across retries and exploration rounds | Cached |
| 7 | Target file contents | Worktree file reads | Stable across exploration rounds; may change between retries | Cached; breakpoint 2 placed after this section |
| 8 | Direct dependencies | Import graph (full file content) | Stable within a task; only present with `--include-deps` | Cached |
| 9 | Interface context | `code_intel/parser.py` (extracted signatures) | Stable within a task | Cached |
| 10 | Exploration results | Accumulated provider responses | Grows during exploration; stable once exploration completes | Cached; breakpoint 3 placed after this section |
| 11 | Previous errors | `ValidationResult` list from prior attempt | Only present on retry; changes every attempt | Not cached (most volatile) |

## Cache breakpoints

Breakpoints are placed using Anthropic's `cache_control` headers. Each breakpoint defines the end of a cacheable prefix.

| Breakpoint | Placed after | Protects | Reuse scope |
|---|---|---|---|
| 1 | Section 2 (Output requirements) | Role + output format | All calls in the same domain |
| 2 | Section 7 (Target file contents) | Sections 1–7 | Retries and exploration rounds within the same step |
| 3 | Section 10 (Exploration results) | Sections 1–10 | Retries within the same step after exploration completes |

Cache savings are cumulative. On a retry, the entire prefix through breakpoint 3 is served from cache; only section 11 (the new error context) is processed from scratch.

## Error injection format

Section 11 is appended only on retry. It contains structured error output followed by AST-derived code context.

**Error output** is the raw validation output:

```text
Previous Attempt Errors

The previous attempt failed validation. Fix these errors:

ruff lint errors:
src/myapp/store.py:47:5 F811 Redefinition of unused `total_runs` from line 38
```

**AST context** follows each error with a file path and line number. Forge parses the source with Python's `ast` module to find the enclosing function or class:

```text
Context around error:
def get_run_statistics(self, since: datetime | None = None) -> dict:
    """Query aggregate run statistics."""
    total_runs = self._count_runs(since)       # line 38
    ...
    total_runs = self._count_all_runs(since)   # <-- ERROR (line 47)
```

The section header includes the attempt number ("Attempt 2 of 3") so the LLM knows how many retries remain.

## Related references

| Topic | What it covers |
|---|---|
| [Context Assembly Reference](context-assembly.md) | Priority ordering table, exploration providers, token budget algorithm, data models, CLI flags |
| [Forge Run Extraction Reference](forge-run-extraction.md) | Playbook table schema (section 5 content source) |
| [Validation and Retries Reference](validation-and-retries.md) | ValidationResult model, transition signals (section 11 trigger conditions) |
