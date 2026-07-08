# Forge Adversarial Test Findings

**Date:** 2026-06-05
**Reviewer:** Codex
**Base commit:** `6df920b`

> **Triaged 2026-07-08** against HEAD (`927ac75`): both findings were
> fixed on 2026-06-09 — Finding 1 by `8418c56` (Confine untrusted
> read/exec paths to the worktree, #24), Finding 2 by `4c06f5c`
> (Auto-discover context in planned-step and sub-task assembly, #25). The
> eight failing tests pass at HEAD (155/155 across the three files), and
> the read-path confinement was independently re-verified as escape-free
> in [`forge-review-2026-07-08.md`](../../forge-review-2026-07-08.md) §8.

## Scope

This follow-up review focused only on the previously identified high-priority test gaps, excluding the separate `sax_llm` test-targeting problem and the repo-wide coverage gate.

New regression tests were added in:

- `tests/test_activity_context.py`
- `tests/test_providers.py`
- `tests/test_activity_exploration.py`

## Summary

The new tests are meaningful and currently expose real defects instead of padding metrics.

Targeted verification:

```bash
uv run pytest --no-cov -q tests/test_activity_context.py tests/test_providers.py tests/test_activity_exploration.py
```

Current result:

- `8 failed, 136 passed`

Those eight failures map directly to two high-priority production gaps.

## Finding 1: Read-path traversal is still possible

**Severity:** High

This violates the local architecture principle in `AGENTS.md` that "Context isolation is a feature."

Affected runtime code:

- `src/forge/activities/context.py:513` in `_read_context_files`
- `src/forge/providers.py:38` in `handle_read_file`
- `src/forge/providers.py:96` in `handle_symbol_list`
- `src/forge/activities/exploration.py:120` in `fulfill_requests`

Observed behavior:

- `../secret.txt` is read successfully from outside the worktree
- absolute paths are read successfully when handed to read-path helpers
- exploration dispatch inherits the same unsafe behavior because it forwards provider params without path confinement

New tests that expose it:

- `TestReadContextFiles.test_skips_parent_traversal`
- `TestReadContextFiles.test_skips_absolute_paths`
- `TestHandleReadFile.test_rejects_parent_traversal`
- `TestHandleReadFile.test_rejects_absolute_paths`
- `TestHandleSymbolList.test_rejects_parent_traversal`
- `TestFulfillRequests.test_path_traversal_request_returns_error`

Why these tests matter:

- They exercise the production read paths that actually feed LLM context.
- They mirror the existing write-path traversal tests, which means the suite now checks both halves of the isolation boundary.
- They fail because production code currently leaks outside-worktree content, not because of brittle mocks or formatting changes.

## Finding 2: Planned and sub-task context assembly still skips auto-discovery

**Severity:** High

This violates the documented execution-mode contract in `AGENTS.md` that all modes include automatic context discovery by default.

Affected runtime code:

- `src/forge/activities/context.py:730` in `assemble_step_context`
- `src/forge/activities/context.py:865` in `assemble_sub_task_context`

Observed behavior:

- planned step context assembly reads only explicit files from the worktree
- sub-task context assembly reads only explicit files from the parent worktree
- neither path calls `discover_context`
- neither path returns `context_stats`, so planned modes lose observability that single-step mode already has

New tests that expose it:

- `TestAssembleStepContext.test_auto_discover_enabled_uses_worktree_context`
- `TestAssembleSubTaskContext.test_auto_discover_uses_parent_worktree_context`

Why these tests matter:

- They pin the documented behavior instead of today’s implementation.
- They assert the critical integration point: planned modes should invoke the same discovery pipeline that single-step mode uses.
- They catch both missing discovery and missing observability data.

## Recommendation

The next code change should:

1. Add a shared read-path confinement helper using `resolve()` and `is_relative_to()` for all context/provider file reads.
2. Route `assemble_step_context` and `assemble_sub_task_context` through the same auto-discovery flow used by `assemble_context`.
3. Preserve `context_stats` in planned and fan-out modes so observability stays consistent across execution paths.

## Notes

These failures are intentional and useful. The new tests are not synthetic coverage padding; they are regression tests for concrete behavior that currently violates the repo's own documented guarantees.
