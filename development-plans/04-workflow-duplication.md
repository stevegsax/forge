# 04 - Workflow Duplication

**Status:** `DONE`
**Priority:** HIGH
**Code Review Section:** [Section 3](../reports/code-review-2026-02-26.md#3-high-structural-duplication-in-workflowspy)
**Depends on:** [01](01-temporal-determinism.md) (both modify `workflows.py`)

## Problem

`workflows.py` is 1,818 lines with two workflow classes sharing ~400 lines of identical code:

| Duplicated Element | ForgeTaskWorkflow | ForgeSubTaskWorkflow |
|---|---|---|
| `_call_llm_batch` | Lines 210-250 | Lines 1347-1386 |
| `_call_generation` | Lines 256-275 | Lines 1392-1411 |
| `_call_conflict_resolution` | Lines 364-400 | Lines 1413-1449 |
| `__init__` / signal / batch fields | Lines 183-190 | Lines 1320-1327 |
| Conflict detection + resolution blocks | Lines 1156-1212 | Lines 1679-1738 |
| Remove worktree pattern | 4 occurrences | 3 occurrences |
| Error summary formatting | 3 occurrences | 2 occurrences |

Temporal workflows can't use inheritance, but these can be extracted into module-level helper functions or a composition object shared by both classes.

## Acceptance Criteria

- [x] Shared logic extracted into module-level functions or a composition object
- [x] No copy-pasted method bodies between `ForgeTaskWorkflow` and `ForgeSubTaskWorkflow`
- [x] `workflows.py` line count reduced meaningfully (1915 → 1819, -96 lines)
- [x] All existing tests pass with no behavioral changes (1060 passed)

## Plan

The three LLM dispatch methods (`_call_llm_batch`, `_call_generation`, `_call_conflict_resolution`) are byte-for-byte identical across both workflow classes. They only depend on `self._batch_results` and `self._sync_mode`, which can be passed as parameters.

**Approach:** Extract these into module-level async functions in `workflows.py` (no new module needed — they use `workflow.*` APIs that only work inside a workflow context). Both classes delegate to the shared functions.

Additionally, the `remove_worktree_activity` call pattern (7 occurrences, ~7 lines each) can be extracted into a small module-level helper.

**Not extracting:**

- `__init__` / `@workflow.signal` boilerplate — Temporal requires decorators on class methods, can't be shared
- Conflict detection/resolution blocks — structurally similar but meaningfully different (different return types, different model routing construction, different error paths)
- `_run_single_step` — completely different implementations despite the same name

## Sub-tasks

- [x] Confirm duplication still exists (verified: all three dispatch methods identical)
- [x] Extract `_call_llm_batch` → module-level `_call_llm_batch_dispatch`
- [x] Extract `_call_generation` → module-level `_call_generation_dispatch`
- [x] Extract `_call_conflict_resolution` → module-level `_call_conflict_resolution_dispatch`
- [x] Extract `_remove_worktree` helper for the repeated `remove_worktree_activity` calls
- [x] Update both workflow classes to delegate to the shared functions
- [x] Run tests and linter

## Files to Modify

- `forge/workflows.py`
- Possibly a new `forge/workflow_helpers.py` module

## Verification

```bash
uv run pytest tests/ -x -q
uv run ruff check src/
```

Review the diff to confirm no behavioral changes, only structural extraction.

## Development Notes

- No new module needed — all shared functions live in `workflows.py` because they call `workflow.*` APIs (e.g. `workflow.execute_activity`, `workflow.wait_condition`, `workflow.info`) which only work inside a Temporal workflow execution context.
- The unique dispatch methods (`_call_planner_llm`, `_call_exploration`, `_call_sanity_check_llm`) on `ForgeTaskWorkflow` still call `self._call_llm_batch`, which is now a thin wrapper delegating to the shared `_call_llm_batch_dispatch`. No changes needed to these methods.
- The `__init__`/`@workflow.signal` boilerplate (~8 lines per class) can't be extracted due to Temporal decorator requirements. Acceptable duplication.
- Pre-existing lint error in `context.py` (unused local `task_id`) — not introduced by this change.
