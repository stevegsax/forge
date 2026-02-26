# 04 - Workflow Duplication

**Status:** `NOT STARTED`
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

- [ ] Shared logic extracted into module-level functions or a composition object
- [ ] No copy-pasted method bodies between `ForgeTaskWorkflow` and `ForgeSubTaskWorkflow`
- [ ] `workflows.py` line count reduced meaningfully
- [ ] All existing tests pass with no behavioral changes

## Plan

*To be written when work begins.*

## Sub-tasks

*To be written when work begins.*

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

*Append discoveries, decisions, and gotchas here during implementation.*
