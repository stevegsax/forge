# 05 - `_run_planned` Decomposition

**Status:** `NOT STARTED`
**Priority:** HIGH
**Code Review Section:** [Section 3](../reports/code-review-2026-02-26.md#3-high-structural-duplication-in-workflowspy) (final paragraph)
**Depends on:** [04](04-workflow-duplication.md) (both refactor `workflows.py`; complete 04 first to avoid conflicts)

## Problem

`_run_planned` (lines 693-1023) is a 331-line method handling 12 distinct responsibilities:

1. Worktree creation
2. Planner context assembly
3. Exploration
4. Planner LLM call
5. Step iteration
6. Fan-out dispatch
7. Step execution
8. Commit
9. Worktree reset
10. Sanity check
11. Plan revision
12. Result assembly

This makes the method difficult to understand, test, and modify.

## Acceptance Criteria

- [ ] `_run_planned` decomposed into 3-5 focused methods with clear responsibilities
- [ ] Each extracted method has a clear name describing its responsibility
- [ ] No behavioral changes
- [ ] All existing tests pass

## Plan

*To be written when work begins.*

## Sub-tasks

*To be written when work begins.*

## Files to Modify

- `forge/workflows.py`

## Verification

```bash
uv run pytest tests/ -x -q
uv run ruff check src/
```

Review the diff to confirm no behavioral changes, only structural decomposition.

## Development Notes

*Append discoveries, decisions, and gotchas here during implementation.*
