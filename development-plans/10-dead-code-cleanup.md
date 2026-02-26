# 10 - Dead Code Cleanup

**Status:** `NOT STARTED`
**Priority:** LOW
**Code Review Section:** [Section 6a, 9e, 9f](../reports/code-review-2026-02-26.md#6-medium-context-assembly-issues-activitiescontextpy) and [Section 10](../reports/code-review-2026-02-26.md#10-low-minor-issues)

## Problem

### Dead constants and functions

- `_OUTPUT_REQUIREMENTS` in `activities/context.py:47-57` -- byte-identical to `_CODE_OUTPUT_REQUIREMENTS` in `domains.py`. No production code uses it; only a test asserts they're equal.
- `compute_budget` and `ContextBudget` in `code_intel/budget.py` -- no production callers. Only referenced in the test file and the module itself.

### Dead code paths

- `code_intel/graph.py:206-215` -- dead `else` branch. The `elif distance <= max_depth` is always true at that point because `distance > max_depth` was already filtered out. The `else` branch assigning `Relationship.DOWNSTREAM` can never execute.
- `llm_client.py` -- `effort` parameter accepted but never used in `build_thinking_param`.
- `eval/judge.py:176` -- `model_name` parameter silently ignored by `judge_plan`.

### Other minor issues

- `_detect_package_name` duplicated in `context.py` and `planner.py`
- `conflict_resolution.py:112` -- unused `domain` parameter in `build_conflict_resolution_system_prompt`
- `_SubprocessResult` / `_GitResult` are structurally identical dataclasses in `git.py` and `validate.py`

## Acceptance Criteria

- [ ] Dead constants removed (with corresponding test updates)
- [ ] Dead code paths removed
- [ ] Duplicated utility functions consolidated
- [ ] Unused parameters removed or connected to their intended behavior
- [ ] Existing tests pass (updated as needed for removed code)

## Plan

*To be written when work begins.*

## Sub-tasks

*To be written when work begins.*

## Files to Modify

- `forge/activities/context.py`
- `forge/code_intel/budget.py`
- `forge/code_intel/graph.py`
- `forge/llm_client.py`
- `forge/eval/judge.py`
- `forge/activities/planner.py`
- `forge/activities/conflict_resolution.py`
- `forge/git.py`
- `forge/activities/validate.py`
- Corresponding test files

## Verification

```bash
uv run pytest tests/ -x -q
uv run ruff check src/
```

Grep for removed symbols to confirm no remaining references.

## Development Notes

*Append discoveries, decisions, and gotchas here during implementation.*
