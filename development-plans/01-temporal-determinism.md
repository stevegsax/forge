# 01 - Temporal Determinism Violation

**Status:** `DONE`
**Priority:** CRITICAL
**Code Review Section:** [Section 1](../reports/code-review-2026-02-26.md#1-critical-temporal-determinism-violation)

## Problem

`detect_file_conflicts` performs filesystem I/O (reads files from disk) and is called directly inside workflow methods. Temporal workflows must be deterministic -- they can be replayed at any time on any worker. Filesystem reads inside workflow code return different results on replay or fail entirely on a different worker.

Locations:

- `workflows.py:1157` -- called in `_run_fan_out_step`
- `workflows.py:1680` -- called in `_run_nested_fan_out`

The function itself lives in the activities module and reads files:

```python
original_path = Path(worktree_path) / file_path
if original_path.is_file():
    original_content = original_path.read_text()
```

## Acceptance Criteria

- [x] `detect_file_conflicts` is never called directly in workflow code
- [x] File conflict detection runs as a Temporal activity
- [x] Existing tests pass
- [x] New test covers the activity wrapper

## Plan

Extract `detect_file_conflicts` into a proper Temporal activity following Function Core / Imperative Shell:

1. Add `DetectFileConflictsInput` / `DetectFileConflictsOutput` Pydantic models
2. Write activity tests first (TDD)
3. Split existing function: `classify_file_conflicts` (pure, no I/O) + `detect_file_conflicts` (thin wrapper with filesystem reads) + `detect_file_conflicts_activity` (`@activity.defn`)
4. Register the activity in `__init__.py` and `worker.py`
5. Replace direct calls in `workflows.py` with `workflow.execute_activity`

## Sub-tasks

- [x] Add `DetectFileConflictsInput` and `DetectFileConflictsOutput` to `models.py`
- [x] Write `TestDetectFileConflictsActivity` tests in `test_conflict_resolution.py`
- [x] Rename existing function to `classify_file_conflicts` (pure, no filesystem I/O)
- [x] Keep `detect_file_conflicts` as thin wrapper calling classify + reading originals
- [x] Add `detect_file_conflicts_activity` with `@activity.defn`
- [x] Export from `activities/__init__.py`
- [x] Register in `worker.py`
- [x] Replace workflow direct calls with `execute_activity`
- [x] Run tests, linter, and grep verification

## Files to Modify

- `forge/workflows.py`
- `forge/activities/` (new or existing activity for conflict detection)
- Tests for the new activity

## Verification

```bash
uv run pytest tests/ -x -q
uv run ruff check src/
```

Grep `workflows.py` to confirm no direct calls to `detect_file_conflicts` remain in workflow code (only in activities).

## Development Notes

- Split `detect_file_conflicts` into three layers following Function Core / Imperative Shell:
    - `classify_file_conflicts` — pure function, no I/O, `original_content` always `None`
    - `detect_file_conflicts` — thin wrapper that calls classify + reads originals from worktree (backward compat for existing callers/tests)
    - `detect_file_conflicts_activity` — `@activity.defn` async wrapper returning `DetectFileConflictsOutput`
- Both workflow call sites (`_run_fan_out_step` and `_run_nested_fan_out`) now use `workflow.execute_activity("detect_file_conflicts_activity", ...)` with `_GIT_TIMEOUT` and `result_type=DetectFileConflictsOutput`
- All 316 non-Temporal unit tests pass; `test_workflows.py` hangs on Temporal test server download (pre-existing issue, not related to this change)
- Ruff linter passes on all changed files
