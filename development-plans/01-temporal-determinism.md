# 01 - Temporal Determinism Violation

**Status:** `NOT STARTED`
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

- [ ] `detect_file_conflicts` is never called directly in workflow code
- [ ] File conflict detection runs as a Temporal activity
- [ ] Existing tests pass
- [ ] New test covers the activity wrapper

## Plan

*To be written when work begins.*

## Sub-tasks

*To be written when work begins.*

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

*Append discoveries, decisions, and gotchas here during implementation.*
