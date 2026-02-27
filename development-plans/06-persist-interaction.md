# 06 - `_persist_interaction` Extraction

**Status:** `DONE`
**Priority:** HIGH
**Code Review Section:** [Section 4](../reports/code-review-2026-02-26.md#4-high-_persist_interaction-duplicated-across-5-files)

## Problem

The observability persistence function is copy-pasted with minor variations across 5 files:

- `activities/llm.py:83-108`
- `activities/sanity_check.py:207-237`
- `activities/conflict_resolution.py:245-276`
- `activities/planner.py:240-271`
- `activities/extraction.py:278-308`

Each copy follows the same pattern: get db_path -> check None -> build AssembledContext -> get engine -> build interaction dict -> save -> catch Exception.

## Acceptance Criteria

- [x] Single shared `persist_interaction` helper in `forge/store.py`
- [x] All 5 activity files call the shared helper instead of their own copy (4 had `_persist_interaction`, 1 had `_persist_extraction_interaction`)
- [x] Existing tests pass (1060 passed)
- [x] The shared helper has its own unit test (existing tests in `test_activity_llm.py::TestPersistInteraction` updated to test `forge.store.persist_interaction`)

## Plan

*To be written when work begins.*

## Sub-tasks

*To be written when work begins.*

## Files to Modify

- `forge/activities/llm.py`
- `forge/activities/sanity_check.py`
- `forge/activities/conflict_resolution.py`
- `forge/activities/planner.py`
- `forge/activities/extraction.py`
- New shared module (e.g., `forge/activities/persistence.py` or `forge/observability/helpers.py`)

## Verification

```bash
uv run pytest tests/ -x -q
uv run ruff check src/
```

Grep for `_persist_interaction` to confirm only one definition remains (plus imports).

## Development Notes

*Append discoveries, decisions, and gotchas here during implementation.*
