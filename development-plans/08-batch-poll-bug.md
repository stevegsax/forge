# 08 - Batch Poll Bug

**Status:** `NOT STARTED`
**Priority:** MEDIUM
**Code Review Section:** [Section 9a](../reports/code-review-2026-02-26.md#9-medium-other-code-quality-issues)

## Problem

In `batch_poll.py` around line 130:

```python
final_status = "succeeded" if signals_sent > 0 else "errored"
```

`signals_sent` is cumulative across all jobs in the polling loop. If job A sent 1 signal, job B gets `"succeeded"` even if job B sent 0 signals. The status should track signals per-job.

## Acceptance Criteria

- [ ] `final_status` is determined per-job based on that job's signal count
- [ ] Existing tests pass
- [ ] New or updated test covers the edge case where one job succeeds and another fails

## Plan

*To be written when work begins.*

## Sub-tasks

*To be written when work begins.*

## Files to Modify

- `forge/activities/batch_poll.py`
- `tests/` (test for the bug fix)

## Verification

```bash
uv run pytest tests/ -x -q
uv run ruff check src/
```

## Development Notes

*Append discoveries, decisions, and gotchas here during implementation.*
