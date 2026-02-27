# 08 - Batch Poll Bug

**Status:** `DONE`
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

1. Add a `job_signals` counter inside the per-job loop, reset each iteration
2. Use `job_signals` (not cumulative `signals_sent`) for `final_status`
3. Add test: job A sends a signal, job B has no results — verify job B gets `"errored"`

## Sub-tasks

- [x] Add per-job `job_signals` counter in `execute_poll_batch_results`
- [x] Use `job_signals` for `final_status` determination
- [x] Add `test_final_status_is_per_job_not_cumulative` test
- [x] All tests pass, ruff clean

## Files to Modify

- `forge/activities/batch_poll.py`
- `tests/` (test for the bug fix)

## Verification

```bash
uv run pytest tests/ -x -q
uv run ruff check src/
```

## Development Notes

- Bug was still present after refactoring (line 131). Straightforward 2-line fix.
