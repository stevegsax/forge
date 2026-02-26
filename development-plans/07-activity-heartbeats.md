# 07 - Activity Heartbeats

**Status:** `NOT STARTED`
**Priority:** HIGH
**Code Review Section:** [Section 2a](../reports/code-review-2026-02-26.md#2-high-temporal-operational-risks)

## Problem

Zero activities in the codebase call `activity.heartbeat()`. LLM-calling activities have 5-minute `start_to_close_timeout` values. Without heartbeats:

- Worker crashes mid-LLM-call are not detected until the full timeout expires
- Activities cannot receive cancellation signals
- The Temporal server has no visibility into activity progress

Affected activities:

- `call_llm`
- `call_planner`
- `call_exploration_llm`
- `call_sanity_check`
- `call_conflict_resolution`
- `call_extraction_llm`
- `poll_batch_results`
- `validate_output`

## Acceptance Criteria

- [ ] Long-running activities (LLM calls, batch polling) call `activity.heartbeat()` at appropriate intervals
- [ ] `heartbeat_timeout` is set on corresponding `execute_activity` calls in `workflows.py`
- [ ] Existing tests pass

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
- `forge/activities/batch_poll.py`
- `forge/activities/validate.py`
- `forge/workflows.py` (add `heartbeat_timeout` to activity invocations)

## Verification

```bash
uv run pytest tests/ -x -q
uv run ruff check src/
```

Grep for `heartbeat` in the activities directory to confirm it's being called.

## Development Notes

*Append discoveries, decisions, and gotchas here during implementation.*
