# 07 - Activity Heartbeats

**Status:** `DONE`
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

- [x] Long-running activities (LLM calls, batch polling) call `activity.heartbeat()` at appropriate intervals
- [x] `heartbeat_timeout` is set on corresponding `execute_activity` calls in `workflows.py`
- [x] Existing tests pass

## Plan

Created a shared `heartbeat_during` async context manager that spawns a background asyncio task to emit heartbeats at 30-second intervals. Wrapped all long-running activity functions with it. Added `heartbeat_timeout` to all corresponding `execute_activity` calls across three workflow files.

## Sub-tasks

- [x] Create `forge/activities/_heartbeat.py` with shared `heartbeat_during` context manager
- [x] Add `heartbeat_during()` to all 6 LLM call activities
- [x] Add `heartbeat_during()` to `poll_batch_results`
- [x] Add `heartbeat_during()` to `validate_output`
- [x] Add `heartbeat_timeout` to `workflows.py` (LLM calls: 60s, validation: 120s)
- [x] Add `heartbeat_timeout` to `extraction_workflow.py` (60s)
- [x] Add `heartbeat_timeout` to `batch_poller_workflow.py` (60s)
- [x] Verify ruff check and all 1060 tests pass

## Files Modified

- `forge/activities/_heartbeat.py` (new — shared context manager)
- `forge/activities/llm.py`
- `forge/activities/planner.py`
- `forge/activities/exploration.py`
- `forge/activities/sanity_check.py`
- `forge/activities/conflict_resolution.py`
- `forge/activities/extraction.py`
- `forge/activities/batch_poll.py`
- `forge/activities/validate.py`
- `forge/workflows.py` (heartbeat_timeout on LLM + validation invocations)
- `forge/extraction_workflow.py` (heartbeat_timeout on extraction LLM invocation)
- `forge/batch_poller_workflow.py` (heartbeat_timeout on poll invocation)

## Verification

```bash
uv run pytest tests/ -x -q
uv run ruff check src/
```

Grep for `heartbeat` in the activities directory to confirm it's being called.

## Development Notes

- The `heartbeat_during` context manager uses `asyncio.create_task` to run heartbeats concurrently. For truly async calls (Anthropic SDK via httpx), heartbeats fire reliably during the await. For `validate_output`, `subprocess.run` blocks the event loop so heartbeats only fire between subprocess calls — `heartbeat_timeout` is set to 120s to accommodate 60s subprocess timeouts.
- Heartbeat interval is 30s; heartbeat timeouts are 2x the interval (60s for LLM, 60s for polling) except validation (120s due to blocking subprocess).
- The `heartbeat_during` helper lives in `_heartbeat.py` (underscore prefix) since it's internal to the activities package.
