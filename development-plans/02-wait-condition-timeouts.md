# 02 - Wait Condition Timeouts

**Status:** `DONE`
**Priority:** HIGH
**Code Review Section:** [Section 2c, 2d](../reports/code-review-2026-02-26.md#2-high-temporal-operational-risks)

## Problem

### Unbounded `wait_condition` for batch signals

```python
await workflow.wait_condition(lambda: len(self._batch_results) > 0)
```

No timeout. If the batch poller fails or the signal is lost, the workflow hangs forever. Anthropic's batch API has a 24-hour expiry -- at minimum, add `timeout=timedelta(hours=25)`.

Locations:

- `workflows.py:234`
- `workflows.py:1370`

### No execution timeout on top-level workflow

Child workflows correctly get `execution_timeout` with depth-based scaling, but `ForgeTaskWorkflow` itself has no execution timeout when started from the CLI. A planned workflow with many steps could run indefinitely.

### No graceful shutdown timeout on worker

`worker.py` uses the default `graceful_shutdown_timeout=0`, meaning on SIGINT, in-flight LLM activities are immediately abandoned.

## Acceptance Criteria

- [x] All `wait_condition` calls have explicit timeouts
- [x] `ForgeTaskWorkflow` has an execution timeout when started from CLI
- [x] Worker has a graceful shutdown timeout configured
- [x] Existing tests pass

## Plan

1. Add `_BATCH_WAIT_TIMEOUT = timedelta(hours=25)` constant to `workflows.py` alongside existing activity timeout presets.
2. Add `timeout=_BATCH_WAIT_TIMEOUT` to both `wait_condition` calls in `ForgeTaskWorkflow._call_llm_batch` and `ForgeSubTaskWorkflow._call_llm_batch`.
3. Add `_WORKFLOW_EXECUTION_TIMEOUT = timedelta(hours=48)` constant to `cli.py`.
4. Add `execution_timeout=_WORKFLOW_EXECUTION_TIMEOUT` to both `client.execute_workflow()` and `client.start_workflow()` calls in CLI.
5. Add `graceful_shutdown_timeout=timedelta(seconds=30)` to `Worker()` constructor in `worker.py`.

## Sub-tasks

- [x] Add batch wait timeout constant and apply to both `wait_condition` calls
- [x] Add workflow execution timeout constant and apply to both CLI workflow start points
- [x] Add graceful shutdown timeout to worker constructor
- [x] Verify ruff passes and imports resolve

## Files Modified

- `src/forge/workflows.py` — added `_BATCH_WAIT_TIMEOUT` constant (line 100), added `timeout=` to both `wait_condition` calls (lines 237, 1382)
- `src/forge/cli.py` — added `timedelta` import, added `_WORKFLOW_EXECUTION_TIMEOUT` constant, added `execution_timeout=` to both workflow start calls
- `src/forge/worker.py` — added `graceful_shutdown_timeout=timedelta(seconds=30)` to `Worker()` constructor

## Verification

```bash
uv run pytest tests/ -x -q
uv run ruff check src/
```

Grep for `wait_condition` in `workflows.py` to confirm all have `timeout=` parameter.

## Development Notes

- `wait_condition` raises `asyncio.TimeoutError` when the timeout fires, which propagates as a workflow failure — correct behavior for a dead batch.
- 25h timeout gives 1h buffer beyond the 24h Anthropic batch API expiry.
- 48h workflow execution timeout is generous enough for large planned workflows (multiple steps x 24h batch ceiling) while preventing indefinite hangs.
- 30s graceful shutdown gives in-flight activities time for git operations and serialization without blocking shutdown indefinitely.
