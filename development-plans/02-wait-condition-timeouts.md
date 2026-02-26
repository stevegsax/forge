# 02 - Wait Condition Timeouts

**Status:** `NOT STARTED`
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

- [ ] All `wait_condition` calls have explicit timeouts
- [ ] `ForgeTaskWorkflow` has an execution timeout when started from CLI
- [ ] Worker has a graceful shutdown timeout configured
- [ ] Existing tests pass

## Plan

*To be written when work begins.*

## Sub-tasks

*To be written when work begins.*

## Files to Modify

- `forge/workflows.py` (wait_condition timeouts)
- `forge/cli.py` (execution timeout on workflow start)
- `forge/worker.py` (graceful shutdown timeout)

## Verification

```bash
uv run pytest tests/ -x -q
uv run ruff check src/
```

Grep for `wait_condition` in `workflows.py` to confirm all have `timeout=` parameter.

## Development Notes

*Append discoveries, decisions, and gotchas here during implementation.*
