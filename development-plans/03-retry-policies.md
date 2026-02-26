# 03 - Retry Policies

**Status:** `NOT STARTED`
**Priority:** HIGH
**Code Review Section:** [Section 2b](../reports/code-review-2026-02-26.md#2-high-temporal-operational-risks)

## Problem

Every `workflow.execute_activity` call uses Temporal's default retry policy: unlimited retries with exponential backoff. This means:

- `commit_changes` raising `CommitError("Nothing to commit")` retries forever (documented in MEMORY.md as causing test hangs)
- Persistent Anthropic API errors (e.g., 400 bad request) retry indefinitely
- There is no `non_retryable_error_types` or `maximum_attempts` anywhere

## Acceptance Criteria

- [ ] All activity invocations have explicit `RetryPolicy` with `maximum_attempts`
- [ ] Non-retryable error types are specified (e.g., `CommitError`, 400-class API errors)
- [ ] LLM-calling activities use a different policy than deterministic activities (e.g., `commit_changes`)
- [ ] Existing tests pass

## Plan

*To be written when work begins.*

## Sub-tasks

*To be written when work begins.*

## Files to Modify

- `forge/workflows.py` (all `execute_activity` calls)

## Verification

```bash
uv run pytest tests/ -x -q
uv run ruff check src/
```

Grep for `execute_activity` in `workflows.py` to confirm all have `retry_policy=` parameter.

## Development Notes

*Append discoveries, decisions, and gotchas here during implementation.*
