# 03 - Retry Policies

**Status:** `DONE`
**Priority:** HIGH
**Code Review Section:** [Section 2b](../reports/code-review-2026-02-26.md#2-high-temporal-operational-risks)

## Problem

Every `workflow.execute_activity` call uses Temporal's default retry policy: unlimited retries with exponential backoff. This means:

- `commit_changes` raising `CommitError("Nothing to commit")` retries forever (documented in MEMORY.md as causing test hangs)
- Persistent Anthropic API errors (e.g., 400 bad request) retry indefinitely
- There is no `non_retryable_error_types` or `maximum_attempts` anywhere

## Acceptance Criteria

- [x] All activity invocations have explicit `RetryPolicy` with `maximum_attempts`
- [x] Non-retryable error types are specified (e.g., `CommitError`, 400-class API errors)
- [x] LLM-calling activities use a different policy than deterministic activities (e.g., `commit_changes`)
- [x] Existing tests pass

## Plan

Define four retry policy presets in `workflows.py` and apply them to all 52 `execute_activity` calls based on activity category:

1. **`_LLM_RETRY`** (max 3 attempts) — for LLM API calls (`call_llm`, `call_planner`, `call_sanity_check`, `call_exploration_llm`, `call_conflict_resolution`, `submit_batch_request`). Non-retryable: `BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError`.
2. **`_LOCAL_RETRY`** (max 2 attempts) — for deterministic/local activities (`assemble_*`, `evaluate_transition`, `validate_output`, `parse_llm_response`, `fulfill_context_requests`, `detect_file_conflicts_activity`, `remove_worktree_activity`).
3. **`_GIT_RETRY`** (max 2 attempts) — for git write operations (`create_worktree_activity`, `commit_changes_activity`, `reset_worktree_activity`). Non-retryable: `CommitError`, `RepoDiscoveryError`.
4. **`_WRITE_RETRY`** (max 2 attempts) — for file write operations (`write_output`, `write_files`). Non-retryable: `OutputWriteError`, `EditApplicationError`.

## Sub-tasks

- [x] Add `from temporalio.common import RetryPolicy` import
- [x] Define `_LLM_RETRY`, `_LOCAL_RETRY`, `_GIT_RETRY`, `_WRITE_RETRY` constants
- [x] Add `retry_policy=` to all 52 `execute_activity` calls
- [x] Verify ruff passes
- [x] Verify all 1060 tests pass (981 unit + 79 workflow)

## Files Modified

- `src/forge/workflows.py` — added `RetryPolicy` import, 4 retry policy constants, `retry_policy=` parameter on all 52 `execute_activity` calls

## Verification

```bash
uv run pytest tests/ -x -q
uv run ruff check src/
```

Grep for `execute_activity` in `workflows.py` to confirm all have `retry_policy=` parameter.

## Development Notes

- Temporal Python SDK converts non-`ApplicationError` exceptions using `type=exception.__class__.__name__` (simple class name, not fully qualified). Confirmed at `temporalio/converter.py:962`. So `non_retryable_error_types=["CommitError"]` matches `forge.git.CommitError`.
- `remove_worktree_activity` uses `_LOCAL_RETRY` (not `_GIT_RETRY`) because it's a cleanup operation — we don't want `WorktreeRemoveError` to be non-retryable since a brief retry may succeed (e.g., transient lock).
- `detect_file_conflicts_activity` uses `_LOCAL_RETRY` despite having `_GIT_TIMEOUT` — it reads files but doesn't modify git state.
- The 4-policy design satisfies the acceptance criteria while keeping the mapping simple: activity name → policy is a pure function of the activity's category.
