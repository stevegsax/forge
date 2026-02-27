# 09 - Model Layer Cleanup

**Status:** `DONE`
**Priority:** MEDIUM
**Code Review Section:** [Section 5](../reports/code-review-2026-02-26.md#5-medium-model-layer-issues-modelspy)

## Problem

### 5a. LLM usage stats duplicated across 5 result models

`LLMCallResult`, `PlanCallResult`, `SanityCheckCallResult`, `ExtractionCallResult`, and `ConflictResolutionCallResult` all repeat the same 6 fields: `model_name`, `input_tokens`, `output_tokens`, `latency_ms`, `cache_creation_input_tokens`, `cache_read_input_tokens`. A shared mixin or base class would eliminate drift risk.

### 5b. `build_llm_stats` / `build_planner_stats` are identical functions

Two functions that do the same thing, differing only in parameter type annotation. A `Protocol` or unified base would allow a single function.

### 5c. `thinking_effort` is an unvalidated `str`

Accepts any string but only `"low"`, `"medium"`, `"high"`, `"max"` are valid. A `StrEnum` or `Literal` type should be used.

### 5d. `ThinkingConfig` exists but is never embedded

The `ThinkingConfig` model (lines 90-98) models exactly the `(thinking_budget_tokens, thinking_effort)` pair that is duplicated across `ConflictResolutionInput`, `PlannerInput`, `SanityCheckInput`, `BatchSubmitInput`, etc. None of them embed it.

## Acceptance Criteria

- [x] Shared mixin or base class for LLM usage stats fields
- [x] Single `build_llm_stats` function (or equivalent) serving all result types
- [x] `thinking_effort` validated with `Literal` or `StrEnum`
- [x] `ThinkingConfig` embedded in models that currently duplicate its fields
- [x] Existing tests pass

## Plan

### 5c: ThinkingEffort type
- Added `ThinkingEffort = Literal["low", "medium", "high", "max"]` to models.py
- Updated `ThinkingConfig.effort` to use it

### 5d: Embed ThinkingConfig
- Removed `enabled` from `ThinkingConfig` (redundant — `budget_tokens=0` means disabled)
- Changed `ThinkingConfig.budget_tokens` default to 0 (disabled)
- `ForgeTaskInput.thinking` overrides default to `budget_tokens=10_000` (enabled)
- Replaced `thinking_budget_tokens: int` + `thinking_effort: str` pairs with `thinking: ThinkingConfig` in: `ConflictResolutionInput`, `ConflictResolutionCallInput`, `PlannerInput`, `SanityCheckInput`, `BatchSubmitInput`
- Updated all workflow and activity callers
- Activity functions unpack `input.thinking.budget_tokens` / `input.thinking.effort` for the lower-level `build_messages_params` API (which keeps primitive params at the boundary)

### 5a: LLMStats base class
- Made `LLMCallResult`, `PlanCallResult`, `SanityCheckCallResult`, `ExtractionCallResult`, `ConflictResolutionCallResult`, `ParsedLLMResponse` inherit from `LLMStats` instead of repeating 6 fields
- `ParsedLLMResponse` overrides `latency_ms` with `float = 0.0` default

### 5b: Unified build function
- Deleted `build_planner_stats`
- Updated `build_llm_stats` to accept `LLMStats` (base type) so it works with any result subclass
- Updated all callers

## Files Modified

- `src/forge/models.py`
- `src/forge/workflows.py`
- `src/forge/activities/planner.py`
- `src/forge/activities/sanity_check.py`
- `src/forge/activities/conflict_resolution.py`
- `src/forge/activities/batch_submit.py`
- `tests/test_models.py`
- `tests/test_activity_planner.py`
- `tests/test_batch_submit.py`

## Verification

```bash
uv run pytest tests/ -x -q     # 1061 passed
uv run ruff check src/          # No new errors (3 pre-existing)
```

## Development Notes

- `ThinkingConfig.enabled` was redundant with `budget_tokens=0`. Removing it simplified the workflow code — no more conditional `if input.thinking.enabled:` guards.
- The `llm_client.py` functions (`build_messages_params`, `build_thinking_param`) intentionally keep primitive params (`int`, `str`) at the boundary. Activities are the adapter layer that unpacks from `ThinkingConfig`.
- `ParsedLLMResponse` needs `latency_ms: float = 0.0` because batch responses don't have a meaningful latency value.
