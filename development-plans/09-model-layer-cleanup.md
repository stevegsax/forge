# 09 - Model Layer Cleanup

**Status:** `NOT STARTED`
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

- [ ] Shared mixin or base class for LLM usage stats fields
- [ ] Single `build_llm_stats` function (or equivalent) serving all result types
- [ ] `thinking_effort` validated with `Literal` or `StrEnum`
- [ ] `ThinkingConfig` embedded in models that currently duplicate its fields
- [ ] Existing tests pass

## Plan

*To be written when work begins.*

## Sub-tasks

*To be written when work begins.*

## Files to Modify

- `forge/models.py`
- Potentially activity files that construct result models
- Tests that create result model instances

## Verification

```bash
uv run pytest tests/ -x -q
uv run ruff check src/
```

## Development Notes

*Append discoveries, decisions, and gotchas here during implementation.*
