# Forge Codebase Review

**Date:** 2026-02-26
**Reviewer:** Claude Opus 4.6
**Commit:** 38bb337 (main)

## Executive Summary

Forge is a well-structured ~6,500 LOC LLM task orchestrator with ~18,000 LOC of tests (nearly 3:1 test-to-source ratio). The architecture is sound: batch-first design, Temporal orchestration, context intelligence via import graph analysis, and a clear phase-based development roadmap. However, the review uncovered **one critical Temporal determinism violation**, several high-severity Temporal operational risks, significant code duplication, and scattered SOLID violations.

---

## 1. CRITICAL: Temporal Determinism Violation

**Filesystem I/O inside workflow code** (`workflows.py:1157, 1680`)

`detect_file_conflicts` is imported from the activities module and called directly inside workflow methods `_run_fan_out_step` and `_run_nested_fan_out`. This function reads files from disk:

```python
original_path = Path(worktree_path) / file_path
if original_path.is_file():
    original_content = original_path.read_text()
```

Temporal workflows must be deterministic -- they can be replayed at any time on any worker. Filesystem reads inside workflow code will return different results on replay (or fail entirely on a different worker). **This must be moved into an activity.**

---

## 2. HIGH: Temporal Operational Risks

### 2a. No heartbeating anywhere

Zero activities in the entire codebase call `activity.heartbeat()`. LLM-calling activities have 5-minute `start_to_close_timeout` values. Without heartbeats:

- Worker crashes mid-LLM-call are not detected until the full timeout expires
- Activities cannot receive cancellation signals
- The Temporal server has no visibility into activity progress

Affected activities: `call_llm`, `call_planner`, `call_exploration_llm`, `call_sanity_check`, `call_conflict_resolution`, `call_extraction_llm`, `poll_batch_results`, `validate_output`.

### 2b. No retry policies on any activity

Every `workflow.execute_activity` call uses Temporal's default retry policy: **unlimited retries with exponential backoff**. This means:

- A `commit_changes` that raises `CommitError("Nothing to commit")` retries forever (this is already documented in MEMORY.md as a known issue causing test hangs)
- Persistent Anthropic API errors (e.g., 400 bad request) retry indefinitely
- There is no `non_retryable_error_types` or `maximum_attempts` anywhere

### 2c. Unbounded `wait_condition` for batch signals (`workflows.py:234, 1370`)

```python
await workflow.wait_condition(lambda: len(self._batch_results) > 0)
```

No timeout. If the batch poller fails or the signal is lost, the workflow hangs forever. Anthropic's batch API has a 24-hour expiry -- at minimum, add `timeout=timedelta(hours=25)`.

### 2d. No execution timeout on the top-level workflow

Child workflows correctly get `execution_timeout` with depth-based scaling, but `ForgeTaskWorkflow` itself has no execution timeout when started from the CLI. A planned workflow with many steps could run indefinitely.

### 2e. Worker has no graceful shutdown timeout (`worker.py`)

Default `graceful_shutdown_timeout=0` means on SIGINT, in-flight LLM activities are immediately abandoned, wasting API calls and potentially leaving worktrees in inconsistent state.

---

## 3. HIGH: Structural Duplication in `workflows.py`

The file is 1,818 lines and contains **two workflow classes that share ~400 lines of identical code**:

| Duplicated Element | ForgeTaskWorkflow | ForgeSubTaskWorkflow |
|---|---|---|
| `_call_llm_batch` | Lines 210-250 | Lines 1347-1386 |
| `_call_generation` | Lines 256-275 | Lines 1392-1411 |
| `_call_conflict_resolution` | Lines 364-400 | Lines 1413-1449 |
| `__init__` / signal / batch fields | Lines 183-190 | Lines 1320-1327 |
| Conflict detection + resolution blocks | Lines 1156-1212 | Lines 1679-1738 |
| Remove worktree pattern | 4 occurrences | 3 occurrences |
| Error summary formatting | 3 occurrences | 2 occurrences |

Temporal workflows can't use inheritance, but these could be extracted into **module-level helper functions** or a **composition object** shared by both classes.

Additionally, `_run_planned` (lines 693-1023) is a **331-line method** handling 12 distinct responsibilities: worktree creation, planner context assembly, exploration, planner LLM call, step iteration, fan-out dispatch, step execution, commit, worktree reset, sanity check, plan revision, and result assembly.

---

## 4. HIGH: `_persist_interaction` Duplicated Across 5 Files

The observability persistence function is copy-pasted with minor variations in:

- `activities/llm.py:83-108`
- `activities/sanity_check.py:207-237`
- `activities/conflict_resolution.py:245-276`
- `activities/planner.py:240-271`
- `activities/extraction.py:278-308`

Each copy follows: get db_path -> check None -> build AssembledContext -> get engine -> build interaction dict -> save -> catch Exception. This should be a single shared helper.

---

## 5. MEDIUM: Model Layer Issues (`models.py`)

### 5a. LLM usage stats duplicated across 5 result models

`LLMCallResult`, `PlanCallResult`, `SanityCheckCallResult`, `ExtractionCallResult`, and `ConflictResolutionCallResult` all repeat the same 6 fields (`model_name`, `input_tokens`, `output_tokens`, `latency_ms`, `cache_creation_input_tokens`, `cache_read_input_tokens`). A shared mixin or base class would eliminate drift risk.

### 5b. `build_llm_stats` / `build_planner_stats` are identical functions

Two functions that do the same thing, differing only in parameter type annotation. A `Protocol` or unified base would allow a single function.

### 5c. `thinking_effort` is an unvalidated `str`

Accepts any string but only `"low"`, `"medium"`, `"high"`, `"max"` are valid. A `ThinkingConfig` model already exists but is not reused in any of the 5 input models that repeat these fields. Use a `StrEnum` or `Literal` type.

### 5d. `ThinkingConfig` exists but is never embedded

The `ThinkingConfig` model (lines 90-98) models exactly the `(thinking_budget_tokens, thinking_effort)` pair that is duplicated across `ConflictResolutionInput`, `PlannerInput`, `SanityCheckInput`, `BatchSubmitInput`, etc. None of them embed it.

---

## 6. MEDIUM: Context Assembly Issues (`activities/context.py`)

### 6a. Dead constant `_OUTPUT_REQUIREMENTS` (line 47-57)

Byte-identical to `_CODE_OUTPUT_REQUIREMENTS` in `domains.py`. No production code in `context.py` uses it -- all prompts now get output requirements from `get_domain_config()`. Only a test asserts they're equal.

### 6b. `_detect_package_name` duplicated in `context.py` and `planner.py`

Character-for-character identical function in both files. Extract to a shared module.

### 6c. `build_error_section` performs I/O (file reads) despite being mixed with pure functions

Reads files from disk at line 199 (`full_path.read_text()`). Should accept file contents as parameters to be truly pure, per the project's "Function Core / Imperative Shell" convention.

### 6d. `infer_task_tags` uses fragile substring matching (lines 424-458)

`"test" in desc_lower` matches "latest", "contest", "attestation". Use word-boundary regex (`\btest\b`).

### 6e. Manual `AssembledContext` reconstruction drops fields on schema change (line 548-555)

Manually copies fields instead of using `model_copy(update={...})`, creating a silent field-dropping bug if `AssembledContext` gains new fields.

---

## 7. MEDIUM: CLI Issues (`cli.py`)

### 7a. `run()` command has 37 parameters

A 130-line monolith handling validation, task building, model config, thinking config, and execution dispatch. Decompose the body into helper functions.

### 7b. `_submit_and_wait` / `_submit_no_wait` are ~100 lines of near-identical code

Same 13 keyword parameters, same client connection, same `ForgeTaskInput` construction. Only the final `execute_workflow` vs `start_workflow` call differs.

### 7c. `format_verbose_result` performs database I/O

Documented as a formatting function but queries the observability store. Violates the project's own Function Core / Imperative Shell convention.

### 7d. `_persist_run` silently swallows all exceptions with bare `pass`

Unlike the activity-level equivalents that at least log a warning, this one has `except Exception: pass` with no logging.

### 7e. `DEFAULT_TEMPORAL_ADDRESS` defined in both `cli.py` and `worker.py`

Duplicated constant. Should be in a shared location.

---

## 8. MEDIUM: Activity Boilerplate Duplication

### 8a. Tracing boilerplate repeated across 7 LLM activity wrappers

The `tracer = get_tracer(); with tracer.start_as_current_span(...); client = get_anthropic_client(); span.set_attributes(...)` pattern is repeated in `llm.py`, `sanity_check.py`, `conflict_resolution.py`, `extraction.py`, `batch_submit.py`, `batch_poll.py`, and `batch_parse.py`. A decorator or context manager would eliminate this.

### 8b. Subprocess handler duplication in `providers.py`

4 subprocess-based handlers (`handle_run_tests`, `handle_lint_check`, `handle_git_log`, `handle_git_diff`) duplicate the same run/catch-timeout/truncate pattern. Extract a `_run_subprocess` helper.

---

## 9. MEDIUM: Other Code Quality Issues

### 9a. `batch_poll.py` likely bug: `final_status` uses cumulative counter (line 130)

```python
final_status = "succeeded" if signals_sent > 0 else "errored"
```

`signals_sent` is cumulative across all jobs in the polling loop. If job A sent 1 signal, job B gets `"succeeded"` even if job B sent 0 signals. Should track signals per-job.

### 9b. `llm_client.py`: `effort` parameter accepted but never used in `build_thinking_param`

The function accepts `effort` but returns the same dict regardless. Either the Opus-specific thinking effort feature is unimplemented, or this is dead code.

### 9c. `write_files` does not populate `output_files` on `WriteResult`

`write_output` populates both `files_written` and `output_files`; `write_files` only populates `files_written`. If callers expect `output_files`, they get an empty dict.

### 9d. `eval/judge.py`: `model_name` parameter silently ignored (line 176)

`judge_plan` accepts `model_name` but never passes it to `execute_judge_call`, which hardcodes `DEFAULT_JUDGE_MODEL`. The caller's intent to override the model is silently dropped.

### 9e. `code_intel/budget.py`: `compute_budget` and `ContextBudget` are dead code

No production callers. Only referenced in the test file and the module itself.

### 9f. `code_intel/graph.py`: Dead `else` branch (lines 206-215)

The `elif distance <= max_depth` is always true at that point because `distance > max_depth` was already filtered out. The `else` branch assigning `Relationship.DOWNSTREAM` can never execute.

### 9g. `planner.py` imports private `_`-prefixed functions from `context.py`

`_read_context_files` and `_read_project_instructions` break encapsulation. Promote them to public API or extract to a shared module.

### 9h. `assemble_planner_context` takes `AssembleContextInput` but ignores half its fields

ISP violation -- the planner activity receives a model with fields it never reads (`prior_errors`, `attempt`, `max_attempts`).

### 9i. `store.py` creates a new SQLAlchemy engine on every call to `get_engine`

`_persist_interaction` is called after every LLM call, creating a new engine (with connection pool) each time. Consider caching engines by path.

---

## 10. LOW: Minor Issues

| Issue | Location |
|---|---|
| `_SubprocessResult` / `_GitResult` are structurally identical dataclasses | `git.py`, `validate.py` |
| `import json as json_mod` appears 3 times to avoid parameter shadowing | `cli.py:801,978,1008` |
| `_validate_task_id` called 2-3 times for the same task_id per git operation | `git.py` |
| Inconsistent heading levels (`##` vs `###`) across system prompt builders | `context.py` |
| `build_extraction_system_prompt` does defensive JSON parsing that duplicates the caller | `extraction.py:96-100` |
| `code_intel/__init__.py` reads direct import files twice when `include_dependencies=True` | `code_intel/__init__.py:167,206` |
| Check functions in `eval/deterministic.py` all accept `(plan, task, known_repo_files)` even when 2 of 3 are unused | `eval/deterministic.py` |
| `resolve_model` uses a throwaway dict instead of `match` statement | `models.py:80-87` |
| Hardcoded `"src"` and `"forge"` in provider handlers | `providers.py:115,132,137` |
| `conflict_resolution.py`: unused `domain` parameter in `build_conflict_resolution_system_prompt` | line 112 |
| `execute_planner_call` hardcodes fallback model string instead of importing from `ModelConfig` | `planner.py:198` |

---

## 11. What's Done Well

- **Test coverage**: 18,000 LOC of tests for 6,500 LOC of source is excellent coverage
- **Pydantic models**: Strong typing throughout with well-documented fields
- **Temporal signal pattern**: Batch signal delivery via `wait_condition` on a list is correct
- **`workflow.unsafe.imports_passed_through()`**: Used correctly at module level for all model imports
- **Deterministic workflow IDs**: `forge-task-{task_id}` and `forge-subtask-{compound_id}` are meaningful and debuggable
- **`pydantic_data_converter`**: Correctly configured on the Client, not the Worker
- **`result_type`**: Consistently passed on all string-based activity references
- **Batch-first architecture**: The entire prompt construction pipeline works for both synchronous and batch modes
- **Code intelligence**: Import graph analysis with PageRank ranking and token budget management is sophisticated and well-tested
- **Phase documentation**: 14 phase specs with clear scope and rationale provide excellent project context
- **No non-deterministic operations**: No `random`, `datetime.now`, or `uuid.uuid4` in workflow code (aside from the filesystem I/O issue above)

---

## Recommended Priority Order

1. **Fix the determinism violation**: Move `detect_file_conflicts` to an activity
2. **Add timeouts to `wait_condition`** calls
3. **Add explicit `RetryPolicy`** with `maximum_attempts` on all activity invocations
4. **Extract shared workflow helpers** to reduce the ~400 lines of duplication between workflow classes
5. **Decompose `_run_planned`** from 331 lines into 3-4 focused methods
6. **Extract `_persist_interaction`** into a shared module
7. **Add heartbeating** to long-running activities
8. **Fix the `batch_poll.py` `signals_sent` bug**
9. **Unify the `*CallResult` models** with a shared mixin for LLM usage stats
10. **Clean up dead code**: `_OUTPUT_REQUIREMENTS`, `compute_budget`/`ContextBudget`, unused parameters
