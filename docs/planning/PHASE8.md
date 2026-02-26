# Phase 8: Error-Aware Retries

## Goal

Feed validation errors back to the LLM on retry so it knows what went wrong and can fix it. Currently, retries start from a blank slate — the LLM receives an identical prompt and has no information about the previous failure. This wastes retry budget on repeating the same mistakes.

The deliverable: when a step fails validation and retries, the retry prompt includes the validation error output (lint errors, test failures) with enough surrounding code context for the LLM to understand and fix the problem.

## Problem Statement

The retry loop in `workflows.py` has three sites (single-step, planned step, sub-task) that all follow the same pattern:

1. Assemble context → call LLM → write output → validate → evaluate transition.
2. On `FAILURE_RETRYABLE`: discard changes (remove worktree or reset), loop back to step 1.
3. On retry, assemble context again — producing an **identical prompt**.

The `validation_results` from the failed attempt are local variables that fall out of scope. They are never passed to context assembly. The `AssembleStepContextInput`, `AssembleContextInput`, and `AssembleSubTaskContextInput` models have no fields for prior errors or attempt number.

This means the LLM retries blind. If a lint error was caused by a missing import, the retry produces the same missing import. Aider and Claude Code both feed error output back — Aider sends lint/test output directly in the conversation, Claude Code returns tool failure output as conversation context.

## Prior Art

- **Aider**: After every edit, runs tree-sitter-based linting and optional external linters/tests. On failure, sends error output to the LLM in the next message. Errors are shown within their containing function/class context (extracted via tree-sitter AST), not as raw line numbers. This AST-contextualized format produces better fix rates.
- **Claude Code**: Returns tool failure output (test errors, lint output, command exit codes) as tool results in the conversation. The model reasons about the failure and decides its next action. No explicit retry count — the model decides when to stop.
- **SWE-bench agents**: Top-performing agents universally include test output in their feedback loops. Agents that retry without error context score significantly lower.

## Scope

**In scope:**

- Add `prior_validation_errors` to context assembly inputs for all three retry paths (single-step, planned step, sub-task).
- Include validation error summaries and details in the retry prompt as a "Previous Attempt Errors" section.
- Format errors with file context: for lint errors, include the enclosing function/class signature (extracted via `ast`) alongside the error message.
- Pass the attempt number into context assembly so the prompt can indicate "Attempt 2 of 2".

**Out of scope (deferred):**

- Tree-sitter-based error formatting (Python `ast` is sufficient for Phase 8).
- Conversation-style retries (appending errors as a follow-up message rather than rebuilding the prompt). This would require switching from stateless prompt construction to stateful conversation management.
- Adaptive retry strategy (varying the model or temperature on retry).
- LLM-based error classification (distinguishing "trivial lint fix" from "fundamental approach error").

## Architecture

### Modified Context Assembly

The three context assembly functions gain optional parameters for retry information:

```
build_step_system_prompt(
    ...,
    prior_errors: list[ValidationResult] | None = None,
    attempt: int = 1,
    max_attempts: int = 2,
)

build_sub_task_system_prompt(
    ...,
    prior_errors: list[ValidationResult] | None = None,
    attempt: int = 1,
    max_attempts: int = 2,
)

build_system_prompt_with_context(
    ...,
    prior_errors: list[ValidationResult] | None = None,
    attempt: int = 1,
    max_attempts: int = 2,
)
```

When `prior_errors` is non-empty, a "Previous Attempt Errors" section is appended before the Output Requirements section:

```
## Previous Attempt Errors (Attempt 2 of 2)

Your previous attempt failed validation. Fix these errors:

### ruff_lint failed
```
src/forge/models.py:42:1: F401 `typing.Optional` imported but unused
```

#### Context around error (models.py, line 42)
```python
from __future__ import annotations

from typing import Optional  # <-- ERROR: F401 unused import

from pydantic import BaseModel, Field
```

### tests failed
```
FAILED tests/test_models.py::TestFoo::test_bar - AssertionError: ...
```

Do NOT repeat the same mistakes. Address each error listed above.
```

### Error Context Enrichment (Pure Function)

A new pure function `enrich_error_context` in `activities/context.py` reads the target file and uses `ast` to find the enclosing scope for each error line number:

```
def enrich_error_context(
    error_output: str,
    file_path: str,
    source: str,
) -> str:
```

This parses line numbers from ruff output (format: `path:line:col: code message`), finds the enclosing `FunctionDef` or `ClassDef` via `ast`, and returns the error message with a code snippet showing the surrounding context.

For test failures, the raw output is included as-is (test output typically already includes enough context).

### Modified Data Models

```
AssembleStepContextInput:
    ...existing fields...
    prior_errors: list[ValidationResult] = Field(default_factory=list)
    attempt: int = Field(default=1)
    max_attempts: int = Field(default=2)

AssembleContextInput:
    ...existing fields...
    prior_errors: list[ValidationResult] = Field(default_factory=list)
    attempt: int = Field(default=1)
    max_attempts: int = Field(default=2)

AssembleSubTaskContextInput:
    ...existing fields...
    prior_errors: list[ValidationResult] = Field(default_factory=list)
    attempt: int = Field(default=1)
    max_attempts: int = Field(default=2)
```

All new fields have defaults, so existing callers and serialized payloads remain backward compatible.

### Modified Workflow Retry Loops

In `workflows.py`, the three retry sites are updated to pass validation results into the next iteration's context assembly:

**Single-step** (line ~297):

```python
if signal == TransitionSignal.FAILURE_RETRYABLE:
    prior_errors = validation_results  # Capture before worktree removal
    await workflow.execute_activity("remove_worktree_activity", ...)
    continue  # Next iteration uses prior_errors
```

The `assemble_context` call at the top of the loop passes `prior_errors` and `attempt`.

**Planned step** (line ~513):

```python
if signal == TransitionSignal.FAILURE_RETRYABLE:
    prior_errors = validation_results  # Capture before reset
    await workflow.execute_activity("reset_worktree_activity", ...)
    continue
```

**Sub-task** (line ~858): Same pattern.

## Project Structure

Modified files:

```
src/forge/
├── models.py                   # Modified: add prior_errors, attempt, max_attempts to assembly inputs
├── activities/
│   └── context.py              # Modified: accept and render prior errors in all prompt builders
└── workflows.py                # Modified: pass validation_results through retry loops
```

## Dependencies

No new dependencies.

## Key Design Decisions

### D51: Error Feedback in Retry Context

**Decision:** Include validation errors from the previous attempt in the retry prompt as a dedicated section, with optional AST-derived code context around error locations.

**Rationale:** Blind retries waste tokens repeating the same mistakes. Both Aider and Claude Code feed error output back to the LLM. The error section is placed before Output Requirements to ensure the LLM sees it before generating. AST-based context enrichment (showing the enclosing function around an error line) helps the LLM understand errors without requiring it to re-read the entire file. Test failure output is included verbatim since it typically already contains sufficient context.

### D52: Backward-Compatible Retry Fields

**Decision:** All retry-related fields (`prior_errors`, `attempt`, `max_attempts`) use defaults (`[]`, `1`, `2`), so existing serialized payloads and callers work unchanged.

**Rationale:** Temporal workflows may have in-flight executions when this change deploys. Default values ensure old payloads deserialize correctly. The first attempt always has `prior_errors=[]`, so the prompt is unchanged for non-retry calls.

## Implementation Order

1. Add `prior_errors`, `attempt`, `max_attempts` fields to `AssembleContextInput`, `AssembleStepContextInput`, `AssembleSubTaskContextInput` in `models.py`.
2. Add `enrich_error_context` pure function to `activities/context.py`.
3. Add `_build_error_section` pure function to `activities/context.py`.
4. Update `build_system_prompt_with_context`, `build_step_system_prompt`, `build_sub_task_system_prompt` to accept and render prior errors.
5. Update `assemble_context`, `assemble_step_context`, `assemble_sub_task_context` activities to pass through the new fields.
6. Update the three retry loops in `workflows.py` to capture `validation_results` and pass them to the next iteration's context assembly input.
7. Tests for each step.

## Edge Cases

- **First attempt (no prior errors):** `prior_errors=[]` produces no error section — prompt is unchanged from current behavior.
- **Error output too large:** Truncate to a configurable limit (default: 2000 characters per error) to avoid blowing the context budget. Include a "... truncated" marker.
- **Ruff error without parseable line number:** Include the raw error text without AST enrichment. The function is best-effort.
- **Target file deleted between attempts (single-step mode, worktree recreated):** AST enrichment reads from the worktree, which starts fresh. Skip enrichment and include raw error text.
- **Multiple failures across retries:** Only the most recent attempt's errors are included (not the full history), keeping the section bounded.

## Definition of Done

Phase 8 is complete when:

- Retry prompts include validation errors from the previous attempt.
- Lint errors include surrounding code context (enclosing function/class).
- Test failure output is included verbatim.
- The attempt number is visible in the prompt ("Attempt 2 of 2").
- First-attempt prompts are unchanged (backward compatible).
- All existing tests pass.
- New unit tests cover: error section rendering, AST context enrichment, workflow retry loop error passing.
