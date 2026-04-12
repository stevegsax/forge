# Validation and Retries

This document explains how Forge validates LLM output, how validation results drive
workflow transitions, and how error context is fed back to the LLM on retry. For
field-level specifications, see [Validation and Retries Reference](../reference/validation-and-retries.md).
For configuring the validation pipeline, see
[How to Configure Validation](../howto/configure-validation.md).

---

## Deterministic Before LLM-Based

Forge's validation pipeline runs deterministic checks — linting, formatting, and optionally
tests — before any LLM-based review. This ordering is deliberate.

Deterministic checks are fast, cheap, and produce precise error messages with exact file
paths and line numbers. A ruff lint failure names the rule that was violated, the line it
occurred on, and a short description of the problem. A test failure includes the test name,
the assertion that failed, and the stack trace. These outputs are directly actionable: the
LLM can read them and know exactly what to fix.

LLM-based review is expensive, slower, and produces probabilistic judgments. It is useful
for subjective quality assessment — coherence, consistency with prior work, adherence to
architectural intent — but it cannot reliably catch a missing import or a formatting
violation. Spending tokens on LLM review before confirming that the output is structurally
correct is wasteful.

The design reflects a general principle: compute what you can compute, and reserve LLM
calls for what only the LLM can do. Deterministic validation is the clearest instance of
this principle in the pipeline.

---

## The Validation Pipeline

The `validate_output` activity runs three potential checks, each producing a
`ValidationResult`:

**ruff lint.** Runs `ruff check` on the modified files. Checks for undefined names, unused
imports, common Python anti-patterns, and other rules configured in the project's
`pyproject.toml`. Fails if any lint errors are reported. Auto-fix mode (`--auto-fix`) can
resolve simple fixable violations automatically before treating the result as a failure.

**ruff format.** Runs `ruff format --check` on the modified files. Checks that the output
matches the project's configured formatting style. Fails if any files require reformatting.
Like lint, this check can be run with auto-fix to apply formatting in place.

**Test execution.** Runs `pytest` against a configured test path. This check is disabled by
default because test execution is slower and depends on a correctly configured environment.
It is most useful for tasks that modify logic with existing test coverage. When enabled, a
test failure is a retryable error.

The pipeline runs only the checks that are enabled in the `ValidationConfig` for the
current task. Checks that are not enabled do not appear in the results and do not influence
the transition signal.

---

## Transition Signal Mapping

After validation completes, the `evaluate_transition` activity maps the collection of
`ValidationResult` objects to a single `TransitionSignal`:

**`SUCCESS`** — All enabled checks passed. The step is committed and execution proceeds
to the next step (or terminates if this was the final step).

**`FAILURE_RETRYABLE`** — One or more checks failed, and the step has remaining retry
attempts. The failed results are collected into an error section that is injected into the
next prompt. The worktree is reset or discarded (depending on execution mode), and the step
runs again.

**`FAILURE_TERMINAL`** — One or more checks failed, and no retry attempts remain. The
workflow reports failure with a structured result containing the full validation output.
Execution halts and escalates to a human.

The retry budget is configured via `--max-retries` (default: 2 total attempts, meaning one
retry after an initial failure). The budget is per step in planned mode — a step exhausts
its own retry allowance without affecting other steps.

---

## Why Blind Retries Waste Budget

Before error-aware retries were introduced (Phase 8), the retry loop rebuilt the prompt
from scratch on each attempt. The LLM received an identical context and produced output
with a high probability of repeating the same mistake.

This is not a theoretical concern. Top-performing agents in SWE-bench benchmarks universally
include test and lint output in their feedback loops. Agents that retry without error context
score significantly lower on repair tasks. Aider feeds lint and test output back to the
model after every edit cycle. Claude Code returns tool failure output as conversation
context so the model reasons about the failure before deciding its next action.

The intuition is straightforward: if the LLM generated a missing import on the first
attempt, the error output will say `F401 'typing.Optional' imported but unused` on a
specific line. That is precise, actionable information. On the next attempt, the LLM can
target exactly that line and fix exactly that problem. Without the error, it has no reason
to produce a different result.

---

## AST-Derived Code Context

Raw error messages like `src/forge/models.py:42:1: F401` are actionable but
require the LLM to mentally reconstruct the code at that location. Forge enriches each
error with a code snippet that shows the surrounding scope.

For lint and format errors, the activity parses the error's file path and line number, then
uses Python's `ast` module to find the enclosing function or class definition. It extracts
a code snippet that includes the scope header (the `def` or `class` line) and the error
line, with the error line annotated as `# <-- ERROR`. This gives the LLM immediate visual
context: it can see what function the problem is inside and what the surrounding code looks
like without needing to search through a large file listing.

The Python `ast` module is used rather than tree-sitter because the codebase is
Python-only in Release 1. Tree-sitter support for multi-language projects is deferred to
Release 2 (Phase 13).

A typical enriched error block looks like this:

```
### ruff_lint failed

src/forge/models.py:42:1: F401 `typing.Optional` imported but unused

#### Context around error (models.py, line 42)

from __future__ import annotations

from typing import Optional  # <-- ERROR: F401 unused import

from pydantic import BaseModel, Field
```

---

## Cache Efficiency of Error Placement

The error section is appended at position ⑪ — the very end of the system prompt, after
all other sections including exploration results. This placement is not arbitrary; it is
required for prompt caching to work correctly.

Anthropic's prompt caching works by hashing prompt prefixes. If stable content appears
before volatile content, the stable prefix can be cached and reused across calls. The error
section is the most volatile part of the prompt: it is absent on the first attempt and
unique on each retry. Placing it last ensures that the preceding sections — role statement,
output requirements, project instructions, repo structure, playbooks, task description,
file contents, exploration results — form a stable prefix that can be cached between the
initial attempt and the retry.

If the error section were placed earlier in the prompt (for example, immediately after
the role statement), every retry would invalidate the cache for all content that follows
it, eliminating the caching benefit for file contents and exploration results that may
be the largest sections in the prompt.

---

## Retry Semantics Per Execution Mode

The retry behavior differs across the three execution modes because each mode has a
different worktree and commit structure.

**Single-step mode.** The worktree for the failed attempt is destroyed. A fresh worktree
is created for the retry, branched from the same base. The retry prompt includes the error
section from the failed attempt. There is no persistent state between attempts; each
attempt is fully independent.

**Planned multi-step mode.** A single worktree is shared across all steps of the plan.
When a step fails, uncommitted changes from that step are reset (`git checkout -- .` on
the affected files), but commits from earlier steps are preserved. The retry runs from the
same worktree with the same committed history, which means the retry has access to the
changes made by prior steps. The error section from the failed attempt is injected into
the retry prompt.

**Fan-out sub-tasks.** Each sub-task runs in its own worktree (a child workflow). If a
sub-task fails and retries, the behavior mirrors single-step mode within that sub-task:
the worktree is discarded and a fresh one is created. Sub-tasks do not commit to git; they
produce output files that are gathered by the parent workflow. The sub-task's retry budget
is independent of the parent step's retry budget.

The practical implication for debugging: in planned mode, a step that fails repeatedly
leaves its worktree intact (up to the retry limit). The error state can be inspected in
the worktree after the workflow halts.

---

## Connection to Output Processing

Validation runs after `write_output` has applied the LLM's edits and written new files.
If `write_output` itself fails — because an edit could not be matched — the failure is
reported before validation runs. The error feedback path is the same: the edit failure
message is included in the retry prompt's error section.

This means that both edit application failures and validation failures are handled by the
same retry mechanism. From the LLM's perspective on retry, it receives a clear description
of what failed, whether that was an edit that could not be located or a lint rule that the
generated code violated.

See [Output Processing](output-processing.md) for the edit matching mechanics and the
conditions under which an edit fails.
