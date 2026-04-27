+++
title = "Validation and Retries Reference"
weight = 84
description = "How Forge validates LLM output with deterministic checks and feeds errors back to the LLM on retry."
topic = "validation-and-retries"
covers = [
    "ValidationResult and ValidationConfig model fields",
    "Validation checks: name, what it runs, pass/fail criteria",
    "TransitionSignal values and mapping rules",
    "Error section format: structure, AST context inclusion rules",
    "CLI flags: --run-ruff-lint, --run-ruff-format, --run-tests, --auto-fix, --max-retries",
]
detail = "Tabular reference for validation configuration and error format."
+++
Technical reference for validation models, checks, transition signal mapping, error
section format, and CLI flags. For design rationale, see
[Validation and Retries](../explanation/validation-and-retries/). For recipes, see
[How to Configure Validation](../howto/configure-validation/).

---

## Data Models

### ValidationResult

The result of a single validation check.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `check_name` | `str` | Yes | Identifier for the check. One of the names in the Validation Checks table below. |
| `passed` | `bool` | Yes | `True` if the check produced no failures. |
| `details` | `str` | No | Raw output from the check tool (lint errors, test failure output, format diff). Present only when `passed` is `False`. |
| `auto_fixed` | `bool` | No | `True` if auto-fix was applied and the check subsequently passed. Defaults to `False`. |

---

### ValidationConfig

Configuration for the validation pipeline, attached to each task definition.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `run_ruff_lint` | `bool` | `True` | Whether to run `ruff check` on modified files. |
| `run_ruff_format` | `bool` | `True` | Whether to run `ruff format --check` on modified files. |
| `run_tests` | `bool` | `False` | Whether to run the test suite via `pytest`. |
| `test_path` | `str` | `"tests/"` | The pytest target path. Used only when `run_tests` is `True`. |
| `test_timeout` | `int` | `60` | Maximum test execution time in seconds. |
| `auto_fix` | `bool` | `False` | If `True`, attempt to auto-fix ruff lint and format violations before treating them as failures. |
| `max_retries` | `int` | `2` | Total number of attempts (initial attempt + retries). A value of 2 means one retry is allowed. |

Domain defaults may differ from the above. See
[Task Domains Reference](task-domains/) for per-domain defaults.

---

## Validation Checks

| Name | Tool | What It Runs | Pass Condition | Fail Condition |
|------|------|--------------|----------------|----------------|
| `ruff_lint` | `ruff check` | Static analysis on modified files using rules from `pyproject.toml`. | Exit code 0 (no violations reported). | Exit code 1 (one or more violations). If `auto_fix=True`, fixable violations are repaired first; non-fixable violations cause failure. |
| `ruff_format` | `ruff format --check` | Format conformance check on modified files. | Exit code 0 (no reformatting needed). | Exit code 1 (one or more files would be reformatted). If `auto_fix=True`, files are reformatted in place; check then re-runs. |
| `tests` | `pytest` | Runs pytest against `test_path` with a `test_timeout` second timeout. | Exit code 0 (all tests collected and passed). | Exit code 1 (one or more tests failed or errored), exit code 2 (collection error), or timeout. |

---

## TransitionSignal Mapping Rules

The `evaluate_transition` activity maps the list of `ValidationResult` objects to a
`TransitionSignal`.

| Signal | Condition |
|--------|-----------|
| `SUCCESS` | All `ValidationResult` objects have `passed=True` (or `auto_fixed=True`). |
| `FAILURE_RETRYABLE` | One or more `ValidationResult` objects have `passed=False`, and the current attempt number is less than `max_retries`. |
| `FAILURE_TERMINAL` | One or more `ValidationResult` objects have `passed=False`, and the current attempt number equals `max_retries`. |

`TransitionSignal` values:

| Value | String Representation | Meaning |
|-------|-----------------------|---------|
| `SUCCESS` | `"success"` | All checks passed. Proceed to commit or next step. |
| `FAILURE_RETRYABLE` | `"failure_retryable"` | Checks failed; retry budget remains. |
| `FAILURE_TERMINAL` | `"failure_terminal"` | Checks failed; no retries remain. Halt and escalate. |

---

## Error Section Format

When a step fails and retries, the error section is appended at position ⑪ in the system
prompt (after all other sections). The structure is:

```
## Previous Attempt Errors (Attempt {N} of {max})

Your previous attempt failed validation. Fix these errors:

### {check_name} failed

{raw tool output}

#### Context around error ({filename}, line {lineno})

{code snippet with annotated error line}
```

### AST Context Inclusion Rules

The code snippet (the "Context around error" block) is generated for lint and format errors
that include a file path and line number. It is not generated for test failures (which may
reference many files) or for checks without location information.

| Condition | Behavior |
|-----------|----------|
| Error has a file path and line number | Python `ast` parses the file and finds the enclosing function or class. The scope header and surrounding lines are included. The error line is annotated with `# <-- ERROR`. |
| Error has a file path but no line number | The code snippet is omitted; only the raw error message is included. |
| Error references a file that cannot be parsed as Python | The code snippet is omitted. |
| Test failure output | No code snippet. The raw pytest output is included as-is. |

The annotated line marker format:

```python
    from typing import Optional  # <-- ERROR: F401 unused import
```

The check name (`F401`) and the short rule description are appended to the comment.

---

## Retry Behavior by Execution Mode

| Execution Mode | On `FAILURE_RETRYABLE` |
|----------------|------------------------|
| Single-step | Worktree is destroyed. Fresh worktree created from the same base. Error section injected into retry prompt. |
| Planned multi-step | Uncommitted changes in the worktree are reset (`git checkout -- .`). Committed history from prior steps is preserved. Error section injected into retry prompt. |
| Fan-out sub-task | Sub-task worktree is destroyed. Fresh sub-task worktree created. Error section injected. Sub-task retry budget is independent of the parent step budget. |

---

## CLI Flags

Flags accepted by `forge run` and related commands that control validation behavior.

| Flag | Type | Default | Effect |
|------|------|---------|--------|
| `--run-ruff-lint` / `--no-ruff-lint` | bool | `True` | Enable or disable the `ruff_lint` check. |
| `--run-ruff-format` / `--no-ruff-format` | bool | `True` | Enable or disable the `ruff_format` check. |
| `--run-tests` | bool | `False` | Enable the `tests` check. |
| `--test-path PATH` | str | `"tests/"` | Set the pytest target for the `tests` check. |
| `--test-timeout SECONDS` | int | `60` | Set the maximum execution time for the test suite. |
| `--auto-fix` | bool | `False` | Apply auto-fix for ruff lint and format violations before treating them as failures. |
| `--max-retries N` | int | `2` | Total number of attempts. `1` means no retries (fail immediately on first failure). |

---

## Related

- [Validation and Retries](../explanation/validation-and-retries/) — Design rationale
  for the validation pipeline and retry semantics.
- [Output Processing Reference](output-processing/) — Edit application errors that
  also feed the retry mechanism.
- [How to Configure Validation](../howto/configure-validation/) — Recipes for
  enabling checks, setting retry limits, and inspecting results.
- [Workflow Step Reference](workflow-step/) — Full activity definitions and timeout
  policies for `validate_output` and `evaluate_transition`.
