# How to Configure Validation

This guide shows how to enable or disable validation checks, add test execution, set
retry limits, and inspect validation results for a failed step.

For background on how validation works, see
[Validation and Retries](../explanation/validation-and-retries.md). For the full list of
flags and model fields, see
[Validation and Retries Reference](../reference/validation-and-retries.md).

---

## Enable or Disable Specific Checks

By default, `ruff_lint` and `ruff_format` are enabled and the `tests` check is disabled.

To disable lint checking for a task:

```
forge run --no-ruff-lint --task-file task.json
```

To disable format checking:

```
forge run --no-ruff-format --task-file task.json
```

To disable both lint and format checking (run with no deterministic checks):

```
forge run --no-ruff-lint --no-ruff-format --task-file task.json
```

To re-enable a check that was disabled in a domain's default `ValidationConfig`, pass
the positive flag explicitly:

```
forge run --run-ruff-lint --task-file task.json
```

---

## Apply Auto-Fix Before Failing

If you want ruff to repair fixable violations automatically before reporting a failure,
use `--auto-fix`:

```
forge run --auto-fix --task-file task.json
```

With `--auto-fix`, ruff applies all safe fixes in place. If all violations were fixable,
the check is recorded as passing (`auto_fixed=True`). If non-fixable violations remain
after auto-fix, the check fails with those violations reported in the error section.

---

## Add Test Execution

To run the project's test suite as part of validation:

```
forge run --run-tests --task-file task.json
```

By default, pytest targets `tests/`. To specify a different path:

```
forge run --run-tests --test-path tests/unit/ --task-file task.json
```

To set a custom timeout (default is 60 seconds):

```
forge run --run-tests --test-timeout 120 --task-file task.json
```

Test execution is disabled by default because it adds latency and requires a correctly
configured test environment. Enable it for tasks that modify logic with existing test
coverage.

---

## Configure Retry Limits

The default is 2 total attempts (one retry after an initial failure). To allow more retries:

```
forge run --max-retries 3 --task-file task.json
```

To disable retries entirely (fail on the first validation failure without retrying):

```
forge run --max-retries 1 --task-file task.json
```

If you are diagnosing a problem and want the workflow to stop on first failure without
consuming retry budget, `--max-retries 1` is the correct flag.

---

## Set Validation Defaults in a Task File

All validation flags can also be specified in the task JSON file rather than on the
command line. This is useful when the same validation configuration should apply every
time a particular task is run.

```json
{
    "task_id": "add-feature-x",
    "description": "...",
    "target_files": ["src/forge/feature_x.py"],
    "validation_config": {
        "run_ruff_lint": true,
        "run_ruff_format": true,
        "run_tests": true,
        "test_path": "tests/test_feature_x.py",
        "test_timeout": 90,
        "auto_fix": false,
        "max_retries": 3
    }
}
```

Command-line flags override the task file values if both are provided.

---

## Inspect Validation Results for a Failed Step

After a workflow fails, use `forge status` to see the validation results:

```
forge status --workflow-id <workflow-id>
```

For detailed output including the full error section that was sent to the LLM on retry:

```
forge status --workflow-id <workflow-id> --verbose
```

The verbose output includes:

- The `ValidationResult` list for each attempt, showing which checks passed and which
  failed.
- The `details` field from each failed check (raw lint output, test failure output).
- The assembled error section with AST-derived code snippets that was injected into the
  retry prompt.

To filter to only failed steps:

```
forge status --workflow-id <workflow-id> --failed-only
```

---

## Diagnose Why a Check Failed Without Running a Full Task

If you want to understand why ruff is failing on a specific file outside of a Forge
workflow, run ruff directly:

```
ruff check src/forge/models.py
ruff format --check src/forge/models.py
```

The output matches what Forge's `validate_output` activity captures. This is useful for
debugging whether a failure is in the generated code or in the project's ruff configuration.

---

## Related

- [Validation and Retries](../explanation/validation-and-retries.md) — How the pipeline
  works and why it is structured as it is.
- [Validation and Retries Reference](../reference/validation-and-retries.md) — Complete
  flag list, model fields, and transition mapping rules.
- [Output Processing Reference](../reference/output-processing.md) — Edit application
  errors that also trigger the retry path.
