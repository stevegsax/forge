# How to Submit Tasks

This guide shows you how to submit tasks to Forge using the CLI. It covers single-step tasks, planned multi-step tasks, JSON task files, fan-out configuration, and sanity checks.

For background on how each execution mode works, see the [task decomposition explanation](../explanation/task-decomposition.md). For full flag and data model documentation, see the [task decomposition reference](../reference/task-decomposition.md).

## Prerequisites

Start the Temporal worker in a separate terminal before submitting tasks:

```bash
forge worker
```

The worker must be running for tasks to execute.

## Submit a single-step task

Use `forge run` without the `--plan` flag. Specify the target files and a description:

```bash
forge run \
    --task-id add-retry-logic \
    --description "Add exponential backoff retry logic to the HTTP client" \
    --target-file src/myapp/http_client.py \
    --context-file src/myapp/config.py
```

The LLM receives the target file contents, auto-discovered imports, and repository structure. It produces edits for existing files or full content for new files. Output is validated with ruff and committed to a worktree branch at `.forge-worktrees/add-retry-logic`.

To submit and return immediately without waiting for the result:

```bash
forge run \
    --task-id add-retry-logic \
    --description "Add exponential backoff retry logic to the HTTP client" \
    --target-file src/myapp/http_client.py \
    --no-wait
```

Check the result later:

```bash
forge status --workflow-id forge-task-add-retry-logic
```

## Submit a planned multi-step task

Add the `--plan` flag. The planner decomposes the task into ordered steps before execution begins:

```bash
forge run \
    --task-id add-auth \
    --description "Add user authentication with password hashing and JWT tokens" \
    --plan
```

When `--plan` is set, `--target-file` is optional -- the planner determines which files each step should touch based on the task description and repository structure.

Each step commits independently on success. The resulting branch has a reviewable commit history showing incremental progress.

## Use a JSON task file

For repeatable or complex task definitions, define the task in a JSON file:

```bash
forge run --task-file task.json
```

Example `task.json`:

```json
{
    "task_id": "add-auth",
    "description": "Add user authentication with password hashing to the API.",
    "domain": "code_generation",
    "target_files": [
        "src/myapp/auth.py",
        "src/myapp/models.py"
    ],
    "context_files": [
        "src/myapp/models.py",
        "src/myapp/api.py"
    ],
    "base_branch": "main",
    "validation": {
        "auto_fix": true,
        "run_ruff_lint": true,
        "run_ruff_format": true,
        "run_tests": false
    },
    "context": {
        "auto_discover": true,
        "include_dependencies": false,
        "token_budget": 100000,
        "max_import_depth": 2
    }
}
```

Field descriptions:

- `task_id` -- Unique identifier. Used in worktree paths and branch names.
- `description` -- What the task should produce. Be specific about the desired behavior.
- `domain` -- Task domain. Options: `code_generation` (default), `research`, `code_review`, `documentation`, `generic`.
- `target_files` -- Files to create or modify.
- `context_files` -- Additional files the LLM should see beyond what auto-discovery provides.
- `base_branch` -- Branch to create the worktree from. Defaults to `main`.
- `validation` -- Controls which deterministic checks run after the LLM produces output.
- `context` -- Controls context assembly behavior: auto-discovery, dependency inclusion, token budget, import depth.

The JSON file and inline CLI options (`--task-id`, `--description`, etc.) are mutually exclusive.

Combine the task file with execution flags:

```bash
forge run --task-file task.json --plan --max-step-attempts 3
```

## Control fan-out depth

Fan-out occurs automatically when the planner declares sub-tasks on a step. Control the maximum nesting depth with `--max-fan-out-depth`:

```bash
# Flat fan-out only (default) -- sub-tasks cannot nest further
forge run \
    --task-id add-tests \
    --description "Add unit tests for the models, api, and utils modules" \
    --plan \
    --max-fan-out-depth 1

# Allow one level of recursive fan-out
forge run \
    --task-id add-tests \
    --description "Add unit tests for the models, api, and utils modules" \
    --plan \
    --max-fan-out-depth 2
```

At depth 1, sub-tasks execute as single-step leaf workflows. At depth 2, sub-tasks can themselves declare sub-tasks that fan out. Each additional level increases coordination overhead.

To disable LLM-based conflict resolution when sub-tasks modify the same file (conflicts become terminal errors instead):

```bash
forge run \
    --task-id add-tests \
    --description "Add unit tests for the models, api, and utils modules" \
    --plan \
    --no-resolve-conflicts
```

## Configure sanity checks

Enable periodic plan re-evaluation during planned execution with `--sanity-check-interval`:

```bash
# Run a sanity check every 3 completed steps
forge run \
    --task-id large-refactor \
    --description "Refactor the validation pipeline to use the strategy pattern" \
    --plan \
    --sanity-check-interval 3
```

After every 3 completed steps, a reasoning-tier LLM reviews the completed results and remaining steps. It can continue with the current plan, revise the remaining steps, or abort the task.

Sanity checks are most useful for long plans (more than five or six steps) where mid-course correction is worth the added latency and token cost. For short plans, the overhead is not justified.

## Configure retries

Control retry limits at each level:

```bash
forge run \
    --task-id fix-parser \
    --description "Fix the off-by-one error in the CSV parser" \
    --target-file src/myapp/parser.py \
    --max-attempts 3

forge run \
    --task-id add-auth \
    --description "Add authentication" \
    --plan \
    --max-step-attempts 3 \
    --max-sub-task-attempts 3
```

- `--max-attempts` -- Task-level retries in single-step mode. Default: 2.
- `--max-step-attempts` -- Per-step retries in planned mode. Default: 2.
- `--max-sub-task-attempts` -- Per-sub-task retries in fan-out. Default: 2.

When a step or sub-task fails validation and retries, the retry prompt includes the validation error output with code context around error locations.

## Enable test validation

By default, only ruff lint and format checks run. To include tests:

```bash
forge run \
    --task-id fix-parser \
    --description "Fix the off-by-one error in the CSV parser" \
    --target-file src/myapp/parser.py \
    --run-tests \
    --test-command "pytest tests/test_parser.py -x"
```

`--test-command` implies `--run-tests`.

## Submit a research task

Research tasks produce markdown files instead of source code. Code validation (ruff lint/format) is disabled:

```bash
forge run \
    --task-id security-audit \
    --description "Conduct a security review of the authentication and authorization modules" \
    --domain research \
    --plan
```

Fan-out works the same as with code tasks -- independent research threads run in parallel and results are gathered.

## Inspect results

After a task completes:

```bash
# Summary
forge status --workflow-id forge-task-add-auth

# Detailed output with token counts, latency, and interaction history
forge status --workflow-id forge-task-add-auth --verbose

# JSON output for programmatic consumption
forge run \
    --task-id add-auth \
    --description "Add authentication" \
    --plan \
    --json
```
