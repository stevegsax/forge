# Submitting Tasks

This guide covers how to submit code development and research tasks to Forge.

## Prerequisites

Before submitting tasks, start the Temporal worker in a separate terminal:

```bash
forge worker
```

The worker must be running for tasks to execute. It connects to the Temporal server (default `localhost:7233`) and polls for queued workflows.

## Code Development Tasks

Code development is the default domain. Forge assembles context from your repository, calls the LLM to generate or modify source files, and validates the output with ruff lint and format checks.

### Single-step (inline)

For focused changes to one or a few files:

```bash
forge run \
    --task-id add-retry-logic \
    --description "Add exponential backoff retry logic to the HTTP client" \
    --target-file src/myapp/http_client.py \
    --context-file src/myapp/config.py
```

- `--task-id` — A unique identifier for this task. Used in worktree branch names and workflow IDs.
- `--description` — What the task should produce. Be specific about the desired behavior.
- `--target-file` — File to create or modify. Repeatable for multiple files.
- `--context-file` — Additional files the LLM should see. Repeatable. Forge also discovers imports automatically.

The LLM produces search/replace edits for existing files and full content for new files. Output is validated with ruff, and the result is committed to a worktree branch.

### Planned multi-step

For larger changes that span multiple files or require ordered steps, use `--plan`. A planner LLM decomposes the task into ordered steps, each of which executes independently and commits on success:

```bash
forge run \
    --task-id add-auth \
    --description "Add user authentication with password hashing and JWT tokens" \
    --plan
```

When `--plan` is set, `--target-file` is optional — the planner determines which files each step should touch.

### Fan-out with sub-tasks

The planner may decompose steps into parallel sub-tasks when work is independent. Each sub-task runs in its own git worktree. Results are gathered, merged, and validated by the parent step. This happens automatically during planning when the planner identifies parallelizable work.

Control fan-out depth with `--max-fan-out-depth`:

```bash
forge run \
    --task-id add-tests \
    --description "Add unit tests for the models, api, and utils modules" \
    --plan \
    --max-fan-out-depth 2
```

- `--max-fan-out-depth 1` (default) — Flat fan-out only (sub-tasks cannot nest further).
- `--max-fan-out-depth 2+` — Allow recursive fan-out (sub-tasks can themselves fan out).

### JSON task file

For repeatable or complex task definitions, use a JSON file:

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

The JSON file and inline CLI options are mutually exclusive.

### Enabling test validation

By default, only ruff lint and format checks run. To also run tests:

```bash
forge run \
    --task-id fix-parser \
    --description "Fix the off-by-one error in the CSV parser" \
    --target-file src/myapp/parser.py \
    --run-tests \
    --test-command "pytest tests/test_parser.py -x"
```

`--test-command` implies `--run-tests`.

## Research Tasks

Research tasks produce markdown files instead of source code. Code linting and formatting checks are disabled. The LLM is prompted as a research assistant and outputs findings as markdown.

### Single-step research

```bash
forge run \
    --task-id analyze-deps \
    --description "Analyze the dependency graph and identify modules with high coupling" \
    --domain research \
    --target-file reports/dependency-analysis.md \
    --context-file src/myapp/api.py \
    --context-file src/myapp/models.py
```

### Planned research

For multi-part research with ordered deliverables:

```bash
forge run \
    --task-id security-audit \
    --description "Conduct a security review of the authentication and authorization modules. Identify vulnerabilities, assess severity, and recommend mitigations." \
    --domain research \
    --plan
```

The planner decomposes the research into steps, each producing markdown files. Fan-out works the same as with code tasks — independent research threads run in parallel and results are gathered.

### Research JSON task file

```json
{
    "task_id": "api-comparison",
    "description": "Compare REST vs GraphQL for our public API. Evaluate performance, developer experience, and migration effort.",
    "domain": "research",
    "target_files": [
        "reports/api-comparison.md"
    ],
    "context_files": [
        "src/myapp/api.py",
        "src/myapp/routes.py"
    ]
}
```

## Other Domains

Two additional domains are available, both producing markdown output with no code validation:

- `--domain code_review` — The LLM acts as a code review assistant, producing review comments and suggestions.
- `--domain documentation` — The LLM acts as a documentation assistant, producing guides and reference material.

```bash
forge run \
    --task-id review-api \
    --description "Review the API module for error handling gaps and naming consistency" \
    --domain code_review \
    --target-file reviews/api-review.md \
    --context-file src/myapp/api.py
```

## Domain Comparison

| Behavior | `code_generation` | `research` | `code_review` | `documentation` |
|----------|-------------------|------------|---------------|-----------------|
| LLM role | Code generation assistant | Research assistant | Code review assistant | Documentation assistant |
| Output format | `files` (new) + `edits` (search/replace) | `files` (markdown) | `files` (markdown) | `files` (markdown) |
| Ruff lint | On | Off | Off | Off |
| Ruff format | On | Off | Off | Off |
| Auto-fix | On | Off | Off | Off |
| Tests | Off (opt-in) | Off | Off | Off |

## Context and Exploration Options

These options apply to all domains:

| Option | Default | Description |
|--------|---------|-------------|
| `--no-auto-discover` | off | Disable automatic import graph analysis |
| `--include-deps` | off | Include dependency file contents in upfront context |
| `--token-budget` | `100000` | Token budget for assembled context |
| `--max-import-depth` | `2` | How deep to trace imports |
| `--max-exploration-rounds` | `10` | Max rounds of LLM-guided context exploration |
| `--no-explore` | off | Disable LLM-guided context exploration |

By default, Forge uses progressive disclosure: only target file contents and a repository map are assembled upfront. The LLM can request additional context on demand (file reads, code search, symbol lookups, import graphs, test files, git history, past run results, and playbooks). Use `--include-deps` to include dependency contents upfront instead.

## Retry Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--max-attempts` | `2` | Task-level retry limit |
| `--max-step-attempts` | `2` | Retry limit per step (planning mode) |
| `--max-sub-task-attempts` | `2` | Retry limit per sub-task (fan-out) |

When a step fails validation and retries, the retry prompt includes the validation error output with code context around error locations, so the LLM can fix the problem rather than retrying blind.

## Model Routing

Override which models handle each capability tier:

```bash
forge run \
    --task-id my-task \
    --description "..." \
    --target-file src/app.py \
    --reasoning-model "anthropic:claude-opus-4-6" \
    --generation-model "anthropic:claude-sonnet-4-5-20250929"
```

| Tier | Default Model | Used For |
|------|---------------|----------|
| `--reasoning-model` | Claude Opus 4.6 | Planning, conflict resolution |
| `--generation-model` | Claude Sonnet 4.5 | Code/content generation |
| `--summarization-model` | Claude Sonnet 4.5 | Knowledge extraction |
| `--classification-model` | Claude Haiku 4.5 | Context exploration |

## Extended Thinking

The planner uses extended thinking by default for higher-quality plans:

```bash
forge run --task-id my-task --description "..." --plan --thinking-budget 20000
forge run --task-id my-task --description "..." --plan --no-thinking
```

## Async Submission

Submit without waiting for the result:

```bash
forge run \
    --task-id background-task \
    --description "Refactor the validation pipeline" \
    --target-file src/myapp/validate.py \
    --no-wait
```

This prints the Temporal workflow ID and exits immediately. Check results with:

```bash
forge status --workflow-id forge-task-background-task
```

## Output Formats

- Default: Human-readable summary with status, validation results, and worktree path.
- `--json`: Full `TaskResult` as JSON for programmatic consumption.
- `--verbose`: Detailed output including LLM stats, token counts, latency, and interaction history.
