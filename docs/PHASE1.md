# Phase 1: The Minimal Loop

## Goal

Prove out the core primitive: the universal workflow step, Temporal activity boundaries, git worktree lifecycle, and OpenTelemetry tracing. The deliverable is a working system that accepts a small task description, executes a single LLM call with assembled context, validates the output, and presents the result in a git worktree for human review.

## Scope

**In scope:**

- Single workflow step: construct, send, receive, serialize, transition.
- One LLM provider: Anthropic (via pydantic-ai).
- One task domain: Python code generation (initial use case).
- Hardcoded context assembly (manually specified files/context, no tree-sitter or LSP yet).
- Deterministic validation: ruff lint, ruff format check, optional test execution.
- Git worktree creation and cleanup.
- OpenTelemetry tracing across the full workflow.
- Pydantic models for task definition, LLM response, and transition signals.
- CLI entry point for submitting a task.

**Out of scope (deferred to later phases):**

- Planning / task decomposition.
- Fan-out / gather / sub-agents.
- Model routing / multiple providers.
- Tree-sitter / LSP context assembly.
- Knowledge extraction / playbooks.
- Conflict resolution.
- Multi-step execution within a single task.

## Architecture

### Workflow

A single Temporal workflow (`ForgeTaskWorkflow`) that executes the following activities in sequence:

1. `assemble_context` — Given a task definition, assemble the prompt and context. In Phase 1, this reads a task description and any explicitly specified files.
2. `call_llm` — Send the assembled message to the LLM via pydantic-ai. Receive and parse the response using a Pydantic output model.
3. `write_output` — Write the LLM's output to the task's git worktree.
4. `validate_output` — Run deterministic validation (ruff lint, ruff format) on the written files. Optionally run tests if specified.
5. `evaluate_transition` — Examine validation results and determine the next state. In Phase 1, this is deterministic: validation passes → `success`, validation fails → `failure_retryable` (up to retry limit) or `failure_terminal`.

### Data Models

#### Task and validation

```
TaskDefinition:
    task_id: str
    description: str                    # What to produce
    target_files: list[str] = []        # Files to create or modify (optional when planning)
    context_files: list[str] = []       # Files to include as context
    validation: ValidationConfig        # What checks to run
    base_branch: str = "main"           # Branch to create worktree from

ValidationConfig:
    auto_fix: bool = True               # Run ruff --fix and ruff format before validation
    run_ruff_lint: bool = True
    run_ruff_format: bool = True
    run_tests: bool = False
    test_command: str | None = None

ValidationResult:
    check_name: str
    passed: bool
    summary: str                        # Concise summary of the result
    details: str | None = None          # Extended details, not sent to LLM by default

TransitionSignal: StrEnum[
    "success",
    "failure_retryable",
    "failure_terminal",
]

TaskResult:
    task_id: str
    status: TransitionSignal
    output_files: dict[str, str]        # path -> content
    validation_results: list[ValidationResult]
    error: str | None = None            # If failed, why
    worktree_path: str | None = None    # Where to review the result
    worktree_branch: str | None = None  # Branch name
    step_results: list[StepResult] = [] # Phase 2: per-step outcomes
    plan: Plan | None = None            # Phase 2: the executed plan
```

#### LLM structured output

```
FileOutput:
    file_path: str                      # Relative path within the worktree
    content: str                        # Complete file content

LLMResponse:
    files: list[FileOutput]             # Files to create or modify
    explanation: str                    # Brief explanation of what was produced
```

#### Inter-activity transport

```
AssembledContext:
    task_id: str
    system_prompt: str
    user_prompt: str

LLMCallResult:
    task_id: str
    response: LLMResponse
    model_name: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

WriteResult:
    task_id: str
    files_written: list[str]
```

#### Activity input models

Temporal activities take a single argument. These models wrap the inputs.

```
ForgeTaskInput:                         # Workflow entry point
    task: TaskDefinition
    repo_root: str
    max_attempts: int = 2
    plan: bool = False                  # Phase 2: enable planning mode
    max_step_attempts: int = 2          # Phase 2: retry limit per step

AssembleContextInput:
    task: TaskDefinition
    repo_root: str
    worktree_path: str

WriteOutputInput:
    llm_result: LLMCallResult
    worktree_path: str

ValidateOutputInput:
    task_id: str
    worktree_path: str
    files: list[str]
    validation: ValidationConfig

TransitionInput:
    validation_results: list[ValidationResult]
    attempt: int
    max_attempts: int = 2
```

#### Git activity I/O models

```
CreateWorktreeInput:
    repo_root: str
    task_id: str
    base_branch: str = "main"

CreateWorktreeOutput:
    worktree_path: str
    branch_name: str

RemoveWorktreeInput:
    repo_root: str
    task_id: str
    force: bool = True

CommitChangesInput:
    repo_root: str
    task_id: str
    status: str
    file_paths: list[str] | None = None
    message: str | None = None          # Override auto-generated commit message

CommitChangesOutput:
    commit_sha: str

ResetWorktreeInput:                     # Phase 2
    repo_root: str
    task_id: str
```

### Git Worktree Lifecycle

1. Before the workflow starts, create a worktree: `git worktree add -b forge/<task_id> <repo_root>/.forge-worktrees/<task_id> <base_branch>`
2. The `write_output` activity writes files into this worktree.
3. The `validate_output` activity runs checks within the worktree.
4. On `success`, the worktree is left in place with a commit for human review.
5. On `failure_retryable`, the worktree is removed and recreated for the retry attempt.
6. On `failure_terminal`, the worktree is left in place with the failure state committed for debugging.
7. Cleanup of merged/abandoned worktrees is manual in Phase 1.

### Tracing

OpenTelemetry spans for:

- The full workflow execution (root span).
- Each activity (child spans): context assembly, LLM call, output writing, validation, transition evaluation.
- The LLM call span should include attributes: model name, token counts (input/output), latency.
- The validation span should include attributes: which checks ran, pass/fail per check.

Exporter types: `CONSOLE` (stdout, default), `OTLP_GRPC`, `OTLP_HTTP`, `NONE`. Configured via `FORGE_OTEL_EXPORTER` environment variable or programmatically.

## Project Structure

```
forge/
├── CLAUDE.md
├── pyproject.toml
├── tool-config/
│   └── ruff.toml              # Shared ruff config for worktrees (see D22)
├── docs/
│   ├── DESIGN.md
│   ├── DECISIONS.md
│   ├── PHASE1.md
│   ├── PHASE2.md
│   └── PHASE3.md
└── src/
    └── forge/
        ├── __init__.py
        ├── cli.py              # CLI entry point (forge run, forge worker)
        ├── models.py           # Pydantic models for all phases
        ├── workflows.py        # Temporal workflows (Phase 1 / Phase 2 / Phase 3 fan-out)
        ├── activities/
        │   ├── __init__.py     # Re-exports all activity functions
        │   ├── context.py      # assemble_context, assemble_planner_context, assemble_step_context, assemble_sub_task_context
        │   ├── git_activities.py  # Git worktree Temporal activities
        │   ├── llm.py          # call_llm activity
        │   ├── planner.py      # call_planner activity (Phase 2)
        │   ├── output.py       # write_output, write_files activities
        │   ├── validate.py     # validate_output activity
        │   └── transition.py   # evaluate_transition activity
        ├── git.py              # Worktree management (pure functions + subprocess shell)
        ├── tracing.py          # OpenTelemetry setup
        └── worker.py           # Temporal worker entry point
```

## Dependencies

Core:

- `temporalio` — Workflow orchestration.
- `pydantic` — Data models and validation.
- `pydantic-ai` — LLM client.
- `opentelemetry-api`, `opentelemetry-sdk` — Tracing.
- `opentelemetry-exporter-otlp` — Trace export (OTLP gRPC and HTTP).
- `click` — CLI.

Development:

- `ruff` — Linting and formatting (both as a dev tool and as a validation dependency).
- `pytest` — Testing Forge itself.

## CLI

Entry point: `forge` (installed via `pyproject.toml` script `forge = forge.cli:main`).

### `forge run`

Submit a task to Temporal and wait for the result.

```bash
# Inline task definition
forge run \
    --task-id my-task \
    --description "Write a greeting module" \
    --target-file hello.py \
    --context-file existing_module.py

# Task definition from JSON file
forge run --task-file task.json
```

Options:

- `--task-id` — Unique task identifier (required for inline).
- `--description` — What the task should produce (required for inline).
- `--target-file` — File to create or modify, repeatable (required for inline unless `--plan`).
- `--context-file` — File to include as context, repeatable.
- `--task-file` — JSON file with a `TaskDefinition`. Mutually exclusive with inline options.
- `--base-branch` — Branch to create worktree from (default: `main`).
- `--max-attempts` — Task-level retry limit (default: `2`).
- `--plan` — Enable planning mode (Phase 2).
- `--max-step-attempts` — Retry limit per step in planning mode (default: `2`).
- `--no-lint` — Disable ruff lint check.
- `--no-format` — Disable ruff format check.
- `--run-tests` — Enable test validation.
- `--test-command` — Custom test command.
- `--json` — Output `TaskResult` as JSON.
- `--no-wait` — Submit and print workflow ID without waiting.
- `--temporal-address` — Temporal server address (default: `localhost:7233`, env: `FORGE_TEMPORAL_ADDRESS`).

Exit codes: `0` success, `1` task failure (terminal), `3` infrastructure error.

### `forge worker`

Start the Temporal worker process.

```bash
forge worker --temporal-address localhost:7233
```

## Implementation Order

1. Project skeleton: `pyproject.toml`, package structure, dependency installation.
2. Pydantic models (`models.py`).
3. Git worktree management (`git.py`).
4. OpenTelemetry setup (`tracing.py`).
5. Individual activities, tested independently.
6. Temporal workflow wiring.
7. Temporal worker.
8. CLI entry point.
9. End-to-end test: submit a task, verify worktree output.

## Definition of Done

Phase 1 is complete when:

- You can run a CLI command that describes a Python coding task.
- Forge creates a git worktree, calls an LLM, writes the output, validates it, and commits.
- The worktree is ready for human review.
- OpenTelemetry traces are visible for the full workflow.
- On validation failure, the system retries (up to a limit) or reports terminal failure.
- The retry creates a fresh worktree and starts clean.
