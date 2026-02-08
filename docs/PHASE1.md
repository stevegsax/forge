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

```
TaskDefinition:
    task_id: str
    description: str                # What to produce
    target_files: list[str]         # Files to create or modify
    context_files: list[str]        # Files to include as context
    validation: ValidationConfig    # What checks to run
    base_branch: str = "main"       # Branch to create worktree from

ValidationConfig:
    auto_fix: bool = True           # Run ruff --fix and ruff format before validation
    run_ruff_lint: bool = True
    run_ruff_format: bool = True
    run_tests: bool = False
    test_command: str | None = None

ValidationResult:
    check_name: str
    passed: bool
    summary: str                    # Concise summary of the result
    details: str | None = None      # Extended details, not sent to LLM by default

TaskResult:
    task_id: str
    status: TransitionSignal
    output_files: dict[str, str]    # path -> content
    validation_results: list[ValidationResult]
    error: str | None = None        # If failed, why
    worktree_path: str | None = None  # Where to review the result
    worktree_branch: str | None = None  # Branch name

TransitionSignal: Literal[
    "success",
    "failure_retryable",
    "failure_terminal",
]
```

### Git Worktree Lifecycle

1. Before the workflow starts, create a worktree: `git worktree add <path> -b forge/<task_id> <base_branch>`
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

Export to a local collector (Jaeger or stdout for initial development).

## Project Structure

```
forge/
├── CLAUDE.md
├── pyproject.toml
├── tool-config/
│   └── ruff.toml              # Shared ruff config for worktrees
├── docs/
│   ├── DESIGN.md
│   ├── DECISIONS.md
│   └── PHASE1.md
└── src/
    └── forge/
        ├── __init__.py
        ├── cli.py              # CLI entry point
        ├── models.py           # Pydantic models (TaskDefinition, TaskResult, etc.)
        ├── workflows.py        # Temporal workflow definitions
        ├── activities/
        │   ├── __init__.py
        │   ├── context.py      # assemble_context activity
        │   ├── git_activities.py  # Git worktree Temporal activities
        │   ├── llm.py          # call_llm activity
        │   ├── output.py       # write_output activity
        │   ├── validate.py     # validate_output activity
        │   └── transition.py   # evaluate_transition activity
        ├── git.py              # Worktree management (pure functions)
        ├── tracing.py          # OpenTelemetry setup
        └── worker.py           # Temporal worker entry point
```

## Dependencies

Core:

- `temporalio` — Workflow orchestration.
- `pydantic` — Data models and validation.
- `pydantic-ai` — LLM client.
- `opentelemetry-api`, `opentelemetry-sdk` — Tracing.
- `opentelemetry-exporter-otlp` — Trace export (or jaeger exporter for local dev).
- `click` — CLI.

Development:

- `ruff` — Linting and formatting (both as a dev tool and as a validation dependency).
- `pytest` — Testing Forge itself.

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
