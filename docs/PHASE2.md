# Phase 2: Planning and Multi-Step Execution

## Goal

Add a planning step that decomposes a task into ordered sub-steps, then executes them sequentially with a commit after each. Still single-model, no fan-out.

The deliverable: describe a larger task, Forge plans the steps, executes them in order, producing a reviewable commit history showing incremental progress.

## Scope

**In scope:**

- Planning activity: LLM decomposes a task into ordered `PlanStep` items.
- Sequential step execution within a single worktree.
- Step-level commit: each successful step produces a git commit.
- Step-level retry: on validation failure, reset the worktree to HEAD and re-run the step.
- Step-level context: later steps can read files created by earlier steps via `context_files`.
- CLI `--plan` flag and `--max-step-attempts` option.
- Backward compatible: `plan=False` (default) runs the existing Phase 1 single-step path.

**Out of scope (deferred to later phases):**

- Task-level retry (re-plan from scratch on failure).
- Fan-out / parallel step execution.
- Model routing for planning vs execution.
- Dynamic re-planning based on step results.

## Architecture

### Workflow Dispatch

`ForgeTaskWorkflow.run()` dispatches based on `input.plan`:

- `plan=False`: Phase 1 single-step execution (unchanged).
- `plan=True`: Planning + multi-step execution.

### Planned Execution Path

```
create_worktree (once)
assemble_planner_context → call_planner → Plan
for step in plan.steps:
    for attempt in 1..max_step_attempts:
        assemble_step_context → call_llm → write_output → validate → transition
        SUCCESS           → commit(step message), break to next step
        FAILURE_RETRYABLE → reset_worktree, continue retry
        FAILURE_TERMINAL  → return TaskResult(FAILURE_TERMINAL)
All steps done → return TaskResult(SUCCESS)
```

### Data Models

New models added to `models.py`:

```
PlanStep:
    step_id: str
    description: str
    target_files: list[str]
    context_files: list[str] = []

Plan:
    task_id: str
    steps: list[PlanStep]    # min_length=1
    explanation: str

StepResult:
    step_id: str
    status: TransitionSignal
    output_files: dict[str, str] = {}
    validation_results: list[ValidationResult] = []
    commit_sha: str | None = None
    error: str | None = None

PlannerInput:
    task_id: str
    system_prompt: str
    user_prompt: str

PlanCallResult:
    task_id: str
    plan: Plan
    model_name: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

AssembleStepContextInput:
    task: TaskDefinition
    step: PlanStep
    step_index: int
    total_steps: int
    completed_steps: list[StepResult] = []
    repo_root: str
    worktree_path: str

ResetWorktreeInput:
    repo_root: str
    task_id: str
```

Modified models:

- `ForgeTaskInput`: added `plan: bool = False`, `max_step_attempts: int = 2`
- `TaskResult`: added `step_results: list[StepResult] = []`, `plan: Plan | None = None`
- `TaskDefinition`: `target_files` now defaults to `[]` (optional when planning)
- `CommitChangesInput`: added `message: str | None = None` for custom commit messages

### Activities

New activities:

- `assemble_planner_context(AssembleContextInput) → PlannerInput` — reuses the Phase 1 `AssembleContextInput` model (task, repo_root, worktree_path); reads context files from repo root, builds planner prompts
- `call_planner(PlannerInput) → PlanCallResult` — calls LLM with structured `Plan` output
- `assemble_step_context(AssembleStepContextInput) → AssembledContext` — reads context files from **worktree** (not repo root), builds step-specific prompts
- `reset_worktree_activity(ResetWorktreeInput) → None` — `git reset --hard HEAD` + `git clean -fd`

### Git Layer

- `reset_worktree(repo_root, task_id)` — discards uncommitted changes in the worktree, preserving committed work.
- `commit_changes` now accepts an optional `message` parameter to override the auto-generated commit message. Step commits use `forge({task_id}): step {step_id} success`.

## Project Structure

New and modified files:

```
src/forge/
├── models.py                  # Modified: 7 new models, 4 modified
├── git.py                     # Modified: reset_worktree, commit message override
├── workflows.py               # Modified: planned execution path
├── worker.py                  # Modified: register new activities
├── cli.py                     # Modified: --plan flag, step-level output
└── activities/
    ├── __init__.py             # Modified: re-exports new activities
    ├── planner.py              # New: planning activity
    ├── context.py              # Modified: step-level context assembly
    └── git_activities.py       # Modified: reset_worktree_activity
```

## Key Design Decisions

- **D23:** Step-level retry over task-level retry (see `docs/DECISIONS.md`).
- **D24:** Planning as a separate activity from step execution.
- **D25:** Single worktree for the entire plan.

## CLI Usage

```bash
# Plan mode: Forge decomposes the task into steps
forge run \
    --task-id my-feature \
    --description "Build a user model and CRUD API" \
    --plan \
    --max-step-attempts 2

# Without --plan: Phase 1 single-step mode (unchanged)
forge run \
    --task-id my-task \
    --description "Write a greeting module" \
    --target-file hello.py
```

When `--plan` is set, `--target-file` is optional — the planner determines which files each step produces.

## Definition of Done

Phase 2 is complete when:

- `forge run --plan` decomposes a task into steps, executes each step, and produces a worktree with incremental commits.
- Each commit in the worktree corresponds to one plan step.
- Step-level retry works: a failing step resets uncommitted changes and retries without losing prior step commits.
- Terminal step failure preserves all prior committed work.
- All Phase 1 tests continue to pass (backward compatible).
- End-to-end tests cover: happy path (multi-step success), step failure, and step retry.
