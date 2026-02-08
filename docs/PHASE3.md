# Phase 3: Fan-Out / Gather

## Goal

Add parallel sub-task execution within planned steps via Temporal child workflows. A plan step can declare independent sub-tasks that execute simultaneously, each in its own git worktree. Results are gathered, merged into the parent worktree, validated, and committed.

The deliverable: describe a task where a plan step has independent sub-tasks, Forge fans out child workflows, gathers their outputs, merges the files into the parent worktree, and commits the result.

## Scope

**In scope:**

- Fan-out: plan steps can declare `sub_tasks` for parallel execution.
- Gather: parent workflow collects sub-task outputs and merges files.
- Sub-task isolation: each sub-task runs in its own git worktree.
- File conflict detection: two sub-tasks writing the same file is a terminal error.
- Sub-task retry: individual sub-tasks retry independently (up to a configurable limit).
- Planner awareness: the planner prompt explains fan-out sub-tasks.
- CLI `--max-sub-task-attempts` option.
- Backward compatible: plans without `sub_tasks` run as Phase 2 sequential steps.

**Out of scope (deferred to later phases):**

- Recursive fan-out (sub-tasks cannot fan out further).
- LLM-based conflict resolution.
- Model routing (all sub-tasks use the same model).
- Dynamic re-planning based on sub-task results.

## Architecture

### Fan-Out Detection

`ForgeTaskWorkflow._run_planned()` inspects each `PlanStep`. If `step.sub_tasks` is populated, the step delegates to `_run_fan_out_step()` instead of the inline retry loop.

### Fan-Out Execution Flow

```
Parent workflow encounters step with sub_tasks:
  1. Validate sub-task ID uniqueness
  2. Start child workflows in parallel (one per sub-task)
     Each child (ForgeSubTaskWorkflow):
       - Creates worktree (compound ID, branched from parent branch)
       - assemble_sub_task_context -> call_llm -> write_output -> validate -> transition
       - SUCCESS: collect output_files from LLM result, remove worktree, return SubTaskResult
       - FAILURE_RETRYABLE: remove worktree, recreate on next attempt, retry
       - FAILURE_TERMINAL: remove worktree, return failure SubTaskResult
  3. Await all children
  4. If any failed -> step fails terminal
  5. Check file conflicts (same path from different children -> error)
  6. Write merged files to parent worktree via write_files activity
  7. Validate merged output
  8. Commit with message: forge({task_id}): step {step_id} fan-out gather
  9. Return StepResult with sub_task_results populated
```

### Compound Task IDs

Sub-tasks use IDs like `my-task.sub.analyze-schema`. Dots are valid in the task ID regex (`^[A-Za-z0-9][A-Za-z0-9._-]*$`), so all existing git functions work unchanged: `worktree_path`, `branch_name`, `create_worktree`, `remove_worktree`. Sub-task worktrees branch from the parent's branch (`forge/<task_id>`), not from `main`, so they see prior committed step outputs.

### Sub-Agents Don't Commit (D16)

Sub-task `output_files` are collected in-memory from `llm_result.response.files` before the worktree is removed. The parent writes the gathered files to its own worktree and commits. Sub-task worktrees are purely execution context for validation.

### Data Models

New models added to `models.py`:

```
SubTask:
    sub_task_id: str
    description: str
    target_files: list[str]
    context_files: list[str] = []

SubTaskResult:
    sub_task_id: str
    status: TransitionSignal
    output_files: dict[str, str] = {}
    validation_results: list[ValidationResult] = []
    digest: str = ""                    # From LLMResponse.explanation (D18)
    error: str | None = None

SubTaskInput:                           # ForgeSubTaskWorkflow input
    parent_task_id: str
    parent_description: str             # Parent task description for context assembly
    sub_task: SubTask
    repo_root: str
    parent_branch: str                  # e.g. "forge/my-task"
    validation: ValidationConfig
    max_attempts: int = 2

WriteFilesInput:
    task_id: str
    worktree_path: str
    files: dict[str, str]               # path -> content

AssembleSubTaskContextInput:
    parent_task_id: str
    parent_description: str
    sub_task: SubTask
    worktree_path: str                  # Parent worktree (for reading context files)
```

Modified models:

- `PlanStep`: added `sub_tasks: list[SubTask] | None = None`
- `StepResult`: added `sub_task_results: list[SubTaskResult] = Field(default_factory=list)`
- `ForgeTaskInput`: added `max_sub_task_attempts: int = Field(default=2)`

### Workflows

**ForgeSubTaskWorkflow** — child workflow for a single sub-task:

1. Create worktree with compound ID, branched from parent branch.
2. Retry loop (1..max_attempts):

    - `assemble_sub_task_context` -> `call_llm` -> `write_output` -> `validate_output` -> `evaluate_transition`
    - SUCCESS: collect output files, remove worktree, return SubTaskResult.
    - FAILURE_RETRYABLE: remove worktree, continue (recreated on next iteration).
    - FAILURE_TERMINAL: remove worktree, return failure SubTaskResult.

3. If all attempts exhausted, return failure SubTaskResult.

Child workflow timeout: 15 minutes.

**ForgeTaskWorkflow._run_fan_out_step** — fan-out coordination:

1. Validate sub-task IDs are unique within the step.
2. Start all child workflows in parallel via `workflow.start_child_workflow`.
3. Await all child results.
4. If any child failed, return StepResult(FAILURE_TERMINAL).
5. Check for file conflicts (same file path from different sub-tasks).
6. Merge all output files into a single dict.
7. Write merged files to parent worktree via `write_files` activity.
8. Run `validate_output` on merged output.
9. Evaluate transition; on success, commit.
10. Return StepResult with sub_task_results.

### Activities

New activities:

- `write_files(WriteFilesInput) -> WriteResult` — writes a `dict[str, str]` to a worktree with path traversal protection.
- `assemble_sub_task_context(AssembleSubTaskContextInput) -> AssembledContext` — reads context files from the parent worktree, builds sub-task-specific prompts.

New pure functions:

- `build_sub_task_system_prompt(parent_task_id, parent_description, sub_task, context_file_contents)` — includes parent task context, sub-task description, target files, context files.
- `build_sub_task_user_prompt(sub_task)` — instruction to produce target files for the sub-task.

### Planner Prompt

The planner system prompt now explains fan-out sub-tasks:

- Steps can optionally include `sub_tasks` for independent parallel work.
- When `sub_tasks` is set, the step's own `target_files` is ignored.
- Sub-tasks cannot see each other's outputs (they run simultaneously).
- Two sub-tasks must not write to the same file.
- Use fan-out only when work items are genuinely independent.

## Project Structure

New and modified files:

```
src/forge/
├── models.py                  # Modified: 5 new models, 3 modified
├── workflows.py               # Modified: ForgeSubTaskWorkflow, _run_fan_out_step
├── worker.py                  # Modified: register new workflow + activities
├── cli.py                     # Modified: --max-sub-task-attempts, format_sub_task_result
└── activities/
    ├── __init__.py             # Modified: re-export new activities
    ├── context.py              # Modified: sub-task context assembly
    ├── output.py               # Modified: write_files activity
    └── planner.py              # Modified: fan-out instructions in planner prompt
```

## Key Design Decisions

- **D26:** Compound task IDs for sub-task isolation (see `docs/DECISIONS.md`).
- **D27:** File conflict as terminal error.
- **D28:** Sub-task context reads from parent worktree.
- **D29:** No recursive fan-out in Phase 3.

## CLI Usage

```bash
# Plan mode with fan-out support
forge run \
    --task-id my-feature \
    --description "Build a user model, API routes, and tests" \
    --plan \
    --max-sub-task-attempts 2

# Without --plan: Phase 1 single-step mode (unchanged)
forge run \
    --task-id my-task \
    --description "Write a greeting module" \
    --target-file hello.py
```

The `--max-sub-task-attempts` option controls the retry limit for individual sub-tasks within fan-out steps (default: 2).

## Edge Cases

- **Empty `sub_tasks` list**: treated same as `None` (no fan-out).
- **Single sub-task**: valid but wasteful; handled correctly (no conflict possible).
- **Sub-task produces no files**: valid; empty `output_files` merged harmlessly.
- **Duplicate sub-task IDs**: detected before starting children; step fails terminal.
- **Orphaned worktrees**: if parent fails mid-fan-out, some child worktrees may remain; acceptable for Phase 3 (worktrees are disposable).

## Definition of Done

Phase 3 is complete when:

- Plan steps with `sub_tasks` fan out to parallel child workflows.
- Each sub-task runs in its own git worktree branched from the parent branch.
- Sub-task outputs are gathered, merged, and committed to the parent worktree.
- File conflicts between sub-tasks are detected and fail the step.
- Sub-task retry works: a failing sub-task destroys its worktree and retries independently.
- Plans without `sub_tasks` run as Phase 2 sequential steps (backward compatible).
- All Phase 1 and Phase 2 tests continue to pass.
- End-to-end tests cover: fan-out success, child failure, and file conflict.
