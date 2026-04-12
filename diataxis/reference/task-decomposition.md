# Task Decomposition Reference

Data models, execution mode selection logic, CLI flags, and git worktree lifecycle for task decomposition and execution.

For background on when to use each execution mode and how the planner, fan-out, and gather mechanisms work, see the [task decomposition explanation](../explanation/task-decomposition.md).

## Plan Data Models

### Plan

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task_id` | `str` | Yes | Identifier of the task this plan decomposes. |
| `steps` | `list[PlanStep]` | Yes | Ordered list of steps (minimum 1). |
| `explanation` | `str` | Yes | Brief explanation of the decomposition strategy. |

### PlanStep

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `step_id` | `str` | Yes | -- | Unique identifier within the plan. |
| `description` | `str` | Yes | -- | What this step should accomplish. |
| `target_files` | `list[str]` | Yes | -- | Files to create or modify in this step. |
| `context_files` | `list[str]` | No | `[]` | Additional files to include as context for this step. |
| `sub_tasks` | `list[SubTask] \| None` | No | `None` | Sub-tasks for fan-out parallel execution. When present, the step executes as a fan-out step. |
| `capability_tier` | `CapabilityTier \| None` | No | `None` | Capability tier override for this step's LLM call. When `None`, the default generation tier is used. |

### SubTask

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `sub_task_id` | `str` | Yes | -- | Unique identifier within the parent step. |
| `description` | `str` | Yes | -- | What this sub-task should produce. |
| `target_files` | `list[str]` | Yes | -- | Files to create or modify. |
| `context_files` | `list[str]` | No | `[]` | Files to include as context (read from parent worktree). |
| `sub_tasks` | `list[SubTask] \| None` | No | `None` | Nested sub-tasks for recursive fan-out. |

## Planner Prompt Structure

The planner receives a set of inputs and produces a `Plan` object. Extended thinking is applied to the planner call by default.

### Inputs to the planner

| Input | Source | Description |
|-------|--------|-------------|
| Task description | `TaskDefinition.description` | The full natural-language task to decompose. |
| Repository map | `code_intel/repo_map.py` | PageRank-ranked file tree with top-level signatures. |
| Exploration results | Accumulated `ContextResult` list | Context gathered during the pre-planning exploration loop, if any. |
| Extended thinking budget | `--thinking-budget` (default `10000`) | Token budget allocated for the planner's internal reasoning. |
| Domain instruction | `DomainConfig.role_prompt` | Role and output requirements for the active domain. |

### Output from the planner

| Output | Type | Description |
|--------|------|-------------|
| Plan | `Plan` | Ordered list of `PlanStep` objects, each with target files, optional context files, and optional sub-tasks for fan-out. |

## Execution Mode Selection

Mode selection is determined by the `plan` flag and the presence of `sub_tasks` on plan steps.

| Condition | Mode | Behavior |
|-----------|------|----------|
| `plan=False` | Single-step | One LLM call per attempt. Worktree created per attempt, destroyed on failure. |
| `plan=True`, no steps have `sub_tasks` | Planned multi-step | Planner decomposes task. Steps execute sequentially in a shared worktree. Each step commits on success. |
| `plan=True`, one or more steps have `sub_tasks` | Fan-out / gather | Steps without sub-tasks execute sequentially. Steps with sub-tasks dispatch child workflows in parallel, gather results, and merge. |

## Sanity Check Verdicts

| Verdict | Meaning | Effect |
|---------|---------|--------|
| `CONTINUE` | Remaining plan is valid. | Execution proceeds with the original remaining steps. |
| `REVISE` | Remaining plan needs changes. | The `revised_steps` replace the remaining steps. Completed steps are preserved. |
| `ABORT` | Task should stop. | Workflow returns `FAILURE_TERMINAL` with the sanity check explanation. |

## CLI Flags

### Execution mode flags

| Flag | Default | Description |
|------|---------|-------------|
| `--plan` | Off | Enable planning mode. The planner decomposes the task into ordered steps before execution. |
| `--max-fan-out-depth N` | `1` | Maximum recursive fan-out depth. `1` allows flat fan-out only (sub-tasks cannot nest further). `2+` allows recursive fan-out. |
| `--sanity-check-interval N` | `0` (disabled) | Run a sanity check every N completed steps. `0` disables sanity checks. |
| `--resolve-conflicts` | On | Attempt LLM-based conflict resolution when fan-out sub-tasks produce different content for the same file. Disable with `--no-resolve-conflicts`. |

### Retry flags

| Flag | Default | Description |
|------|---------|-------------|
| `--max-attempts N` | `2` | Task-level retry limit (single-step mode). |
| `--max-step-attempts N` | `2` | Retry limit per step (planned mode). On failure, uncommitted changes are reset and the step retries with error context. |
| `--max-sub-task-attempts N` | `2` | Retry limit per sub-task (fan-out). Each sub-task retries independently. |

### Task definition flags

| Flag | Description |
|------|-------------|
| `--task-id ID` | Unique task identifier. Used in worktree paths and branch names. |
| `--description TEXT` | What the task should produce. |
| `--target-file PATH` | File to create or modify. Repeatable. Optional in planned mode (the planner determines target files). |
| `--context-file PATH` | Additional file to include as context. Repeatable. |
| `--task-file PATH` | JSON file containing the full task definition. Mutually exclusive with inline flags. |
| `--domain DOMAIN` | Task domain: `code_generation` (default), `research`, `code_review`, `documentation`, `generic`. |

### Extended thinking flags

| Flag | Default | Description |
|------|---------|-------------|
| `--thinking-budget N` | `10000` | Token budget for extended thinking in the planner. |
| `--no-thinking` | Off | Disable extended thinking for the planner. |

## Git Worktree Lifecycle

### Path and branch naming

| Item | Pattern | Example |
|------|---------|---------|
| Worktree directory | `<repo_root>/.forge-worktrees/<task_id>` | `/code/.forge-worktrees/add-auth` |
| Branch name | `forge/<task_id>` | `forge/add-auth` |
| Sub-task worktree | `<repo_root>/.forge-worktrees/<task_id>.sub.<sub_task_id>` | `/code/.forge-worktrees/add-auth.sub.models` |
| Sub-task branch | `forge/<task_id>.sub.<sub_task_id>` | `forge/add-auth.sub.models` |

Task IDs must match `^[A-Za-z0-9][A-Za-z0-9._-]*$`.

### Lifecycle by execution mode

**Single-step:**

1. Create worktree branched from `base_branch` (default `main`).
2. Execute the universal workflow step.
3. On success: commit changes, worktree remains for human review.
4. On retryable failure: remove worktree, create a fresh one, retry with error context.
5. On terminal failure: remove worktree, return failure result.

**Planned multi-step:**

1. Create worktree branched from `base_branch` (once, at the start).
2. For each step: execute, validate, commit on success.
3. On step failure: reset uncommitted changes in the worktree (`git checkout -- .`). Prior committed steps are preserved.
4. On terminal failure of any step: worktree remains with committed progress for inspection.

**Fan-out sub-tasks:**

1. Parent worktree exists from the planned execution.
2. Each child creates a worktree branched from the parent's branch (`forge/<task_id>`).
3. Children execute and produce output files (no commits).
4. Parent gathers output, children's worktrees are removed.
5. Merged output is written to the parent worktree, validated, and committed.

### Cleanup

Worktrees are removed automatically on:

- Sub-task completion (success or failure)
- Single-step retryable failure (before recreating)

Worktrees are preserved on:

- Successful task completion (for human review and merge)
- Terminal failure in planned mode (committed steps remain for inspection)

## JSON Task File Format

The `--task-file` flag accepts a JSON file with the following structure:

```json
{
    "task_id": "string",
    "description": "string",
    "domain": "code_generation",
    "target_files": ["path/to/file.py"],
    "context_files": ["path/to/context.py"],
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

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `task_id` | `string` | Yes | -- | Unique task identifier. |
| `description` | `string` | Yes | -- | What the task should produce. |
| `domain` | `string` | No | `"code_generation"` | Task domain. |
| `target_files` | `string[]` | No | `[]` | Files to create or modify. |
| `context_files` | `string[]` | No | `[]` | Additional context files. |
| `base_branch` | `string` | No | `"main"` | Branch to create the worktree from. |
| `validation.auto_fix` | `bool` | No | `true` | Auto-fix lint issues before validation. |
| `validation.run_ruff_lint` | `bool` | No | `true` | Run ruff lint check. |
| `validation.run_ruff_format` | `bool` | No | `true` | Run ruff format check. |
| `validation.run_tests` | `bool` | No | `false` | Run test suite. |
| `context.auto_discover` | `bool` | No | `true` | Enable automatic import graph analysis. |
| `context.include_dependencies` | `bool` | No | `false` | Include dependency file contents upfront. |
| `context.token_budget` | `int` | No | `100000` | Token budget for assembled context. |
| `context.max_import_depth` | `int` | No | `2` | Depth limit for import graph traversal. |

## See Also

- [Task Decomposition (explanation)](../explanation/task-decomposition.md)
- [How to Submit Tasks](../howto/submit-tasks.md)
