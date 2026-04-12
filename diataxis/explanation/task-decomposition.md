# Task Decomposition and Execution

Forge supports three execution modes for processing tasks, each built on the same [universal workflow step](workflow-step.md). The choice of mode determines how much decomposition occurs before execution begins and whether work runs sequentially or in parallel. This document explains how tasks are decomposed, how parallel sub-tasks are gathered and reconciled, and how git worktrees provide isolation throughout.

## The Three Execution Modes

### Single-step

Single-step is the default mode. The task runs as a single LLM call: assemble context, call the model, write output, validate, commit. One worktree is created per attempt. On a retryable failure, the worktree is destroyed and a fresh one is created for the next attempt, with the previous error injected into the prompt.

Single-step is appropriate when the task is focused -- a change to one or two files where the LLM can produce the correct output in a single pass. There is no planning overhead, so turnaround is fast.

### Planned multi-step

When a task spans multiple files or requires ordered changes where later work depends on earlier results, planned mode decomposes the task into a sequence of steps before any execution begins. A planning LLM receives the full task description, target files, repository structure, and context, then produces an ordered list of steps. Each step specifies its own target files, description, and optional context.

Execution proceeds sequentially within a single shared worktree. Each step runs the universal workflow step and commits on success. If a step fails validation, uncommitted changes are reset (preserving commits from prior steps) and the step retries with the error context included. Later steps see the files created or modified by earlier steps because they share the same worktree.

The key property of planned mode is that the planner makes all decomposition decisions upfront. This is a deliberate design choice -- it follows the plan-then-execute model where the full scope of work is known before execution starts. Steps do not discover new work or re-plan during execution. If the plan becomes stale as steps complete, a separate sanity check mechanism (described below) can revise the remaining steps.

### Fan-out / gather

Fan-out is an extension of planned mode. When the planner identifies work within a step that can proceed independently -- for example, writing unit tests for three unrelated modules -- it declares sub-tasks on that step. Each sub-task runs as a Temporal child workflow in its own git worktree, executing in parallel. After all children complete, the parent gathers and merges their output.

Fan-out is appropriate when a step contains genuinely independent units of work that do not share context expensive to reconstruct. If sub-tasks would need to read each other's output, they belong in sequential steps instead.

## Why Planning Gets Premium Resources

Planning is the most consequential LLM call in the entire pipeline. A plan that misidentifies file boundaries, orders steps incorrectly, or fails to anticipate conflicts produces cascading failures downstream -- each step retries, worktrees are recreated, and token budget is spent on work that was doomed from the start.

For this reason, the planner always uses the reasoning capability tier (the most capable model available). It also receives extended thinking tokens so it can reason through the decomposition before committing to a plan. The rationale is that a few thousand extra tokens spent on planning save orders of magnitude more tokens on failed execution attempts.

The planner receives a rich context: the full task description, repository structure ranked by structural importance, target file contents, project conventions, and any relevant playbooks from prior tasks. The exploration loop runs before the planning call as well, so the planner can request additional context (file reads, code search, import graphs) before deciding how to decompose the task.

### What the planner produces

The planner outputs a `Plan` containing an ordered list of `PlanStep` objects. Each step specifies:

- An identifier and description of what the step should accomplish
- Target files to create or modify
- Context files to include (beyond what auto-discovery provides)
- Optional sub-tasks for fan-out parallel execution
- Optional capability tier override (so a particularly complex step can use the reasoning tier instead of the default generation tier)

The plan also includes an explanation of the decomposition strategy, which is recorded in the observability store for later inspection.

For the precise field definitions of these models, see the [task decomposition reference](../reference/task-decomposition.md).

## Fan-Out Mechanics

### Dispatch

When the parent workflow encounters a plan step with sub-tasks, it starts one Temporal child workflow per sub-task. Each child gets its own git worktree, branched from the parent's branch (not from `main`), so it sees the results of prior committed steps. All children start concurrently.

### Execution

Each child runs the universal workflow step independently -- assemble context, call LLM, write output, validate, transition -- with its own retry budget. Children do not commit to git. They produce output files that are returned to the parent.

### Gather

The parent awaits all children. If any child fails terminally, the entire step fails. Otherwise, the parent collects output files from all successful children.

### Conflict detection and resolution

When multiple sub-tasks produce content for the same file path, a conflict exists. If conflict resolution is enabled (the default), a reasoning-tier LLM receives all conflicting versions alongside the task description, step description, and sub-task descriptions. It produces a merged version of each conflicting file. This conflict resolution step is itself an instance of the universal workflow step -- same pattern, specialized prompt.

Non-conflicting files are collected directly without LLM involvement.

### Merge and validate

All files -- non-conflicting originals plus resolved versions -- are written to the parent worktree. Validation runs on the merged output. If validation passes, the step is committed.

### Nested fan-out

Sub-tasks can themselves contain sub-tasks, creating recursive fan-out. The recursion depth is bounded by a configurable maximum (default 1, meaning flat fan-out only). Each child workflow checks its current depth against the maximum and either executes as a leaf (single-step) or recurses (nested fan-out).

Nested fan-out is useful when a sub-task is itself large enough to benefit from parallel decomposition, but each additional level adds coordination overhead and increases the likelihood of file conflicts. In practice, flat fan-out covers most use cases.

## Sanity Checks

In planned mode, the planner commits to a full decomposition before any execution begins. As steps complete, the assumptions behind the original plan may become stale -- an early step might have restructured files in a way that invalidates later steps, or validation failures might reveal that the approach needs adjustment.

Sanity checks address this by periodically re-evaluating the plan. When enabled, a reasoning-tier LLM reviews the completed step results and remaining steps after every N completed steps. It can issue one of three verdicts:

- **CONTINUE** -- the remaining plan is still valid.
- **REVISE** -- replace the remaining steps with a revised list. The LLM provides the replacement steps.
- **ABORT** -- the task should stop. The remaining work is not salvageable from the current state.

Sanity checks are disabled by default (interval of 0). They add latency and token cost, so they are most useful for long plans (more than five or six steps) where mid-course correction is worth the investment.

## Git Strategy

Git worktrees are the isolation mechanism for all execution modes. Every task runs in its own worktree, separate from the main repository working directory.

### Worktree-per-task isolation

Each task-level workflow creates a worktree at `<repo_root>/.forge-worktrees/<task_id>` on a branch named `forge/<task_id>`. The worktree is branched from the specified base branch (usually `main`). Sub-tasks within fan-out steps create their own worktrees with compound IDs (e.g., `my-task.sub.analyze-schema`), branched from the parent task's branch so they see prior committed step outputs.

### Human-gated merges

Forge never auto-merges worktree branches into `main`. All merges are human-gated. The system produces results in isolated branches; a human reviews and decides whether to merge. This is a fundamental safety property -- the system cannot corrupt the main branch regardless of what the LLM produces.

### Disposable worktrees

Worktrees are treated as disposable. In single-step mode, on a retryable failure the worktree is destroyed and recreated from scratch for the next attempt. In planned mode, uncommitted changes within a step are reset while prior committed steps are preserved. Sub-task worktrees are removed after their output is collected by the parent.

This disposability simplifies the failure model. When something goes wrong, the worktree can be discarded without side effects on other tasks or the main repository. On repeated failure, the workflow halts and escalates to a human with a structured report of what went wrong.

## How the Final Result Is Assembled

The path from task submission to final committed result depends on the execution mode, but the end state is the same: a set of committed changes on a worktree branch, ready for human review.

In single-step mode, the result is a single commit containing the LLM's output after validation passes.

In planned mode, the result is a series of commits, one per step, representing incremental progress. Each commit is independently valid (it passed validation before being committed). A human reviewing the branch sees a clean history showing how the task was built up step by step.

In fan-out mode, each fan-out step produces a single commit containing the merged output of all sub-tasks. Sequential steps before and after the fan-out step produce their own commits. The resulting branch history interleaves sequential commits with gather-merge commits.

In all modes, the final `TaskResult` returned to the CLI includes the status, output files, validation results, step results (in planned mode), and aggregate statistics (token usage, latency, model calls). The worktree remains on disk for human inspection until explicitly removed.

## Further Reading

- [The Golden Path: A Planned Task End-to-End](../tutorials/golden-path.md) -- a walkthrough of a planned task from submission to committed output
- [The Universal Workflow Step](workflow-step.md) -- the five-phase pattern that every execution mode builds on
- [Task Decomposition Reference](../reference/task-decomposition.md) -- data model fields, execution mode selection logic, CLI flags, and git worktree lifecycle details
