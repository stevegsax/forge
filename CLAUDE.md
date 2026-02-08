# Forge

Forge is a general-purpose LLM task orchestrator built around batch mode with document completion. It decomposes tasks into independent work units, executes them as single-step state machine transitions, and reconciles results. The architecture is task-agnostic: code generation, research, analysis, and other domains are all instances of the same universal workflow step with different prompts and context.

## Project Status

Phase 3 is implemented. The system supports fan-out/gather: plan steps can declare independent sub-tasks that execute in parallel via Temporal child workflows. Each sub-task runs in its own git worktree, and results are gathered, checked for file conflicts, merged into the parent worktree, validated, and committed. Phase 2 planning and multi-step execution remains fully backward compatible. Phase 1 single-step mode remains the default.

## Key Documents

- `docs/DESIGN.md` — Full architecture and design document.
- `docs/DECISIONS.md` — Key design decisions and rationale.
- `docs/PHASE1.md` — Detailed specification for Phase 1 (the minimal loop).
- `docs/PHASE2.md` — Detailed specification for Phase 2 (planning and multi-step).
- `docs/PHASE3.md` — Detailed specification for Phase 3 (fan-out / gather).

## Development Conventions

- Python package management: `uv`
- Linting and formatting: `ruff`
- Data models: `pydantic`
- LLM client library: `pydantic-ai`
- File search: `fd`
- Content search: `rg` (uses Rust regex syntax)
- Terminal: `tmux` / `ghostty`
- Platform: macOS

## Architecture Principles

1. **Deterministic work should be deterministic.** Never ask the LLM to figure out something you can compute. Pre-calculate facts and include them in context.
2. **Context isolation is a feature.** Each task gets a tightly constrained definition of "done" and a customized context assembled fresh for each request.
3. **Planning is the hard part.** Invest the most expensive models and highest token budgets in planning. Everything downstream is bounded by plan quality.
4. **Halt when confused.** When the orchestrator encounters a situation it cannot classify, it stops and escalates to a human.
5. **The LLM call is the universal primitive.** Every task is an instance of: construct message, send, receive, serialize, transition.

## The Universal Workflow Step

Every operation follows this pattern:

1. Construct message (assemble prompt + context)
2. Send to LLM
3. Receive response
4. Serialize result
5. Evaluate transition (success, retry, escalate, new tasks discovered, etc.)

Temporal provides the workflow engine. The LLM call and transition evaluation are separate Temporal activities.

## Git Strategy

- Each task-level agent works in its own git worktree branched from `main`.
- Merges to `main` are always human-gated.
- Worktrees are disposable: on failure, document the problem, create a fresh worktree, start over.
- Task ordering from the plan is the primary conflict avoidance mechanism.

## Current Phase: Phase 3 — Fan-Out / Gather (Complete)

Phase 3 is implemented. The system supports three modes:

- **Single-step** (`plan=False`, default): Phase 1 behavior. Assemble context, call LLM, write, validate, commit. Retries from a clean worktree.
- **Planned** (`plan=True`): A planner LLM decomposes the task into ordered steps. Each step executes the universal workflow step and commits on success. Step-level retry resets uncommitted changes without losing prior commits.
- **Fan-out** (planned steps with `sub_tasks`): Steps with sub-tasks fan out to parallel child workflows. Each sub-task runs in its own worktree, results are gathered and merged, then validated and committed by the parent.

See `docs/PHASE1.md`, `docs/PHASE2.md`, and `docs/PHASE3.md` for specifications.
