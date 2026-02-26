# Forge

Forge is a general-purpose LLM task orchestrator built around batch mode with document completion. It decomposes tasks into independent work units, executes them as single-step state machine transitions, and reconciles results. The architecture is task-agnostic: code generation, research, analysis, and other domains are all instances of the same universal workflow step with different prompts and context.

## Project Status

Phases 1–9 are implemented. The system supports single-step execution, planned multi-step execution, fan-out/gather with parallel sub-tasks via Temporal child workflows, intelligent context assembly with automatic import graph discovery, PageRank ranking, and token budget management, an observability store with SQLite persistence, Alembic migrations, and CLI inspection commands, knowledge extraction with playbook generation and injection into future task contexts, LLM-guided context exploration where the LLM requests context from providers before generating code, error-aware retries that feed validation errors back to the LLM on retry, and prompt caching via Anthropic cache control headers with cache-efficient prompt ordering and cache token tracking. A planner evaluation framework with deterministic checks and LLM-as-judge scoring is also implemented.

## Documentation

See [docs/TOC.md](docs/TOC.md) for a full table of contents covering design docs, phase specifications, research, and reference material.

## Development Conventions

- Python package management: `uv`
- Linting and formatting: `ruff`
- Data models: `pydantic`
- LLM client library: `anthropic`
- File search: `fd`
- Content search: `rg` (uses Rust regex syntax)
- Terminal: `tmux` / `ghostty`
- Platform: macOS

## Architecture Principles

1. **Batch-first.** The system is designed to operate in batch mode, with orchestration handled by Temporal workflows. Any proposed change must be evaluated for batch compatibility. If a change requires synchronous, interactive, or low-latency LLM calls that are incompatible with batch mode, flag it early.
2. **Deterministic work should be deterministic.** Never ask the LLM to figure out something you can compute. Pre-calculate facts and include them in context.
3. **Context isolation is a feature.** Each task gets a tightly constrained definition of "done" and a customized context assembled fresh for each request.
4. **Planning is the hard part.** Invest the most expensive models and highest token budgets in planning. Everything downstream is bounded by plan quality.
5. **Halt when confused.** When the orchestrator encounters a situation it cannot classify, it stops and escalates to a human.
6. **The LLM call is the universal primitive.** Every task is an instance of: construct message, send, receive, serialize, transition.
7. **Follow Temporal best practices.** Before planning changes that touch Temporal workflows, activities, or worker configuration, check [Temporal Best Practices](https://docs.temporal.io/best-practices) and [docs/planning/WORKERS.md](docs/planning/WORKERS.md) to ensure the approach aligns with Temporal's guidance.

## The Universal Workflow Step

Every operation follows: construct message, send to LLM, receive response, serialize result, evaluate transition. Temporal provides the workflow engine; the LLM call and transition evaluation are separate activities. Every LLM call is structured as a document completion for batch API compatibility. See [docs/DESIGN.md](docs/planning/DESIGN.md) for details.

## Git Strategy

- Each task-level agent works in its own git worktree branched from `main`.
- Merges to `main` are always human-gated.
- Worktrees are disposable: on failure, document the problem, create a fresh worktree, start over.
- Task ordering from the plan is the primary conflict avoidance mechanism.

## Execution Modes

- **Single-step** (`plan=False`, default): Assemble context, call LLM, write, validate, commit.
- **Planned** (`plan=True`): Planner decomposes task into ordered steps; each step commits on success.
- **Fan-out** (planned steps with `sub_tasks`): Parallel child workflows per sub-task, gathered and merged by the parent.

All modes include automatic context discovery (Phase 4), LLM-guided exploration (Phase 7), diff-based output (D50), and error-aware retries (Phase 8) by default. See [docs/ARCHITECTURE.md](docs/planning/ARCHITECTURE.md) for details and CLI flags.

## Release Roadmap

- **Release 1** (current): Phases 1–14 — the core orchestrator with batch processing. Focus on hardening and confidence before expanding scope.
- **Release 2** (future): Phase 13 (tree-sitter multi-language support) and additional enhancements. See `docs/PHASE13.md`.
