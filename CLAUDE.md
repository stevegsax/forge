# Forge

Forge is a general-purpose LLM task orchestrator built around batch mode with document completion. It decomposes tasks into independent work units, executes them as single-step state machine transitions, and reconciles results. The architecture is task-agnostic: code generation, research, analysis, and other domains are all instances of the same universal workflow step with different prompts and context.

## Project Status

Phases 1–9 are implemented. The system supports single-step execution, planned multi-step execution, fan-out/gather with parallel sub-tasks via Temporal child workflows, intelligent context assembly with automatic import graph discovery, PageRank ranking, and token budget management, an observability store with SQLite persistence, Alembic migrations, and CLI inspection commands, knowledge extraction with playbook generation and injection into future task contexts, LLM-guided context exploration where the LLM requests context from providers before generating code, error-aware retries that feed validation errors back to the LLM on retry, and prompt caching via Anthropic cache control headers with cache-efficient prompt ordering and cache token tracking. A planner evaluation framework with deterministic checks and LLM-as-judge scoring is also implemented.

## Key Documents

- `docs/DESIGN.md` — Full architecture and design document.
- `docs/DECISIONS.md` — Key design decisions and rationale.
- `docs/PHASE1.md` — Detailed specification for Phase 1 (the minimal loop).
- `docs/PHASE2.md` — Detailed specification for Phase 2 (planning and multi-step).
- `docs/PHASE3.md` — Detailed specification for Phase 3 (fan-out / gather).
- `docs/PHASE4.md` — Detailed specification for Phase 4 (intelligent context assembly).
- `docs/PHASE5.md` — Detailed specification for Phase 5 (observability store).
- `docs/PHASE6.md` — Detailed specification for Phase 6 (knowledge extraction).
- `docs/PHASE7.md` — Detailed specification for Phase 7 (LLM-guided context exploration).
- `docs/PHASE8.md` — Detailed specification for Phase 8 (error-aware retries).
- `docs/PHASE9.md` through `docs/PHASE12.md` — Specifications for Phases 9–12.
- `docs/PHASE14.md` — Detailed specification for Phase 14 (batch processing: 14a infrastructure, 14b workflow integration, 14c poller + scheduling).
- `docs/PHASE13.md` — Phase 13: Tree-Sitter multi-language support (deferred to Release 2).

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

## The Universal Workflow Step

Every operation follows this pattern:

1. Construct message (assemble prompt + context)
2. Send to LLM
3. Receive response
4. Serialize result
5. Evaluate transition (success, retry, escalate, new tasks discovered, etc.)

Temporal provides the workflow engine. The LLM call and transition evaluation are separate Temporal activities.

The goal is to provide the same high-level functionality as an agentic loop — LLMs can request follow-up information across rounds, task agents can update plans and leave notes for one another, and tools provide connections to the outside world — but every LLM call is structured as a document completion so it is compatible with batch APIs and local LLM tools. The orchestrator (Temporal) owns the control loop, not the LLM.

## Git Strategy

- Each task-level agent works in its own git worktree branched from `main`.
- Merges to `main` are always human-gated.
- Worktrees are disposable: on failure, document the problem, create a fresh worktree, start over.
- Task ordering from the plan is the primary conflict avoidance mechanism.

## Execution Modes

- **Single-step** (`plan=False`, default): Phase 1 behavior. Assemble context, call LLM, write, validate, commit. Retries from a clean worktree.
- **Planned** (`plan=True`): A planner LLM decomposes the task into ordered steps. Each step executes the universal workflow step and commits on success. Step-level retry resets uncommitted changes without losing prior commits.
- **Fan-out** (planned steps with `sub_tasks`): Steps with sub-tasks fan out to parallel child workflows. Each sub-task runs in its own worktree, results are gathered and merged, then validated and committed by the parent. Sub-tasks can themselves contain nested `sub_tasks` for recursive fan-out, bounded by `--max-fan-out-depth` (default 1 = flat fan-out only).

All modes use automatic context discovery (Phase 4) by default: import graph analysis via `grimp`, PageRank ranking via `networkx`, symbol extraction via `ast`, and token budget packing. By default, only target file contents and the repo map are assembled upfront (progressive disclosure); dependency file contents and transitive signatures are omitted to keep prompts lean. The LLM can pull dependencies on demand via exploration providers (Phase 7). Use `--include-deps` to include dependency contents upfront. Disable auto-discovery entirely with `--no-auto-discover`.

All modes support LLM-guided context exploration (Phase 7) by default: the LLM requests context from providers (file reads, code search, symbols, import graphs, tests, lint, git history, repo maps, past runs, playbooks) before generating code. Disable with `--no-explore` or set `--max-exploration-rounds 0`.

All modes use diff-based output (D50): the LLM produces search/replace edits (`edits` list) for existing files and full content (`files` list) for new files. Step and sub-task contexts always include current target file contents from the worktree so the LLM can produce precise diffs. The `write_output` activity applies edits sequentially, requiring each search string to match exactly once.

All modes use error-aware retries (Phase 8): when a step fails validation and retries, the retry prompt includes the validation error output with AST-derived code context around error locations, so the LLM knows what went wrong and can fix it instead of retrying blind.

## Release Roadmap

- **Release 1** (current): Phases 1–14 — the core orchestrator with batch processing. Focus on hardening and confidence before expanding scope.
- **Release 2** (future): Phase 13 (tree-sitter multi-language support) and additional enhancements. See `docs/PHASE13.md`.
