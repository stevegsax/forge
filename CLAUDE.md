# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Forge

Forge is a general-purpose LLM task orchestrator built around batch mode with document completion. It decomposes tasks into independent work units, executes them as single-step state machine transitions, and reconciles results. The architecture is task-agnostic: code generation, research, analysis, and other domains are all instances of the same universal workflow step with different prompts and context.

## Project Status

Phases 1–12 and 14 are implemented (Phase 13, tree-sitter multi-language support, is deferred to Release 2). The system supports single-step execution, planned multi-step execution, fan-out/gather with parallel sub-tasks via Temporal child workflows, intelligent context assembly with automatic import graph discovery, PageRank ranking, and token budget management, an observability store with SQLite persistence, Alembic migrations, and CLI inspection commands, knowledge extraction with playbook generation and injection into future task contexts, LLM-guided context exploration where the LLM requests context from providers before generating code, error-aware retries that feed validation errors back to the LLM on retry, prompt caching via Anthropic cache control headers with cache-efficient prompt ordering and cache token tracking, fuzzy edit matching with a four-level fallback chain, model routing with capability tiers, extended thinking for planning, and batch processing via the Anthropic Batch API. The OCR pipeline supports both synchronous and batch modes. The synchronous path (`OcrSyncWorkflow`) calls the Mistral OCR API directly and returns results immediately. The batch path (`OcrSubmitWorkflow`) submits to the Mistral Batch API and waits for a polling signal. Both paths extract images from documents, store them in the `ocr_images` table, and rewrite markdown references to unique `ocr-image://` URIs. A planner evaluation framework with deterministic checks and LLM-as-judge scoring is also implemented. Transcript ingestion (`forge ingest`) reads Claude Code JSONL session files, analyzes them via the batch API, and hands extracted experiences to pbook's ExtractionWorkflow cross-queue for storage as playbook entries.

## Cross-Project Dependencies

Forge depends on two sibling editable packages (via `[tool.uv.sources]` in pyproject.toml):

- **sax-llm** (`../sax-llm`) — shared LLM provider abstraction, output type registry, and batch response parsing.
- **pbook** (`../pbook`, optional) — cross-project knowledge store. Forge's transcript ingestion workflows call pbook's `ExtractionWorkflow` and `record_ingested_session` activity cross-queue on `pbook-task-queue`. The dependency is guarded: if pbook is not installed, the worker starts without ingestion workflows and `forge ingest` exits with a clear error. Forge's own playbook store (`forge.db` `playbooks` table) is separate from pbook's `entries` table — the two stores are parallel and do not share data. See `diataxis/explanation/learning-loops.md` for the design discussion.

## Documentation

See [TOC.md](TOC.md) for a full table of contents covering design docs, phase specifications, research, and reference material.

## Build, Test, and Lint Commands

```bash
uv sync                          # Install dependencies
uv run pytest                    # Run all tests (excludes e2e by default)
uv run pytest tests/test_foo.py  # Run a single test file
uv run pytest tests/test_foo.py::TestClass::test_name  # Run a single test
uv run pytest --no-cov           # Skip coverage (faster iteration)
uv run pytest -m e2e             # Run e2e tests (requires Temporal + APIs)
uv run ruff check .              # Lint
uv run ruff format .             # Format
uv run ruff check --fix .        # Auto-fix lint issues
```

Coverage is enforced at 85% by default (`--cov-fail-under=85` in pyproject.toml). When running a single test file, use `--no-cov` to avoid failing on aggregate coverage.

The test suite uses `asyncio_mode = "auto"` and a session-scoped Temporal time-skipping environment (`WorkflowEnvironment.start_time_skipping` in `conftest.py`). Tests that require external services (Mistral API, Temporal server) are marked `e2e` and excluded from default runs.

## Development Conventions

- Python package management: `uv`
- Linting and formatting: `ruff`
- Data models: `pydantic`
- LLM client library: `anthropic`
- File search: `fd`
- Content search: `rg` (uses Rust regex syntax)
- Document conversion: `pandoc`
- JSON processing: `jq`
- Markdown querying: `mq` (jq-like syntax for markdown)
- Markdown linting and structure checking: `markdownlint-cli2`
- YAML/INI processing: `yq` (jq-like syntax for YAML and INI)
- XML processing: `xmlstarlet`
- HTML querying: `htmlq` (CSS selector syntax)
- Natural language linting: `textlint`
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

## Test Patterns

Workflow tests use the Temporal time-skipping test server with a session-scoped `env` fixture (`conftest.py`). Activities are mocked by name using `@activity.defn(name="activity_name")` and registered on a per-test `Worker`. Batch signal delivery is tested by starting the workflow with `env.client.start_workflow`, then calling `handle.signal(WorkflowClass.batch_result_received, BatchResult(...))` to deliver the batch result.

CLI tests use Click's `CliRunner` and mock async submission helpers with `_async_result(value)` (a helper that returns an async function wrapping `value`, avoiding orphaned coroutine warnings from `asyncio.run`).

Activity tests call the activity function directly as a plain async function (not through a Temporal worker), mocking external dependencies with `unittest.mock.patch`.

Cross-queue workflow tests (e.g., ingestion) run two `Worker` instances in parallel — one on `forge-task-queue` and one on `pbook-task-queue` — with mock activities/workflows registered on each. The workflow sandbox restricts module-level mutable state access inside workflows, so canned results must be returned from activities rather than read from module globals.

## The Universal Workflow Step

Every operation follows: construct message, send to LLM, receive response, serialize result, evaluate transition. Temporal provides the workflow engine; the LLM call and transition evaluation are separate activities. Every LLM call is structured as a document completion for batch API compatibility. See [docs/planning/DESIGN.md](docs/planning/DESIGN.md) for details.

## Git Strategy

- Each task-level agent works in its own git worktree branched from `main`.
- Merges to `main` are always human-gated.
- Worktrees are disposable: on failure, document the problem, create a fresh worktree, start over.
- Task ordering from the plan is the primary conflict avoidance mechanism.

## Execution Modes

- **Single-step** (`plan=False`, default): Assemble context, call LLM, write, validate, commit.
- **Planned** (`plan=True`): Planner decomposes task into ordered steps; each step commits on success.
- **Fan-out** (planned steps with `sub_tasks`): Parallel child workflows per sub-task, gathered and merged by the parent.

All modes include automatic context discovery (Phase 4), LLM-guided exploration (Phase 7), diff-based output (D50), and error-aware retries (Phase 8) by default. See [docs/planning/ARCHITECTURE.md](docs/planning/ARCHITECTURE.md) for details and CLI flags.

## Release Roadmap

- **Release 1** (current): Phases 1–14 — the core orchestrator with batch processing. Focus on hardening and confidence before expanding scope.
- **Release 2** (future): Phase 13 (tree-sitter multi-language support) and additional enhancements. See [docs/planning/PHASE13.md](docs/planning/PHASE13.md).

## Development Plans

When working on tasks from `development-plans/`, follow the process described in [development-plans/PROCESS.md](development-plans/PROCESS.md): read the task file's Problem and Acceptance Criteria, write a Plan section describing your approach, break it into sub-tasks, update status to IN PROGRESS, check off sub-tasks as you go, and append to Development Notes when you discover something unexpected or change the plan. Task files are living documents — accurate status documentation is as important as writing code.
