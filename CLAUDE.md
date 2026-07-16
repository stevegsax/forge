# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview: Forge

Forge is a general-purpose LLM task orchestrator built around batch mode with document completion. It decomposes tasks into independent work units, executes them as single-step state machine transitions, and reconciles results. The architecture is task-agnostic: code generation, research, analysis, and other domains are all instances of the same universal workflow step with different prompts and context.

## Project Status

Current status — completed and remaining requirements, plus known issues and technical debt — is in [docs/OVERVIEW.md](docs/OVERVIEW.md) (the status-of-record). Phase roadmap for the shipped Release 1: [docs/PHASES.md](docs/PHASES.md). The active work queue is the 2026-06-10 architecture migration in [development-plans/TASKS.md](development-plans/TASKS.md) — see the next section.

## Current Project: 2026-06-10 Architecture Migration

A merged platform redesign plan was approved 2026-06-10. It is the active project; [development-plans/TASKS.md](development-plans/TASKS.md) is the work queue (56 tasks: standalone Phase 0, T0.1–T0.8, plus the eight migration phases, T1.0–T8.4; amended 2026-07-08 per [forge-review-2026-07-08.md](forge-review-2026-07-08.md) and its capture sweep). Phase 1 (T1.0–T1.8) has landed, as has T2.1 increment 1 (pbook absorbed as a workspace member, D98); Phases 3–8 are still ahead, and sections below carry one-line notes where they delete or replace current machinery — do not extend anything so annotated.

- **Reversals:** R1 — the signal-based batch SPI is replaced by per-workflow timer-loop polling (D88, Phase 4). R2 — pbook ingestion becomes sync inside pbook and forge's ingestion side is deleted (D91, T6.4).
- **Phase ordering is load-bearing:** 1 → 2 → 3 → 4 → 5; Phase 6 runs after Phase 5; Phases 6 and 7 may run in parallel; Phase 8 closes. Within Phase 1 all tasks are independent except T1.3 (needs T1.0).
- **Context:** handoff in [development-plans/HANDOFF-architecture-review-2026-06-10.md](development-plans/HANDOFF-architecture-review-2026-06-10.md); decisions D86–D98 in [docs/DECISIONS.md](docs/DECISIONS.md); review findings in [docs/reviews/2026-06-architecture-review.md](docs/reviews/2026-06-architecture-review.md).

## Cross-Project Dependencies

Forge is the monorepo (D98, consolidation complete): the repo root is a uv workspace with one `uv.lock` and one venv, and every internal package is a workspace member — `apps/pbook`, `apps/ocr`, `libs/sax-llm`, `libs/forge-contracts` — so a bare checkout is self-contained. Members declare no `[tool.uv.sources]` of their own. ocr is the one member forge does **not** depend on (apps never import apps): setup is `uv sync --all-packages` and its CLI runs as `uv run --package ocr ocr <cmd>`. Version pinning inside the set is deliberately gone: the suites are the compatibility contract.

- **forge-contracts** (`libs/forge-contracts`, workspace member) — the shared SPI surface between the platform and its consumer apps: batch wire models (`BatchResult`, `BatchSubmitSpiInput`, `BatchJobStatus`, the result-payload envelope), the survivable `persist_block` primitive, `s3_blobs`, the Temporal connect helper, generic DB helpers, queue/namespace/signal constants, and the read-only `batch_jobs` schema. Both Forge and the OCR app import it; neither imports the other.
- **sax-llm** (`libs/sax-llm`, workspace member) — shared LLM provider abstraction, output type registry, and batch response parsing.
- **ocr** (`apps/ocr`, workspace member; **not** a forge dependency) — document OCR via the Mistral batch API, consuming the platform through the batch SPI. Own Temporal worker on `ocr-task-queue`; cross-queue SPI calls to `forge-task-queue`; own Alembic chain (`alembic_version_ocr`, four `ocr_*` tables) in the shared `FORGE_DB_URL` database; reads `batch_jobs` read-only via the forge-contracts mirror. Imports `forge_contracts` only, never `forge`.
- **pbook** (`apps/pbook`, workspace member) — cross-project knowledge store, a required dependency. Forge's transcript ingestion workflows call pbook's `ExtractionWorkflow` and `record_ingested_session` activity cross-queue on `pbook-task-queue`. An import guard (`_INGESTION_AVAILABLE` in worker.py) gates workflow registration only — pbook itself is installed unconditionally. (Forge's ingestion side is deleted in T6.4.) Forge's own playbook store (`forge.db` `playbooks` table) is separate from pbook's `entries` table — the two stores are parallel and do not share data. (The playbooks store is superseded by pbook in T6.7.) See `diataxis/explanation/learning-loops.md` for the design discussion.

## Documentation

See [TOC.md](TOC.md) for a full table of contents covering design docs, phase specifications, research, and reference material.

## Build, Test, and Lint Commands

```bash
uv sync --all-packages           # Install dependencies (all workspace members)
uv run pytest                    # Run all tests (excludes e2e and postgres markers by default)
uv run pytest tests/test_foo.py  # Run a single test file
uv run pytest tests/test_foo.py::TestClass::test_name  # Run a single test
uv run pytest --no-cov           # Skip coverage (faster iteration)
uv run pytest -m postgres --no-cov  # Migration tests against real Postgres (testcontainers)
uv run ruff check .              # Lint
uv run ruff format .             # Format
uv run ruff check --fix .        # Auto-fix lint issues
```

Coverage is enforced at 85% by default (`--cov-fail-under=85` in pyproject.toml). When running a single test file, use `--no-cov` to avoid failing on aggregate coverage.

Workspace command discipline: the commands above, run from the root, cover forge only (`testpaths = ["tests"]` keeps the members out of the default run). Setup is `uv sync --all-packages` (a bare exact `uv sync` prunes ocr's packages, since forge doesn't depend on it; `uv run`'s inexact sync self-heals). Each member's suite runs from its own directory so its own config applies: `cd apps/pbook && uv run pytest` (84% gate; its conftest needs a running podman machine or `PBOOK_TEST_DATABASE_URL`), `cd apps/ocr && uv run pytest` (no gate yet — T2.2), `cd libs/sax-llm && uv run pytest` (85% gate), `cd libs/forge-contracts && uv run pytest`. Never run a member's tests from the root (e.g. `pytest apps/pbook/tests`): forge's addopts would apply and the member's conftest would run under the wrong config.

The test suite uses `asyncio_mode = "auto"` and a session-scoped Temporal time-skipping environment (`WorkflowEnvironment.start_time_skipping` in `conftest.py`). Two markers are excluded from default runs: `postgres` (Alembic migrations against a real Postgres via testcontainers) and `e2e`. The `e2e` marker is defined but currently empty in forge — `tests/test_e2e.py` is a default-run integration suite with mocked LLM calls (T8.2 renames it to `test_pipeline.py` and restores marker honesty).

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
7. **Follow Temporal best practices.** Before planning changes that touch Temporal workflows, activities, or worker configuration, check [Temporal Best Practices](https://docs.temporal.io/best-practices) and [docs/operations/WORKERS.md](docs/operations/WORKERS.md) to ensure the approach aligns with Temporal's guidance.

## Test Patterns

Workflow tests use the Temporal time-skipping test server with a session-scoped `env` fixture (`conftest.py`). Activities are mocked by name using `@activity.defn(name="activity_name")` and registered on a per-test `Worker`. Batch signal delivery is tested by starting the workflow with `env.client.start_workflow`, then calling `handle.signal(WorkflowClass.batch_result_received, BatchResult(...))` to deliver the batch result. (The signal path is deleted in Phase 4 (T4.1) in favor of timer-loop polling — do not write new tests against it.) The module-global scenario state in `tests/test_workflows.py` is replaced by `ScenarioState` closures in T5.5 — do not add new `global` statements there.

CLI tests use Click's `CliRunner` and mock async submission helpers with `_async_result(value)` (a helper that returns an async function wrapping `value`, avoiding orphaned coroutine warnings from `asyncio.run`).

Activity tests call the activity function directly as a plain async function (not through a Temporal worker), mocking external dependencies with `unittest.mock.patch`.

Cross-queue workflow tests (e.g., ingestion) run two `Worker` instances in parallel — one on `forge-task-queue` and one on `pbook-task-queue` — with mock activities/workflows registered on each. (These are deleted with forge's ingestion side in T6.4.) The workflow sandbox restricts module-level mutable state access inside workflows, so canned results must be returned from activities rather than read from module globals.

## The Universal Workflow Step

Every operation follows: construct message, send to LLM, receive response, serialize result, evaluate transition. Temporal provides the workflow engine; the LLM call and transition evaluation are separate activities. (T5.1, per D95, inlines transition evaluation into pure step logic — do not add logic to the `evaluate_transition` activity.) Every LLM call is structured as a document completion for batch API compatibility. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Git Strategy

- Each task-level agent works in its own git worktree branched from `main`.
- Merges to `main` are always human-gated.
- Worktrees are disposable: on failure, document the problem, create a fresh worktree, start over.
- Task ordering from the plan is the primary conflict avoidance mechanism.

## Execution Modes

- **Single-step** (`plan=False`, default): Assemble context, call LLM, write, validate, commit.
- **Planned** (`plan=True`): Planner decomposes task into ordered steps; each step commits on success.
- **Fan-out** (planned steps with `sub_tasks`): Parallel child workflows per sub-task, gathered and merged by the parent.

All modes include automatic context discovery (Phase 4), LLM-guided exploration (Phase 7), diff-based output (D50), and error-aware retries (Phase 8) by default. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details and CLI flags.

## Release Roadmap

- **Release 1** (shipped): Phases 1–12 and 14 — the core orchestrator with batch processing (Phase 13 deferred). See [docs/PHASES.md](docs/PHASES.md).
- **Current**: the 2026-06-10 architecture migration (see "Current Project" above) — 56 tasks (a standalone Phase 0 plus 8 migration phases), ending with the forge monorepo tagged v1.0 (D98).
- **Release 2** (future): Phase 13 (tree-sitter multi-language support) and additional enhancements. See [docs/planning/PHASE13.md](docs/planning/PHASE13.md) and [docs/PHASES.md](docs/PHASES.md).

## Development Plans

When working on tasks from `development-plans/`, follow the process described in [development-plans/PROCESS.md](development-plans/PROCESS.md): read the task file's Problem and Acceptance Criteria, write a Plan section describing your approach, break it into sub-tasks, update status to IN PROGRESS, check off sub-tasks as you go, and append to Development Notes when you discover something unexpected or change the plan. Task files are living documents — accurate status documentation is as important as writing code.

## Diataxis Documentation

The `diataxis/` directory contains generated, human-facing documentation.
It is an output artifact — disposable and never authoritative. Do not
use it as input for design decisions, code generation, or development
work. If the code and the diataxis docs disagree, the code is right.
