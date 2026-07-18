# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview: Forge

Forge is a general-purpose LLM task orchestrator built around batch mode with document completion. It decomposes tasks into independent work units, executes them as single-step state machine transitions, and reconciles results. The architecture is task-agnostic: code generation, research, analysis, and other domains are all instances of the same universal workflow step with different prompts and context.

## Project Status

Current status — completed and remaining requirements, plus known issues and technical debt — is in [docs/OVERVIEW.md](docs/OVERVIEW.md) (the status-of-record). Phase roadmap for the shipped Release 1: [docs/PHASES.md](docs/PHASES.md). The active work queue is the 2026-06-10 architecture migration in [development-plans/TASKS.md](development-plans/TASKS.md) — see the next section.

## Current Project: 2026-06-10 Architecture Migration

A merged platform redesign plan was approved 2026-06-10. It is the active project; [development-plans/TASKS.md](development-plans/TASKS.md) is the work queue (56 tasks: standalone Phase 0, T0.1–T0.8, plus the eight migration phases, T1.0–T8.4; amended 2026-07-08 per [forge-review-2026-07-08.md](forge-review-2026-07-08.md) and its capture sweep). Sections below carry one-line notes where a later phase deletes or replaces current machinery — do not extend anything so annotated.

**Landed** (as of 2026-07-17): **Phases 1 and 2 are complete.** Phase 1 (T1.0–T1.8) stopped the bleeding; Phase 2 delivered the monorepo (T2.1, D98; Python pinned to 3.14), the root gates + GitHub Actions CI (T2.2 — `make gates` mirrors CI; import-linter DAG contracts; every package coverage-gated at 85%), and mypy-strict across the workspace's six packages at the time (T2.3a–d). **T0.7** retired EC2 for the local-first deployment (D99). **Phase 3 is underway:** T3.1–T3.3 landed the `sax_platform` library — the both-lane structured-outputs LLM client, the single model-tier registry, and `MistralOcr` — and made both forge and ocr depend on it. **T3.4 landed:** the `sax_platform.{temporal,db,embeddings,config,logging}` plumbing modules plus the `sax_platform.contracts` sandbox-light layer (batch wire models, `persist_block`, `s3_blobs`, `UTCDateTime`, constants, the read-only `batch_jobs` mirror) — forge-contracts is retired and forge, ocr, and pbook are migrated onto the platform. **T3.5 landed:** forced tool use is retired platform-wide — forge and pbook now complete structured outputs through `sax_platform.llm` (typed refusal/truncation/mismatch outcomes), both string-keyed output-type registries are replaced by frozen `OUTPUT_TYPES` mappings, and `libs/sax-llm` is deleted. **The workspace is four packages now** (forge, pbook, ocr, sax-platform). **Next up: T3.6** closes Phase 3. Phase 0's other tasks (T0.1–T0.6, T0.8) are independent of every phase and can land anytime; Phases 4–8 are ahead. Per-task detail is in [development-plans/TASKS.md](development-plans/TASKS.md) and [development-plans/CHANGELOG.md](development-plans/CHANGELOG.md).

**Still current until later phases — do not "fix" these as if done:** the signal-based batch transport and its shared poller live until Phase 4 (T4.1); pbook keeps its own duplicated tier map until T6.4.

- **Reversals:** R1 — the signal-based batch SPI is replaced by per-workflow timer-loop polling (D88, Phase 4). R2 — pbook ingestion becomes sync inside pbook and forge's ingestion side is deleted (D91, T6.4).
- **Phase ordering is load-bearing:** 1 → 2 → 3 → 4 → 5; Phase 6 runs after Phase 5; Phases 6 and 7 may run in parallel; Phase 8 closes.
- **Context:** handoffs in [development-plans/](development-plans/) (`HANDOFF-architecture-review-2026-06-10.md` for the plan's origin; `HANDOFF-2026-07-17-phase3.md` for the current state, with `HANDOFF-2026-07-16-monorepo-deployment.md` for the monorepo/deployment lead-up); per-task history in [development-plans/CHANGELOG.md](development-plans/CHANGELOG.md); decisions D86–D99 in [docs/DECISIONS.md](docs/DECISIONS.md); review findings in [docs/reviews/2026-06-architecture-review.md](docs/reviews/2026-06-architecture-review.md).

## Running the System (D99)

Deployment is **local-first on this desktop** — there is no cloud host and no remote access. Temporal self-hosts in the podman stack (`make stack-up`; frontend `127.0.0.1:7233`, UI `http://localhost:8233`, Postgres, MinIO), the forge/pbook/ocr workers run as launchd-supervised host processes reading `~/.config/forge/forge.env`, and the application stores stay managed (Supabase Postgres, S3). Full detail: [docs/operations/DEPLOYMENT.md](docs/operations/DEPLOYMENT.md), [deploy/local-stack/README.md](deploy/local-stack/README.md), [deploy/launchd/README.md](deploy/launchd/README.md).

> **The ambient shell env points at production** (`FORGE_DB_URL` → Supabase, `AWS_*` → real S3). Override every relevant var before any local DB/blob command. On this machine the local stack's Postgres is on **port 5434** (5433 is taken by another project; the override lives in the gitignored `deploy/local-stack/.env`).

## Cross-Project Dependencies

Forge is the monorepo (D98, consolidation complete): the repo root is a uv workspace with one `uv.lock` and one venv, and every internal package is a workspace member — `apps/pbook`, `apps/ocr`, plus `libs/sax-platform` (born in Phase 3, not an absorbed repo) — so a bare checkout is self-contained. Members declare no `[tool.uv.sources]` of their own. ocr is the one member forge does **not** depend on (apps never import apps): setup is `uv sync --all-packages` and its CLI runs as `uv run --package ocr ocr <cmd>`. Version pinning inside the set is deliberately gone: the suites are the compatibility contract.

- **sax-platform** (`libs/sax-platform`, workspace member; **a dependency of forge, ocr, and pbook**) — the D89 platform library. `sax_platform.llm`: structured-outputs LLM client on both lanes — the sync `complete[T]` helper (`messages.parse`) and the batch request builder plus submit/status/fetch helpers (`output_config.format`, with stored-bytes classification into typed non-retryable failures at the fetch/parse seam) — typed refusal/truncation/mismatch outcomes, `max_retries=0`, required `max_tokens`, thinking selected via `ThinkingPolicy`, opt-in prompt caching. As of T3.5 this is the runtime LLM completion path for both forge and pbook. `sax_platform.llm.tiers`: the single model-tier registry (REASONING → `claude-opus-4-8`, GENERATION/SUMMARIZATION → `claude-sonnet-5`, CLASSIFICATION → `claude-haiku-4-5`); `budget_tokens` is gone platform-wide — thinking is adaptive or explicitly disabled, and the forge CLI flag is `--effort`, not `--thinking-budget`. `sax_platform.ocr`: `MistralOcr` (owns the `mistralai` dependency). `sax_platform.contracts` (T3.4; sandbox-light, forbidden from importing SDKs or shell siblings) absorbed the retired forge-contracts' SPI surface: batch wire models (`BatchResult`, `BatchSubmitSpiInput`, `BatchJobStatus`, the result-payload envelope), `persist_block`, `s3_blobs`, `UTCDateTime`, queue/namespace/signal constants, and the read-only `batch_jobs` mirror. `sax_platform.{temporal,db,embeddings,config,logging}` (also T3.4) hold the rest of the shared plumbing — Temporal connect + worker scaffold + retry presets, the DB engine factory + generic helpers, the `Embedder` protocol, frozen pydantic-settings config groups, and shared logging setup. T3.5 finished the LLM-client consumer migration (forge and pbook both run on `sax_platform.llm`); T3.6 (composition roots) removes the remaining module-global client/provider seams.
- **ocr** (`apps/ocr`, workspace member; **not** a forge dependency) — document OCR via the Mistral batch API, consuming the platform through the batch SPI. Own Temporal worker on `ocr-task-queue`; cross-queue SPI calls to `forge-task-queue`; own Alembic chain (`alembic_version_ocr`, four `ocr_*` tables) in the shared `FORGE_DB_URL` database; reads `batch_jobs` read-only via the `sax_platform.contracts` mirror. Imports `sax_platform` only (its `MistralOcr` DI seam lives in `ocr/deps.py` and is installed at worker startup, T3.3), never `forge`.
- **pbook** (`apps/pbook`, workspace member) — cross-project knowledge store, a required dependency. Forge's transcript ingestion workflows call pbook's `ExtractionWorkflow` and `record_ingested_session` activity cross-queue on `pbook-task-queue`. An import guard (`_INGESTION_AVAILABLE` in worker.py) gates workflow registration only — pbook itself is installed unconditionally. As of T3.4, pbook also imports `sax_platform` directly — its LLM completion, embeddings, and Temporal heartbeat run on the platform's `sax_platform.llm`/`sax_platform.embeddings`/`sax_platform.temporal`. (Forge's ingestion side is deleted in T6.4.) Forge's own playbook store (`forge.db` `playbooks` table) is separate from pbook's `entries` table — the two stores are parallel and do not share data. (The playbooks store is superseded by pbook in T6.7.) See `diataxis/explanation/learning-loops.md` for the design discussion.

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

Workspace command discipline: the commands above, run from the root, cover forge only (`testpaths = ["tests"]` keeps the members out of the default run). Setup is `uv sync --all-packages` (a bare exact `uv sync` prunes ocr's packages, since forge doesn't depend on it; `uv run`'s inexact sync self-heals). Each member's suite runs from its own directory so its own config applies: `cd apps/pbook && uv run pytest` (85% gate; its conftest needs a running podman machine or `PBOOK_TEST_DATABASE_URL`), `cd apps/ocr && uv run pytest` (85% gate), `cd libs/sax-platform && uv run pytest` (85% gate). `make gates` runs the workspace gates — lint, mypy (all four packages), lint-imports, and the four test suites — mirroring `.github/workflows/ci.yml`, which additionally runs the `postgres` migration suite (`pytest -m postgres`) as its own job. Never run a member's tests from the root (e.g. `pytest apps/pbook/tests`): forge's addopts would apply and the member's conftest would run under the wrong config.

The test suite uses `asyncio_mode = "auto"` and a session-scoped Temporal time-skipping environment (`WorkflowEnvironment.start_time_skipping` in `conftest.py`). Two markers are excluded from default runs: `postgres` (Alembic migrations against a real Postgres via testcontainers) and `e2e`. The `e2e` marker is defined but currently empty in forge — `tests/test_e2e.py` is a default-run integration suite with mocked LLM calls (T8.2 renames it to `test_pipeline.py` and restores marker honesty).

## Development Conventions

- Python: **3.14**, standard GIL (pinned in `.python-version`; `requires-python` floors stay `>=3.12`)
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
- **Current**: the 2026-06-10 architecture migration (see "Current Project" above) — 56 tasks (a standalone Phase 0 plus 8 migration phases); Phases 1–2 are complete and Phase 3 is in progress. The migration ends with the forge monorepo tagged v1.0 (T8.4's final sweep — not before).
- **Release 2** (future): Phase 13 (tree-sitter multi-language support) and additional enhancements. See [docs/planning/PHASE13.md](docs/planning/PHASE13.md) and [docs/PHASES.md](docs/PHASES.md).

## Development Plans

When working on tasks from `development-plans/`, follow the process described in [development-plans/PROCESS.md](development-plans/PROCESS.md): read the task file's Problem and Acceptance Criteria, write a Plan section describing your approach, break it into sub-tasks, update status to IN PROGRESS, check off sub-tasks as you go, and append to Development Notes when you discover something unexpected or change the plan. Task files are living documents — accurate status documentation is as important as writing code.

## Diataxis Documentation

The `diataxis/` directory contains generated, human-facing documentation.
It is an output artifact — disposable and never authoritative. Do not
use it as input for design decisions, code generation, or development
work. If the code and the diataxis docs disagree, the code is right.
