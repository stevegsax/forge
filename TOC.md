# Documentation Table of Contents

Start here: **[docs/OVERVIEW.md](docs/OVERVIEW.md)** — current project status, completed/remaining requirements, and known issues.

This repo is the monorepo root (D98). Workspace members: [apps/pbook/](apps/pbook/) (own CLAUDE.md and `design/` notes), [apps/ocr/](apps/ocr/), [libs/sax-llm/](libs/sax-llm/), and [libs/forge-contracts/](libs/forge-contracts/).

## Status & Planning

- [OVERVIEW.md](docs/OVERVIEW.md) — status-of-record: implemented capabilities, requirements complete/remaining, known issues & technical debt.
- [PHASES.md](docs/PHASES.md) — the 14-phase roadmap (1–12 + 14 done with module map; 13 deferred).
- [development-plans/TASKS.md](development-plans/TASKS.md) — live task list (completed vs uncompleted).
- [development-plans/PROCESS.md](development-plans/PROCESS.md) — how to pick up and work a task.
- [development-plans/HANDOFF-2026-07-16-monorepo-deployment.md](development-plans/HANDOFF-2026-07-16-monorepo-deployment.md) — current state: the monorepo consolidation (D98) and local-first deployment (D99), what to know before touching anything, and where to start.

## Architecture & Decisions

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — how Forge works: universal workflow step, execution modes, context assembly, batch-vs-sync, key data models, module map, and subsystems beyond the core loop.
- [DECISIONS.md](docs/DECISIONS.md) — design decisions D1–D98, with supersession/stale markers.

## Requirements

- [requirements/](docs/requirements/) — Gherkin behavioral specs (18 feature files; 16 implemented, 2 specified-but-unbuilt) plus the index with capability→source mapping.

- [README.md](docs/requirements/README.md) — BDD requirements framework with Gherkin feature files, tag taxonomy, and capability mappings.
- [STANDARD.md](docs/requirements/STANDARD.md) — Autonomous-agent requirements standard: paired feature/core artifacts, static-first contracts, functional core / imperative shell split, and handoff gate.
- [TEMPLATE.md](docs/requirements/TEMPLATE.md) — Template for writing a structured `<requirement-id>.core.md` requirement sidecar.
- [REVIEW_CHECKLIST.md](docs/requirements/REVIEW_CHECKLIST.md) — Reviewer checklist for approving a requirement package before autonomous implementation.
- [examples/README.md](docs/requirements/examples/README.md) — Worked examples, including the toy `Inspira` web-app requirement package.

## Operations

- [operations/USAGE.md](docs/operations/USAGE.md) — submitting code and research tasks to Forge.
- [operations/WORKERS.md](docs/operations/WORKERS.md) — worker overview, identity, and scaling.
- [operations/DEPLOYMENT.md](docs/operations/DEPLOYMENT.md) — local-first deployment: podman stack (Temporal + Postgres + MinIO), launchd workers, Supabase + S3 stores (D99).
- [operations/DEBUGGING.md](docs/operations/DEBUGGING.md) — logging, observability store, API message logs, OTel tracing, env vars.
- [operations/ADDING_A_DOMAIN.md](docs/operations/ADDING_A_DOMAIN.md) — parameterizing LLM behavior through task domains.
- [operations/test-strategy.md](docs/operations/test-strategy.md) — testing pyramid and signal quality.

## Reference

- [reference/mistral.md](docs/reference/mistral.md) — Mistral API reference: auth, batch API, OCR, SDK, curl.

## User Guides

- [user/playbooks.md](docs/user/playbooks.md) — playbook system: extraction, manual add, storage.

## Remaining-Work Specs (`docs/planning/`)

- [planning/PHASE13.md](docs/planning/PHASE13.md) — tree-sitter multi-language support (deferred to Release 2).
- [planning/LSP_INTEGRATION_PLAN.md](docs/planning/LSP_INTEGRATION_PLAN.md) — LSP-based context generation (deferred).
- [planning/task-management/DECOMPOSITION.md](docs/planning/task-management/DECOMPOSITION.md) — multi-transform DAG planner (**draft; not implemented**).
- [planning/task-management/DECOMPOSITION_SCENARIOS.md](docs/planning/task-management/DECOMPOSITION_SCENARIOS.md) — behavioral scenarios for the DECOMPOSITION draft.
- [planning/task-management/DECOMPOSITION_REVIEW.md](docs/planning/task-management/DECOMPOSITION_REVIEW.md) — design review of the DECOMPOSITION draft (required + recommended changes, scenario gaps).

## Worked Examples

- [llm-examples/README.md](docs/llm-examples/README.md) — worked LLM examples tracing the universal workflow step.

## Archive

Superseded and exploratory material, **not authoritative** — see [archive/README.md](archive/README.md). Contains the original `DESIGN.md`, completed phase specs (`PHASE1`–`PHASE12`, `PHASE14`), `research/`, and the unmerged `to-merge/` reports.
