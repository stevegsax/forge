# Documentation Table of Contents

Start here: **[docs/OVERVIEW.md](docs/OVERVIEW.md)** — current project status, completed/remaining requirements, and known issues.

## Status & Planning

- [OVERVIEW.md](docs/OVERVIEW.md) — status-of-record: implemented capabilities, requirements complete/remaining, known issues & technical debt.
- [PHASES.md](docs/PHASES.md) — the 14-phase roadmap (1–12 + 14 done with module map; 13 deferred).
- [development-plans/TASKS.md](development-plans/TASKS.md) — live task list (completed vs uncompleted).
- [development-plans/PROCESS.md](development-plans/PROCESS.md) — how to pick up and work a task.

## Architecture & Decisions

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — how Forge works: universal workflow step, execution modes, context assembly, batch-vs-sync, key data models, module map, and subsystems beyond the core loop.
- [DECISIONS.md](docs/DECISIONS.md) — design decisions D1–D85, with supersession/stale markers.

## Requirements

- [requirements/](docs/requirements/) — Gherkin behavioral specs (18 feature files; 16 implemented, 2 specified-but-unbuilt) plus the index with capability→source mapping.

## Operations

- [operations/USAGE.md](docs/operations/USAGE.md) — submitting code and research tasks to Forge.
- [operations/WORKERS.md](docs/operations/WORKERS.md) — worker overview, identity, and scaling.
- [operations/DEPLOYMENT.md](docs/operations/DEPLOYMENT.md) — self-hosted AWS EC2 + Supabase Postgres + S3 deployment.
- [operations/SECURE-REMOTE-ACCESS.md](docs/operations/SECURE-REMOTE-ACCESS.md) — mTLS-secured remote Temporal access.
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

## Worked Examples

- [llm-examples/README.md](docs/llm-examples/README.md) — worked LLM examples tracing the universal workflow step.

## Archive

Superseded and exploratory material, **not authoritative** — see [archive/README.md](archive/README.md). Contains the original `DESIGN.md`, completed phase specs (`PHASE1`–`PHASE12`, `PHASE14`), `research/`, and the unmerged `to-merge/` reports.
