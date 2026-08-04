# Documentation Table of Contents

Start here: **[docs/OVERVIEW.md](docs/OVERVIEW.md)** — current project status, completed/remaining requirements, and known issues.

This repo is the monorepo root (D98). Workspace members: [apps/pbook/](apps/pbook/) (own CLAUDE.md and `design/` notes), [apps/ocr/](apps/ocr/), and [libs/sax-platform/](libs/sax-platform/) (the Phase 3 platform library; absorbed `libs/forge-contracts` at T3.4; `libs/sax-llm` was deleted at T3.5).

## Status & Planning

- [OVERVIEW.md](docs/OVERVIEW.md) — status-of-record: implemented capabilities, requirements complete/remaining, known issues & technical debt.
- [PHASES.md](docs/PHASES.md) — the 14-phase roadmap (1–12 + 14 done with module map; 13 deferred).
- [development-plans/TASKS.md](development-plans/TASKS.md) — live task list (completed vs uncompleted).
- [development-plans/PROCESS.md](development-plans/PROCESS.md) — how to pick up and work a task.
- [development-plans/SDLC.md](development-plans/SDLC.md) — the end-to-end development lifecycle: roles (the agent model split), the task flow from queue to production, the records-ownership map, and the standing directives.
- [development-plans/HANDOFF-2026-08-03-t10.1-start.md](development-plans/HANDOFF-2026-08-03-t10.1-start.md) — **current state**: five of T10.1's ten remaining items landed (`d77761f` verify-only worker startup at all three workers, `7b5d531` the `make db-change` generator + Squawk gate, `22a8730` the process into PROCESS.md/SDLC.md, `086f4a8` the canonical test Postgres image + the advisory-lock/CIC deadlock fix, `3812a32` the WORKERS.md/DEBUGGING.md rewrites), and the maiden change request 0001 (`b31ca4b`) was **applied to production 2026-08-03**. Read items 1–3 first: the prod deploy is now unblocked and a prod worker restart fails until it lands (`forge_prod` is at revision 004, the deployed `104c14b` chain stops at 003), pbook stays down under an owner-deferred credential rotation, and the `_test` trust-path gate has opened.
- [development-plans/HANDOFF-2026-08-02-t10.1-start.md](development-plans/HANDOFF-2026-08-02-t10.1-start.md) — D104 landed (`104c14b` — forge owns no infrastructure; `deploy/local-stack/`, the `forge-stack` agent, and the backup leg deleted) and production was deployed to it; the schema-change request process was agreed end to end with the sax-datastores operator (`a6d7768` + `15f184b`; canonical doc `sax-datastores/docs/schema-changes.md`) — no product DDL, verify-only worker startup, binding expand/contract; T10.1 resumed at the ten-item list in its 2026-08-02 Development Note.
- [development-plans/HANDOFF-2026-07-30-t10.1-start.md](development-plans/HANDOFF-2026-07-30-t10.1-start.md) — T10.1's ST3 Temporal half landed (`a7a1f0b` — `resolve_temporal_target` derives a frozen `TemporalTarget` from `FORGE_ENV`; `FORGE_TEMPORAL_NAMESPACE` deleted) along with the amendment carrying the junk-data and pbook-first-class rulings plus both datastore registrations (`6173166`) and the pbook DATA_MODEL truth fix (`a768d7b`); the dev lane cut over to the shared `:7236` server. **Superseded in part by the 2026-07-31 rebuild** — production followed onto `:7243`/`forge-prod` with its store on `:5442` (T10.1's execution note is the record); the namespace-convention contradiction it flagged has since been resolved in `sax-temporal/docs/namespaces.md`.
- [development-plans/HANDOFF-2026-07-29-t10.1-start.md](development-plans/HANDOFF-2026-07-29-t10.1-start.md) — a records-only change-set complete (`bd7b4d1` + `cfe6bc1` — CLAUDE.md's Landed narrative restructured into a phase ledger, the paired handoff-sweep Step 3(b) amendment that keeps it one, the cwd-proof markdownlint `--config` invocation, root/pbook truth fixes); production unchanged on `d6af5be`; T10.1 is still IN PROGRESS mid-ST1, with a third Plan decision (boot ownership) now superseded and the external repo's prod-placement decision landed.
- [development-plans/HANDOFF-2026-07-28-t10.1-start.md](development-plans/HANDOFF-2026-07-28-t10.1-start.md) — the T5.6 follow-up (`887bbdc` — the halt reports every attempt; failure-path replay histories 13 → 16; batch-lane preflight observed) and the DEPLOYMENT.md pre-deploy in-flight check (`47dd284`) complete; production deployed to `d6af5be`, so T5.6 and its follow-up went live; T10.1 (migrate forge onto sax-datastores, `d6af5be`) IN PROGRESS mid-ST1.
- [development-plans/HANDOFF-2026-07-27-t5.7-start.md](development-plans/HANDOFF-2026-07-27-t5.7-start.md) — T5.6 (plan preflight gate — pure recursive finders in `plan_checks.py`, the live gate at the one `dispatch_planner` seam, three capped planner attempts, the catching REVISE splice) complete as `36e1609`, with the SDLC documentation change-set (`d7a2d1f`) and the D93 banner amendment (`e0e3214`); next task is T5.7 (execution-time reference repair).
- [development-plans/HANDOFF-2026-07-27-t5.6-start.md](development-plans/HANDOFF-2026-07-27-t5.6-start.md) — T5.5 (harness rebuild + replay tests — `ScenarioState` harness, identity-keyed scripting, histories 6 → 13, the ~1/8 flake fixed at its mechanism) complete as `66b5ec2`; next task is T5.6 (plan preflight gate, owner-ruled retry policy).
- [development-plans/HANDOFF-2026-07-27-t5.5-start.md](development-plans/HANDOFF-2026-07-27-t5.5-start.md) — T5.4 (split the monolith, `7dd43f7`) complete + the premise-level plan review of the remaining queue and the first owner rulings; next task is T5.5.
- [development-plans/HANDOFF-2026-07-26-t5.4-start.md](development-plans/HANDOFF-2026-07-26-t5.4-start.md) — T5.3 (single gather + dispatch; `83a2e07` + `006e599`) complete, replay 7/7 with zero regenerations; next task is T5.4.
- [development-plans/HANDOFF-2026-07-25-t5.3-start.md](development-plans/HANDOFF-2026-07-25-t5.3-start.md) — T5.2 (single step block) complete + versioned worker identities + the D103 pinned-prod deploy model (adopted live); operating notes (prod deploys via `make prod-deploy`, step-block conventions, replay status); next task is T5.3.
- [development-plans/HANDOFF-2026-07-24-t5.2-start.md](development-plans/HANDOFF-2026-07-24-t5.2-start.md) — T5.1 (pure step logic) complete + lane-scoped worker restarts + pbook in prod; operating notes (which lane restarts how, step_logic conventions).
- [development-plans/HANDOFF-2026-07-23-phase5-start.md](development-plans/HANDOFF-2026-07-23-phase5-start.md) — Phases 0–4 + T4.4 + T0.9 + staging lane complete; operating notes (env guard, dev namespace, worker pair).
- [development-plans/HANDOFF-2026-07-21-phase5-start.md](development-plans/HANDOFF-2026-07-21-phase5-start.md) — the Phase 4 close record (T4.1–T4.3 detail; T4.4 spec adoption).
- [development-plans/HANDOFF-2026-07-19-phase4-start.md](development-plans/HANDOFF-2026-07-19-phase4-start.md) — Phases 0–3 complete; Phase 0 closeout details; the Phase 4 kickoff.
- [development-plans/HANDOFF-2026-07-18-phase3-complete.md](development-plans/HANDOFF-2026-07-18-phase3-complete.md) — the Phase 3 record: the platform library as it stands, composition roots (D93), the four-package workspace; its "What to know" section still applies.
- [development-plans/HANDOFF-2026-07-17-phase3.md](development-plans/HANDOFF-2026-07-17-phase3.md) — lead-up: Phase 2 close (root gates/CI, mypy strict) and Phase 3 T3.1–T3.3.
- [development-plans/HANDOFF-2026-07-16-monorepo-deployment.md](development-plans/HANDOFF-2026-07-16-monorepo-deployment.md) — lead-up: the monorepo consolidation (D98) and local-first deployment (D99).

## Architecture & Decisions

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — how Forge works: universal workflow step, execution modes, context assembly, batch-vs-sync, key data models, module map, and subsystems beyond the core loop.
- [DECISIONS.md](docs/DECISIONS.md) — design decisions D1–D104, with supersession/stale markers.
- [proposals/db-change-request-process.md](docs/proposals/db-change-request-process.md) — forge's proposal for schema change requests under the consumer/operator model (offline-SQL artifacts, verify-only startup, expand/contract). **Accepted and amended 2026-08-02** on sax-datastores issue #2; the canonical process is now `sax-datastores/docs/schema-changes.md`, and this file is the point-in-time proposal it came from. What it settles for forge is recorded in T10.1's 2026-08-02 Development Note.

## Requirements

- [requirements/](docs/requirements/) — Gherkin behavioral specs (18 feature files; 16 implemented, 2 specified-but-unbuilt) plus the index with capability→source mapping.

- [README.md](docs/requirements/README.md) — BDD requirements framework with Gherkin feature files, tag taxonomy, and capability mappings.
- [STANDARD.md](docs/requirements/STANDARD.md) — Autonomous-agent requirements standard: paired feature/core artifacts, static-first contracts, functional core / imperative shell split, and handoff gate.
- [TEMPLATE.md](docs/requirements/TEMPLATE.md) — Template for writing a structured `<requirement-id>.core.md` requirement sidecar.
- [REVIEW_CHECKLIST.md](docs/requirements/REVIEW_CHECKLIST.md) — Reviewer checklist for approving a requirement package before autonomous implementation.
- [examples/README.md](docs/requirements/examples/README.md) — Worked examples, including the toy `Inspira` web-app requirement package.

## Operations

- [operations/USAGE.md](docs/operations/USAGE.md) — submitting code and research tasks to Forge.
- [operations/WORKERS.md](docs/operations/WORKERS.md) — workers: where they run, startup sequence, the environment guard, schema verification (workers never migrate), the staging lane, identity, restarts and scaling. Rewritten 2026-08-03; its "Deployed state" section is dated and asks to be re-derived.
- [operations/DEPLOYMENT.md](docs/operations/DEPLOYMENT.md) — local-first deployment: launchd workers out of a commit-pinned checkout, the shared sax-datastores/sax-temporal stacks, the `FORGE_ENV` guard (D99, D102, D103, D104). Partly pre-D104 — banner at the top.
- [deploy/README.md](deploy/README.md) — deployment quick start and directory map; points into the guides below (D99).
- [deploy/launchd/README.md](deploy/launchd/README.md) — launchd worker agents: install, operate, restart, logs.
- [deploy/s3/README.md](deploy/s3/README.md) — S3 blob-bucket lifecycle policy: what it does and deliberately does not expire.
- [datastore-changes/](datastore-changes/) — committed schema-change request artifacts, one directory per request (`request.md` plus per-phase `change-<n>.sql`), generated by `make db-change` and linted by `make lint-sql` against the vendored `.squawk.toml`. **The commit is the request**: nothing under a submitted directory is ever modified. pbook's requests live under `apps/pbook/datastore-changes/`. Canonical process: `sax-datastores/docs/schema-changes.md`; forge-side duties: [development-plans/PROCESS.md](development-plans/PROCESS.md).
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

Superseded and exploratory material, **not authoritative** — see [archive/README.md](archive/README.md). Contains the original `DESIGN.md`, completed phase specs (`PHASE1`–`PHASE12`, `PHASE14`), `research/`, and the dispositioned `merged/` reports.
