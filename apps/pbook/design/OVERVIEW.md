# pbook — Knowledge Playbook Service

*Describes the target architecture adopted in the June 2026 review; implementation is phased —
see [REVIEW-2026-06.md](REVIEW-2026-06.md).*

pbook is a knowledge management service that stores, retrieves, and curates actionable lessons
for LLM-assisted workflows: a Python library and CLI over a dedicated PostgreSQL schema, plus a
Temporal worker for the two durable jobs (ingestion, curation). It lives in the `sax`
uv-workspace monorepo as `apps/pbook`, with shared plumbing in `libs/sax-platform`.

## What It Does

pbook manages two types of knowledge entries:

- **Pitfalls** — specific, unexpected situations extracted from real project experience by the
  `IngestWorkflow`, validated by an LLM judge, and stored on probation until confirmed useful.
- **Curated advice** — general best practices submitted by humans via `pbook add`; the human
  submission is the review, so entries enter `active` directly (probation with `--needs-review`).

All entries carry namespaced tags (`lang:python`, `lib:sqlalchemy`, `project:forge`). Entries
move through an explicit lifecycle (`probation → active → stale / superseded / rejected`) and
are never hard-deleted except by the explicit `pbook purge` admin command. Retrieval fuses
lexical, semantic, and tag signals in one pure RRF-based scoring function and packs ranked
results within a token budget.

## Quality Bar

The highest-priority design constraint: **better to miss a note than add a misleading one**.

Entries must be minimal (avoid over-constraining future decisions) and accurate (factually
correct). The extraction LLM is instructed to produce nothing rather than something generic or
misleading; a validation judge rejects ungrounded, vague, or generic candidates before storage.
Eval suites (extraction goldens, judge calibration, retrieval goldens) make the bar testable.

## Architecture

pbook is library-first: CRUD, retrieval, and export are synchronous calls into
`pbook/service.py` over a frozen `AppContext` built once at process entry — no module-level
singletons. Only the durable jobs run as Temporal workflows.

```text
┌──────────────────────┐      ┌──────────────────────────┐
│ Claude Code skill    │      │ Temporal worker          │
│ CLI (uvx pbook)      │      │ (pbook-task-queue)       │
│ Python library       │      │ ├─ IngestWorkflow        │
└──────────┬───────────┘      │ └─ CurationWorkflow      │
           │                  └────────┬─────────┬───────┘
           ▼                           │         ▼
┌──────────────────────┐               │   ┌─────────────────┐
│ pbook.service        │               │   │ Anthropic (LLM) │
│ (sync functions over │               │   │ OpenAI (embed)  │
│  frozen AppContext)  │               │   └─────────────────┘
└──────────┬───────────┘               │
           ▼                           ▼
   ┌────────────────────────────────────────────┐
   │ PostgreSQL — pbook schema, pbk_ tables     │
   │ (halfvec embeddings + tsvector lexical)    │
   └────────────────────────────────────────────┘
```

- **Monorepo**: `apps/pbook` in the `sax` uv-workspace; LLM client, embeddings, Temporal
  scaffolding, DB engine factory, and config live in `libs/sax-platform`.
- **Database**: PostgreSQL (Supabase), `pbook` schema with `pbk_`-prefixed tables, Alembic
  migrations; configured via `PBOOK_DATABASE_URL`.
- **Temporal**: one task queue, `pbook-task-queue`, two workflows — `IngestWorkflow` and
  `CurationWorkflow`; house rules in [TEMPORAL_PATTERNS.md](TEMPORAL_PATTERNS.md).
- **CLI**: `pbook` subcommands; most run without the worker (direct service calls). The skill
  invokes it via `uvx --from "git+https://github.com/stevegsax/sax.git#subdirectory=apps/pbook"`.

## Documents

- [OVERVIEW.md](OVERVIEW.md) — this file
- [DECISIONS.md](DECISIONS.md) — architectural decisions and rationale
- [DATA_MODEL.md](DATA_MODEL.md) — database schema, entry types, and tag system
- [WORKFLOWS.md](WORKFLOWS.md) — Temporal workflows, activities, and the extract/judge pipeline
- [CLI.md](CLI.md) — command reference
- [INTEGRATION.md](INTEGRATION.md) — how clients interact with pbook
- [REVIEW-2026-06.md](REVIEW-2026-06.md) — June 2026 architecture review and migration plan
- [TEMPORAL_PATTERNS.md](TEMPORAL_PATTERNS.md) — house rules for SAX Temporal services
