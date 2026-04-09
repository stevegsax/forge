# pbook — Knowledge Playbook Service

pbook is a knowledge management service that stores, retrieves, and curates actionable lessons for LLM-assisted workflows. It runs as an independent Temporal worker with its own SQLite database and CLI.

## What It Does

pbook manages three types of knowledge entries:

- **Pitfalls** — specific, unexpected situations extracted from real project experience. Created automatically when a client pushes experience data through the extraction workflow.
- **Curated advice** — general best practices submitted by humans. Reviewed by an LLM for quality before storage.
- **API doc records** — library documentation with method signatures and working examples. Built incrementally as methods are encountered.

All entries are tagged with namespaced tags (`lang:python`, `lib:sqlalchemy`, `project:forge`) and retrieved via a token-budgeted query that ranks results by relevance to the caller's context.

## Quality Bar

The highest-priority design constraint: **better to miss a note than add a misleading one**.

Entries must be minimal (avoid over-constraining future decisions) and accurate (factually correct). The extraction LLM is instructed to produce nothing rather than produce something generic or misleading. The review LLM is instructed to reject rather than accept a vague or over-prescriptive entry.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Client      │     │  pbook       │     │  sax-llm     │
│  (CLI, API,  │────▶│  Worker      │────▶│  Provider    │
│   Temporal)  │     │  (Temporal)  │     │  (Anthropic) │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                     ┌──────▼───────┐
                     │  SQLite DB   │
                     │  (pbook.db)  │
                     └──────────────┘
```

- **Temporal task queue**: `pbook-task-queue` (separate from any client's queue)
- **Database**: SQLite with WAL mode, Alembic migrations, XDG-compliant path
- **LLM provider**: Uses `sax-llm` for Anthropic API calls (extraction and review only). See the sax-llm project for provider documentation.
- **CLI**: `pbook` command with subcommands for all operations

## Documents

- [OVERVIEW.md](OVERVIEW.md) — this file
- [DECISIONS.md](DECISIONS.md) — architectural decisions and rationale
- [DATA_MODEL.md](DATA_MODEL.md) — database schema, entry types, and tag system
- [WORKFLOWS.md](WORKFLOWS.md) — Temporal workflows, activities, and the extraction/review quality bar
- [CLI.md](CLI.md) — command reference
- [INTEGRATION.md](INTEGRATION.md) — how clients interact with pbook
