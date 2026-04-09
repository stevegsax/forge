# Architectural Decisions

## Separate database

pbook uses its own SQLite database (`pbook.db`), not a shared database with any client. This means:

- Clients cannot accidentally corrupt playbook data through schema migrations or direct writes.
- The database can be backed up, migrated, or reset independently.
- Multiple projects share the same playbook store without coordination.

Database path follows XDG: `$PBOOK_DB_PATH` > `$XDG_STATE_HOME/pbook/pbook.db` > `~/.local/state/pbook/pbook.db`. Set `PBOOK_DB_PATH=""` to disable the store entirely.

## Separate Temporal queue

pbook runs on `pbook-task-queue`, a dedicated Temporal task queue with its own worker process. Clients call pbook workflows via cross-queue workflow execution. This decouples deployment — pbook can be started, stopped, or scaled independently.

## Namespaced tags with controlled vocabulary

Tags use `namespace:value` format with five namespaces:

| Namespace | Tier | Purpose |
|-----------|------|---------|
| `lang:` | General | Programming language |
| `lib:` | General | Library or framework |
| `domain:` | General | Problem domain (testing, api, database) |
| `project:` | Extracted | Source project identifier |
| `pattern:` | Extracted | Lesson type (failure-pattern, retry-pattern) |

The two tiers serve different purposes:

- **General tags** (`lang:`, `lib:`, `domain:`) are attached to curated advice. They surface whenever the technology is present, regardless of project. A `lib:sqlalchemy` entry appears in every project that uses SQLAlchemy.
- **Extracted tags** (`project:`, `pattern:`) are attached to pitfalls extracted from experience. They surface within their project context.

Tags are validated on write (CLI rejects invalid namespaces) but the controlled vocabulary is primarily enforced on the read/query side. LLM extraction may produce imperfect tags.

## Two ingestion paths

1. **Extraction path** (automated): A client pushes raw experience data (`PushExperienceInput`) describing a problem and its resolution. The extraction LLM analyzes it and produces entries tagged `needs_review=True`. These are included in query results by default.

2. **Direct submission** (manual): A human submits a `PlaybookEntry` via the CLI or Temporal workflow. The review LLM checks it for quality, accuracy, and duplication before storing it.

The extraction path exists for project-specific pitfalls. The direct path exists for curated general knowledge.

## Intent-based retrieval ranking

Retrieval accepts a `mode` parameter: `create` or `fix`.

- **`create`** (writing new code): Boosts general knowledge (`lang:`, `lib:`, `domain:`) and API doc entries. The LLM needs best practices and reference implementations.
- **`fix`** (debugging): Boosts project-specific entries (`project:`, `pattern:`) and pitfall entries. The LLM needs "this specific codebase has this specific gotcha."

Both modes return all matching entries — the mode adjusts ranking weights, not filtering. The retrieval workflow packs entries within a token budget (default 5,000 tokens), so ranking determines what fits.

## Optimistic review with fallback

LLM-extracted entries are tagged `needs_review=True` and included in query results by default. This is optimistic — extracted entries are usually useful. If unreviewed entries cause problems, consumers can pass `approved_only=True` to exclude them.

This avoids a separate staging table and prevents entries from rotting in a review queue that nobody checks.

## Extraction targets unexpected + actionable only

The extraction LLM is not a summarizer. It looks for two specific signals:

1. **Unexpected**: The default or obvious approach did not work. Multiple retries were needed, or an API behaved differently than documented.
2. **Actionable**: There is specific advice that would help someone encountering this situation for the first time.

Generic advice ("use proper error handling"), standard rules, and expected outcomes are explicitly excluded from extraction. The system prompt states: "It is better to extract NOTHING than to extract a misleading or overly generic entry."

## Pluggable LLM provider

pbook uses `sax-llm` for LLM calls but does not hardcode a provider. The worker registers a provider at startup via `pbook.llm.set_provider()`. Activities access it via `pbook.llm.get_provider()`. This allows:

- Testing with mock providers (no API calls in tests)
- Swapping providers without changing activity code
- Running without a provider for operations that don't need LLM (list, get, approve)

## Function Core / Imperative Shell

All modules follow this pattern:

- **Pure functions** build prompts, score entries, validate data. They take inputs and return outputs with no side effects. They are tested directly.
- **Imperative shell** functions perform I/O: database queries, LLM calls, Temporal activity registration. They are thin wrappers that call pure functions.
- **Testable functions** accept an injected provider, making LLM calls testable without mocking the entire call chain.

## Three content types, one table

Pitfalls, curated advice, and API doc records share a single `entries` table with an `entry_type` discriminator column. API doc records store structured data (signature, examples) as serialized JSON in the `content` field.

One table simplifies queries — retrieval doesn't need to join across tables — and the tag system handles filtering by content type when needed.
