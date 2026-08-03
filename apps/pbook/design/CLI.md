# CLI Reference

*Describes the target architecture adopted in the June 2026 review; implementation is phased —
see [REVIEW-2026-06.md](REVIEW-2026-06.md).*

Entry point: `pbook`. The Claude Code skill invokes it via uvx:

```text
uvx --from "git+https://github.com/stevegsax/sax.git#subdirectory=apps/pbook" pbook <cmd>
```

## Execution model

Every command except `worker`, `ingest`, `push`, and `curate` is a direct call into
`pbook.service` — no Temporal server or worker involved. These commands require only
`PBOOK_DATABASE_URL`; if it is unset, the command emits the `db_disabled` JSON envelope and
exits non-zero. `search` additionally uses `OPENAI_API_KEY` when present and degrades to
lexical + tag ranking when it is absent (see [Search](#search)) — it never hard-fails on a
missing key.

Each invocation builds a frozen `AppContext(settings, engine, embedder | None)` once at
`__main__` entry; environment variables are read only inside the `Settings` class. The four
Temporal commands additionally need a reachable Temporal server (`--temporal-address`, default
`localhost:7233`); `ingest`, `push`, and `curate` need a running `pbook worker`.

## Worker

```text
pbook worker [--temporal-address localhost:7233]
```

Start the pbook Temporal worker on `pbook-task-queue`. Verifies the schema once at
startup (verify-only — it applies no DDL; a database behind `pbk_alembic_version`'s
expected head refuses to start),
registers the two workflows — `IngestWorkflow` and `CurationWorkflow` (see
[WORKFLOWS.md](WORKFLOWS.md)) — with their activities (bound methods on frozen activity
classes built at the composition root), and idempotently creates the weekly
`pbook-curation-weekly` Schedule (already-exists ignored).

Required environment:

- `PBOOK_DATABASE_URL` — Postgres connection string.
- `ANTHROPIC_API_KEY` — extraction, validation judge, and consolidation calls.
- `OPENAI_API_KEY` — embeddings (`text-embedding-3-small`).

A missing or invalid key is classified non-retryable (`classify_llm_error` from
`sax_platform.temporal`), so a misconfigured worker fails the workflow and the ingestion
session records `error` instead of hanging at `running`.

## Database

```text
pbook migrate
```

Apply Alembic migrations (version table `pbk_alembic_version`). The migration preflights
pgvector ≥ 0.7 (halfvec) and ≥ 0.8 (iterative scans) and refuses to run otherwise. Emits
`db_disabled` when `PBOOK_DATABASE_URL` is unset. This is the only command that
migrates: the worker verifies at startup and never applies DDL, and production schema
changes go through the sax-datastores change-request process
(`sax-datastores/docs/schema-changes.md`), not this command.

## Entry management

### List entries

```text
pbook list [--tag TAG]... [--type TYPE] [--project PROJECT] [--limit 20] [--json]
```

List entries newest-first. Tags use OR matching — an entry matching any specified tag is
included. `--type` filters by entry type (`pitfall`, `curated`). `--project` filters by source
project.

### Get a single entry

```text
pbook get ENTRY_ID [--json]
```

Fetch one entry by its database id. Emits `not_found` if it does not exist.

### Add an entry

```text
pbook add [--file FILE] [--needs-review] [--schema] [--json]
```

Add an entry from JSON on stdin or from `--file`. Tags are validated against the closed
namespace vocabulary — invalid tags emit `tag_invalid`. New entries insert as `active` by
default (the skill-plus-human submission is the review); `--needs-review` inserts as
`probation` instead. `--schema` prints the input JSON schema without adding anything.

Example entry file (`entry_type` is `pitfall` or `curated`):

```json
{
    "title": "Use dispose() in SQLAlchemy test fixtures",
    "content": "SQLAlchemy's create_engine caches connections by URL. Call engine.dispose() in test teardown to prevent connection leaks across tests.",
    "tags": ["lib:sqlalchemy", "domain:testing"],
    "entry_type": "curated"
}
```

### Update an entry

```text
pbook update ENTRY_ID --file FILE
```

Update fields of an existing entry. The JSON file contains only the fields to change. Tags in
the update are validated.

## Lifecycle commands

Entries move through the `probation → active → stale / rejected / superseded` state machine
(see [DATA_MODEL.md](DATA_MODEL.md)). These commands drive the human transitions; transition
legality is checked by the pure `lifecycle.transition` function.

### Approve

```text
pbook approve ENTRY_ID [--json]
```

Promote a `probation` entry to `active`. On a `stale` entry, `approve` is revalidation:
`stale → active` with `last_validated_at` refreshed.

### Reject

```text
pbook reject ENTRY_ID --reason TEXT [--json]
```

Set status to `rejected` (`rejected_by = 'human'`, reason recorded in `status_reason`). Never
deletes — `rejected` is a terminal status and the row remains for audit and eval mining.

### Invalidate

```text
pbook invalidate ENTRY_ID --reason TEXT [--json]
```

Mark an `active` entry `stale` — known wrong or outdated, but not yet rejected. Stale entries
surface in `pbook review --stale` for triage.

### Purge

```text
pbook purge ENTRY_ID
```

Admin hard-delete with a confirmation prompt. The only command that removes rows; workflows
never hard-delete. Needed when a corrected experience must be re-ingested: the entry's
`origin_hash` would otherwise block the re-insert (purge-then-reingest).

## Duplicate checking

```text
pbook check-duplicate --title TITLE [--tag TAG]...
```

Search for existing entries with similar titles (case-insensitive substring match). If tags
are provided, results are sorted by tag overlap (most overlapping first). Returns up to 10
matches.

## Search

```text
pbook search [QUERY] [--tag TAG]... [--mode create|fix] [--token-budget 5000]
             [--active-only] [--json]
```

Hybrid retrieval as a direct service call (`service.search`). Provide a query, tags, or both.
With a query and `OPENAI_API_KEY` set, the query is embedded once; candidates are the union
of three top-50 SQL rank lists — lexical (`search_tsv`), semantic (halfvec `<=>` cosine), and
tag — all filtered to status `probation`/`active` in SQL. The pure
`ranking.score_candidates` function fuses them (RRF base, k=60) with bounded factors for tag
overlap, mode alignment, feedback ratio, and probation (×0.7), then results are packed within
the token budget (default 5,000).

- `--mode` reweights ranking — it never filters. `create` boosts curated entries and
  `lang:`/`lib:`/`domain:` tag overlap; `fix` boosts pitfall entries and
  `project:`/`pattern:` overlap.
- `--active-only` excludes probation entries (replaces the old `approved_only`). By default
  probation entries are served and carry `"status": "probation"` in packed output so
  consumers see confidence.
- Results include similarity and status.
- Degraded mode: without `OPENAI_API_KEY`, ranking is lexical + tag only
  (`ORDER BY ts_rank_cd(...) DESC, tag_overlap DESC`) and the JSON envelope gains
  `"degraded": "no_embedding_key"`.
- Each search records a `pbk_retrieval_events` row and increments `retrieval_count` on served
  entries (best-effort) so feedback can be attributed later.

## Review

```text
pbook review [--stale] [--by-experience] [--limit 20] [--json]
```

Default: list the probation queue. `--stale` lists stale entries for triage instead —
revalidate with `pbook approve` or retire with `pbook reject`. `--by-experience` groups the
probation queue by source experience.

## Provenance and metadata

### Sources

```text
pbook sources ENTRY_ID [--json]
```

List an entry's provenance rows (`pbk_entry_sources`): ingestion session, experience hash,
and source context. Provenance survives consolidation (rows are reparented, never deleted).

### Session text

```text
pbook session-text SESSION_ID [--path PATH] [--json]
```

Print the transcript text for an ingested session. `--path` points at a transcript JSONL
outside the default discovery location.

### Tags

```text
pbook tags [--json]
```

Print the tag vocabulary: five closed namespaces (`lang:`, `lib:`, `domain:`, `project:`,
`pattern:`). Values are open but pattern-validated (`^[a-z0-9][a-z0-9-]{0,30}$`) and
canonicalized by `normalize_tags` (lowercase, strip, alias map — e.g. `postgresql → postgres`,
`py → python`).

### Sessions

```text
pbook sessions [--project PROJECT] [--limit 20] [--json]
```

List ingestion sessions (`pbk_ingested_sessions`) with status (`running`, `completed`,
`error`), experience/entry counts, and workflow ids.

## Feedback

```text
pbook feedback ENTRY_ID (--helpful | --harmful) [--context TEXT]
```

Record a feedback event (`pbk_feedback_events`) and bump the matching counter on the entry in
the same transaction. Idempotent per session: `UNIQUE(entry_id, session_id, polarity)`, where
the session id is the Claude Code session UUID supplied by the calling skill (a distinct
namespace from ingestion session ids). The first helpful feedback on a `probation` entry
promotes it to `active`. Feedback can demote or flag an entry but never auto-rejects it.

## Ingestion and curation (worker required)

### Push

```text
pbook push --file FILE [--temporal-address localhost:7233]
```

Start an `IngestWorkflow` with the `InlineExperiences` input variant and wait for the result
(`{entries_created, entries_rejected, failures}`). Candidates pass the extraction and
validation-judge pipeline like any transcript-derived experience.

Example experience file:

```json
{
    "project": "my-project",
    "problem": "SQLite check_same_thread error in Temporal activities",
    "resolution": "Pass check_same_thread=False to create_engine because Temporal activities run on different threads",
    "context": "Using SQLAlchemy with SQLite inside async Temporal activities"
}
```

### Ingest

```text
pbook ingest [TRANSCRIPT_PATH] [--all] [--project NAME] [--dry-run] [--force]
             [--temporal-address localhost:7233]
```

Discover session transcript JSONLs, filter out already-ingested sessions via a direct DB
read, and start one `IngestWorkflow` per session (workflow id `pbook-ingest-{session_id}`).
Transcripts never enter workflow history: the workflow receives `{path, session_id}` and the
analysis activity reads the file. The session row is owned by the workflow — its first
activity writes `running`, the error handler writes `error` — so the CLI seeds no orphan rows.
`--force` re-submits already-ingested sessions; `origin_hash` dedup still applies on insert.

### Curate

```text
pbook curate [--temporal-address localhost:7233]
```

Start an on-demand `CurationWorkflow` (the weekly run is the `pbook-curation-weekly`
Schedule). It sweeps staleness, consolidates near-duplicate clusters through the validation
gate, and flips sessions stuck `running` > 48 h to `error`; returns
`{staled, consolidated, failures}`. See [WORKFLOWS.md](WORKFLOWS.md).

## Skill support

```text
pbook skill-prompt [--operation NAME] [--json]
```

Print machine-readable command guidance for the Claude Code skill: per-command descriptions,
arguments, and examples. `--operation` narrows output to one operation.

## JSON output contract

Commands that produce data accept `--json` for machine-readable output. On a `--json` failure
path the error envelope goes to stdout (one parseable stream) and the command exits non-zero:

```json
{"error": "Entry 42 not found.", "code": "not_found"}
```

| Code | Meaning |
| --- | --- |
| `not_found` | Entry or session id does not exist |
| `validation_error` | Input JSON failed model validation |
| `tag_invalid` | Tag fails the namespace vocabulary or value pattern |
| `db_disabled` | `PBOOK_DATABASE_URL` unset — `ConfigError` mapped to this envelope |
| `config_error` | Other required configuration missing or malformed |

## Removed commands

- `prune` — superseded by the `CurationWorkflow` staleness sweep (automatic flagging, never
  deletion) plus `pbook purge` for explicit hard deletes.
