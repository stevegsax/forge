# Client Integration

*Describes the target architecture adopted in the June 2026 review; implementation is phased —
see [REVIEW-2026-06.md](REVIEW-2026-06.md).*

pbook has three integration surfaces: the Claude Code skill driving the CLI, the CLI itself,
and direct Python library access. There is no Temporal API surface for consumers — pbook's two
workflows ([WORKFLOWS.md](WORKFLOWS.md)) are internal durable jobs, not an RPC layer.

```text
┌─────────────────────┐
│  Claude Code skill  │
│  (skill-pbook)      │
└──────────┬──────────┘
           │ shells out (uvx, --json)
           ▼
┌─────────────────────┐      ┌────────────────────────────┐      ┌────────────────────┐
│      pbook CLI      │ ───▶ │  pbook.service             │ ───▶ │  Postgres          │
└─────────────────────┘      │  synchronous functions     │      │  (pbook schema)    │
┌─────────────────────┐ ───▶ │  over a frozen AppContext  │      └────────────────────┘
│  Python consumers   │      └────────────────────────────┘
└─────────────────────┘
```

## Claude Code skill

The skill exists in the skill-pbook repo. It drives the CLI via uvx from the `sax` monorepo and
parses the `--json` envelope:

```bash
uvx --from "git+https://github.com/stevegsax/sax.git#subdirectory=apps/pbook" pbook search \
    --query "connection pooling" --tag lang:python --json
```

The skill bootstraps its server-provided instructions via `pbook skill-prompt`. Contract points
it (and any other envelope consumer) relies on:

- Every command supports `--json` with a stable envelope. Errors carry one of the codes
  `not_found`, `validation_error`, `tag_invalid`, `db_disabled`, `config_error` — never a stack
  trace. A missing `PBOOK_DATABASE_URL` surfaces as the `db_disabled` envelope.
- Search without `OPENAI_API_KEY` degrades to lexical + tag ranking and the envelope gains
  `"degraded": "no_embedding_key"`. Search never hard-fails on a missing key.
- Packed entries that are still in probation carry `"status": "probation"` so the consumer can
  weight its trust; `--active-only` excludes them entirely (this replaces the removed
  `approved_only`).
- Feedback is recorded only after the human confirms it; the skill supplies its Claude Code
  session UUID with the feedback command (see the feedback contract below).

## CLI

The CLI calls `pbook.service` directly — CRUD, retrieval, review, and export need only a
database connection, not a worker. Only `worker`, `ingest`, `push`, and `curate` talk to
Temporal. See [CLI.md](CLI.md) for the full reference. Key commands for integration:

```bash
pbook add --file entry.json            # direct write; default status active
pbook search --query "..." --json     # hybrid retrieval (lexical + semantic + tags)
pbook list --tag lib:sqlalchemy --json
pbook review                           # probation triage queue
pbook approve 42                       # probation -> active (revalidates a stale entry)
pbook feedback 42 --helpful            # idempotent per session
pbook ingest                           # worker required: one IngestWorkflow per session
```

## Python library access

pbook is library-first: every CLI verb is a synchronous function in `pbook/service.py`
(`search`, `list_entries`, `get_entry`, `add_entry`, `update_entry`, `approve_entry`,
`reject_entry`, `invalidate_entry`, `purge_entry`, `record_feedback`, `review_queue`,
`list_sources`, `list_tags`, `list_sessions`, `session_text`, `check_duplicate`,
`export_entries`). All take a frozen `AppContext(settings, engine, embedder | None)` built once
at the consumer's entrypoint — the composition root. There are no module-level singletons to
register or monkeypatch.

```python
from openai import OpenAI

from pbook import service
from pbook.config import Settings
from pbook.context import AppContext
from pbook.models import RetrievalMode
from sax_platform.db import create_engine
from sax_platform.embeddings import OpenAIEmbeddings

# Composition root: build once at the entrypoint, pass inward.
settings = Settings()  # the only place env is read; missing PBOOK_DATABASE_URL -> ConfigError
engine = create_engine(settings)
embedder = (
    OpenAIEmbeddings(OpenAI(api_key=settings.openai_api_key), model="text-embedding-3-small")
    if settings.openai_api_key
    else None  # search degrades to lexical + tag ranking
)
ctx = AppContext(settings=settings, engine=engine, embedder=embedder)

result = service.search(
    ctx,
    query="connection pooling with pgbouncer",
    tags=["lang:python", "lib:sqlalchemy"],
    mode=RetrievalMode.CREATE,  # or RetrievalMode.FIX ("create"/"fix" on the CLI)
    token_budget=5000,
)
```

### Pure helpers

The functional core is importable without an `AppContext` or a database:

```python
from pbook import tags

query_tags = tags.normalize_tags(["lang:PY", " lib:PostgreSQL "])
# -> ["lang:python", "lib:postgres"]  (lowercase, strip, alias map)

hints = tags.infer_tags_from_context(file_extensions=[".py"], description="fix migration test")
# -> ["domain:bug-fix", "domain:migration", "domain:testing", "lang:python"]
```

`ranking.score_candidates` (with its frozen `ScoringWeights`) is the only ranking
implementation — the pure seam used by `service.search`, by tests, and by any consumer that
wants to re-rank its own candidate set. Lifecycle logic (`lifecycle.transition`,
`lifecycle.compute_staleness`) is equally pure and importable.

## Consuming retrieval results

- `mode` reweights ranking, never filters: `CREATE` boosts curated entries and
  `lang:`/`lib:`/`domain:` tag overlap; `FIX` boosts pitfall entries and
  `project:`/`pattern:` overlap.
- Results are packed within a token budget (default 5,000) and include similarity and status.
- Probation entries are served by default with `"status": "probation"`; pass `--active-only`
  (CLI) or `active_only=True` (library) to exclude them.
- Each search records a `pbk_retrieval_events` row and bumps `retrieval_count` (best-effort);
  this feeds the feedback ratio and dead-weight staleness trigger, so served-but-never-helpful
  entries decay. See [DATA_MODEL.md](DATA_MODEL.md).

## Feedback contract

`pbook feedback ENTRY_ID --helpful|--harmful [--context]` (or `service.record_feedback`) writes
a `pbk_feedback_events` row and bumps the entry counters in the same transaction.

- **Session-id idempotency.** Events are unique per `(entry_id, session_id, polarity)`; the
  `session_id` is the Claude Code session UUID supplied by the skill — a distinct namespace
  from ingestion session ids. Re-running the command in the same session is a no-op.
- **Human confirmation.** The skill confirms with the human before recording; feedback is an
  explicit human-confirmed signal, not an inferred one.
- **Bounded effect.** The first helpful feedback promotes a probation entry to active; harmful
  signals contribute to staleness triggers. Feedback can demote or flag, but never auto-rejects
  and never deletes.

## No cross-queue Temporal integration

There are no cross-queue Temporal calls in either direction: forge does not call pbook, and
pbook does not submit to forge's queue. Ingestion is pbook-owned end-to-end — `pbook ingest`
discovers session transcripts and starts one `IngestWorkflow` per session on
`pbook-task-queue`. Services wanting playbook context use the CLI or the library; they do not
need a Temporal client at all.

If a cross-service trigger is ever genuinely needed, the documented escape hatch is
[TEMPORAL_PATTERNS.md](TEMPORAL_PATTERNS.md) rule 9: start a workflow by string name with wire
models pinned in the platform contracts module — never call another service's activities or
child workflows cross-queue.

## Forge consumption: read-only SQL view

**Added 2026-06-10 (merged platform plan).** Forge consumes pbook knowledge through a
read-only view contract, not the CLI, the library, or Temporal:

- pbook publishes a `knowledge.approved_entries` SQL view exposing the non-vector entry
  columns plus the `search_tsv` generated tsvector column.
- forge consumes it via `sax_platform.contracts.knowledge`: a read-only SQLAlchemy `Table`, a
  frozen `KnowledgeEntry` model, and a query helper.
- forge's retrieval is the union of a lexical rank list (`websearch_to_tsquery` +
  `ts_rank_cd`) and tag-overlap candidates, fused by one small pure scoring function in which
  tags boost but never gate, then sliced to a token budget. Active-only entries are served.
- There are **no embeddings and no OpenAI dependency on forge's hot path** — the read is
  deterministic SQL plus pure scoring.
- forge has an explicit degraded mode: with `knowledge_db_url` unset it returns empty results
  and logs, never hard-fails.
- pbook owns a schema-sync test asserting the view shape after every pbook migration, so a
  migration cannot silently break the contract.

pbook's full hybrid RRF retrieval (lexical + semantic + tags) remains pbook's own surface for
skill, CLI, and library consumers; the view contract is forge's only path.

## Environment requirements

Environment variables are read only inside the `Settings` class.

| Consumer | Required environment |
| --- | --- |
| Skill / CLI — CRUD, review, lexical search | `PBOOK_DATABASE_URL` |
| Skill / CLI — semantic search | `PBOOK_DATABASE_URL` + `OPENAI_API_KEY` (absent → degraded lexical mode) |
| Python library | `PBOOK_DATABASE_URL`; `OPENAI_API_KEY` only if wiring an embedder into `AppContext` |
| Worker (`pbook worker`, `ingest`, `push`, `curate`) | `PBOOK_DATABASE_URL` + `OPENAI_API_KEY` + `ANTHROPIC_API_KEY` |

Worker-backed commands additionally need a reachable Temporal server (default
`localhost:7233`).
