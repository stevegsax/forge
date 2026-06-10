# Architectural Decisions

*Describes the target architecture adopted in the June 2026 review; implementation is phased —
see [REVIEW-2026-06.md](REVIEW-2026-06.md).*

## Dedicated schema in the shared Postgres

pbook stores its data in a dedicated `pbook` schema (all tables/indexes prefixed `pbk_`) inside
the Supabase project's PostgreSQL database, with its own Alembic version table
(`pbk_alembic_version`). It does not own a separate database: a separate database within a
Supabase project is reachable only over the direct 5432 connection and is invisible to the
Supabase tooling (MCP, pooler, dashboard, advisors), whereas a dedicated schema keeps all of
that working while still isolating pbook. This means:

- Another tenant of the database cannot collide with pbook's tables or its migration chain.
- The schema can be migrated or reset independently of any client's tables.
- Multiple projects share the same playbook store without coordination.

The connection string comes from `PBOOK_DATABASE_URL` (a `postgresql://` /
`postgresql+psycopg://` URL; bare URLs are normalized to the psycopg v3 driver). It is
required: a missing value raises `ConfigError`, which the CLI maps to the `db_disabled` JSON
envelope. There is no SQLite fallback. Use the direct 5432 connection for migrations; if
connecting through Supabase's transaction-mode pooler (6543), psycopg prepared statements are
disabled automatically.

## One `sax` monorepo with a `sax-platform` library

pbook lives at `apps/pbook` in a single `sax` uv-workspace monorepo alongside `apps/ocr`,
`apps/forge`, and `libs/sax-platform`. The sax-llm and forge-contracts repos are absorbed into
the platform library and retired. `sax_platform` owns exactly the concerns that were previously
implemented two or three times across repos:

- `llm/` — `AnthropicLLM` built on first-class structured outputs (`client.messages.parse`),
  `MistralOcr`, and one `tiers.py` model registry.
- `embeddings/` — `Embedder` protocol plus `OpenAIEmbeddings`.
- `temporal/` — connect and `run_worker` scaffolding, retry presets, `classify_llm_error`,
  heartbeat helpers.
- `db/` — engine factory, `run_migrations`, `insert_or_ignore`.
- `config.py` (frozen pydantic-settings; the only place env is read), `logging.py`, and
  `contracts/`.

One `.python-version`, one `uv.lock`, one CI workflow; cross-cutting changes land as a single
atomic commit. The skill invokes pbook via
`uvx --from "git+https://github.com/stevegsax/sax.git#subdirectory=apps/pbook" pbook <cmd>`.
See [REVIEW-2026-06.md](REVIEW-2026-06.md).

## Temporal for durable jobs only

Exactly two workflows exist, both on `pbook-task-queue`: `IngestWorkflow` (per-session
transcript ingestion) and `CurationWorkflow` (weekly staleness sweep plus consolidation). They
are the only jobs that earn Temporal's keep: multi-minute, multi-step, with partial progress
worth preserving across a crash. Everything else — CRUD, retrieval, export — is a synchronous
function in `pbook/service.py`; Postgres is already the durable store, and a failed read is
just rerun. Workflow-as-RPC (one-activity wrapper workflows) is banned, and the agent hot path
(`pbook search`) never touches Temporal.

There are no cross-queue calls in either direction: forge does not call pbook workflows and
pbook does not submit to forge's queue. Ingestion is pbook-owned end-to-end. House rules live
in [TEMPORAL_PATTERNS.md](TEMPORAL_PATTERNS.md). See [REVIEW-2026-06.md](REVIEW-2026-06.md).

## Dependencies injected at the composition root

There are no module-level singletons and no string-keyed registries. Each entrypoint builds
its dependencies exactly once:

- The CLI builds a frozen `AppContext(settings, engine, embedder | None)` at `__main__` entry
  and passes it to the `pbook.service` functions.
- The worker constructs frozen activity classes with their dependencies (LLM client, engine,
  embedder) and registers their bound methods — Temporal's sanctioned DI.

Env is read only inside the frozen pydantic-settings `Settings` class. LLM calls go through
`sax_platform`'s `AnthropicLLM` on first-class structured outputs (`client.messages.parse`);
the output class appears in the activity signature, so mypy checks the contract and there is
no name registry to drift. Tests construct the same classes with fakes — no monkeypatching of
module globals. See [REVIEW-2026-06.md](REVIEW-2026-06.md).

## Namespaced tags with controlled vocabulary

Tags use `namespace:value` format with five closed namespaces:

| Namespace | Tier | Purpose |
| --- | --- | --- |
| `lang:` | General | Programming language |
| `lib:` | General | Library or framework |
| `domain:` | General | Problem domain (testing, api, database) |
| `project:` | Extracted | Source project identifier |
| `pattern:` | Extracted | Lesson type (failure-pattern, retry-pattern) |

The two tiers serve different purposes:

- **General tags** (`lang:`, `lib:`, `domain:`) attach to curated advice and surface wherever
  the technology is present, regardless of project.
- **Extracted tags** (`project:`, `pattern:`) attach to pitfalls extracted from experience and
  surface within their project context.

Namespaces are closed; values are open but pattern-validated (`^[a-z0-9][a-z0-9-]{0,30}$`) and
canonicalized by the pure `normalize_tags` (lowercase, strip, alias map — e.g.
`postgresql → postgres`, `py → python`). LLM-produced tags are normalized, never save-failing:
an invalid namespace is dropped and logged rather than rejecting the entry.

Tags are demoted from retrieval gate to capped ranking boost (plus list/triage filtering).
They never gate retrieval — zero-tag-overlap entries still surface via the lexical and
semantic candidate lists. See [REVIEW-2026-06.md](REVIEW-2026-06.md).

## Two content types, one table

`pitfall` (extracted) and `curated` (human-submitted) entries share the single `pbk_entries`
table with an `entry_type` discriminator. One table keeps retrieval a single query — no joins
across content-type tables — and tags handle filtering when needed. The former `api_doc` type
is removed: serialized-JSON-in-content was a poor fit for a structured record and had no
consumer.

## Two ingestion paths

1. **Extraction path** (automated): `IngestWorkflow` takes a union input —
   `TranscriptSource(path, session_id, project)` for session transcripts read inside the first
   activity, or `InlineExperiences(experiences, project)` pushed directly. Extraction →
   validation judge → save; surviving entries enter `probation`. pbook owns this path
   end-to-end; no other service's workflows are involved.
2. **Direct submission** (manual): `pbook add` inserts a curated entry as `active` — the skill
   plus the human operating it *is* the review, so a second LLM review gate would add latency
   without signal. `--needs-review` opts into `probation` instead.

The extraction path exists for project-specific pitfalls; the direct path for curated general
knowledge. See [REVIEW-2026-06.md](REVIEW-2026-06.md).

## Deterministic idempotency

Every durable write is keyed by a retry-safe natural key: each extracted entry carries
`origin_hash = sha256(session_id + experience_hash + normalized_title)` (UNIQUE; NULL for
manual and legacy entries) and inserts with `ON CONFLICT (origin_hash) DO NOTHING`, so an
activity retry can never duplicate an entry. The cosine thresholds (0.85 entry-match, 0.92
source-dedup) are demoted to cross-session match-or-attach *policy* — deciding whether a new
experience attaches to an existing entry — never retry protection, because embeddings drift.
Documented limitation: re-ingesting a corrected experience requires purge-then-reingest.
See [REVIEW-2026-06.md](REVIEW-2026-06.md).

## Entry lifecycle

`status ∈ {probation, active, stale, rejected, superseded}` replaces the old
`needs_review`/`rejected` boolean pair and the CLI hard-delete. All transitions go through one
pure function, `lifecycle.transition(current, event)`, with exhaustive `match` +
`assert_never`; illegal transitions return `IllegalTransition` instead of constructing illegal
states.

- Extracted entries start in `probation`; human approval or the first helpful feedback
  promotes them to `active`. Probation entries are served by default, annotated
  `"status": "probation"` and penalized ×0.7 in ranking; `--active-only` excludes them
  (replacing the old `approved_only`).
- Staleness (age, harmful signal, dead weight — pure `compute_staleness` over a frozen
  `StalenessConfig`) and `pbook invalidate` move entries to `stale` for triage; they are
  flagged, never deleted.
- Consolidation marks parents `superseded` (with `superseded_by_id`) and inserts the survivor
  in `probation` with zeroed counters — merged knowledge re-earns trust.
- Nothing is ever hard-deleted by a workflow; `pbook purge` is an explicit admin command.

This keeps the old optimistic-review benefit — extracted entries visible by default, no
staging table to rot in — while making confidence explicit and preserving audit history.
See [REVIEW-2026-06.md](REVIEW-2026-06.md).

## Intent-based ranking over hybrid retrieval

Candidates are the union of three top-50 SQL rank lists — lexical (`search_tsv` /
`websearch_to_tsquery`), semantic (`embedding <=>` query vector), and tag overlap — all
status-filtered in SQL. One pure function, `ranking.score_candidates` with a frozen
`ScoringWeights`, does all scoring: a reciprocal-rank-fusion base (k=60) over the lexical and
semantic rank lists, multiplied by bounded factors — capped tag overlap, mode alignment,
feedback ratio (gated at ≥ 3 retrievals), probation ×0.7.

Retrieval accepts a `mode`:

- **`create`** (writing new code) boosts curated entries and `lang:`/`lib:`/`domain:` overlap.
- **`fix`** (debugging) boosts pitfall entries and `project:`/`pattern:` overlap.

Mode adjusts ranking weights only — it never filters. Ranked candidates are packed within a
token budget (default 5,000), so ranking determines what fits. Without `OPENAI_API_KEY`,
search degrades to lexical + tag ordering and flags `"degraded": "no_embedding_key"` rather
than hard-failing. Hybrid fusion replaces the old hierarchical cosine-then-tags ranking;
pure-vector retrieval is a documented dead end for agent memory.
See [REVIEW-2026-06.md](REVIEW-2026-06.md).

## Feedback is explicit, human-confirmed events

Two event tables record the loop: `pbk_retrieval_events` (query, tags, mode, served entry ids,
client, session) and `pbk_feedback_events` (entry, polarity, source, context, session) with
`UNIQUE(entry_id, session_id, polarity)` for per-session idempotency — `session_id` here is
the Claude Code session UUID, a distinct namespace from ingestion session ids. The counters on
`pbk_entries` remain as cheap scoring aggregates, bumped in the same transaction as the event
insert.

Attribution policy, stated honestly for a one-developer shop: explicit human-confirmed
feedback plus served-but-never-helpful decay; no synthetic outcome pipeline inferring success
from downstream behavior. Poisoning bounds: per-session idempotency, human confirmation before
recording, and feedback can demote or flag an entry but never auto-reject it.
See [REVIEW-2026-06.md](REVIEW-2026-06.md).

## Extraction targets unexpected + actionable only

The extraction LLM is not a summarizer. It looks for two signals:

1. **Unexpected**: the default or obvious approach did not work — retries were needed, or an
   API behaved differently than documented.
2. **Actionable**: there is specific advice that would help someone hitting this situation for
   the first time.

Generic advice ("use proper error handling"), standard rules, and expected outcomes are
excluded. The system prompt states: "It is better to extract NOTHING than to extract a
misleading or overly generic entry."

This bar is no longer prompt text alone. A validation judge (CLASSIFICATION tier, one call per
experience batching its candidates) applies four binary checks — grounded-in-resolution,
specific, non-generic, actionable — and failures are saved as `rejected_by='validator'` rows,
auditable and minable as eval goldens. The eval suites then pin the bar as failing tests.
See [REVIEW-2026-06.md](REVIEW-2026-06.md).

## Evals gate prompts and model pins

Hand-rolled pytest suites under `tests/evals/` (run via `make evals`, excluded from the
default run) are the regression gate for every prompt or model change:

- Extraction goldens — including must-extract-zero negatives, gated at 100%.
- Judge calibration — hand-graded cases, gated at ≥ 85% agreement (100% on generic-advice
  traps).
- Retrieval goldens — frozen pre-embedded corpus, gated at recall@5 ≥ 0.8 and MRR ≥ 0.6.

Model pins move only on green paired-delta runs (old vs. new, with the delta table in the
commit message). This is a regression gate, not a tuning loop.
See [REVIEW-2026-06.md](REVIEW-2026-06.md).

## halfvec embeddings with model metadata

Embeddings are stored as `HALFVEC(1536)` (half precision halves storage at ~99% recall) with
an HNSW index using `halfvec_cosine_ops`, and every row carries `embedding_model` /
`embedding_dim` columns so a future model change is data, not archaeology. The model stays
`text-embedding-3-small`; it is re-evaluated only on retrieval-eval evidence, and an
embedding-model change additionally re-embeds the frozen eval corpus. The migration preflights
pgvector ≥ 0.7 (halfvec) and ≥ 0.8 (iterative scans) and refuses to run otherwise.
See [REVIEW-2026-06.md](REVIEW-2026-06.md).

## Considered and rejected

- **Anthropic Message Batches** for extraction: rejected at current volume — batch latency
  buys nothing at a handful of sessions per day. Revisit above ~500 sessions/month.
- **Temporal Nexus** for cross-service calls: rejected — with ingestion inverted to pbook and
  the forge coupling deleted, there is nothing left to mediate. Promote only if a second
  consumer or a namespace split appears.

See [REVIEW-2026-06.md](REVIEW-2026-06.md).

## Function Core / Imperative Shell

All modules separate pure logic from I/O:

- **Pure core**: `ranking.score_candidates`, `lifecycle.transition`, `compute_staleness`,
  `normalize_tags`, packing and prompt builders. Inputs to outputs, no side effects; tested by
  direct import — never through a workflow or a database fixture.
- **Imperative shell**: the `pbook/service.py` functions, store I/O, activity methods, and the
  worker. Thin wrappers that fetch, call the core, and persist.

Dependencies enter by parameter (see the composition-root decision above), so the shell is
testable with fakes and the core needs no test doubles at all.
