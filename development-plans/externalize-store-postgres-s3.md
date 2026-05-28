# Externalize the Forge store: Postgres backend + S3 OCR blobs + survivable writes

**Status:** PHASES A & B DONE. Phase C (survivable writes) not started.
**Last updated:** 2026-05-28
**Owner:** stevegsax

> **Resume pointer — next action:** Phases A and B are complete on SQLite.
> `FORGE_DB_URL` is the single required store config (SQLite + Postgres engines
> build; `psycopg2-binary` locked). OCR blobs live in S3: `data` columns dropped,
> `s3_key` added (migration `014`), all blob I/O encapsulated in `ocr/s3_blobs.py`;
> tests mock S3 with `moto` (session-scoped conftest fixture). Full suite: 1470
> passed; only failures are the 3 pre-existing environmental `TestExecuteCheckOcrDuplicate`
> cases (missing `test-inputs/2311.06440v1.pdf`). **Still owed for full confidence:**
> validate Alembic migrations (incl. `014`'s `batch_alter_table`) against a real
> Postgres — engine build + SQLite DDL are tested, PG DDL is not. **Next:** Phase C
> (survivable store writes; migration `015` follows `014`).

Prerequisite work for the AWS EC2 deployment in
[docs/planning/DEPLOYMENT.md](../docs/planning/DEPLOYMENT.md). Today the Forge store
is SQLite-only and OCR blobs live in `LargeBinary` columns. The target deployment
keeps the whole store in Supabase Postgres and OCR blobs in S3, leaving the EC2
instance with no durable local state and unlocking multi-host workers. **Phase C**
then makes every store write *survivable*: a DB blip pauses and retries the write
via Temporal rather than losing data or re-running an expensive LLM/OCR call.

---

## 1. Problem

Two coupled limitations block the deployment:

1. **The store is SQLite on the worker's local disk.** `get_engine()`
   (`src/forge/store.py:322`) hard-codes `sqlite:///{db_path}` and attaches a
   SQLite-only `PRAGMA journal_mode=WAL` listener. Records (interactions, runs,
   playbooks, batch_jobs, OCR) are pinned to one host, preventing multi-host
   workers.
2. **OCR image/file bytes live in `LargeBinary` columns** — `file_content_blobs.data`
   (`store.py:199`) and `ocr_images.data` (`store.py:215`). In Postgres these become
   `bytea`: storage/egress cost on Supabase and full-row loads into memory.

A third problem surfaces once the store is remote (Phase C): **store writes today
are not survivable.** They are either swallowed best-effort (lost on a blip) or
embedded in expensive activities (a retry re-runs the LLM/OCR/batch call).

### What is already portable (verified)

- All column types are `sa.Text`/`sa.String`/`sa.Integer`/`sa.Boolean`/
  `sa.LargeBinary`/`UTCDateTime` — no SQLite-only types.
- `UTCDateTime` (`store.py:58`) is a `TypeDecorator` over `sa.DateTime` storing
  tz-naive UTC; bind/result processing is dialect-agnostic → works on Postgres.
- `run_migrations` (`store.py:336`) already overrides `sqlalchemy.url`
  programmatically, so Alembic targets whatever URL we pass.
- Store access happens only in activities and CLI code (never in workflow code),
  so none of this affects Temporal determinism.

### What is NOT in scope

- **pbook's store.** It uses SQLite-specific `json_each(tags_json)` and
  `PRAGMA foreign_keys=ON`; its Postgres migration is a separate, larger effort.
  Keep pbook on local SQLite (see DEPLOYMENT.md). The forge→pbook call at
  `src/forge/cli.py:1249` (`pbook_get_engine`) is intentionally left untouched.
- **Data backfill.** This targets a *fresh* deployment (empty store). Migrating an
  existing populated SQLite DB to Postgres / existing blobs to S3 is a separate
  one-off (noted under [Risks](#8-risks--open-questions)).

---

## 2. Design decisions (made)

| Decision | Rationale |
|---|---|
| Introduce `get_store_engine() -> Engine` that resolves `FORGE_DB_URL` and builds the engine. | ~30 call sites currently repeat `db_path = get_db_path(); if db_path is None: ...; engine = get_engine(db_path)`. A single resolver shrinks churn and centralizes the sqlite-vs-postgres choice. |
| `FORGE_DB_URL` is the **single, required** store config: a `sqlite:///…` URL for dev/tests, `postgresql+psycopg2://…` for production. **Unset → hard error** (no implicit default path; `FORGE_DB_PATH` retired). | One explicit knob, no silent defaulting. Tests pass a sqlite URL; prod passes the Supabase URL. **No runtime failover — see invariant.** |
| Gate the WAL pragma on `engine.dialect.name == "sqlite"`. | WAL is meaningless/erroring on Postgres. |
| Use **sync `psycopg2`** (`postgresql+psycopg2://`), not an async driver. | The store is synchronous SQLAlchemy. psycopg2 issues no server-side prepared statements, so it tolerates Supabase's pooler; avoids the prepared-statement landmine. |
| Postgres engine: `pool_pre_ping=True`, small `pool_size`/`max_overflow`. | Supabase drops idle connections and caps total connections; pre-ping avoids stale-connection errors, small pool respects caps. |
| **Encapsulate all S3 I/O inside the store blob functions.** Keep their byte-in / byte-out signatures; only internals change (upload on write, fetch on read). The DB row stores `s3_key` instead of `data`; `get_*` returns the same dict shape with `data` populated from S3. | Leaves the ~8 OCR-activity call sites unchanged — minimal blast radius. |
| S3 auth via the default AWS credential chain (EC2 instance role). | No static keys in code or env; matches DEPLOYMENT.md security model. |
| S3 is the **only** OCR blob store. `FORGE_OCR_S3_BUCKET` unset **or** S3 unavailable → the OCR *task* fails with a clear log message (never inline-in-DB). | One blob store, no dual schema. Only OCR workflows need S3; non-OCR work is unaffected. **No runtime failover — see invariant.** |
| **Survivable writes (Phase C):** extract every store write into a dedicated, idempotent `persist_to_store` activity invoked by the workflow, with a generous-but-finite retry policy; abandon the D42 best-effort swallow. | A DB blip retries only the *cheap* write (the expensive LLM/OCR/batch call already returned to the workflow and is never re-run); nothing is silently lost; a prolonged outage fails loudly. This is the Temporal-idiomatic realization of "pause and retry later." |

### Invariant: no runtime failover (fail-fast)

Backends are selected **once at startup from env vars** — never switched at runtime
based on availability. A configured backend that is unavailable **raises**; the
store must **never** divert writes to a different store, because that fragments the
dataset. Two severities:

**Database (`FORGE_DB_URL`) — hard error.** The DB is core infrastructure.
- Unset → hard error (refuse to start; no implicit default path).
- A non-sqlite URL whose server is unreachable → hard error (the worker fails to
  start, since migrations run at boot; activities raise). Never write to SQLite.
- A `sqlite:///` URL is the dev/test configuration; no availability check applies
  (local file).
- **Phase C** turns this raise into *survivability*: the `persist_to_store` activity
  raising on an unreachable DB is exactly what makes Temporal pause-and-retry the
  write until the DB recovers (bounded by the generous retry cap).

**OCR blobs (`FORGE_OCR_S3_BUCKET`) — fail the task, not the worker.** S3 is needed
only by OCR workflows.
- Unset **or** S3 unreachable → the OCR activity raises with a clear log message;
  Temporal records the failed workflow and applies its retry policy; the worker and
  Temporal server keep running. Never store blobs in the DB.
- Non-OCR work proceeds normally without `FORGE_OCR_S3_BUCKET`.

---

## 3. Acceptance Criteria

- [ ] `FORGE_DB_URL` is required: a `sqlite:///` URL (dev/tests) or
      `postgresql+psycopg2://` URL (prod). Unset → hard error at startup.
- [ ] WAL pragma applied only for SQLite.
- [ ] Alembic migrations run cleanly on **both** SQLite and a real Postgres,
      including migration `008` (`batch_alter_table`) and the new Phase B/C migrations.
- [ ] OCR blobs always go to S3 (DB holds `s3_key` + metadata + the `ocr-image://`
      URI). `FORGE_OCR_S3_BUCKET` unset or S3 unavailable → the OCR task fails with a
      clear log message; no inline-in-DB storage; worker/Temporal unaffected.
- [ ] S3 access uses the AWS default credential chain; no static keys.
- [ ] **No runtime failover:** a configured-but-unavailable remote backend (Postgres
      or S3) raises; the store never writes to SQLite/inline as a substitute.
- [ ] **Survivable writes (C):** a transient DB failure during a write triggers a
      Temporal retry of the cheap `persist_to_store` activity — the expensive
      LLM/OCR/batch call is **not** re-run (proven by a time-skipping test: expensive
      activity ran once while persist retried).
- [ ] **Loud on prolonged outage (C):** a DB outage longer than the retry cap
      (~20 min) fails the workflow (ScheduleToClose), not hang.
- [ ] **Idempotent writes (C):** every persist is safe to re-apply; `ocr_images`,
      `interactions`, and `playbooks` use deterministic keys; the D42 best-effort
      swallow is removed for store writes.
- [ ] Full test suite passes on SQLite with coverage ≥ 85%.
- [ ] New tests cover: the Postgres engine branch (URL selection, no WAL), the
      unset-URL hard error, the S3 write/read/delete paths (mocked), the
      bucket-unset → OCR-task-fails behavior, and the survivable-write/idempotency
      paths.
- [ ] `docs/planning/DEPLOYMENT.md` env vars/behavior match the implementation.

---

## 4. Phase A — Postgres-backed store

**Goal:** the store is configured by a required `FORGE_DB_URL` (a `sqlite:///` URL
for dev/tests, Postgres for prod). Independently valuable: this alone unlocks
multi-host workers.

### Current state

- `get_db_path() -> Path | None` (`store.py:233`): `FORGE_DB_URL`-unaware; resolves
  `FORGE_DB_PATH` → `$XDG_STATE_HOME/forge/forge.db` → `~/.local/state/...`; `None`
  if `FORGE_DB_PATH==""`.
- `get_engine(db_path: Path) -> Engine` (`store.py:322`): `create_engine(sqlite:///…)`
  + WAL pragma listener.
- `run_migrations(db_path: Path)` (`store.py:336`): builds Alembic `Config`, sets
  `sqlalchemy.url = sqlite:///{db_path}`, `command.upgrade(cfg, "head")`.
- **~30 call sites** follow `get_db_path()` → null-check → `get_engine(db_path)`:
  - `cli.py`: 190, 447, 897, 1033, 1350, 1951 (line 1249 is pbook — skip)
  - `providers.py`: 340, 369
  - `activities/batch_submit.py`: 91
  - `activities/extraction.py`: 252, 327
  - `activities/batch_poll.py`: 275
  - `activities/playbook_export.py`: 59, 78
  - `activities/playbook_review.py`: 171
  - `activities/context.py`: 495
  - `ocr/activities.py`: 343, 715, 750, 846, 899, 918, 977, 1003, 1030, 1084, 1101, 1119, 1277
  - `store.py`: 388 (internal, in `persist_interaction`)
- `worker.py:_init_store()` (`worker.py:111`) calls `run_migrations(get_db_path())`
  on startup.

### Changes

**`src/forge/store.py`**
- Add `get_store_url() -> str`: returns `FORGE_DB_URL`; **raise a clear error if
  unset**. No implicit `get_db_path()` default. Retire `FORGE_DB_PATH` (a
  `sqlite:///path` URL supersedes it).
- Add `get_store_engine() -> Engine`: build the engine from the URL. If the URL is
  sqlite → attach the WAL listener; if postgres → `pool_pre_ping=True`,
  `pool_size=5`, `max_overflow=5`. Do not catch connection errors.
- Remove/retire `get_db_path()` and `get_engine(db_path)` (keep a thin internal
  sqlite helper only if it reduces churn). The store is now **mandatory** (see
  resolved disable-store note in [Risks](#8-risks--open-questions)).
- **Do not** wrap engine creation/use in a try/except that substitutes a local
  store on connection failure (no-failover invariant). Let connection errors
  propagate.
- Update `run_migrations` to accept/resolve the URL (Postgres or SQLite); keep the
  `mkdir` only for the SQLite branch.

**`src/forge/worker.py`**
- `_init_store()` resolves the URL (hard error if unset) and runs migrations against
  it; the worker must not start without a reachable store.

**Call sites (~30)**
- Replace the `get_db_path()`+null-check+`get_engine()` triple with
  `engine = get_store_engine()`.
- Mechanical; do file-by-file and run that file's tests after each.

**`pyproject.toml`**
- Add `psycopg2-binary>=2.9` to `dependencies`. Run `uv lock`.

### Sub-tasks
- [x] A1. `get_store_url()` (required; raise if unset) + `get_store_engine()` in `store.py` (WAL gated to sqlite) — added `StoreConfigError`
- [x] A2. `run_migrations` accepts a URL; Postgres + SQLite both upgrade to head
- [x] A3. `worker.py:_init_store()` uses the URL resolver (hard error if unset; logs password-redacted URL)
- [x] A4. Added `psycopg2-binary>=2.9`; `uv lock` succeeds; Postgres pool settings in `get_store_engine`. Unblocked by fixing the pbook/sax-llm source conflict (see Progress log)
- [x] A5. Migrate ~30 call sites to `get_store_engine()` (by file); added `_require_store_engine()` CLI helper
- [x] A6. Retire `FORGE_DB_PATH`; store is mandatory; updated DEBUGGING.md, DEPLOYMENT.md, playbooks.md, scripts/ocr-results.sh
- [x] A7. Rewired store-touching test files to `FORGE_DB_URL`; added autouse `forge_db_url` conftest fixture; store-disabled tests → unset-URL hard-error tests
- [x] A8. Tests written (sqlite→WAL, postgres→no-WAL [skipped pending psycopg2], unset→raises). Full suite green on SQLite **except** 3 pre-existing environmental failures + the 85% gate (see Progress log)

### Tests
- `get_store_url`/`get_store_engine`: sqlite URL → sqlite engine + WAL; postgres URL
  → postgres engine, no WAL; **unset → raises**.
- Migrations against a real/throwaway Postgres (testcontainers or a local PG):
  `upgrade head` succeeds; key tables exist; verify migration `008`.
- Rewired SQLite store tests stay green under `FORGE_DB_URL=sqlite:///…`.

### Verification
```bash
uv run pytest tests/ -k store --no-cov
# against a scratch Postgres:
FORGE_DB_URL="postgresql+psycopg2://postgres:pwd@localhost:5432/forge_test" \
  uv run python -c "from forge.store import run_migrations, get_store_url; run_migrations(get_store_url())"
```

---

## 5. Phase B — OCR blobs to S3

**Goal:** with `FORGE_OCR_S3_BUCKET` set, OCR image/file bytes live in S3; the DB
stores references. Depends on Phase A (shared store/migration touch points).

### Current state

Blob tables (`store.py`):
- `file_content_blobs`: `id, data(LargeBinary), mime_type, file_size_bytes, created_at` (migration `007`).
- `ocr_images`: `id, document_id, page_index, original_image_id, data(LargeBinary), mime_type, file_size_bytes, top_left_x/y, bottom_right_x/y, created_at` (migration `009`).

Store functions (byte-in / byte-out):
- Write: `save_file_content` (`store.py:843`), `save_ocr_image` (`store.py:887`).
- Read: `get_file_content` (`store.py:863`, returns `data`), `get_ocr_image`
  (`store.py:974`, returns `data`). `get_ocr_images` (`store.py:951`) is metadata
  only — **no S3 fetch needed**.
- Delete: `delete_file_content` (`store.py:875`), `delete_ocr_images_by_document`
  (`store.py:986`).

Callers (should stay unchanged if S3 is encapsulated):
- Write: `ocr/activities.py:236`, `:452`, `:930`; `activities/batch_poll.py:319`.
- Read: `ocr/activities.py:414`, `:618`, `:754`, `:900`.
- Delete: `ocr/activities.py:474`, `:790`.
- The `ocr-image://` resolver/export that materializes images to local files:
  `ocr/activities.py:189` (and the export flow around `:588`–`:618`).

### Changes

**Alembic migration (new, `014_ocr_blobs_to_s3.py`)**
- Add `s3_key: String, NOT NULL` to `file_content_blobs` and `ocr_images`, and
  **drop** the `data` column from both (S3 is the only blob store — no inline mode).
- Use `op.batch_alter_table(...)` (SQLite can't drop/alter columns natively; batch
  mode handles it — pattern in migration `008`). Fresh deploy: no backfill.

**`src/forge/store.py` models + functions**
- Replace `data: LargeBinary` with `s3_key: str` on both ORM classes.
- `save_file_content` / `save_ocr_image`: upload `data` to
  `s3://{bucket}/{prefix}{content_id|image_id}` and store `s3_key`. If
  `FORGE_OCR_S3_BUCKET` is unset or the upload fails → **raise** with a clear
  message (the OCR activity fails; see invariant). Never store bytes in the DB.
- `get_file_content` / `get_ocr_image`: read `s3_key`, fetch bytes from S3, return
  under the existing `data` key (preserve dict shape). S3 fetch failure → **raise**.
- `delete_file_content` / `delete_ocr_images_by_document`: also delete the S3 object(s).

> **No inline mode.** S3 is the only OCR blob store. When `FORGE_OCR_S3_BUCKET` is
> unset or S3 is unreachable, the OCR activity raises with a clear log message and
> the workflow fails (Temporal applies its retry policy); the worker keeps running.
> Tests mock S3 (e.g. `moto`) rather than storing bytes in SQLite. Only OCR
> workflows need S3 — other Forge tasks run without it.

**`src/forge/ocr/s3_blobs.py` (new)**
- Thin boto3 wrapper: `put(key, data, content_type)`, `get(key) -> bytes`,
  `delete(key)`; bucket from `FORGE_OCR_S3_BUCKET`, optional `FORGE_OCR_S3_PREFIX`;
  default credential chain; lazy client.

**`pyproject.toml`**
- Add `boto3>=1.34`. Run `uv lock`.

### Sub-tasks
- [x] B1. Migration `014`: add `s3_key` (NOT NULL), drop `data` from both blob tables (batch_alter_table; verified on SQLite)
- [x] B2. `ocr/s3_blobs.py` boto3 wrapper (`get_bucket`/`build_key`/`put`/`get`/`delete`; `S3ConfigError`; per-call client, lazy boto3 import)
- [x] B3. Update ORM models (`FileContentBlob`, `OcrImage`) — `data` → `s3_key`
- [x] B4. Rewrite `save_file_content` / `save_ocr_image` (upload to S3 first, then store `s3_key`)
- [x] B5. Rewrite `get_file_content` / `get_ocr_image` (fetch from S3, return under `data`)
- [x] B6. Update `delete_*` to remove S3 objects (query key → delete row → delete object)
- [x] B7. Confirmed OCR-activity callers + `ocr-image://` export unchanged (all use `blob["data"]` / byte-in)
- [x] B8. Added `boto3>=1.34` (+ `moto[s3]>=5.0` dev); `uv lock`
- [x] B9. Tests: `tests/test_s3_blobs.py` (moto) — write/read/delete, prefix, bucket-unset-raises; existing blob tests pass via a session-scoped moto fixture in conftest
- [x] B10. Updated DEPLOYMENT.md (no inline mode; Phase A+B marked implemented)

### Tests
- `save_*`/`get_*` with S3 (mocked via `moto`): upload called, row has `s3_key`,
  `get_*` returns original bytes.
- Bucket unset / S3 error → `save_*` (and the OCR activity) raise with a clear
  message; nothing is written to a DB `data` column (which no longer exists).
- `delete_*` removes the S3 object.
- An OCR export round-trip writes the expected image files from S3-backed rows.

### Verification
```bash
uv run pytest tests/ -k "ocr or blob" --no-cov
# live smoke (on the instance, post-deploy):
uv run forge start OcrSyncWorkflow '{"file_path":"…/sample.pdf"}' --wait
aws s3 ls s3://$FORGE_OCR_S3_BUCKET/
```

---

## 6. Phase C — Survivable store writes (pause-and-retry on DB blips)

**Goal:** a DB blip pauses and retries only the *cheap* write (the expensive
LLM/OCR/batch call is never re-run); nothing is silently lost; a prolonged
(>~20 min) outage fails the workflow loudly. **Depends on Phase A** — its
`get_store_engine()` raising on an unreachable Postgres is the retry trigger.
(Designed and refined via Ultraplan; user-approved 2026-05-27.)

### Current state

- **No dedicated persistence activity** — every write is embedded inside an activity
  that also does the expensive LLM/OCR/batch call.
- **Best-effort swallowers** (to remove): `persist_interaction` (`store.py:358`),
  `_record_submission` (`batch_submit.py:80`), `_safe_update_status` + image-store
  try/except (`batch_poll.py`), `_persist_run` (`cli.py:438`). Each catches the
  exception and lets the activity *succeed* → Temporal never retries → record lost.
- **Non-idempotency bug** — a store failure after a successful OCR/batch API call
  retries the *whole* activity and re-calls the API (`submit_ocr_batch`,
  `call_ocr_sync`; code comments "a duplicate on retry is better than a lost batch").
- **Verified:** `AssembledContext` + `LLMCallResult` already cross the workflow
  boundary (`workflows.py:531`, `:540`; `call_llm` takes context, returns result),
  so moving persistence to a workflow-invoked activity adds no major new history
  payload.

### Changes

**New `src/forge/persist_models.py`** — pure-Pydantic discriminated union
`PersistRequest` (`Field(discriminator="kind")`; kinds: `interaction`, `run`,
`batch_submission`, `batch_failure`, `batch_status`, `ocr_result`, `playbooks`) +
`PersistResult`. **No** `sqlalchemy`/`forge.store` imports → safe to import in the
workflow sandbox under `workflow.unsafe.imports_passed_through()`.

**New `src/forge/activities/persist.py`** — `persist_to_store(req: PersistRequest)
-> PersistResult`: `get_store_engine()` (raises if DB down → Temporal retries),
dispatch on `req.kind`, idempotent writes. Register in `worker.py` + export from
`activities/__init__.py`. Blob writes (`save_file_content`, `save_ocr_image`) stay
their own activities (they carry bytes/S3) — made idempotent in place, not folded in.

**Idempotency (`store.py`)** — add `insert_or_ignore(engine, table, values, *,
index_elements)` dispatching on `engine.dialect.name` to
`sqlite.insert(...).on_conflict_do_nothing` / `postgresql.insert(...)
.on_conflict_do_nothing`. Rewrite `save_run`, `record_batch_submission`,
`record_batch_failure`, `save_ocr_result`, `save_ocr_image`, `save_file_content`,
`save_interaction`, `save_playbooks` to use it; `update_batch_status` stays a plain
UPDATE. Keys: `runs.workflow_id` (UNIQUE), `batch_jobs.id` (PK),
`ocr_results.document_id` (UNIQUE). **Fix non-deterministic keys:** `ocr_images.id`
→ `uuid5(NS, f"{request_id}:{original_image_id}:{page_index}")` (`batch_poll.py:311`,
`ocr/activities.py:923`); add UNIQUE `idempotency_key` to `interactions` (built in
the workflow: `f"{workflow_id}:{role}:{step_id}:{sub_task_id}:{attempt}:{seq}"`) and
`playbooks` (`uuid5(extraction_workflow_id, title)`).

**Migration `015_interactions_idempotency_key.py`** (number **after** Phase B's
`014`) — add `idempotency_key` (UNIQUE) to `interactions` and `playbooks` via
`op.batch_alter_table`; update the `Interaction`/`Playbook` ORM models.

**Retry policy (`workflow_blocks.py`, reused at every persist call site):**
```
_PERSIST_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1), backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=60), maximum_attempts=20,
    non_retryable_error_types=["ValueError"],   # validation errors never succeed
)
# call-site options:
start_to_close_timeout=timedelta(seconds=30)        # one write attempt
schedule_to_close_timeout=timedelta(minutes=20)     # the real cap across retries
```
`schedule_to_close=20min` is the governor: backoff 1,2,4,8,16,32,60,60… fits ~18–20
attempts in 20 min, then the activity fails `ScheduleToClose` and (uncaught) fails
the workflow loudly. DB-unreachable surfaces as psycopg2 `OperationalError`/
`InterfaceError` (retryable); duplicate-key inserts never raise (absorbed by
`insert_or_ignore`).

**Re-wire call sites (strip embedded writes → workflow persists):**
- **LLM family** (`call_llm`, `call_planner`, `call_sanity_check`,
  `call_conflict_resolution`, `call_extraction_llm`): remove the trailing
  `persist_interaction(...)`. Add `persist_interaction_block(context, result, role,
  …)` in `workflow_blocks.py`; call from `generation_dispatch`,
  `conflict_resolution_dispatch`, `_plan_task`, `_run_sanity_check`,
  `ForgeExtractionWorkflow.run`. Fold `save_extraction_results` into `persist_to_store`.
- **Batch submit** (`submit_batch_request`): delete `_record_submission`; add
  `provider` to `BatchSubmitResult`; persist in `batch_submit_and_wait` after the
  activity returns.
- **Batch poll** (`poll_batch_results`): remove swallowing; idempotent status/image
  writes propagate (existing whole-loop retry then covers DB errors).
- **OCR submit** (`submit_ocr_batch`): strip `record_batch_submission`/
  `record_batch_failure`/`delete_file_content`; return the chosen `request_id`;
  `OcrSubmitWorkflow` persists the submission, *then* deletes the blob; on API
  failure the workflow persists the failure. (Fixes the double-submit-on-DB-error bug.)
- **OCR sync** (`call_ocr_sync`): API call + dedupe-keyed image store only; the
  `ocr_results` row write moves to `OcrSyncWorkflow` via `persist_to_store`.
- **OCR store/reassembly** (`store_ocr_result`, `reassemble_ocr_chunks`): already
  workflow-invoked — make their store functions idempotent and bump call-site retry
  to `_PERSIST_RETRY`. Preserve the `ocr_results`-commit-before-`SUCCEEDED` ordering.
- **Run** (`_persist_run`): remove from `cli.py`; add a final
  `persist_to_store(PersistRunRequest(workflow_id=..., task_result=result))` step to
  `ForgeTaskWorkflow.run` (also fixes that fire-and-forget submissions never persisted).

### Sub-tasks
- [ ] C0. Confirm Phase A `get_store_engine()` has landed; number the migration after Phase B (`015`)
- [ ] C1. `persist_models.py` union + `PersistResult`; `_PERSIST_RETRY` consts in `workflow_blocks.py`; add `provider` to `BatchSubmitResult`
- [ ] C2. `insert_or_ignore` + rewrite `save_*`/`record_*` idempotent; migration `015` (idempotency_key on interactions+playbooks); deterministic `uuid5` ids (ocr_images, request_id threading)
- [ ] C3. `activities/persist.py` `persist_to_store`; register in `worker.py` + `activities/__init__.py`
- [ ] C4. Remove swallowers (`persist_interaction`, `_record_submission`, `_safe_update_status` + try/excepts, `_persist_run`)
- [ ] C5. Re-wire LLM family; `persist_interaction_block`; fold `save_extraction_results`
- [ ] C6. Re-wire batch submit/poll
- [ ] C7. Re-wire OCR submit/sync/store/reassembly (deterministic image ids; blob-delete-after-persist ordering)
- [ ] C8. Run persistence into `ForgeTaskWorkflow`; remove CLI `_persist_run`
- [ ] C9. Tests (below); update ~5 tests asserting old swallow behavior; full suite ≥ 85% on SQLite

### Tests
- **Pause-and-retry (time-skipping env):** fake `persist_to_store` raises `K` times
  then succeeds; run the workflow with other activities mocked; assert it completes
  AND the expensive activity (e.g. `call_llm`) ran **exactly once** while
  `persist_to_store` ran `K+1` times (backoff time skipped instantly).
- **Prolonged outage:** fake `persist_to_store` always raises → workflow fails
  (ScheduleToClose) within the skipped 20 min — no hang.
- **Idempotency:** unit-test each `kind` against SQLite (`tmp_path` + migrations):
  double-apply → one row, second call `applied=False`. Cover `ocr_images` with the
  deterministic `uuid5` key.
- **Regression:** update the ~5 tests tied to old swallow behavior
  (`test_activity_llm.py`, `test_batch_submit.py`, `test_batch_poll.py`,
  `test_ocr_sync.py`, `test_cli.py`).

### Verification
```bash
uv run pytest tests/ -k "persist or store or workflow or ocr" --no-cov
uv run pytest                 # full suite; coverage ≥ 85% on SQLite
```

---

## 7. Cross-cutting

**New dependencies:** `psycopg2-binary` (A), `boto3` (B) → `uv lock`. Phase C adds
no new runtime dependency (only `moto` as a dev/test dep for S3/idempotency tests).

**New / changed config (reflect in DEPLOYMENT.md config table):**

| Var | Phase | Meaning |
|---|---|---|
| `FORGE_DB_URL` | A | **Required.** `sqlite:///` (dev/tests) or `postgresql+psycopg2://` (prod). Unset → hard error |
| `FORGE_OCR_S3_BUCKET` | B | Required for OCR. Unset/unavailable → OCR task fails (no inline) |
| `FORGE_OCR_S3_PREFIX` | B | Optional key prefix |
| `FORGE_DB_PATH` | A | **Retired** — superseded by a `sqlite:///` `FORGE_DB_URL` |

Phase C introduces no env vars (it adds the `persist_to_store` activity, the
`idempotency_key` columns, and the `_PERSIST_RETRY` policy).

**Docs:** DEPLOYMENT.md already references these as "prerequisite code changes";
update its [Prerequisite code changes](../docs/planning/DEPLOYMENT.md) section to
"implemented" once merged, and reconcile any naming differences.

---

## 8. Risks & open questions

- **Real Postgres needed to validate.** SQLite tests won't catch PG-specific DDL/SQL
  issues. Use `testcontainers`/`moto` or a scratch Supabase `forge_test` DB. Decide
  the test approach at A8 and record it.
- **`batch_alter_table` drop/alter on Postgres** — confirm migrations `014`/`015`
  apply on PG (batch mode → native `ALTER`).
- **Disable-store mode — RESOLVED: store is mandatory.** The old `FORGE_DB_PATH=""`
  "store disabled" feature (tested in `test_cli.py`, documented in DEBUGGING.md) is
  removed by the URL-required rule and reinforced by Phase C (the store is core
  infrastructure that workflows now depend on). Update those tests + DEBUGGING.md.
- **Existing data backfill** — out of scope here; if a populated SQLite store must
  move, write a one-off `forge` migration script (read SQLite rows → insert Postgres
  / upload blobs to S3). Track separately if needed.
- **Supabase connection caps** vs SQLAlchemy pool size × worker count — keep pool
  small; revisit if adding worker hosts.
- **Phase C — migration ordering.** Phase C's migration (`015`) must be numbered
  after Phase B's (`014`); if B is deferred, renumber.
- **Phase C — transcript payload size (R1).** Large prompts already transit history
  as `call_llm` input; the persist activity input adds a second copy. Watch
  Temporal's 2 MB/payload limit; if heavy, pass an S3 ref (Phase B). Measure first.
- **Phase C — non-deterministic ids (R6).** Any `uuid4` minted inside an activity
  that still does the write must become `uuid5` (deterministic) or be chosen by the
  workflow, else retries duplicate.
- **Phase C — `_persist_run` move is a behavior change (R7).** Fire-and-forget
  submissions now persist a run (a fix, but observable); remove the CLI write to
  avoid a double-insert and update affected tests.
- **Phase C — persistence now blocks workflow progress (R9).** One extra fast
  activity round-trip per write under a healthy DB; under a sick DB the workflow
  visibly pauses (intended). Fan-out children persist in parallel on the worker pool.

---

## 9. Definition of Done

All Acceptance Criteria boxes checked; all three phases' sub-tasks checked; full
suite green on SQLite (≥85%); migrations verified on Postgres; survivable-write and
idempotency tests pass; DEPLOYMENT.md updated; PR(s) opened against `main`
(human-gated merge per CLAUDE.md Git Strategy — each phase can merge independently).

---

## 10. Progress log

> Append a dated entry whenever you make a decision, hit a gotcha, or pause. Keep
> the **Resume pointer** at the top of this file current.

- **2026-05-27** — Plan authored from code recon (store.py, ocr/activities.py,
  migrations, sibling repos). No code written yet. Verified: ~30 engine call sites;
  `UTCDateTime` is PG-portable; `run_migrations` already URL-driven; blob S3 logic
  can be encapsulated in store functions leaving OCR-activity callers unchanged.
  pbook store deliberately excluded (SQLite `json_each`). Next: Phase A / A1.
- **2026-05-27** — Decision (user): **no runtime failover.** A configured remote
  backend (Postgres/S3) that is unavailable must raise, never silently write to a
  local store — that would fragment the dataset. SQLite/inline is strictly the
  config-time dev/test default when no remote is set. Added the no-failover
  invariant, an acceptance criterion, and fail-fast notes on the read/write paths.
- **2026-05-27** — Decision (user), refined the rules: (1) `FORGE_DB_URL` is the
  **single required** store config — a sqlite URL for tests, Supabase for prod;
  unset → hard error; `FORGE_DB_PATH` retired. (2) Remote DB unreachable → hard
  error (worker won't start). (3) **No inline blob mode** — S3 is the only OCR blob
  store; bucket unset or unreachable → fail the OCR *task* with a clear log message,
  leave the worker/Temporal running. Schema simplifies to `data`→`s3_key` (drop
  `data`, NOT NULL). Disable-store mode resolved: store is now mandatory.
- **2026-05-27** — Added **Phase C: survivable store writes** (designed, refined via
  Ultraplan, user-approved). Pattern: extract every write into a dedicated
  idempotent `persist_to_store` activity invoked by the workflow; `_PERSIST_RETRY`
  (20 attempts / 20-min schedule_to_close) rides out DB blips; the D42 best-effort
  swallow is removed; the non-idempotent OCR/batch double-call bug is fixed by
  separating the API call from the write; fire-and-forget runs now persist. User
  chose: all writes (incl. transcripts) survivable; generous-but-finite cap.
  Sequencing **A → B → C** (C depends on A's `get_store_engine()`); migration `015`
  follows B's `014`. Next: Phase A / A1.
- **2026-05-28** — Implemented Phase A. `store.py`: added `StoreConfigError`,
  `get_store_url()` (raises if `FORGE_DB_URL` unset/empty), `get_store_engine()`
  (sqlite → WAL + parent mkdir; postgres → `pool_pre_ping`, `pool_size=5`,
  `max_overflow=5`; no failover); `run_migrations(url)`; removed `get_db_path()`
  and `get_engine()`. Migrated all ~30 forge call sites to `get_store_engine()`
  (CLI inspection commands via a new `_require_store_engine()` helper that exits
  cleanly on `StoreConfigError`; best-effort sites keep swallowing). `worker.py`
  `_init_store()` resolves the URL and logs a password-redacted form via
  `make_url(...).render_as_string(hide_password=True)`. Tests: added an autouse
  `forge_db_url` conftest fixture (per-test isolated sqlite URL) so the mandatory
  store doesn't break unrelated tests; rewired test_store/test_ocr/test_ocr_sync/
  test_cli/test_worker/test_activity_llm/test_store_batch/test_store_utc_datetime/
  test_playbook_*; the OCR/sync activity tests now patch the single
  `store.get_store_engine` instead of the old `get_db_path`+`get_engine` pair;
  store-disabled tests became unset-URL hard-error tests. Full suite: **1459
  passed, 1 skipped** (postgres engine test `importorskip("psycopg2")`), only
  failures are **3 pre-existing, environmental** `TestExecuteCheckOcrDuplicate`
  cases that read `test-inputs/2311.06440v1.pdf` — a file absent from HEAD's tree;
  `execute_check_ocr_duplicate` (unchanged here) reads it before any store logic.
  Coverage **79.1%** vs **78.3%** baseline on HEAD in this env (verified by stash) —
  the 85% gate is unmet here purely because the missing `test-inputs/` fixtures
  leave OCR paths unexercised; Phase A added no regression and slightly improved it.
  ruff clean on `src/`; introduced no new lint errors (pre-existing TC002/TC003/
  SIM117 in untouched-by-Phase-A test regions remain). **A4 blocked:** `uv lock`
  (and `uv lock --check`) fail to resolve — `pbook` declares `sax-llm` via
  `git+https://…@v0.1.0` while `forge` declares it as a local editable path, and uv
  cannot unify the two sources in the `python_full_version >= '3.14' and
  sys_platform == 'win32'` split. Until that sibling-repo source conflict is fixed,
  `psycopg2-binary` can't be added (adding it without a successful relock would
  break `uv run --frozen`). Worked around the moved-git-tag re-resolution during
  development by running tests with `uv run --frozen`.
- **2026-05-28** — A4 unblocked and done. Fixed the conflict by changing **pbook**'s
  `[tool.uv.sources]` for `sax-llm` from `{ git = …@v0.1.0 }` to
  `{ path = "../sax-llm", editable = true }`, matching forge's local-editable source
  (the established sibling-repo pattern). Regenerated both lockfiles: forge's
  `uv.lock` was unchanged (it already recorded the local editable source — the
  conflict only bit on re-resolution); pbook's `uv.lock` now points sax-llm at
  `../sax-llm`. `uv lock`/`uv run` work without `--frozen` again. Added
  `psycopg2-binary>=2.9` to forge deps and re-locked (psycopg2-binary v2.9.12). The
  postgres engine test now runs (no longer `importorskip`-skipped): 1459 passed.
  Phase A complete. Remaining validation gap: Alembic migrations on a real Postgres
  (engine build is tested; PG DDL is not — needs a scratch PG/testcontainers).
- **2026-05-28** — Implemented Phase B (OCR blobs → S3). New `ocr/s3_blobs.py`
  encapsulates all S3 I/O (`get_bucket` raising `S3ConfigError` when
  `FORGE_OCR_S3_BUCKET` is unset, `build_key` applying `FORGE_OCR_S3_PREFIX`,
  `put`/`get`/`delete`; boto3 imported lazily, fresh client per call so there's no
  module-level I/O state and moto intercepts cleanly). Migration `014` adds
  `s3_key` (NOT NULL) and drops `data` on both blob tables via `batch_alter_table`
  (verified on SQLite). ORM models updated. `save_file_content`/`save_ocr_image`
  upload to S3 first then store `s3_key` (a DB failure leaves only a harmless orphan
  object, never a row without bytes); `get_*` fetch from S3 and return under the
  historical `data` key so the ~8 OCR-activity callers and the `ocr-image://` export
  are unchanged; `delete_*` query the key, delete the row, then delete the object.
  Added `boto3>=1.34` + `moto[s3]>=5.0` (dev); re-locked. Tests: new
  `tests/test_s3_blobs.py` (write/read/delete, prefix, bucket-unset-raises,
  no-row-on-failure) plus a session-scoped `moto` S3 backend + autouse
  `forge_ocr_s3` fixture in conftest so existing blob tests pass unchanged. Full
  suite: **1470 passed** (same 3 environmental PDF failures), ruff clean on `src/`
  and new tests. No inline-in-DB fallback — S3 is the only OCR blob store. Next:
  Phase C.

---

> `development-plans/` currently has only `PROCESS.md` (no `TASKS.md`/`CHANGELOG.md`
> index yet). When those are introduced, register this task per [PROCESS.md](PROCESS.md).
