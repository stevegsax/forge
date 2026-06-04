# Separate OCR into its own repo (Forge = Temporal platform, OCR = consumer)

**Status:** NOT STARTED (plan only)
**Last updated:** 2026-06-04
**Owner:** stevegsax

> **Resume pointer — next action:** Nothing implemented yet. This plan is the
> decision basis turned into an ordered, cross-repo build sequence. Start at
> **Phase 0** (`forge-contracts` foundation) — specifically extract `s3_blobs`
> first, because `forge/store.py` imports `forge.ocr.s3_blobs` (a layering
> inversion) and that cycle must break before anything else moves. The settled
> design lives in [`grill-me-sessions/separate-ocr-modules.grill.md`](../grill-me-sessions/separate-ocr-modules.grill.md);
> the current-state inventory that grounds the file/line references here was
> produced by workflow `w98hu9ev1` (5-slice inventory) and `wkwh24osi` (the
> adversarial B1–B6 sweep). **Do not re-open the locked decisions** — this plan
> implements them.

## 1. Problem

OCR is entangled into Forge across data, control flow, and packaging. The goal is
to make **Forge a generic Temporal-based platform** (LLM connectivity, a generic
batch service, observability store) and extract **OCR into its own repository**
that consumes the platform as a customer — with a shared `forge-contracts` package
as the only thing both import.

Three repos are in play (all exist locally):

- `/Users/stevengreenberg/repos-sax/forge` — the platform (this repo).
- `/Users/stevengreenberg/repos-sax/forge-contracts` — NEW shared package, currently empty (`.git`/`.envrc`/`.gitignore` only).
- `/Users/stevengreenberg/repos-sax/ocr` — NEW OCR app repo, currently empty (`.gitignore` only).

### Current entanglement (verified)

- **Dependency cycle:** `forge/store.py` imports `forge.ocr.s3_blobs` at 6 sites
  (`store.py:897,921,937,951,1069,...`); `src/forge/ocr/*` imports `forge.store`
  (~25 sites), `forge.models`, `forge.persist_models`, `forge.workflow_blocks`.
- **Platform poller is OCR-contaminated:** `activities/batch_poll.py:160-170,270-305`
  decodes images, writes `ocr_images` via `save_ocr_image`/`ocr_image_id`, and
  injects a private `_image_mapping` into `raw_response_json`.
- **Shared store/migrations:** one `Base` (`store.py:96`), one linear Alembic chain
  (001→015, head 015), `ocr_results`(006)/`file_content_blobs`(007)/`ocr_images`(009)
  interleaved with platform tables; OCR columns grafted onto generic `batch_jobs`
  (`file_path` in 011, `document_id` in 013).
- **Hard-wired worker:** `worker.py:79-104,260-267,311-325` imports/registers all
  OCR workflows + activities (no guard), including `OcrSyncWorkflow`.
- **OCR CLI in platform:** `cli.py:1841` (`ocr-jobs`), `cli.py:1891` (`backfill-hashes`).

### Not in scope

- Re-opening any locked decision (batch stays; same namespace; same DB; etc.).
- Unifying Forge's *own* LLM batch builder with the new opaque-blob submit SPI
  (the generic path keeps its internal builder; only the cross-queue SPI is new).
- Temporal Nexus / separate namespace / separate DB (explicitly rejected).
- Production deploy topology + the deferred two-worker DB connection-pressure
  question (revisit pre-production).

## 2. Design decisions (made)

The full DECIDED log + adversarial blockers (B1–B6) + the **SOUNDNESS INVARIANT**
are in [`separate-ocr-modules.grill.md`](../grill-me-sessions/separate-ocr-modules.grill.md).
Compact restatement of what this plan implements:

- OCR = own repo, own worker on **`ocr-task-queue`**, **same** Temporal namespace,
  **same** Postgres DB, **own** Alembic chain (distinct `version_table` +
  `include_object`), `ocr_`-prefixed tables, **zero `forge.*` imports**.
- `forge-contracts` holds: wire models (`BatchResult` = inline `raw_response_json`
  **or** optional `s3_key`; the new opaque submit-SPI request; narrowed
  `BatchJobStatus`), queue/namespace/signal-name constants, `s3_blobs`, the Temporal
  connect helper, the survivable-write primitive (`persist_block` + presets), the
  shared `UTCDateTime` type, and the `batch_jobs` READ model.
- Platform owns the batch service: **Option 1** — platform submits an opaque
  pre-built request blob (submit writes nothing; a separate persist writes
  `batch_jobs`); poll + signal. `batch_jobs` becomes generic provider-only state;
  `file_path`/`document_id` removed. Result delivery: poller stashes the raw
  provider result to S3 and signals a `BatchResult` pointer **chosen by size**.
- Image work: `sax-llm` parses provider FORMAT (stays); OCR owns
  storage/bbox/markdown rewrite; `save_ocr_image` + `_image_mapping` leave the poller.
- Status = coarse SQL projection (`submitted/processing/stored/failed`): provider
  state in `batch_jobs` (platform single-writer), processing/terminal state in OCR's
  own `ocr_` status table (OCR single-writer); one SQL join on
  **`request_id == provider custom_id == batch_jobs PK`** (minted once).
- `OcrSyncWorkflow` **removed** (generic sync LLM kept). Client retries already
  disabled (committed `7cce5dc`). Big-bang cutover, no corpus → squash migrations.
  Blob GC via bucket TTL.

## 3. Acceptance Criteria (overall)

The split is correct only if the **soundness invariant** holds. Concretely:

1. `rg "forge\.ocr" src/forge` (excluding `ocr/`) returns **nothing**; `rg "forge\."`
   in the OCR repo returns **nothing** (only `forge_contracts` + `sax_llm`).
2. Forge builds and its worker starts **without OCR installed**.
3. The platform poller never imports OCR, never writes `ocr_images`, never injects
   `_image_mapping`; it stashes a verbatim provider result blob and signals a typed
   pointer (size-chosen).
4. `batch_jobs` has no `file_path`/`document_id`; one mint point for
   `request_id == custom_id == batch_jobs PK`; the status join works across the
   contracts `batch_jobs` read model and OCR's `ocr_` status table.
5. Two Alembic chains upgrade against one Postgres without dropping each other's
   tables (distinct `version_table` + `include_object`), validated by a `postgres`
   testcontainers test.
6. Orphan/expiry **signals** the waiting OCR workflow (no silent timeout); the 25h
   `wait_condition` timeout is caught → terminal `failed` (already committed `7cce5dc`).
7. Full forge suite green; OCR suite green standalone; a cross-repo two-worker
   integration test exercises the end-to-end OCR batch.

## 4. Phase 0 — `forge-contracts` foundation

**Goal:** create the shared package and break the `store → ocr.s3_blobs` cycle.

### Current state
- `forge-contracts` empty. `s3_blobs.py` (`src/forge/ocr/s3_blobs.py`) has **zero
  `forge.*` imports** (clean leaf). `connect_temporal`/`build_tls_config`
  (`temporal_client.py:61-119`) never set a namespace (implicit `default`).
  `persist_block` + `_PERSIST_RETRY`/`_PERSIST_SCHEDULE_TO_CLOSE`
  (`workflow_blocks.py:79-92,68-76`). `UTCDateTime` (`store.py:64-88`) used by all
  tables. Signal name `'batch_result_received'` is a bare literal in 5+ places
  (`batch_poll.py:133,188`; `workflows.py:266,1451`; `ocr/workflow_store.py:58`).

### Changes
- Scaffold `forge-contracts` (hatchling, `src/` layout, `sax-llm` editable pin) using
  `../pbook/pyproject.toml` as the template.
- Move **`s3_blobs.py`** → `forge_contracts.s3_blobs`. Generalize the
  `FORGE_OCR_S3_BUCKET`/prefix env names (or keep by convention — decide once) and
  add a **per-kind key namespace** so reapable request/result blobs vs durable
  `ocr_images`/file blobs can carry separate TTLs.
- Move **`UTCDateTime`** → `forge_contracts.types`.
- Move the Temporal connect helper (`connect_temporal` + `build_tls_config` +
  `pydantic_data_converter` wiring) → `forge_contracts.temporal`. Add an explicit
  **namespace constant** (currently none exists — define `default` or a chosen name;
  both repos must use the same value).
- Move the survivable-write primitive: `persist_block` + the retry/timeout presets →
  `forge_contracts.persist`. NOTE: `persist_block` targets the activity by **string
  name** `'persist_to_store'` — the helper moves, but **each repo registers its own
  `persist_to_store` activity** on its own queue (see Cross-cutting §10).
- Define **constants**: `FORGE_TASK_QUEUE` (from `workflows.py:76`), **new**
  `OCR_TASK_QUEUE = "ocr-task-queue"`, the namespace, and
  `BATCH_RESULT_SIGNAL = "batch_result_received"`.
- Define **wire models**: `BatchResult` (move from `models.py:1103`, add optional
  `s3_key`), narrowed `BatchJobStatus` (provider-only — see §10), the new opaque
  **submit-SPI request** `{s3_key, model, endpoint, provider, custom_id}`, and the
  **`batch_jobs` READ model** (pydantic mirror of the slimmed `batch_jobs`; no
  pydantic read model exists today — `get_pending_batch_jobs`/`get_batch_job` return
  raw mapping dicts).
- In forge: add `forge-contracts` editable pin to `pyproject.toml`
  (`[tool.uv.sources]`, alongside `../sax-llm`); re-point `store.py`'s blob funcs and
  `temporal_client` usage to `forge_contracts`; replace the bare signal literals with
  the constant. `uv lock`.

### Sub-tasks
- [ ] Scaffold `forge-contracts` package (pyproject, src layout, README).
- [ ] Move `s3_blobs` to contracts; re-point `forge.store` blob funcs; delete `forge.ocr.s3_blobs`.
- [ ] Move `UTCDateTime`, connect helper (+ namespace const), `persist_block` + presets.
- [ ] Define `BatchResult` (+`s3_key`), `BatchJobStatus` (narrowed), submit-SPI request, `batch_jobs` read model.
- [ ] Define queue/namespace/signal constants; replace bare literals in forge.
- [ ] Wire forge editable pin; `uv lock`; forge imports from contracts.

### Tests / Verification
- `rg "forge\.ocr\.s3_blobs" src/forge` → only inside `ocr/` (cycle from `store.py` gone).
- Forge suite green (OCR still present, now importing contracts for the moved bits).
- `uv run python -c "import forge_contracts"` OK; forge worker boots.

### Acceptance
Forge depends on `forge-contracts`; the `store → ocr.s3_blobs` inversion is gone;
all shared primitives have a single home.

## 5. Phase 1 — make the Forge platform OCR-agnostic

**Goal:** strip OCR knowledge out of the platform while OCR still lives in-repo
(keeps the suite runnable between phases).

### Current state
- `batch_jobs` carries `file_path`(`store.py:147`)/`document_id`(`store.py:148`);
  written by `record_batch_submission`(627-658)/`record_batch_failure`(661-691);
  mirrored in `PersistBatchSubmission`/`PersistBatchFailure`
  (`persist_models.py:81-82,93-94`) + persist dispatch (`persist.py:83-94`).
- `batch_poll.py` owns image storage (`store_images_fn` 270-305, `_image_mapping`
  164-170) and signals via a module-global client (187-189).
- `persist.py:105-118` has a `PersistOcrResult` arm (+ `assert_never` 122-123).
- `batch_submit.py:46-70` builds the request internally (no endpoint).
- `workflow_blocks.batch_submit_and_wait` **asserts** `raw_response_json is not None`
  (`workflow_blocks.py:145`).
- `worker.py` imports/registers OCR + `OcrSyncWorkflow`.

### Changes
- **Drop `file_path`/`document_id` from `batch_jobs`** (lockstep across 4 files,
  gotcha): `store.py` columns + `record_batch_submission`/`record_batch_failure`
  params; `persist_models.py` fields; `persist.py` dispatch kwargs;
  `ocr/workflow_submit.py:205-206,219-220` persist calls (OCR side, adjust now).
- **Narrow `BatchJobStatus`** to provider-only `{submitted, processing, failed,
  expired, missing}` (add `processing`; move `storing/succeeded/errored/canceled` to
  the OCR side). Re-validate the poller mapping `BatchPollStatus.value ==
  BatchJobStatus.value` (`batch_poll.py:147-151,205-206`) and the store validation
  (`store.py:707`). **Not a clean subset** — coordinate enum + poller + store.
- **De-contaminate the poller:** remove `store_images_fn`/`_image_mapping`/
  `save_ocr_image`/`ocr_image_id`. The poller stashes the verbatim provider result
  to S3 (`forge_contracts.s3_blobs`) and signals `BatchResult` with `s3_key` when the
  payload exceeds a **size threshold**, else inline `raw_response_json`.
- **Size-based delivery breaks the generic path too:** update
  `batch_submit_and_wait` to branch on `s3_key` (fetch the blob) instead of asserting
  `raw_response_json` non-null.
- **Opaque-blob submit SPI:** add a platform activity accepting the submit-SPI
  request `{s3_key, model, endpoint, provider, custom_id}` that fetches the blob and
  calls `provider.submit_batch(requests, model, endpoint=...)` verbatim — **writes
  nothing**; a separate persist writes `batch_jobs` (preserves the no-store-on-submit
  double-submit fix). Endpoint is a first-class field so the platform routes `/v1/ocr`
  without knowing it's OCR.
- **Remove `PersistOcrResult`** arm + `assert_never` update.
- **Remove OCR from `worker.py`** (imports + registration) and **delete
  `OcrSyncWorkflow`** registration (the workflow itself is deleted, not moved).
  Keep `set_temporal_client` + `_register_output_types` (generic).

### Sub-tasks
- [ ] Remove `file_path`/`document_id` from `batch_jobs` + all writers/models (4-file lockstep).
- [ ] Narrow `BatchJobStatus`; fix poller mapping + store validation.
- [ ] De-contaminate `batch_poll.py`; poller stashes raw blob + size-based pointer signal.
- [ ] Branch `batch_submit_and_wait` on `s3_key` (fetch blob); drop the non-null assert.
- [ ] Add opaque-blob submit-SPI activity (submit writes nothing; separate persist).
- [ ] Remove `PersistOcrResult` arm + `assert_never`.
- [ ] Remove OCR imports/registration from `worker.py`; delete `OcrSyncWorkflow`.

### Tests / Verification
- Generic (Anthropic) batch path green through size-based delivery (small=inline, large=pointer).
- `rg "ocr" src/forge/activities/batch_poll.py` → none.
- Platform suite green with OCR temporarily disabled in the worker.

### Acceptance
Platform poll + submit are OCR-agnostic; `batch_jobs` generic; `BatchResult` carries
both delivery arms; no OCR symbols on the platform path.

## 6. Phase 2 — scaffold the OCR repo and move the code

**Goal:** stand up `ocr` as an independent app importing only `forge_contracts` + `sax_llm`.

### Current state
- `src/forge/ocr/` = 11 files (~2428 LOC). OCR store tables/functions live in
  `forge/store.py`; OCR persist variants in `persist_models.py`; OCR CLI in `cli.py`;
  3 test files + conftest fixtures; `tests/fixtures/ocr/` empty (e2e expects
  `hello_jpeg.jpg`/`hello_png.png`, regenerated by `scripts/generate_ocr_fixtures.py`
  via pymupdf). OCR-only deps `mistralai` + `pymupdf` sit in forge `pyproject.toml`.

### Changes
- Scaffold `ocr` repo (pyproject: deps `forge-contracts` + `sax-llm` + `mistralai` +
  `pymupdf`; **not** forge; hatchling src layout per pbook).
- **Move** `src/forge/ocr/*` → `ocr/src/ocr/`. Rewrite every `forge.*` import to
  `forge_contracts.*` or OCR-local. **Delete** `OcrSyncWorkflow` + `call_ocr_sync` +
  `execute_call_ocr_sync` + `OcrSyncInput`/`OcrSyncCallResult` + the now-dead
  `sax_llm` private import (`activities.py:652`).
- **Move OCR store** → `ocr/src/ocr/store.py` with its **own `Base`** (using
  `forge_contracts.UTCDateTime`): `OcrResult`(177-200), `OcrImage`(216-233),
  `FileContentBlob`(203-213, rename table → `ocr_file_content_blobs`), + the ~20 OCR
  store funcs (`save_ocr_result`, `get/find/delete_ocr_*`, `save_ocr_image`,
  `ocr_image_id`, `mark/clear_ocr_removal`, hash helpers). Add the **NEW `ocr_`
  status table** (PK == `request_id`; holds coarse processing/terminal status +
  `document_id`/`file_path` that left `batch_jobs`).
- **OCR persist:** OCR registers its **own `persist_to_store` activity** on
  `ocr-task-queue` writing the OCR store; OCR workflows call `persist_block` (from
  contracts) which dispatches to that queue's impl.
- **Move OCR CLI** (`ocr-jobs`, `backfill-hashes`) into the OCR repo CLI with its own
  store-engine helper (same `FORGE_DB_URL` Postgres) and an `ocr start` path
  defaulting to `ocr-task-queue`.
- **OCR worker** on `ocr-task-queue` registering OCR workflows + activities +
  `persist_to_store`, with its own `_init_store`/`run_migrations` (replicate the
  `%`→`%%` escape, `store.py:389`).
- **Move tests + fixtures** + `generate_ocr_fixtures.py`; OCR conftest patches the OCR
  store module's `create_engine` (not `forge.store`), moto S3, two-worker harness.

### Sub-tasks
- [ ] Scaffold `ocr` pyproject/worker/CLI; move OCR-only deps off forge.
- [ ] Move `ocr/` code; rewrite imports to contracts/local; delete sync path + dead imports.
- [ ] Move OCR store (own `Base`) + add `ocr_` status table; OCR-side `persist_to_store`.
- [ ] Move OCR CLI + `ocr start` (ocr-task-queue) + OCR store-engine helper.
- [ ] Move OCR tests/fixtures/conftest; standalone OCR suite green.

### Tests / Verification
- `rg "forge\." ocr/src` → only `forge_contracts`.
- OCR suite green **standalone** (its own worker/queue/store/conftest).

### Acceptance
OCR is an independent package; zero `forge.*` imports; its tests pass on their own.

## 7. Phase 3 — cross-queue wiring, correlation key, status projection

**Goal:** connect OCR ↔ platform purely through `forge-contracts` + Temporal string-name calls.

### Current state
- `request_id == custom_id == batch_jobs PK` minted once in
  `execute_submit_ocr_batch` (`activities.py:263`). `OcrSubmitWorkflow` uses
  `ParentClosePolicy.ABANDON` children (`workflow_submit.py:143,167`). Cross-queue
  template = `ingestion_workflow.py` (string-name child wf/activity + `task_queue`,
  zero pbook imports). `execute_list_ocr_jobs` (`activities.py:1043-1171`) joins on
  `batch_jobs.document_id`/`file_path`.

### Changes
- **OCR → platform (submit):** `OcrSubmitWorkflow` builds the OCR body, writes the
  pre-built request blob to S3 (contracts), and calls the platform submit-SPI activity
  **cross-queue** (`execute_activity(..., task_queue=FORGE_TASK_QUEUE)`) with
  `{s3_key, model, endpoint="/v1/ocr", provider, custom_id}`. **`custom_id` minted
  once** here == `batch_jobs.id`.
- **platform → OCR (result):** the poller signals `OcrStoreWorkflow`'s
  `batch_result_received` (by `workflow_id`, same namespace) with `BatchResult`
  (pointer or inline). On orphan/expiry/failure the poller **signals with an error**
  (not a silent timeout) so OCR fails cleanly.
- **OCR result handling:** `OcrStoreWorkflow` fetches the raw blob from S3 when
  `s3_key` is set, then does **all** image extraction/storage/bbox/markdown rewrite
  (`activities.py:130-143,283-298,665-670`) — reworking `parse_ocr_result` which today
  expects `_image_mapping` pre-injected (`activities.py:289-291`). Writes
  `ocr_results` + `ocr_images` + the `ocr_` status table (OCR single-writer).
- **Status projection:** rewrite `execute_list_ocr_jobs` as a join between the
  contracts `batch_jobs` read model (provider status) and OCR's `ocr_` status table
  (`document_id`/`file_path`/processing status) on **`request_id`**; preserve the
  root-vs-chunk `document_id` asymmetry.

### Sub-tasks
- [ ] OCR submit → S3 blob + cross-queue submit-SPI call; single mint of `custom_id`.
- [ ] Poller signals OCR by workflow_id (success + error/expiry/orphan).
- [ ] Rework OCR result path: fetch blob, extract/store images, markdown rewrite, write status.
- [ ] Rewrite status join on `request_id` (read model ⋈ ocr_ status table).

### Tests / Verification
- Two-worker test (forge-task-queue + ocr-task-queue, shared DB, moto S3): submit →
  poller signal → OCR store → status join returns `stored`.
- Inject expiry → OCR receives error signal → terminal `failed` (no 25h hang).

### Acceptance
End-to-end OCR batch runs across the queue boundary using only contracts; the status
join works without OCR reading `forge.store`.

## 8. Phase 4 — migrations: squash to clean baselines, two isolated chains

**Goal:** one Postgres, two independent Alembic chains, no cross-drops.

### Current state
- One chain 001→015 (head 015), one `Base.metadata` (`env.py:9`), **no**
  `version_table`, **no** `include_object`. `batch_jobs`=004, `ocr_results`=006,
  `file_content_blobs`=007, `ocr_images`=009; grafts 011/013; s3 swap 014 (013 has a
  data backfill). Zero ForeignKeys. `run_migrations` (`store.py:372`) called from
  `worker.py:125`.

### Changes
- **Forge baseline (squash):** single clean baseline creating platform tables +
  **generic `batch_jobs`** (no `file_path`/`document_id`). `env.py`: set
  `version_table="alembic_version_forge"` + `include_object` **excluding** `ocr_*`/
  `ocr_file_content_blobs`. Preserve `sa.false()` (migration 010 rationale) and the
  `%`→`%%` escape.
- **OCR baseline (new chain):** `ocr/alembic` with its **own `env.py`** importing
  OCR's `Base`, `version_table="alembic_version_ocr"`, `include_object` **limited to**
  OCR tables. Baseline creates `ocr_results`, `ocr_images`, `ocr_file_content_blobs`,
  and the `ocr_` status table directly in **post-014 `s3_key` shape** (do NOT replay
  the LargeBinary→s3_key transition; drop the 013 backfill — no corpus).
- Both run against the same `FORGE_DB_URL`; order between chains is irrelevant (no FKs,
  separate `version_table`s).

### Sub-tasks
- [ ] Squash forge chain to a slimmed-`batch_jobs` baseline; add `version_table` + `include_object`.
- [ ] Create OCR chain (own env.py, version_table, include_object, post-014 shapes).
- [ ] Verify `%` escape + `sa.false()` carried into the relevant baseline.

### Tests / Verification
- `postgres`-marked testcontainers test: run **both** `alembic upgrade head` against
  one fresh Postgres; assert neither chain drops the other's tables and both
  `*_alembic_version` tables coexist. Run with
  `TESTCONTAINERS_RYUK_DISABLED=true uv run pytest -m postgres` (podman; see memory).

### Acceptance
Two chains coexist on one Postgres; autogenerate on either side ignores the other's tables.

## 9. Phase 5 — cutover, cross-repo tests, cleanup

**Goal:** remove OCR from forge entirely; prove the whole thing end-to-end.

### Changes
- Delete `src/forge/ocr/`, the OCR store tables/funcs from `forge/store.py`, OCR
  persist variants, OCR scripts (`scripts/*ocr*`, `scripts/nushell/ocr.nu`,
  `scripts/list_ocr_jobs.nu`), `queries/ocr-results.sql`, OCR diataxis docs +
  `docs/requirements/ocr_*.feature`, and OCR-only deps (`mistralai`, `pymupdf`) from
  forge `pyproject.toml`. `uv lock`.
- Cross-repo integration test (two workers, two queues, shared Postgres, moto S3)
  exercising submit → poll → signal → store → status.
- Update top-level docs/CLAUDE.md project-status to reflect the split.

### Sub-tasks
- [ ] Remove all OCR code/tables/scripts/docs/deps from forge; `uv lock`.
- [ ] Cross-repo e2e integration test green.
- [ ] Docs updated; `rg "forge\.ocr|ocr_results|ocr_images" src/forge` → none.

### Acceptance
Forge has zero OCR; full forge suite green; OCR suite green; cross-repo e2e green;
the §3 + soundness-invariant checklist passes.

## 10. Cross-cutting concerns

- **Per-queue `persist_to_store`.** `persist_block` (contracts) targets the activity
  by string name; the *impl* writes a store. Forge registers a `persist_to_store`
  writing the platform store; OCR registers its own writing the OCR store. A workflow
  hits the impl on **its own** task queue — so OCR workflows MUST run on
  `ocr-task-queue` to persist to the OCR store. No shared generic persist activity is
  built.
- **Lockstep edits.** `file_path`/`document_id` removal and `BatchJobStatus`
  narrowing each span 4+ files; change them together or persist-request validation /
  poller mapping breaks.
- **Single mint point.** `request_id == custom_id == batch_jobs PK` minted once
  (OCR submit), threaded through the SPI → `batch_jobs.id` → `BatchResult.request_id`
  → OCR status join. Never re-mint.
- **Namespace.** None is set today (implicit `default`). Define one constant in
  contracts; both repos must pass the same value to `connect_temporal`.
- **Signal name** must be byte-identical across repos — use the contracts constant.
- **Env var generalization.** `FORGE_OCR_S3_*` / `FORGE_TEMPORAL_TLS_*` are read by
  shared helpers; either rename generically or keep by convention (decide in Phase 0).
- **`UTCDateTime`** shared via contracts (used by both Bases).

## 11. Risks & open questions

- **SECURITY (act now):** `forge-contracts/.envrc` contains plaintext live
  `AWS_KEY`/`AWS_SECRET` (gitignored, but real on disk) — rotate / move to a secrets
  manager; shared `s3_blobs` should rely on the default AWS credential chain, not
  committed/dotenv creds.
- **Generic LLM batch path is touched** by size-based delivery (the `s3_key` branch in
  `batch_submit_and_wait`) and `BatchResult` shape — regression-test the Anthropic
  path, not just OCR.
- **`persist_block` entanglement** is the trickiest move; if the per-queue
  `persist_to_store` pattern proves awkward, the fallback is OCR owning a bespoke OCR
  persist activity not derived from `persist_block`.
- **`BatchJobStatus` narrowing** adds `processing` (new) and removes OCR states — the
  poller's `BatchPollStatus.value == BatchJobStatus.value` mapping must be re-verified.
- **Two-worker DB connection pressure** (DEFERRED in the grill) — revisit before any
  production deploy; two bounded pools against one managed Postgres.
- **No corpus assumption** underpins the squash (drops migration 013's backfill) —
  confirm the target DB is empty before squashing.

## 12. Definition of Done

All §3 acceptance criteria met; every clause of the SOUNDNESS INVARIANT in
[`separate-ocr-modules.grill.md`](../grill-me-sessions/separate-ocr-modules.grill.md)
satisfied; forge + OCR suites green; cross-repo e2e green; `forge-contracts`,
`forge`, and `ocr` each lock cleanly with the editable sibling pins; AWS creds rotated.

## 13. Progress log

- 2026-06-04 — Plan written from grill decisions + 5-slice inventory (`w98hu9ev1`)
  and the B1–B6 adversarial sweep (`wkwh24osi`). Nothing implemented. Prereqs already
  landed: client retries disabled + OCR wait-timeout caught (commit `7cce5dc`); pbook
  ingest API fix (commit `a981d2a`).
