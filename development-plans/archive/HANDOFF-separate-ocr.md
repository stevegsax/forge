# HANDOFF — Separate OCR into its own repo

**Date:** 2026-06-04
**Branch (forge):** `separate-ocr`
**Status:** ✅ **CUT COMPLETE.** All of "What remains" below is done. Forge is an
OCR-agnostic batch platform; OCR is a runtime-complete consumer app; both import only
`forge-contracts`. Verification: forge `uv run pytest` → 1325 passed; ocr → 23 passed;
forge-contracts → 10 passed; ruff clean across all three. Postgres (podman): forge 4 +
ocr 1 postgres-marked tests pass, and a manual both-real-chains `run_migrations` against
ONE Postgres shows both chains' tables + `alembic_version_forge`/`alembic_version_ocr`
coexisting with a slim `batch_jobs`. See the Progress log in
`development-plans/archive/separate-ocr-into-its-own-repo.md` (entry dated 2026-06-04
"CUT COMPLETE") for the per-increment detail and the residual follow-ups (AWS cred
rotation, env-name generalization, connection-pressure review). Merges to `main` remain
human-gated.

The remainder of this document is the original navigation layer, kept for context.

Read these two first — they are the authority; this handoff is the navigation layer:

- **Design / decision basis:** `grill-me-sessions/separate-ocr-modules.grill.md`
  (full DECIDED log, the B1–B6 adversarial blockers, and the **SOUNDNESS INVARIANT**
  paragraph = acceptance criteria).
- **Implementation plan + running progress log:** `development-plans/archive/separate-ocr-into-its-own-repo.md`
  (phases, per-phase changes/acceptance, and a dated Progress log of every increment
  landed so far, including discovered sequencing corrections).

## 1. Goal

Make **Forge** a generic Temporal-based platform (LLM connectivity, a generic batch
service, observability store) and extract **OCR** into its own repo that consumes the
platform as a customer. A shared **`forge-contracts`** package is the only thing both
import; neither imports the other.

## 2. The three repos

| Repo | Path | Role |
| --- | --- | --- |
| forge | `/Users/stevengreenberg/repos-sax/forge` | the platform (this repo; branch `separate-ocr`) |
| forge-contracts | `/Users/stevengreenberg/repos-sax/forge-contracts` | NEW shared SPI package (branch `main`) |
| ocr | `/Users/stevengreenberg/repos-sax/ocr` | NEW OCR app (branch `main`) |

All three use local editable `[tool.uv.sources]` pins (`../sax-llm`, `../forge-contracts`).
Per the user's standing constraint, sibling packages must use the **local editable**
source — do not re-pin to git (it breaks `uv lock`). See the user memory note.

## 3. What's done (committed)

**forge-contracts** (`main`) — the full SPI foundation OCR needs:

- `78bcc8f` scaffold + `s3_blobs` (moved out of forge.ocr — broke the platform→plugin cycle) + `types` (`UTCDateTime`)
- `290b0df` `temporal` (connect_temporal + build_tls_config + pydantic converter; explicit namespace) + `constants` (TEMPORAL_NAMESPACE, FORGE_TASK_QUEUE, OCR_TASK_QUEUE, BATCH_RESULT_SIGNAL)
- `236ebe6` `models` (`BatchResult` with the new optional `s3_key`; `BatchJobStatus` — kept FULL, narrowing deferred)
- `52c08f5` `db` (StoreConfigError, get_store_url, get_store_engine, ensure_sqlite_parent, insert_or_ignore)

**ocr** (`main`):

- `694656d` scaffold (pyproject deps = forge-contracts + sax-llm + mistralai + pymupdf …; never forge)
- `ab62d00` `ocr/store.py` — own `Base`; tables `ocr_results`, `ocr_images`, `file_content_blobs`, and the NEW `ocr_job_status` projection (PK `request_id`); ~20 functions; round-trip smoke green
- `b55cea1` leaf modules moved import-clean: `models.py` (sync model classes deleted), `persist.py` (retry/timeout presets), `workflow_export/gather/list_jobs/mark_removal/store.py`

**forge** (`separate-ocr`) — platform side, all green (1499 tests):

- `7cce5dc` disable client-side LLM retries; catch OCR `wait_condition` timeout (pre-req fixes)
- `a981d2a` fix `forge ingest` to pbook's `get_store_engine` API
- `77d6fc5` grill doc · `57b468a` plan doc
- `9480026` re-point `forge.store` to `forge_contracts.s3_blobs` (+`UTCDateTime`); delete `forge/ocr/s3_blobs.py`
- `0ba80bf` `forge.temporal_client` → re-export shim over `forge_contracts.temporal`; `batch_poll.py` signals via `BATCH_RESULT_SIGNAL`
- `06a2047` `forge.models` re-exports `BatchResult`/`BatchJobStatus` from contracts
- `6f86f46` `forge.store` re-exports the generic DB helpers from `forge_contracts.db`
- `1d54cf8`, `3566d0d` plan progress updates

> forge re-exports everything it moved, so all existing `from forge.store import …` /
> `from forge.models import …` / `from forge.temporal_client import …` call sites still
> work unchanged. `src/forge/ocr/` is **still present and registered** in forge — it is
> deleted only in the final cutover (step 5 below), which keeps forge green meanwhile.

## 4. What remains — one coordinated re-architecture (in order)

The OCR `activities.py` and `workflow_submit.py` were deliberately **not** moved: they
are built on the OLD model (OCR writes `batch_jobs` directly via
`persist_block`/`PersistBatchSubmission`/`update_batch_status`, submits to the provider
itself, and `execute_list_ocr_jobs` joins the platform `BatchJob` table). Moving them
mechanically would mean recreating throwaway machinery. So the rest is design-bearing
and must be done together:

1. **Platform batch-service SPI (forge).** Add an opaque-blob submit activity
   (Option 1): takes `{s3_key, model, endpoint, provider, custom_id}`, fetches the
   pre-built request blob from S3, calls `provider.submit_batch(...)` verbatim — **writes
   nothing**; a separate persist writes `batch_jobs` (preserves the no-store-on-submit
   double-submit fix). Make `batch_jobs` generic: drop `file_path`/`document_id`
   (`store.py` columns + `record_batch_submission`/`record_batch_failure` params +
   `persist_models.PersistBatchSubmission/Failure` fields + `persist.py` dispatch — 4-file
   lockstep). Narrow `BatchJobStatus` to provider-only `{submitted, processing, failed,
   expired, missing}` and re-validate the poller mapping (`batch_poll.py:147-151`) +
   `store.update_batch_status` validation. (NOTE: narrowing also edits the contracts copy
   in `forge_contracts/models.py`.)
2. **De-contaminate the poller (forge).** Remove `store_images_fn`/`_image_mapping`/
   `save_ocr_image`/`ocr_image_id` from `activities/batch_poll.py`. Poller stashes the
   verbatim provider result to S3 (`forge_contracts.s3_blobs`) and signals a `BatchResult`
   with `s3_key` when over a size threshold, else inline `raw_response_json`. Update the
   generic `workflow_blocks.batch_submit_and_wait` to branch on `s3_key` (fetch the blob)
   instead of asserting `raw_response_json is not None` (`workflow_blocks.py:145`).
   Orphan/expiry must **signal** the waiting workflow with an error (not a silent
   timeout). Remove the `PersistOcrResult` arm + the `assert_never` in `persist.py`.
3. **contracts `batch_jobs` read model.** Add a read-only schema/Table for `batch_jobs`
   to `forge-contracts` so OCR can `SELECT` it for the status JOIN **without importing
   `forge.store.BatchJob`** (the zero-forge-import rule allows table access, not Python
   import). Rewrite OCR's `execute_list_ocr_jobs` as the join `ocr_job_status ⋈
   batch_jobs(read model)` on `request_id`; redirect status writes from `update_batch_status`
   to `ocr.store.upsert_ocr_job_status`.
4. **Move `activities.py` + `workflow_submit.py` into `ocr`** onto the new SPI; delete
   the sync path (`call_ocr_sync`, `execute_call_ocr_sync`, and the `_extract_images_from_response`
   import). Add the OCR-side `persist_to_store` activity + `persist_block` (in `ocr.persist`).
   OCR submit: build body → write request blob to S3 → call the platform submit SPI
   cross-queue (`execute_activity(..., task_queue=FORGE_TASK_QUEUE)`); **mint `request_id`
   once** (== custom_id == batch_jobs PK). OCR result: receive `batch_result_received`
   signal → fetch blob if `s3_key` → extract images + markdown rewrite + store
   `ocr_results`/`ocr_images` + `ocr_job_status` (all OCR-owned).
5. **OCR worker + CLI** on `ocr-task-queue` (registering OCR workflows/activities + the
   OCR `persist_to_store`); move `ocr-jobs`/`backfill-hashes` CLI. **Two isolated Alembic
   chains** (squash to clean baselines — no corpus): forge baseline with
   `version_table=alembic_version_forge` + `include_object` excluding `ocr_*`; ocr baseline
   with `version_table=alembic_version_ocr` + `include_object` limited to OCR tables; both
   post-014 `s3_key` shapes; preserve the `%`→`%%` escape (`store.run_migrations`) and
   `sa.false()` default. **Then delete `src/forge/ocr/`** from forge + remove OCR-only deps
   (`mistralai`, `pymupdf`) + OCR scripts/queries/docs; forge goes green again.
6. **Cross-repo integration test** — two workers (forge + ocr queues), shared Postgres,
   moto S3 — exercising submit → poll → signal → store → status join.

The implementation plan's Phase sections (4–9) have the per-item detail and acceptance
criteria; the plan's Progress log has the rationale for every deviation.

## 5. Deferred / smaller items (don't lose these)

- `persist_block` relocation to contracts was deferred (its `PersistRequest`/`PersistResult`
  embed OCR variants that step 1 removes) — do it as part of step 4.
- `FORGE_TASK_QUEUE` single-sourcing: forge still defines its own literal in `workflows.py:76`
  (value equals `forge_contracts.constants.FORGE_TASK_QUEUE`); single-source it when convenient.
- `file_content_blobs` table → rename to `ocr_*` prefix in the OCR squash baseline (step 5).
- The lone `# (OcrSyncInput removed …)` comment in `ocr/models.py` can be tidied.

## 6. How to verify

- forge: `cd forge && uv run pytest` (1499 passing; `--no-cov` to skip the 85% gate, which
  is unmet locally because e2e-only modules like `providers.py` aren't exercised — a
  pre-existing CI-with-services matter, not a regression).
- forge-contracts: `cd forge-contracts && uv run pytest && uv run ruff check src`.
- ocr: `cd ocr && uv run ruff check src` + `uv run python -c "import ocr.store, ocr.models, ocr.workflow_store"`.
- postgres migration tests (later): `TESTCONTAINERS_RYUK_DISABLED=true uv run pytest -m postgres`
  (this machine uses **podman**, not docker — see user memory).
- **Editable-LSP caveat:** Pyright/IDE shows `forge_contracts` as "unresolved" in forge/ocr
  files. That is an editor venv-config artifact only — `uv run`, ruff, and the test suite all
  resolve the editable installs. Trust the test suite, not the squiggles.

## 7. Risks / gotchas

- **SECURITY (act):** `forge-contracts/.envrc` contains plaintext live AWS credentials
  (gitignored, but real on disk). Rotate / move to a secrets manager; `s3_blobs` uses the
  default AWS credential chain, so the dotenv keys shouldn't be needed.
- **Two-worker DB connection pressure** (DEFERRED): forge + ocr workers each hold a bounded
  pool against the same managed Postgres. Revisit before any production deploy.
- **Single correlation key invariant:** `request_id == provider custom_id == batch_jobs PK`,
  minted once (in OCR submit). The whole status model depends on it.
- **Same Temporal namespace** (`default`) for both workers — cross-queue signals + child
  workflows rely on it. No namespace was set historically; `connect_temporal` now passes it
  explicitly.
- The full set of must-hold invariants is the **SOUNDNESS INVARIANT** paragraph in the grill
  doc — treat it as the definition of done.

## 8. Key files for the remaining work

- forge: `src/forge/activities/batch_poll.py`, `src/forge/activities/batch_submit.py`,
  `src/forge/activities/persist.py`, `src/forge/persist_models.py`, `src/forge/workflow_blocks.py`,
  `src/forge/store.py` (BatchJob + record_batch_* + update_batch_status), `src/forge/worker.py`
  (OCR registration to remove), `src/forge/alembic/` (chain to squash), `src/forge/models.py`
  (BatchJobStatus narrowing — also in contracts).
- ocr: `src/ocr/store.py` (has `upsert_ocr_job_status`/`get_ocr_job_status` ready), `src/ocr/persist.py`
  (presets; add persist_block + persist_to_store), `src/ocr/workflow_store.py` (already moved;
  has the caught wait-timeout). To create: `src/ocr/activities.py`, `src/ocr/workflow_submit.py`,
  `src/ocr/worker.py`, `src/ocr/cli.py`, `src/ocr/alembic/`.
- The not-yet-moved source of truth for OCR `activities.py`/`workflow_submit.py` is still
  `forge/src/forge/ocr/` — read it there when migrating (it gets deleted in step 5).
- Cross-queue template: `forge/src/forge/ingestion_workflow.py` (pbook string-name cross-queue
  calls). pbook (`../pbook`) is the separate-repo + own-alembic template.
