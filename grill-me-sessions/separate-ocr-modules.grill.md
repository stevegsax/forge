# Grill Session: separate-ocr-modules

Started: 2026-06-04
Last updated: 2026-06-04
Status: complete
Domain: Software architecture — layering Forge as a Temporal-based platform with OCR as a plugin/consumer application

## Summary

COMPLETE. Examined and stress-tested the plan to split OCR out of Forge so Forge is a
Temporal-based platform (LLM connectivity, batch service, storage) and OCR is an independent
repo consuming it. Verdict on the core question — Forge is *mostly* using Temporal correctly:
durability is NOT re-implemented (persist = observability + cross-process coordination, itself
Temporal activities); the batch poller is necessary (Temporal has no batch primitive); the one
real duplicated flow-control is client-side LLM retries stacked under Temporal's RetryPolicy
(fix: max_retries=0).

**Converged target architecture (all decided):**
- OCR → separate repo, own worker on `ocr-task-queue`, SAME namespace, SAME Postgres DB, own
  Alembic chain (distinct version_table + include_object filter), `ocr_`-prefixed tables, zero
  `forge.*` imports.
- `forge-contracts` (new shared pkg) is the SPI surface: batch wire models (BatchResult with
  inline+pointer arms, the submit-SPI request, BatchJobStatus), queue/namespace/signal-name
  constants, `s3_blobs`, Temporal connect helper, survivable-write primitives, and the
  `batch_jobs` read schema. forge and OCR both depend on it; neither imports the other → forge
  no longer needs OCR installed.
- Platform owns the batch service: Option 1 (platform makes the provider call from an opaque
  pre-built request blob; submit writes nothing, separate persist writes `batch_jobs`),
  poll + signal. Result delivery via S3 pointer chosen by SIZE.
- Lifecycle split: `batch_jobs` = provider-only states (platform single-writer, generic — no
  file_path/document_id); processing/terminal status in OCR's own table (OCR single-writer).
  Status query = one SQL join in the shared DB on `request_id` (== provider custom_id ==
  batch_jobs PK, minted once). Coarse states only; no `@workflow.query` in the core path.
- Image work: sax-llm parses provider FORMAT (stays); OCR owns STORAGE/bbox/markdown rewrite;
  `save_ocr_image` + `_image_mapping` leave the platform poller.
- `OcrSyncWorkflow` removed (generic sync LLM kept). Big-bang cutover (no corpus) → migrations
  are a clean redefinition. Blob GC via bucket TTL (≥ the ~25-26h wait), reapable vs durable
  keys namespaced.
- Failure handling: orphan/expiry SIGNALS the waiting OCR workflow (not a silent timeout); the
  25h `wait_condition` timeout is caught → terminal `failed`.

The full DECIDED log below plus the **SOUNDNESS INVARIANT** paragraph are the acceptance
criteria for whoever writes the implementation plan. Output of this session was the plan's
decision basis — not code.

## Stated positions (to be adversarially tested)

- **Layering**: Forge = platform (LLM connectivity, job management, storage); OCR =
  application/workflow that plugs in and consumes platform features.
- **S3 / external-API functions** (hypothesis, unproven): pure external-API access (e.g. S3)
  should move from the OCR package into Forge as a platform facility / workflow step.
  - Code note: `s3_blobs.py` lives in `src/forge/ocr/` yet `forge/store.py` imports from it —
    so the platform already depends on the plugin for S3. Instinct is half-confirmed.
- **Storage & schema (#3)**: platform owns storage and schema.
- **Plugin registration (#4)**: open to pbook's model unless there is a downside.
- **Packaging (#5, firm)**: OCR becomes a SEPARATE REPO; Forge is the orchestrator, OCR an app on it.
- **Temporal-first principle**: maximize use of Temporal's native flow control; do not
  duplicate what Temporal already provides; prefer expressing processes as workflow steps and
  abstracting common functionality into independently invokable/awaitable workflows.

## Open Threads

1. Platform SPI contract — is the interface genuinely generic, or OCR-shaped?
2. Breaking the platform→plugin cycle (store↔ocr.s3_blobs, batch_poll↔ocr store funcs)
3. Storage & schema ownership + single Alembic migration chain
4. Plugin registration (pbook try/except model, cross-queue calls, Temporal Nexus?)
5. Separate-repo packaging — uv sources, worker/task-queue topology, deploy boundary
6. Temporal correctness — hand-rolled vs native (persist_block, batch submit/poll/signal, child workflows)

### Gating threads exposed by homework (decide these first; everything forks on them)

7. **Separation DEPTH** — pbook-parity (own queue/worker/DB/chain, zero forge imports, string-name
   cross-queue invocation) vs cosmetic import-guard on the existing single-queue worker.
8. **Namespace boundary** — same namespace as Forge (pbook model; allows cross-queue child
   workflows + the current global-client signaling) vs OCR's own namespace (forces Nexus, since
   child workflows cannot cross namespaces).
9. **Fate of the shared batch poller + `batch_jobs`** — keep as a generic platform batch-completion
   service (and remove OCR contamination via an extension point) vs dissolve it into a
   Temporal-native primitive (async activity completion / activity-retry polling) so no shared
   poller exists. THIS is the "are we using Temporal correctly?" question.
10. Q1 — one shared DB (two Alembic chains) vs OCR's own DB. (pbook chose own DB.)
11. Q2 — platform mediates writes (write-SPI) vs platform only provisions store + OCR writes its
    own tables. Homework: shared `persist_to_store` already mediates, but leaks `PersistOcrResult`.
12. Sync OCR path — first-class plugin capability or deprecate (batch-first violation).
13. Packaging optionality — hard editable dep (like pbook today) vs truly optional/absent.

## Decision Log

### RESOLVED: Owning plugin tables needs no new subsystem
- **Resolution**: `store.py` has ZERO foreign keys (`rg ForeignKey src/forge/store.py` → none).
  Tables are linked by string convention only (`document_id`, `batch_id`, `workflow_id`). A
  plugin can own its tables via its own `DeclarativeBase` + its own Alembic chain with a distinct
  `version_table` — stock SQLAlchemy/Alembic, no new abstraction. pbook already does exactly this
  (`PBOOK_DATABASE_URL`, `pbk_alembic_version`). The new-subsystem trap (dynamic plugin schema
  registry / Base-merging) is only needed for runtime-dynamic plugins, which the user has ruled
  out ("infrequent, install-time customization").
- **Correction logged**: Temporal is NOT a substitute for these tables. It persists workflow
  execution state (event history), not queryable domain data; `ocr_results`/`ocr_images` exist
  because the CLI queries them after the workflow is gone.
- **Date**: 2026-06-04

### DECIDED: Separation depth = pbook parity, EXCEPT shared database
- **Decision**: OCR becomes a separate repo with its own worker and its own task queue, zero
  `forge.*` imports, invoked by Temporal string-name cross-queue calls (full pbook parity) —
  with one exception: it uses the SAME database as Forge (not pbook's own-DB model). OCR owns
  its own Alembic chain / `version_table`; its tables are prefixed `ocr_`.
- **Rationale**: User sometimes needs to query processing status by document metadata, joining
  workflow/job data with OCR data — wants those joins available in one DB. Separate worker accepted.
- **Date**: 2026-06-04

### DECIDED: Same Temporal namespace as Forge (like pbook)
- **Rationale**: Avoids Nexus (docs say GA, but installed sdk-python 1.27.2 marks it
  experimental/unstable). Keeps cross-queue child workflows + signaling available.
- **Date**: 2026-06-04

### DECIDED: Batch is a cornerstone — polling stays, not open for discussion
- **Decision**: The submit→poll→signal batch mechanism is retained. Option to dissolve the poller
  into async-activity-completion / activity-retry-polling is OFF the table.
- **Rationale**: Batch mode is the reason Forge exists; the principle is always maintained.
- **Date**: 2026-06-04

### DECIDED: Platform owns "the batch service"; OCR is a consumer (Collision 1)
- **Decision**: Platform owns batch submit + poll + signal AND the `batch_jobs` coordination table.
  OCR submits batches via a platform SPI cross-queue (pbook-style, like `record_ingested_session`),
  never touches `batch_jobs`, and owns only its `ocr_*` result schema + all OCR-specific parsing.
- **Date**: 2026-06-04

### ACTION (plan item, not yet executed): disable client-side LLM retries
- `AsyncAnthropic(max_retries=0)` and Mistral equivalent. Rely solely on Temporal RetryPolicy.
- Kept out of this session per the "plan, not code" framing; trivial one-liner per client.

### OPEN: two-writer batch_jobs state machine — leaning "split the state"
- Today one `batch_jobs.status` column is written by TWO processes (poller: STORING/FAILED/
  EXPIRED/MISSING; OcrStoreWorkflow: SUCCEEDED) — conflates two lifecycles, caused a double-submit bug.
- Reframe under separation: split into (1) provider-batch lifecycle = platform/poller single-writer
  in `batch_jobs`; (2) consumer-processing lifecycle = OCR single-writer in its own `ocr_*` table.
  Same DB → status-by-document joins still work. Secondary options (workflow Query; custom Search
  Attributes + advanced visibility) rejected because they don't serve the SQL join requirement.

### DECIDED: Status = coarse SQL projection, no @workflow.query in core path
- **Decision**: Coarse states `submitted / processing / stored / failed` are enough. Status is a SQL
  projection: provider-batch states in platform `batch_jobs` (single writer), processing states in
  OCR's own `ocr_*` status table (single writer, keyed by document_id + correlation IDs). "Status of
  myfile.pdf" = one SQL join across both, in the shared DB. `@workflow.query` dropped from the core
  path (kept as a future optional "live tail" for page-level progress only).
- **Rationale**: query has a retention cliff (history purged → NotFound), splits across two workflow
  owners, breaks on chunk fan-out (1 doc → N workflows), and adds worker-availability coupling. SQL
  projection is durable, joinable, worker-independent.
- **Date**: 2026-06-04

### DECIDED: Collision 2 = (a) explicit S3 pointer
- **Decision**: Platform poller stashes the raw provider result to S3 (generic blob capability) and
  signals OCR a pointer; OCR fetches and does all OCR-specific image extraction/parsing on its own
  worker. Claim Check codec shelved as a documented future option.
- **Date**: 2026-06-04

### DECIDED: Shared wire contract lives in a new `forge-contracts` package
- **Decision**: Create a dedicated `forge-contracts` sibling package holding the batch wire models
  (BatchResult, submit-request shape, BatchJobStatus) + service/queue name constants. forge and OCR
  both depend on it; neither imports the other. User will create the repo and provide access.
- **Consequence**: forge stops hard-importing OCR (`worker.py:79-104`); forge no longer needs OCR
  installed → resolves packaging-optionality (#13). Platform treats provider request/response bodies
  as OPAQUE, so it never needs OCR's output-type registry.
- **Note**: contracts package CONTENTS are a plan design item (define the boundary), not to be
  implemented in this grill session. Adds one strand to the uv-source sibling web (known fragility).
- **Date**: 2026-06-04

### DECIDED: Submit fork = Option 1 (platform makes the provider call)
- **Decision**: The platform owns the actual `provider.submit_batch` call. OCR builds its request
  body, writes large items (the pre-built request body) to blob storage, and passes the platform a
  POINTER. The platform fetches the opaque request blob and submits — never parsing it. Centralizes
  provider credentials / rate-limiting / quota in the platform.
- **Date**: 2026-06-04

### DECIDED: Remove OcrSyncWorkflow; keep generic sync LLM capability
- **Decision**: Drop the OCR synchronous path entirely (it violated batch-first and was the only
  non-batch OCR path). The platform RETAINS generic synchronous LLM calls (`generation_dispatch`
  sync_mode) for non-OCR needs. Latency-sensitive OCR will be handled another way later.
- **Date**: 2026-06-04

### DECIDED: Big-bang cutover (no strangler-fig)
- **Decision**: Not in production, no document corpus to preserve, still in development → cut over in
  one shot. **Consequence**: migration untangling largely evaporates — this is a clean schema
  REDEFINITION, not a data migration. Can squash/reset chains to a clean baseline.
- **Date**: 2026-06-04

### DEFERRED: Two-worker DB connection pressure
- **Reason**: User not concerned right now (development).
- **Risk if ignored**: forge + ocr workers each hold a bounded pool against the same managed Postgres;
  the externalize-store work tuned pools to respect connection caps. Two workers ≈ double the
  connections. Revisit before any production deploy.
- **Date**: 2026-06-04

### DECIDED: s3_blobs → forge-contracts (shared); blobs by key in contract messages; bucket TTL GC
- **Decision**: The S3 access library moves into `forge-contracts` (NOT platform-internal), because
  blob I/O cannot be mediated cross-queue (payload limits). Blobs are addressed by KEY carried in
  contract messages (BatchResult/submit). Both workers hold S3 credentials + share one bucket + key
  scheme. "Platform owns storage" softens to: platform owns bucket provisioning + GC policy; access
  library + key scheme are shared contract. GC = bucket TTL/lifecycle policy (no cross-repo delete).
- **Revises** the earlier "move s3_blobs into the platform" framing.
- **Date**: 2026-06-04

### DECIDED: OCR CLI commands move to the OCR repo
- `ocr-jobs` (cli.py:1841, status query) and `backfill-hashes` (cli.py:1891, reaches into both
  forge.ocr + forge.store) are OCR-specific → move to OCR's own CLI. `backfill-hashes` is moot under
  big-bang/no-corpus.

### STATE: forge-contracts repo exists but is EMPTY (0 files) — clean slate. Contents = plan design item.

### ADVERSARIAL SWEEP COMPLETE (workflow wkwh24osi, 6 agents) — 6 BLOCKERS

Full output: `/private/tmp/.../tasks/wkwh24osi.output`.

**B1 — Who writes terminal status?** Today OcrStoreWorkflow is the ONLY writer of SUCCEEDED
(`workflow_store.py:127-135`); the poller stops at STORING (`batch_poll.py:200-205`). If batch_jobs is
platform-single-writer, OCR can't advance it → jobs stick in STORING. RESOLUTION (confirm): batch_jobs
holds PROVIDER-ONLY states (submitted/processing/failed/expired/missing); stored/succeeded lives ONLY
in OCR's status table; "done" computed by the join.

**B2 — Image extraction is ALSO in sax-llm, not just the poller.** `batch_poll.py:158-170,270-305`
stores images + injects `_image_mapping`; but upstream `sax-llm` `poll_batch` already extracts images
(`mistral.py:134-174` `_extract_images_from_response` → `BatchResultEntry.extracted_images`).
RESOLUTION (confirm the line): sax-llm parsing the Mistral OCR FORMAT is a provider concern (stays);
the contamination is `save_ocr_image` + `_image_mapping` in the poller (must move to OCR). Platform
stashes the parsed result (incl. base64 images) to S3; OCR does STORAGE/bbox/markdown rewrite. No
sax-llm raw-passthrough needed IF "parse format = provider, store = domain" is accepted.

**B3 — BatchResult has no s3 pointer; 3 consumers read raw_response_json inline** (`models.py:1103-1110`;
OCR store, generic `batch_submit_and_wait` which ASSERTS not-None `workflow_blocks.py:145-149`, ingestion
`ingestion_workflow.py:54`). RESOLUTION (confirm): BatchResult carries BOTH optional inline
raw_response_json AND optional s3_key; the poller picks pointer-vs-inline by SIZE (domain-agnostic
threshold), so generic/ingestion keep inline and large OCR results go via S3.

**B4 — The "one SQL JOIN" status needs OCR to read batch_jobs, which imports forge.store.BatchJob**
(`ocr/activities.py:1075`). Also no defined correlation key, and root-vs-chunk document_id asymmetry
(batch_jobs chunk rows carry ROOT document_id; ocr_results carry chunk doc_id). FORK: (i) contracts
ships the batch_jobs READ schema so OCR can SELECT it (table access ≠ Python import); or (ii) OCR mirrors
provider status into its own table via the signal (no cross-repo read). Correlation key must be ONE
value: request_id == provider custom_id == batch_jobs PK, minted once.

**B5 — Option 1 submit is unbuilt + reintroduces the double-submit bug.** Today OCR submits
(`ocr/activities.py:257-265`, Mistral `/v1/ocr` file-upload path); the no-store-on-submit fix works
because submit writes nothing and persist happens after (`activities.py:746-750`). RESOLUTION (confirm):
platform submit activity makes the provider call and writes NOTHING; a SEPARATE persist writes
batch_jobs; request_id minted once (by OCR, as custom_id) and reused as batch_jobs PK so the poller's
per-entry signal (keyed on custom_id, `batch_poll.py:172-184`) hits the right row. New submit-SPI model:
{s3_key_of_prebuilt_body, model, endpoint, provider, custom_id}; preserve Mistral file-upload path.

**B6 — Two Alembic chains, one DB.** Today ONE chain, ONE `Base.metadata` (`alembic/env.py:9`), default
`alembic_version`. RESOLUTION (confirm): distinct version_tables (alembic_version_forge /
alembic_version_ocr) + `include_object` filters in BOTH env.py so neither repo's autogenerate DROPs the
other's tables. Decide file_content_blobs ownership (backs OCR input files → likely OCR).

### forge-contracts MUST CONTAIN (the SPI surface)
1. BatchResult (final shape: inline raw_response_json + optional s3_key). 2. BatchJobStatus enum.
3. New submit-SPI request model {s3_key, model, endpoint, provider, custom_id}. 4. ParseResponseInput/
ParsedLLMResponse if shared (NO image fields). 5. Signal-name constant + binding convention (Temporal
binds by METHOD NAME — a constant alone doesn't enforce; needs a typed wrapper/convention). 6. Queue +
namespace name constants (FORGE_TASK_QUEUE + new OCR_TASK_QUEUE + explicit namespace). 7. s3_blobs (after
generalizing FORGE_OCR_S3_BUCKET env + per-kind key namespace for separate GC TTLs). 8. Temporal connect
helper (connect_temporal + build_tls_config + pydantic_data_converter) — single-source or TLS drift.
9. Survivable-write primitives OCR imports today (persist_block + retry/timeout presets). 10. batch_jobs
READ model for the JOIN (no forge.store import). 11. forge-contracts editable pin in both pyprojects.

### SOUNDNESS INVARIANT (acceptance criteria for the plan)
Correct only if: one stable correlation key (request_id==custom_id==batch_jobs PK, minted once);
platform poll+submit genuinely OCR-agnostic (poller stashes verbatim result blob + signals typed pointer,
never stores images / never injects _image_mapping; submit sends opaque pre-built blob preserving
/v1/ocr file-upload); ALL image storage/bbox/markdown in OCR; every cross-queue interaction mediated by
forge-contracts constants/models with single-sourced TLS; result blobs outlive the full ~25-26h wait,
request blobs survive submit retries, reapable vs durable blobs separated for GC; double-submit safe
(provider submit and batch_jobs write never share a re-runnable activity); MISSING/orphan actively
SIGNALS the waiting OCR workflow rather than letting it time out; two Alembic chains isolated by distinct
version_tables + include_object. If any one fails, the split breaks at RUNTIME, not import time.

### RESOLVED after sweep (2026-06-04)
- B1 CONFIRMED: batch_jobs = provider-only states; stored/succeeded only in OCR's status table.
- B2 CONFIRMED: boundary = "parse provider format = sax-llm (stays); store/transform = OCR". Delete
  save_ocr_image + _image_mapping from the platform poller; poller stashes the parsed result blob.
- B3 CONFIRMED: BatchResult carries inline raw_response_json + optional s3_key; poller picks by SIZE.
- B4 DECIDED → (i): forge-contracts ships the batch_jobs READ schema; OCR SELECTs it for the join
  (table access ≠ Python import). Single correlation key request_id==custom_id==batch_jobs PK, minted
  once. OCR status table preserves root-vs-chunk document_id asymmetry.
- B5 CONFIRMED: platform submit activity writes nothing; separate persist writes batch_jobs; request_id
  minted once and reused as PK.
- B6 CONFIRMED: distinct version_tables + include_object filters; file_content_blobs → OCR side.

### DECIDED: Timeout handling — catch the wait_condition timeout, not the workflow execution timeout
- The 25h `wait_condition` timeout raises `asyncio.TimeoutError` INSIDE the workflow (catchable).
  Today it is NOT caught (`workflow_store.py:70-74`) → on timeout the status row sticks. FIX: wrap in
  try/except, write terminal `failed` status, then raise ApplicationError.
- A workflow Execution/Run Timeout is NOT catchable (Temporal terminates; no cleanup runs) → do not
  rely on it for cleanup. Use the catchable wait_condition timeout as the internal backstop.
- Safety net = orphan signal (primary) + caught wait_condition timeout (backstop), both → terminal
  `failed`. Accepted caveat: a late result after timeout is dropped (workflow closed); 25h > provider SLA.
- **Date**: 2026-06-04

### FINDING: The SPI is NOT zero-coupling — there is an irreducible shared WIRE CONTRACT
- Forge connects with `pydantic_data_converter` (`temporal_client.py:111-114`). Cross-queue payloads
  are pydantic models rehydrated by the receiver's DECLARED parameter type. So OCR's
  `batch_result_received` signal handler must declare a `BatchResult`-typed param — but OCR has zero
  `forge.*` imports. Therefore the batch wire contract (`BatchResult`, the submit-request shape,
  `BatchJobStatus`) must live in a place BOTH repos import.
- Implication: "OCR is a customer of the platform" → the platform must ship a thin **contracts/SDK
  package** (or extend `sax-llm`). With that, forge depends on the contract NOT on OCR, and OCR
  depends on the contract NOT on forge → forge no longer needs OCR installed (fixes the hard-import
  in worker.py and resolves packaging-optionality #13). Neither imports the other's code; both import
  the contract. Cleaner than pbook (which is a hard dep of forge today).
- The string-keyed output-type registry (`llm_client.py:189`, `worker.py:132-166`) is OCR-side only
  IF the platform treats provider responses as OPAQUE strings (it should) — then the platform never
  deserializes OCR types.
- Cost/risk: a shared package adds to the uv-source web the user has already been bitten by (see
  memory: re-pinning siblings breaks `uv lock`; all siblings must use local editable `../sax-llm`).

### FACT: OCR-specific post-batch handling is substantial (justifies the separate repo)
- Beyond page chunking: image extraction (base64 decode, data-URI strip, S3 upload, `ocr_images`
  rows with bounding boxes + page_index), markdown reference rewriting `img-N.jpeg`→`ocr-image://{uuid}`
  (`activities.py:130-143,283-313`), reverse rewrite for export, OCR response-format parsing
  (`pages[].markdown` + `usage_info`), hash-based dedup, chunk gather/merge with image document_id
  reassignment. Request sets `include_image_base64:True` (`activities.py:108`) — which is WHY the raw
  result is large (Collision 2).

### FINDING: Retries are duplicated (Temporal flow-control re-implemented)
- LLM clients constructed with default retries ON (`llm_client.py:260` bare `AsyncAnthropic()`;
  `mistral.py:221` bare `Mistral()`). SDK retries stack on top of Temporal `RetryPolicy`. Cookbook
  rule is `max_retries=0` everywhere. This is the clearest duplicated-flow-control instance.
- Other "duplication" checks: durable state/replay NOT re-implemented (persist = observability +
  coordination, itself Temporal activities). `runs` table partially duplicates Temporal's result
  record but serves the `get_unextracted_runs` query Temporal can't. `batch_jobs` two-writer
  status machine is genuine cross-process coordination (poller can't read event history) but fragile.

## Homework — COMPLETE (run w11y0bhc6, 9 agents)

Full output: `/private/tmp/.../tasks/w11y0bhc6.output`. Headline findings:

- **The real coupling is the batch machinery, not the `ocr/` files.** OCR's submit→poll→signal
  handshake is Forge-invented (no precedent in the Temporal AI cookbook — zero `batch` matches
  there). The poller (`forge/batch_poller_workflow.py` + `activities/batch_poll.py`) and the
  `batch_jobs` table are SHARED by the generic-LLM batch path AND OCR.
- **The "generic" poller is contaminated with OCR.** `batch_poll.py:164-170,270-305` imports
  `save_ocr_image`/`ocr_image_id` and injects a private `_image_mapping` side-channel into the
  raw response. The platform is not domain-agnostic today.
- **Cycle even affects generic blobs.** `store.py`'s generic `save_file_content` imports
  `forge.ocr.s3_blobs` (`store.py:897,921,937`). `s3_blobs.py` is provider-neutral and has zero
  forge imports — it is "a platform capability misfiled into OCR."
- **A generic write-SPI already exists but leaks OCR.** `persist_block` → `persist_to_store`
  activity (`workflow_blocks.py:79-92`, `activities/persist.py`) is a generic survivable-write
  mechanism — but its `PersistRequest` union embeds `PersistOcrResult` (`persist.py:105-118`).
- **OCR is hard-wired into the worker with NO guard** (`worker.py:79-104`), unlike ingestion
  (try/except) and unlike pbook (separate queue/worker/DB/chain, zero forge imports, invoked by
  string name cross-queue). pbook is the real template — but pbook has no batch-poller dependency,
  so OCR separation is strictly HARDER than pbook.
- **Result delivery uses a module-global Temporal client inside the poll activity**
  (`worker.py:234` set, `batch_poll.py:187-189` used) — assumes ONE namespace. Temporal-native
  alternative: async activity completion by task token / Workflow Id, or activity-retry polling
  (zero history).
- **`OcrSyncWorkflow` violates batch-first** but is the LEAST-entangled path (no poller, no
  `batch_jobs`) and is closer to Temporal-canon (synchronous activities) than the batch path.

## Parking Lot

- Temporal Nexus as the cross-repo/cross-service boundary mechanism (directly relevant to #5)
- Whether OCR is even "the same kind of thing" as Forge's claimed "universal workflow step"
- Async activity completion vs the current batch-result signal mechanism
