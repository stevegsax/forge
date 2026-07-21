# Handoff — Phase 4 complete; start T4.4 or Phase 5

**Date:** 2026-07-21 (updated later the same day; see the Addendum)
**Status:** Landed. **Phases 0–4 are complete** (T0.1–T0.8, T1.0–T4.3) —
except **T4.4 (ocr Mistral status tracker)**, added 2026-07-21 by owner
adoption of the parked T4.2 proposal, NOT STARTED, fully specced in
[tasks/T4.4-mistral-status-tracker.md](tasks/T4.4-mistral-status-tracker.md).
`make gates` is EXIT=0 (forge **1348** / pbook **353** / ocr **152** /
sax-platform **439**; re-verified 2026-07-21); everything after the T4.2
close is docs-only. Nothing is in flight; local main is pushed and level
with origin/main. Next task: **T4.4 or T5.1 — owner's choice** (they are
file-disjoint; see the Addendum).

This is a state-of-the-world note for the next session. It does not restate
the work queue ([TASKS.md](TASKS.md)), the decisions
([../docs/DECISIONS.md](../docs/DECISIONS.md)), or the status-of-record
([../docs/OVERVIEW.md](../docs/OVERVIEW.md)) — it says what changed since the
[Phase 4 handoff](HANDOFF-2026-07-19-phase4-start.md), what to know before
touching anything, and where to start. The Phase 4 handoff (and the
[Phase 3 handoff](HANDOFF-2026-07-18-phase3-complete.md) behind it) remain
the detailed record of the transport-before state and the platform library;
their "What to know" sections still hold except where superseded below.

## Status — Phase 4, task by task

- **T4.1 — forge timer-loop transport** (DONE 2026-07-19). The signal path is
  gone: `workflow_blocks.batch_submit_and_wait` mints `request_id` via
  `workflow.uuid4()` in the workflow (closing the submit-retry orphan window),
  submits, `workflow.sleep()` + a thin `batch_status` activity loop (default
  600s, floor 300s twice) until `ended`, then one `fetch_batch_result`
  activity (provider threaded; claim-check inline ≤256KB else S3). Deleted:
  `BatchPollerWorkflow` + Schedule, `poll_batch_results`/`batch_poll.py`,
  every signal handler, the T1.2/T1.3 interims, forge's `BatchResult`.
  `batch_jobs` became a behavioral audit ledger (no migration): monotonic
  `update_batch_status` (WHERE-guard on `submitted`, error-preserving),
  statuses `submitted → ended | failed | expired | missing` (`processing`
  legacy). The **timeout tree** the June review missed was closed —
  mode-aware `_child_timeout` and `derive_execution_timeout` size timeouts
  from the permitted-wait budget against the 25h `BATCH_WAIT_CEILING`, with
  `MAX_PLAN_STEPS = 25` bounding planned mode. Replay coverage landed with it.
- **T4.2 — ocr owns its batches; SPI deleted** (DONE 2026-07-19). ocr submits
  (`submit_ocr_batch`) and polls its own Mistral batches on the **shared**
  `sax_platform.temporal.polling` loop (`wait_batch_ended` + `BackoffSchedule`
  300s → ×2 → 1800s cap, 10% jitter); forge cut over as the loop's first
  consumer with the T4.1 replay histories passing **unregenerated** (the
  extraction's determinism proof). `OcrGatherWorkflow` is deleted for
  parent-awaited children — a failed chunk fails the document promptly, the
  verified 26h hang is structurally dead, and one `fetch_and_store` activity
  keeps result bytes out of workflow history. The cross-queue batch SPI is
  deleted end to end (`submit_batch_blob`, `BatchSubmitSpiInput`, the
  `BatchResult` envelope, `BATCH_RESULT_SIGNAL`); **forge is anthropic-only**
  (non-anthropic providers raise) and **zero `@workflow.signal` remain
  platform-wide**. Six ocr activities gained typed inputs;
  `OcrProcessingStatus`/`OcrJobDerivedStatus` enums with an exhaustive pure
  `_derive_status`.
- **T4.3 — transport decisions sweep** (DONE 2026-07-20; docs-only). D77/D78
  already carried their D88 supersession banners and D81 already recorded the
  600s-vs-60s drift, so the real work was reaffirmation banners on D76/D79, an
  as-shipped note under D80, an `Amended by D100` banner on D88, the new
  **D100** entry (Phase 4 transport as shipped), and pbook
  `TEMPORAL_PATTERNS.md` rule 8 gaining the scoping note (interval ≥300s;
  ~11 history events/poll; continue-as-new escape) pointing at
  `sax_platform.temporal.polling`.
- **Session extras.** (a) A post-T4.1 owner-directed follow-up split
  `MistralOcr.poll_batch` into `get_batch_status` (one `jobs.get`, never
  downloads) and `fetch_batch_results`, killing the duplicate `output_file`
  download per completed batch. (b) An **OCR docs/scripts/skills audit**
  (bundled into T4.2): fixed the `ocr list --status` filter bug (it filtered
  the raw column while help advertised the derived vocabulary); rebuilt the
  broken `batch-status` skill on FORGE_DB_URL/SQLAlchemy (read-only,
  parameterized — was SQLite-only with a wrong filter and a nonexistent
  column); rewrote `docs/reference/mistral.md` off the deleted chat lane;
  **deleted five redundant/dead ocr scripts** (kept `nushell/ocr.nu`); fixed
  the `mistral-batch` skill jq, launchd interval, DEPLOYMENT diagram, TOC.
  (c) **PROCESS.md gained a "Specification Changes" policy** — a change
  contradicting a decision needs a new DECISIONS entry plus a
  supersession/amendment banner **in the same change-set** (R-number if it
  reverses direction); unbuilt-task scope changes land as dated task
  amendments; unadopted designs get parked-proposal notes.

Gates at the T4.2 close: forge **1348** / pbook **353** / ocr **152** /
sax-platform **439**; mypy-strict ×4; import-linter 6 kept; ruff +
markdownlint clean.

## What Phase 5 needs to know

Phase 5 rewrites the workflow layer (`workflows.py` monolith → shared
`step_logic.py` + `blocks/`), which is exactly what the Phase 4 nets guard.

- **The committed replay histories under `tests/replay/` are the determinism
  guard for the Phase 5 rewrites.** `tests/test_replay.py` replays six
  committed histories through a temporalio `Replayer` and fails on
  divergence. Each rewrite (T5.1–T5.4) regenerates and replays as it lands;
  **regenerate ONLY on deliberate logic changes** via `make replay-histories`
  (or `uv run python -m tests.replay.regenerate`). Histories are
  **shape-stable, not byte-stable** — run ids/timestamps/minted uuids differ
  every regeneration, so expect diffs; the event shape is what must hold. T5.5
  maintains and completes this scaffold, it does not introduce it.
- **The batch transport surface Phase 5 refactors against** is the timer loop
  in `workflow_blocks.batch_submit_and_wait` over
  `sax_platform.temporal.polling` (`wait_batch_ended`, `FixedInterval`,
  `BATCH_WAIT_CEILING`). `BATCH_WAIT_FAILURES` is now `(ApplicationError,)` —
  the signal-era failure tuple collapsed when the signal path died.
- **The batch test pattern is three mocks by name** — submit/status/fetch
  activities registered on a per-test `Worker` — which replaced the signal
  stubs. There is no signal path left to test against.
- **The timeout-tree formulas live in the [T4.1 Dev Notes](tasks/T4.1-forge-timer-loop-transport.md)**
  (per-path wait table, `_child_timeout`, `derive_execution_timeout`).
  `MAX_PLAN_STEPS = 25` is a `max_length` validator on `Plan.steps` (oversized
  plan → retryable parse failure) — any planner-touching rewrite (T5.6) must
  respect it.
- **T5.5's pre-existing ScenarioState plan** replaces the 30 module-level
  `global` statements + 11 `_reset_*` functions in `test_workflows.py` with
  per-test closures. **Do not add new `global` statements** there meanwhile.
- **The T5.x amendment sections each carry non-obvious scope** — read them,
  not just the Problem: T5.1 makes result builders carry file contents exactly
  once (today 3–4×, nearing the 2MB payload cap on large fan-outs) and
  **deletes the `evaluate_transition` activity + its mocks**; T5.2 must wrap
  the worktree in try/finally (no leaked `forge/<task_id>` branch) and make
  `commit_changes`/`create_worktree` idempotent under retry; T5.3 must isolate
  per-child failure (a raising child → failed SubTaskResult, siblings finish),
  choose an **explicit ParentClosePolicy**, and give all four dispatch arms
  interaction records (exploration writes none today); T5.6 must gate
  REVISE-spliced plans with a revision/step cap (the one unbounded history
  path) and reject an empty `LLMResponse` (`files=[] + edits=[]`) at parse.

## Operational notes

- **The podman machine must be running** for pbook's suite (pgvector
  testcontainer) and the `-m postgres` tests — it does not restart itself
  after a reboot; a stopped machine surfaces as all-353-errors in pbook.
  `podman machine start`, then re-run.
- The ambient shell env points at production (`FORGE_DB_URL` → Supabase,
  `AWS_*` → real S3); the local stack's Postgres is on **5434** on this
  machine. Override before any local DB/blob command.
- `uv sync --all-packages`; run each member's suite from its own directory;
  `make gates` is the CI mirror.
- **New in Phase 4:** `ocr submit` is **start-only** (echoes the workflow_id;
  the submit workflow runs up to the batch ceiling); the ocr worker **fails
  fast without `MISTRAL_API_KEY`** (was optional); the rebuilt **`batch-status`
  skill needs `FORGE_DB_URL`** and is read-only.

## Parked and operator-optional (carried forward)

- **T4.2 cache-refresher discussion.** Parked; the cache-refresher variant is
  the owner's frontrunner. Now grounded by the recorded burst data (one
  Mistral batch per *chunk* → a 1,000-doc burst = ≥1,000 concurrent waiters;
  per-doc 1–1440 min, mode ≈ 30) plus a third option — batch **coalescing**
  (≤100,000 requests/batch dissolves the waiter count at the source). Full
  agenda in the
  [T4.2 Dev Notes](tasks/T4.2-ocr-own-polling-gather-restructure.md); **still
  blocked on the owner reading real Mistral RPS/TPM from
  `admin.mistral.ai/plateforme/limits`** before anything is sized.
- **`batch_jobs` spend columns** (token/model at final fetch) were considered
  and deliberately deferred — no decision specced them (D100); a candidate D80
  amendment if wanted.
- **The env-gated live ocr e2e** is runnable on demand (real key + bucket +
  `OCR_E2E_PLATFORM=1`; command in the
  [T3.3](tasks/T3.3-mistral-ocr-chat-deleted.md) /
  [T4.2](tasks/T4.2-ocr-own-polling-gather-restructure.md) headers) — never
  fired autonomously (real spend).
- **The eval-judge benchmark** stays deferred — the three coded ACs shipped in
  [T0.6](tasks/T0.6-eval-judge-integrity.md); the benchmark that would settle
  the Opus judge pin empirically is operator-optional.
- **The frozen-Gherkin falseness list** is recorded in the
  [T8.4 Dev Notes](tasks/T8.4-final-sweep.md):
  `docs/requirements/batch_processing.feature` still describes the deleted
  signal poller. The specs are frozen; T8.4 or T8.2 owns their disposition —
  informational, no action now.

## Addendum (2026-07-21, later)

**T4.4 was added after this handoff was written** — owner adoption of the
parked T4.2 batch-tracker proposal in its stateless-broadcast form (a
2-minute list-endpoint sweep signaling status hints to signal-wait store
children; design settled and recorded in
[tasks/T4.4-mistral-status-tracker.md](tasks/T4.4-mistral-status-tracker.md)).
It is ocr-local and disjoint from Phase 5's forge files; sequencing
(T4.4 first vs parallel with T5.1) is the owner's call. The "parked
cache-refresher" item below is thereby resolved — superseded by T4.4's
design; the Mistral admin-console rate numbers are no longer gating.

## Where to start

**Either T4.4 or T5.1 — the owner sequences.** For **T4.4** (ocr Mistral
status tracker): the design is fully settled in the task file — a
stateless 2-minute tracker sweeping the Mistral list endpoint and
broadcasting status hints to signal-wait store children; read its Design
section, then write the Plan per [PROCESS.md](PROCESS.md). It is ocr-local
(plus one new `MistralOcr` method and one platform retirement) and cannot
collide with Phase 5's forge files. Its D101/banner obligations land in
the same change-set per PROCESS.md's "Specification Changes".

**T5.1 — pure step logic.** Read the T5.1 task file's Problem/Scope/amendment
notes first (it deletes the `evaluate_transition` activity and fixes the
result-builder content duplication), then write the Plan section per
[PROCESS.md](PROCESS.md) before coding. Phase ordering is load-bearing:
5 → 6 (serialized; Phase 6 after Phase 5), Phases 6 and 7 may run in parallel,
Phase 8 closes and tags v1.0. Within Phase 5 the chain is strict —
T5.1 → T5.2 → T5.3 → T5.4, with T5.6 depending on T5.1 and T5.5 on T5.4.
