# Handoff — T4.4 + T0.9 + staging lane complete; start T5.1

**Date:** 2026-07-23
**Status:** Landed. **Phases 0–4 are fully complete** (T0.1–T0.9,
T1.0–T4.4) plus the owner-directed staging-lane work that followed. All
suites green as of the last full run: forge **1368** / pbook **373** /
ocr **236** / sax-platform **482**; mypy ×4; import-linter 6 kept; the
last `make gates` EXIT=0 plus per-suite verification after each later
change. Nothing is in flight; the working tree is clean. Next task:
**T5.1 — pure step logic (D95)**, opening Phase 5.

This is a state-of-the-world note for the next session. It does not
restate the work queue ([TASKS.md](TASKS.md)), the decisions
([../docs/DECISIONS.md](../docs/DECISIONS.md)), or the status-of-record
([../docs/OVERVIEW.md](../docs/OVERVIEW.md)) — it says what changed since
the [2026-07-21 handoff](HANDOFF-2026-07-21-phase5-start.md), what to know
before touching anything, and where to start.

## What landed since the last handoff

- **T4.4 — ocr Mistral status tracker (D101, 2026-07-22).** Per-waiter
  provider polling is gone: the stateless `OcrBatchTrackerWorkflow` (120s
  Schedule, overlap=SKIP, installed by the reintroduced `_ensure_schedule`
  guard at ocr worker startup) sweeps the Mistral batch **list** endpoint
  once per cycle and broadcasts `ocr_status_hint` signals to the
  ledger-recorded store children; `OcrStoreWorkflow` is a signal-wait state
  machine (pure `next_state` table; unchanged 25h ceiling → MISSING; hints
  never authoritative). Idle cycles make zero Mistral calls. The single-row
  `ocr_tracker_heartbeat` table (ocr Alembic 002) makes a stale tracker a
  system alert; there is deliberately no per-task fallback. Deleted:
  `ocr_batch_status`, platform `BackoffSchedule` (forge's `FixedInterval`
  loop untouched; replay histories green unregenerated).
- **`ocr tracker-status`** — operator health probe, Temporal-free by
  design (the sanctioned infrastructure exception to Principle 8): prints
  `checked_at_gmt` first always, then heartbeat fields and a
  `fresh`/`stale`/`never-ran` verdict. Exit codes: 0 fresh · 1 stale or
  never-ran, no live jobs · 2 stale **with** live jobs (work queued but
  tracker not cycling — check worker + Mistral) · 3 probe error (missing
  `FORGE_DB_URL`/unreachable DB) · 78 guard failure.
- **T0.9 — store rehomed + explicit-environment guard (D102, 2026-07-22).**
  Application store of record moved from Supabase to the local
  `forge-postgres` podman instance: databases `forge` (forge + ocr) and
  `pbook` beside Temporal's, migrated with exact row-count verification;
  `PBOOK_DATABASE_URL` set for the first time (pbook's store had been
  silently disabled on workers). Supabase is retired/frozen; final dumps in
  `~/.local/state/forge/supabase-migration-2026-07-22/`. Nightly
  `com.saxcapital.db-backup` agent (03:30 → S3 `db-backups/`), verified by
  a manual run. The `FORGE_ENV` guard fronts all six entrypoints: prod/dev/
  test, **no default**, prod requires the tagged profile **and**
  `FORGE_PROD_ACK=yes`, failures exit 78.
- **Staging lane (owner-directed, 2026-07-22/23).** Per-environment
  Temporal namespace isolation: prod = `default`, dev = `forge-dev`
  (registered, 72h retention), with `require_namespace_coherence` enforced
  before every connect (prod must use `default`; dev/test must not — an
  incoherent process fails fast instead of polling prod queues). `--env
  <name-or-path>` on every CLI in **either position** (subcommand-level
  wins; never supplies the prod ack; `--help` works env-less). `ocr
  migrate` runs the ocr chain standalone. `make dev-worker
  [WORKER=ocr|forge|pbook]` starts a crash-safe detached tmux worker (dead
  panes retained, output tee'd to
  `~/.local/state/forge/logs/dev-<worker>-worker.log`, honest
  running/CRASHED/died-at-startup status).
- **Proven end to end:** a 43-page document ran the full pipeline in
  `forge-dev` — chunked submit → real Mistral batches → tracker sweep →
  hint broadcast → signal-wait store → reassembly, `stored` ×2 — without
  touching production.
- **Architecture Principle 8** (owner, 2026-07-23): application access is
  through Temporal; no direct DB connections outside workers except the
  named infrastructure tooling. See the principle text in CLAUDE.md and
  the parked corollaries in TASKS.md.

## What to know before touching anything

1. **No process reaches a database without `FORGE_ENV`.** A bare shell has
   neither `FORGE_DB_URL` nor `FORGE_ENV`. Interactive prod:
   `set -a; source ~/.config/forge/envs/prod.env; set +a; export
   FORGE_ENV=prod FORGE_PROD_ACK=yes` — the `set -a` is load-bearing (plain
   `source` doesn't export; the guard rejects unexported tags by design).
   Dev: `--env dev` on any CLI command, either position.
2. **The deployment is the working tree.** launchd workers exec `uv run`
   from this repo — `make workers-restart` deploys whatever is on disk,
   and a KeepAlive relaunch mid-edit can crash-loop on mixed code
   (observed 2026-07-22). Exercise changes in the staging lane first;
   restart prod only on a committed, gates-green tree.
   (Deploy-from-committed-ref is a parked follow-up.)
3. **The dev lane needs the worker pair.** ocr's ledger writes
   (`persist_block`) are cross-queue to `forge-task-queue`; with only a
   dev ocr worker, submissions block at the first persist and the tracker
   logs "no routable batch_jobs row" (observed live). `make dev-worker` +
   `make dev-worker WORKER=forge`.
4. **Dev tmux workers may still be running from this session**
   (`dev-ocr-worker`, `dev-forge-worker`, plus the owner's
   `dev-pbook-worker`). They are disposable: `tmux kill-session -t <name>`.
   Their crash forensics persist in the dev-worker logs.
5. **Health checks:** `ocr tracker-status --env dev` (or prod with the
   full declaration) is the first probe; `temporal task-queue describe
   [-n forge-dev] --task-queue <q>` shows pollers;
   `make workers-status` shows processes. Prod tracker heartbeat has been
   `fresh` since deployment.
6. **Owner process preferences** (recorded in session memory, restated
   here for continuity): implementation work goes to Opus subagents with
   small, hard-bounded scopes; commits happen only on the owner's explicit
   word; the owner wants weak arguments challenged, not praised.
7. **Supabase is frozen, not gone.** Do not reconnect anything to it; the
   owner decommissions on their own schedule.

## Where to start

- **T5.1 — pure step logic (D95)**: read the task file's Problem/AC, D95,
  and the "Universal Workflow Step" section of CLAUDE.md. Note the
  standing warning: do not add logic to the `evaluate_transition` activity
  — T5.1 inlines it into pure step logic.
- Alternatively, the owner may direct grooming of the **parked follow-ups**
  (TASKS.md, 2026-07-23 section): bounded first-step CLI waits (owner-
  adopted direction), deploy-from-ref, CLI glue consolidation, e2e → dev
  lane, pbook direct-DB review, large-result claim-check (parked).
- Phase ordering unchanged: 5 → 6 (6∥7) → 8; v1.0 tags at T8.4, not
  before.
