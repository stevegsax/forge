# Handoff — Phase 0 complete; start Phase 4

**Date:** 2026-07-19
**Status:** Landed. **Phases 0–3 are all complete** (T0.1–T0.8, T1.0–T3.6).
`make gates` is EXIT=0. Nothing is in flight. Next task: **T4.1**.

This is a state-of-the-world note for the next session. It does not restate
the work queue ([TASKS.md](TASKS.md)), the decisions
([../docs/DECISIONS.md](../docs/DECISIONS.md)), or the status-of-record
([../docs/OVERVIEW.md](../docs/OVERVIEW.md)) — it says what changed since the
[Phase 3 handoff](HANDOFF-2026-07-18-phase3-complete.md), what to know before
touching anything, and where to start. The Phase 3 handoff remains the
detailed record of the platform library and composition-root state; everything
in its "What to know" section still holds.

## Status — Phase 0, task by task (all closed 2026-07-19)

- **T0.1 — observability defaults.** The root logger sits at INFO with the
  `forge` logger at DEBUG, so third-party DEBUG (SDK payload logging — the
  prompt-leak channel) no longer reaches `forge.log`/`worker.log` while
  forge's own DEBUG still does. The tracing exporter defaults to **`none`**
  (was console): a bare worker emits no per-span stdout; set
  `FORGE_OTEL_EXPORTER=console|otlp` to opt in. `FORGE_OTEL_ENDPOINT` was
  verified already deleted (T3.6).
- **T0.2 — CLI helper fixes.** The two ~45-line duplicated submit helpers are
  one overloaded `_submit(task_input: ForgeTaskInput, temporal_address, *,
  wait)` — the 14-parameter `ForgeTaskInput` mirror is gone and the
  `sync_mode=True` default inversion is structurally impossible (the flag
  rides inside `ForgeTaskInput`, default batch). Eval cases are discovered
  once per `eval-planner` run and passed as a list.
- **T0.3 — dead code.** `build_interaction_dict`, `scripts/basenames.py`, and
  the forge-level `scripts/nushell/` are deleted (`apps/ocr/scripts/nushell/`
  is the real, kept module). `boto3` moved from forge's `[project]`
  dependencies to the dev group (forge's moto fixture imports it; runtime
  access is transitive via sax-platform).
- **T0.4 — docs truth sweep.** Organizational pass the earlier truth passes
  never did: the stale 2026-06-10 tasks-handoff copy and three completed
  top-level plans moved into `development-plans/archive/` (all links swept);
  `archive/to-merge/` → `archive/merged/`; D1 and D8 got their missing
  supersession banners; **`diataxis/` is deleted** (stale generated output —
  regenerate later if wanted, never authoritative); README/TOC/OVERVIEW/
  requirements staleness fixed.
- **T0.5 — idempotent, path-safe edit application.** `write_output` is
  stage-then-write: all paths resolved, the files/edits mutual-exclusion
  check runs on **resolved** paths, and every output is computed in memory
  before the first disk write. Retries idempotent-skip already-applied edits
  (residual ambiguities documented in `activities/output.py`). Proven by an
  OSError-mid-write retry regression test and the repo's first `hypothesis`
  property suite (`tests/test_output_properties.py`, default suite).
- **T0.6 — eval judge integrity.** Judge prompts now carry a capped repo
  file listing (`format_repo_context`, max 200 files) from both callers;
  `JudgeVerdict` validates exactly one score per criterion (violations
  surface as retryable `LLMSchemaMismatch` at the parse seam); the ±0.5 band
  is `JUDGE_SCORE_SIGNIFICANCE_BAND` with a standing small-sample caveat in
  `EvalComparison.summary`. The judge **benchmark** stays deferred
  (operator-optional; see below).
- **T0.8 — operating decision.** Closed retrospectively: forge did operate
  unattended pre-Phase-4 (D99), the T1.2/T1.3 interims were justified and
  shipped, and T4.1 owns their deletion. Recorded in the TASKS.md Phase 1
  preamble.

Gates at close: forge **1327** / pbook **353** / ocr **118** / sax-platform
**398** tests; mypy-strict ×4; import-linter 6 kept / 0 broken.

## What Phase 4 needs to know

- **The transport surface T4.1 replaces is exactly as T3.6 left it** — Phase 0
  did not touch it. The signal path lives in: `BatchActivities` bound methods
  (`submit_batch_request`, `submit_batch_blob`, `poll_batch_results`,
  `parse_llm_response` in `forge/activities/roots.py`, delegating to
  `batch_submit.py`/`batch_poll.py`/`batch_parse.py`), `BatchPollerWorkflow` +
  the `forge-batch-poller` Schedule, `BATCH_RESULT_SIGNAL` consumption in
  `workflow_blocks.batch_submit_and_wait`, and the T1.2 correlation + T1.3
  poller interims. T4.1 deletes all of it in favor of per-workflow timer-loop
  polling (D88, reversal R1). The platform batch lane it builds on
  (`sax_platform.llm.batch`: builder, `submit_batch`, `get_batch_status`,
  `fetch_batch_result_lines`, `classify_result_json`) is stable and tested.
- **T4.1's `batch_jobs` reduction is the first ALTER-shaped migration** — the
  Alembic hardening (render_as_batch, compare flags, advisory lock) landed in
  T3.4 and is verified green on SQLite and Postgres; autogenerate is
  trustworthy on both chains.
- **Debugging note:** with T0.1's default, a bare worker emits no spans — set
  `FORGE_OTEL_EXPORTER=console` when you want them during Phase 4 work.
- **Everything in the [Phase 3 handoff](HANDOFF-2026-07-18-phase3-complete.md)
  "What to know" section still applies**: settings are the only keyed env
  readers (documented exceptions); activities register bound methods; the
  sandbox-light discipline (`tests/test_sandbox_light.py`); no env-reading
  fallbacks; the operator-visible behavior changes.

## Operational notes

- **The podman machine must be running** for pbook's suite (pgvector
  testcontainer) and the `-m postgres` migration tests — it does not start
  itself after a reboot; a stopped machine surfaces as all-353-errors in
  pbook. `podman machine start`, then re-run. (Bit us at Phase 0 close.)
- The ambient shell env points at production (`FORGE_DB_URL` → Supabase,
  `AWS_*` → real S3); local stack Postgres is on **5434** on this machine.
- `uv sync --all-packages`; per-package suites from each package's own
  directory; `make gates` is the CI mirror.

## Parked and operator-optional (carried forward)

- **T4.2 owner proposal — batch-tracker for the peaky Mistral case.** Parked
  in the [T4.2 Dev Notes](tasks/T4.2-ocr-own-polling-gather-restructure.md)
  (pointer from [T4.1](tasks/T4.1-forge-timer-loop-transport.md)); the
  **cache-refresher variant is the owner-designated frontrunner** — a tracker
  refreshes `batch_jobs` from the provider list endpoint (~0.5 RPM regardless
  of burst size); waiters poll the local ledger. Operational data recorded
  there: 1,000-doc bursts are ordinary; per-doc processing 1–1440 min,
  mode ≈ 30. **Read this before Phase 4 planning** and confirm Mistral's rate
  limits + batch-jobs-per-burst shape first.
- **Operator-optional, pending:** the ocr env-gated e2e (runnable — command in
  the [T3.3 task file](tasks/T3.3-mistral-ocr-chat-deleted.md)); the eval
  judge benchmark ([T0.6](tasks/T0.6-eval-judge-integrity.md) Dev Notes —
  T0.6's three coded ACs shipped; the benchmark that would settle the Opus
  judge pin empirically remains deferred).

## Where to start

**T4.1 — forge: submit → poll-loop → fetch.** Read the T4.1 task file's
Problem/Scope/amendments and the **parked T4.2 owner proposal first**, then
plan per [PROCESS.md](PROCESS.md). Phase ordering is load-bearing
(4 → 5; Phase 6 after Phase 5; Phases 6 and 7 may run in parallel; Phase 8
closes and tags v1.0). With Phase 0 done there is no independent side queue —
the path is strictly through the phases.
