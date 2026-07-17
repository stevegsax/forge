# Handoff — Phase 2 close + Phase 3 platform library (T3.1–T3.3)

**Date:** 2026-07-17
**Status:** Landed and shipped. PRs #32–#41 are merged to `main`; the deployment
is live and running graceful-drain workers. Nothing is in flight. Next task:
**T3.4**.

This is a state-of-the-world note for the next session. It does not restate the
work queue ([TASKS.md](TASKS.md)), the decisions ([../docs/DECISIONS.md](../docs/DECISIONS.md)),
or the status-of-record ([../docs/OVERVIEW.md](../docs/OVERVIEW.md)) — it says
what changed since the [monorepo/deployment handoff](HANDOFF-2026-07-16-monorepo-deployment.md),
what to know before touching anything, and where to start.

## What changed

**Phase 2 closed** (PRs #32–#36):

- **T2.2 — root gates.** GitHub Actions CI (`.github/workflows/ci.yml`, **8
  jobs**: `lint` + the six per-package suites + `test-postgres-migrations` for
  the postgres marker); `import-linter` with **6 DAG contracts** in
  `pyproject.toml` (apps↛forge; forge↛ocr; libs↛apps/each-other;
  contracts↛LLM-SDKs; pbook⊥ocr; the three libs⊥each-other); every package
  coverage-gated at 85%; `make gates` mirrors CI exactly and bare `make` prints
  help.
- **T2.3a–d — mypy strict** across all six packages. Zero ignores except the
  **16 recorded temporalio stub-boundary ignores** in
  `apps/pbook/src/pbook/workflows/cli_ops.py` (`[no-any-return]` on
  `execute_activity`).

**Phase 3 T3.1–T3.3 landed** (PRs #37–#39) — `libs/sax-platform` is born:

- **T3.1 — platform LLM client** ([task](tasks/T3.1-platform-llm-client.md)).
  `sax_platform.llm` structured-outputs client on both lanes, typed
  refusal/truncation/mismatch outcomes (exceptions on the sync lane, values on
  the batch lane), classify-before-parse, `max_retries=0`, required
  `max_tokens`, opt-in caching with per-model minimum prefixes. Recorded
  pre-work: the production interactions store had **zero rows** (settled rec 7 →
  caching off by default) and the structured-outputs/batch/caching claims were
  re-verified against the live API.
- **T3.2 — one tier registry (D94)**
  ([task](tasks/T3.2-tier-registry-thinking-migration.md)).
  `sax_platform.llm.tiers` is THE registry: `opus-4-8` / `sonnet-5` /
  `haiku-4-5`, live-verified 2026-07-16. `budget_tokens` is deleted
  platform-wide (400 on the new pins, proven live; omitting `thinking` runs
  adaptive BY DEFAULT, so *disabled* is now an explicit wire shape); CLI
  `--effort` (low..max incl. xhigh) replaced `--thinking-budget` and warns in
  single-step mode; 11 shadow fallback literals are registry-resolved; forge now
  depends on sax-platform; SDK-importing surfaces are lazy-exported for
  workflow-sandbox safety.
- **T3.3 — MistralOcr in the platform**
  ([task](tasks/T3.3-mistral-ocr-chat-deleted.md)). `sax_platform.ocr` owns
  `MistralOcr` (injected client; file-based `/v1/ocr` submit,
  error-file-merging poll, image extraction, new sync `process()`); forge's SPI
  mistral branch uses it; `sax_llm`'s `mistral.py` **and** chat support are
  deleted (zero users verified), `mistralai` moved to sax-platform; ocr installs
  a `MistralOcr` DI seam (`apps/ocr/src/ocr/deps.py`) at worker startup for
  Phase 4's self-polling.

**Phase 3 review + fixes** (PR #40): a workflow-backed review of the whole
Phase 3 diff confirmed 10 defects; **9 fixed** — explicit 16384 caps on
planner/sanity/conflict (adaptive thinking competes inside the cap now);
`interactions.stop_reason` (migration **003**) + WARN logs on `max_tokens`
truncation (owner-mandated token telemetry); a before-validator so legacy
`ThinkingConfig` payloads deserialize correctly (they used to silently flip
thinking ON); shared batch fallback is thinking-disabled; null-`response` guards
in both Mistral result-file paths; `make_mistral_client` raises on missing
`MISTRAL_API_KEY`; empty-endpoint normalization; CLI provider validation; one
cached mistral resolver. **1 deferred**: the eval judge's Sonnet→Opus tier jump
stands until [T0.6](tasks/T0.6-eval-judge-integrity.md) runs an empirical judge
benchmark (its own Problem statement says the current baselines are unreliable).

**Worker ops** (PR #41 + Makefile commits): all three workers drain gracefully
on SIGTERM/SIGINT (30s) and exit 0; launchd `KeepAlive` relaunches on on-disk
code; `make workers-restart` / `workers-status` (friendly no-match output). The
standard restart is now `make workers-restart`.

## What to know before touching anything

- **Ambient env still points at production** — unchanged (`FORGE_DB_URL` →
  Supabase, `AWS_*` → real S3; local stack Postgres on **5434**). Override before
  any local DB/blob command.
- **The deployment is live and now drains gracefully.** Verified up this
  session: launchd `forge-worker-1/2` + `ocr-worker` running on `main`; podman
  `forge-postgres` / `forge-minio` / `forge-temporal`(+UI) healthy;
  `MISTRAL_API_KEY` present in `~/.config/forge/forge.env` (the ocr worker
  fail-fasts without it). Migration **003** is the alembic head in-repo
  (`down_revision = "002"`) and, per the operator, is applied to production
  (`alembic_version_forge=003`; `interactions.stop_reason` exists). Restarted
  2026-07-17 ~11:26 (operator-reported).
- **forge's runtime LLM path has NOT moved yet.** It still uses `sax_llm`'s
  anthropic provider + forced tool use until **T3.5/T3.6**; the signal-based
  batch transport + shared poller are live until **Phase 4**. `sax_platform`
  exists and forge depends on it, but the step/planner calls don't route through
  it yet.
- **Setup/test discipline unchanged**: `uv sync --all-packages`; suites run
  per-package from each package's own directory (`make gates` runs the CI set).
  sax-platform is the sixth package now.

## Parked and operator-optional

- **T4.2 owner proposal — batch-tracker for the peaky Mistral case.** Parked in
  the [T4.2 Dev Notes](tasks/T4.2-ocr-own-polling-gather-restructure.md)
  (pointer from [T4.1](tasks/T4.1-forge-timer-loop-transport.md)); the
  **cache-refresher variant is the owner-designated frontrunner** — a tracker
  refreshes `batch_jobs` from the provider list endpoint (~0.5 RPM regardless of
  burst size); waiters poll the local ledger; continuous two-way reconciliation.
  Operational data recorded there: 1,000-doc bursts are ordinary; per-doc
  processing 1–1440 min, mode ≈ 30. **Read this before T4.2 planning** and
  confirm Mistral's rate limits + batch-jobs-per-burst shape first.
- **Operator-optional, pending**: the ocr env-gated e2e (now runnable — command
  in the [T3.3 task file](tasks/T3.3-mistral-ocr-chat-deleted.md)); the judge
  benchmark ([T0.6](tasks/T0.6-eval-judge-integrity.md)).

## Where to start

**T3.4 — platform plumbing modules** is the next task. **T3.5**
(forced-tool-use retirement) and **T3.6** (composition roots everywhere) then
close Phase 3 and move forge's runtime onto `sax_platform`. **Phase 4**
(timer-loop batch transport) follows — read the parked T4.2 proposal first.
Phase 0 (T0.1–T0.6, T0.8) remains independent of everything and can land anytime.
