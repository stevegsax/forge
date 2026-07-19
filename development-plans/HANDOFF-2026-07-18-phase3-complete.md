# Handoff — Phase 3 complete (composition roots, T3.6)

**Date:** 2026-07-18
**Status:** Landed. **Phase 3 is complete** (T3.1–T3.6). `make gates` is
EXIT=0. Nothing is in flight. Next phase: **Phase 4 (T4.1)**.

This is a state-of-the-world note for the next session. It does not restate the
work queue ([TASKS.md](TASKS.md)), the decisions ([../docs/DECISIONS.md](../docs/DECISIONS.md)),
or the status-of-record ([../docs/OVERVIEW.md](../docs/OVERVIEW.md)) — it says
what changed since the [T3.1–T3.3 handoff](HANDOFF-2026-07-17-phase3.md), what
to know before touching anything, and where to start.

## Status — Phase 3, task by task

- **T3.1 — platform LLM client.** `sax_platform.llm`: structured-outputs client
  on both lanes (sync `complete[T]` via `messages.parse`; batch request builder +
  submit/status/fetch via `output_config.format` with stored-bytes classification
  at the fetch/parse seam), typed refusal/truncation/mismatch outcomes,
  `max_retries=0`, required `max_tokens`, opt-in caching.
- **T3.2 — one tier registry (D94).** `sax_platform.llm.tiers` is THE registry
  (REASONING → `claude-opus-4-8`, GENERATION/SUMMARIZATION → `claude-sonnet-5`,
  CLASSIFICATION → `claude-haiku-4-5`); `budget_tokens` deleted platform-wide,
  thinking adaptive-by-default or explicitly disabled; CLI `--effort` replaced
  `--thinking-budget`.
- **T3.3 — MistralOcr in the platform.** `sax_platform.ocr` owns `MistralOcr`
  (injected client) and `mistralai`; forge's SPI mistral branch uses it.
- **T3.4 — platform plumbing.** `sax_platform.contracts` (sandbox-light: batch
  wire models, `persist_block`, `S3Blobs`, `UTCDateTime`, constants, the
  read-only `batch_jobs` mirror) + `sax_platform.{temporal,db,embeddings,config,logging}`;
  forge-contracts retired; forge/ocr/pbook migrated onto the platform.
- **T3.5 — forced tool use retired.** forge and pbook complete structured
  outputs through `sax_platform.llm`; both string-keyed output-type registries
  replaced by frozen `OUTPUT_TYPES` mappings; `libs/sax-llm` deleted.
- **T3.6 — composition roots everywhere (D93; this task).** Detailed below.

## What changed (T3.6)

**The platform library as it now stands.** `sax_platform` is the shared runtime
for forge, ocr, and pbook: the both-lane LLM client + tier registry (`.llm`,
`.llm.tiers`), `MistralOcr` (`.ocr`), the sandbox-light contracts layer
(`.contracts` — now exposing the **`S3Blobs(bucket, prefix)` class** instead of
the old `s3_blobs` module functions), the plumbing modules (`.temporal`, `.db`,
`.embeddings`, `.config`, `.logging`), and a new test-only module
(`sax_platform.testing`: `temporal_env`, `FakeLLM`, `FakeMistralOcr`).

**Composition roots.** Every process (forge worker + CLI, ocr worker, pbook
worker + CLI) now has a composition root:

- **Frozen settings, read once.** `ForgeSettings` / `OcrSettings` /
  `PbookSettings` are built once at each worker main and fail fast on missing
  required config. CLI commands construct only the settings groups they need
  (so `forge run` doesn't demand a `FORGE_DB_URL` it never uses). Settings
  groups are the **only keyed environment readers** left.
- **Class-based activities with bound methods.** forge: `StoreActivities`,
  `ContextActivities`, `LlmActivities`, `BatchActivities` (in
  `forge/activities/roots.py`); ocr: `OcrStoreActivities`; pbook:
  `StoreActivities`, `LlmActivities`, `EmbeddingActivities` (in `pbook/roots.py`).
  Bound methods keep their `__name__`, so workflows still invoke by string name
  and the by-name workflow-mock test files were untouched. No-dependency
  activities (git, validate, output, the pure context assemblers,
  `evaluate_transition`, `prepare_transcript`) stay free functions.
- **One engine + one SDK client per process.** The store engine, Anthropic SDK
  client, S3 blob client, and (when `MISTRAL_API_KEY` is set) the Mistral client
  are each built exactly once at startup and injected. This fixed a live defect:
  the platform engine factory has no cache, so ocr previously built a fresh
  Postgres pool (`pool_size=5 + max_overflow=5`) per activity call (×12).
- **All nine module-global seams deleted:** forge `get_llm`, both batch-client
  caches, `set_temporal_client`, the `_mistral` cache; ocr `deps.py`; pbook
  `set_provider`/`_client`/`_engines`. `sax_platform` was already at zero.

**Also in T3.6:** `get_store_url` deleted; `make_mistral_client(api_key)` is now
required (no env fallback); pbook's worker adopted `connect_temporal` + the
platform `run_worker` scaffold; pbook migrations pass the URL via alembic
`cfg.attributes` (the runtime `os.environ` write is gone); the tracing
private-API reset (`_TRACER_PROVIDER_SET_ONCE`) and the dead
`FORGE_OTEL_ENDPOINT` constant were deleted; ocr gained logging for the first
time (`setup_logging` + `silence_noisy_loggers`).

**The four-package workspace.** forge, pbook, ocr, sax-platform. `make gates`
EXIT=0: forge 1299 / pbook 353 / ocr 118 / sax-platform 398 tests; mypy ×4;
import-linter 6 contracts kept; postgres migration suite 7 passed. The T3.6 AC
sweeps (no module-level mutable clients/engines/registries; tests construct
classes with fakes, no monkeypatched globals) all returned ZERO.

## What to know before touching anything

- **Settings are the only keyed env readers.** Add new configuration as a field
  on a settings group, not a point-of-use `os.environ` read. The deliberate,
  documented exceptions are: XDG path-convention reads (e.g. `eval/runner.py`'s
  `XDG_DATA_HOME`, ocr's export dir), the two `allowlist_env(os.environ)`
  subprocess passthroughs, and click `envvar=` option declarations (CLI-argument
  surface, not point-of-use config).
- **Activities register bound methods, not module functions.** To add a
  dependency-carrying activity, add a method to the relevant `*Activities` class
  and register `instance.method` in the worker main. No-dep activities stay free
  functions. Tests construct the class with a fake (`LlmActivities(FakeLLM())`,
  `OcrStoreActivities(engine, blobs)`, …) and call the bound method directly.
- **Sandbox-light discipline is enforced.** `forge/activities/roots.py` is
  chain-imported into the Temporal workflow sandbox, so it stays pydantic-only
  with SDK types under `TYPE_CHECKING` and lazy imports; `tests/test_sandbox_light.py`
  guards this. `sax_platform.contracts` must not import SDKs or shell siblings
  (import-linter's forbidden-externals contract enforces it).
- **The transitional-fallback pattern is GONE.** Wave 1 landed the platform
  signature changes with temporary env fallbacks (marked
  `# T3.6 transitional env fallback — deleted in ST7`); ST7 deleted every one.
  `build_tls_config` / `connect_temporal` / `get_store_engine` / `S3Blobs` /
  `make_mistral_client` now **require** their settings/argument inputs — do not
  reintroduce env-reading fallbacks. (`connect_temporal(settings=…)` stays
  optional, defaulting to `TemporalSettings()`, so the CLI's 8 connect sites stay
  unswollen while settings remain the only env reader.)
- **Behavior changes an operator will notice:** ocr fails fast at startup on a
  missing `FORGE_OCR_S3_BUCKET` (was a first-use error); `MISTRAL_API_KEY` is
  **not** a fail-fast for either worker (unset → Mistral OCR client not built);
  pbook's pg pool is now capped 5+5 (was uncapped); pbook log rotation is
  10 MB × 5 (was 5 MB × 3). See [../docs/operations/DEPLOYMENT.md](../docs/operations/DEPLOYMENT.md)
  and [../docs/operations/WORKERS.md](../docs/operations/WORKERS.md).
- **Setup/test discipline unchanged:** `uv sync --all-packages`; suites run
  per-package from each package's own directory; `make gates` runs the CI set.

## Parked and operator-optional

- **T4.2 owner proposal — batch-tracker for the peaky Mistral case.** Parked in
  the [T4.2 Dev Notes](tasks/T4.2-ocr-own-polling-gather-restructure.md) (pointer
  from [T4.1](tasks/T4.1-forge-timer-loop-transport.md)); the **cache-refresher
  variant is the owner-designated frontrunner** — a tracker refreshes
  `batch_jobs` from the provider list endpoint (~0.5 RPM regardless of burst
  size); waiters poll the local ledger. Operational data recorded there:
  1,000-doc bursts are ordinary; per-doc processing 1–1440 min, mode ≈ 30.
  **Read this before Phase 4 planning** and confirm Mistral's rate limits +
  batch-jobs-per-burst shape first.
- **Operator-optional, pending:** the ocr env-gated e2e (runnable — command in
  the [T3.3 task file](tasks/T3.3-mistral-ocr-chat-deleted.md)); the eval judge
  benchmark ([T0.6](tasks/T0.6-eval-judge-integrity.md), the one deferred Phase 3
  review item — the current judge baselines are unreliable by its own Problem
  statement).

## Where to start

**Phase 4 (batch transport simplification) is next.** **T4.1 — forge:
submit → poll-loop → fetch** replaces the signal-based batch SPI + shared poller
with per-workflow timer-loop polling (D88, reversal R1). **Read the parked T4.2
owner proposal first** — it shapes the ocr side and the shared transport
decisions. Phase ordering is load-bearing (1 → 2 → 3 → 4 → 5; Phase 6 after
Phase 5; Phases 6 and 7 may run in parallel; Phase 8 closes). Phase 0
(T0.1–T0.6, T0.8) remains independent of every phase and can land anytime.
