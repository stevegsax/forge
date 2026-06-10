# Architecture Review Migration — Detailed Task List

Companion to [HANDOFF-architecture-review-2026-06-10.md](HANDOFF-architecture-review-2026-06-10.md).
Source of truth for design rationale: the merged plan (`~/.claude/plans/perform-a-thorough-adversarial-vectorized-barto.md`; durable copy in `~/.claude/projects/-Users-stevengreenberg-repos-sax-forge/review-artifacts-2026-06-10/merged-plan.md`).

**Status: APPROVED 2026-06-10 (including reversals R1/R2) and converted.** Each task below now has a file under [`tasks/`](tasks/) per PROCESS.md, indexed by [TASKS.md](TASKS.md) in this order. The task files are the working documents; this list remains as the conversion source. Every task is sized for one agent worktree and must land green (tests + ruff + mypy where gated) and independently mergeable.

## Phase dependency graph

```text
Phase 1 (stop the bleeding, current repos)
   └─▶ Phase 2 (monorepo `sax`)
          └─▶ Phase 3 (sax_platform lib + structured outputs)
                 ├─▶ Phase 4 (batch transport → timer-loop)
                 │      └─▶ Phase 5 (workflow consolidation)
                 │             ├─▶ Phase 6 (knowledge: pbook product + forge consumption)  ← serialized after 5
                 │             └─▶ Phase 7 (context engine)
                 └──────────────────────────┘
Phase 8 (docs/decisions/honesty) — after 1–7
```

Within Phase 1 all tasks are independent except T1.3 (needs T1.0). Phases 6 and 7 may run in parallel with each other (disjoint files) but both after Phase 5.

---

## Phase 1 — Stop the bleeding (in the current five repos)

### T1.0 — Uniform editable sibling sources

- **Repos:** forge (pyproject), CLAUDE.md
- **Depends on:** —
- **Problem:** forge pins siblings by git tag (sax-llm v0.1.1, pbook v0.2.1, forge-contracts v0.1.1) while ocr uses editable paths — the two sides of one runtime wire contract can silently run different versions. The checked-in CLAUDE.md still claims editable sources (stale since commit `19382b3`).
- **Scope:** switch forge's `[tool.uv.sources]` to editable path sources for all three siblings; `uv lock`; fix the CLAUDE.md "Cross-Project Dependencies" sentence.
- **Acceptance:** `uv lock` clean; full forge suite green; CLAUDE.md matches reality. Unblocks T1.3's cross-repo enum change.

### T1.1 — Delete the dead provider stack; repatriate sax-llm's tests

- **Repos:** forge, sax-llm
- **Depends on:** —
- **Problem:** `src/forge/llm_providers/` (1,135 LOC) + `src/forge/llm_client.py` (265 LOC) are a dead duplicate of sax-llm — zero live imports, yet maintained (PRs #26/#27 touched it) and documented in OVERVIEW.md as the live provider layer. sax-llm's real tests (4 files, 2,515 LOC: `test_llm_client.py` 427, `test_llm_providers_registry.py` 150, `test_llm_providers_anthropic.py` 301, `test_llm_providers_mistral.py` 1,637) live in forge under misleading names while sax-llm's own coverage gate is 25%.
- **Scope:** delete the two dead modules + their dead-copy tests (`test_llm_providers_models`); `git mv` the four sax-llm test files into `sax-llm/tests/`; raise sax-llm `--cov-fail-under` 25→85; update `docs/OVERVIEW.md` lines 26/31/76 + the multi-provider-parity tech-debt row to cite sax-llm.
- **Acceptance:** rg shows zero `forge.llm_providers|forge.llm_client` references; **`test_llm_client.py:363` imports `forge.models.LLMResponse` — split or rewrite that case** (flagged by verification); sax-llm suite green at 85%; **forge still clears its own 85% gate after losing 1,400 LOC of covered-dead code**; OVERVIEW.md accurate.

### T1.2 — INTERIM batch-result correlation stopgap

- **Repos:** forge, ocr
- **Depends on:** —
- **Problem (critical, live):** `batch_result_received` appends to `list[BatchResult]` and consumers take by count (`wait_condition(len>0)` + `pop(0)` — workflow_blocks.py:119-123, ocr workflow_store.py:62). At-least-once signal delivery means a duplicate/stale signal becomes the *wrong call's* result.
- **Scope:** replace the list with `dict[str, BatchResult]`; handler = `setdefault(result.request_id, result)`; `batch_submit_and_wait` waits on **its own** `request_id`; never pop. Apply in ForgeTaskWorkflow, ForgeSubTaskWorkflow, TranscriptIngestionWorkflow, ocr OcrStoreWorkflow (~20 lines each).
- **Acceptance:** duplicate-delivery regression test (re-signal the same request_id → no misattribution); ocr request_id-match test.
- **Note:** **INTERIM** — the whole signal path is deleted in T4.1/T4.2. The ingestion workflow copy moves to pbook in Phase 6: port, don't re-derive. Say this in the task file so nobody "finishes" it.

### T1.3 — INTERIM minimal poller patch

- **Repos:** forge, forge-contracts (enum member — needs T1.0)
- **Depends on:** T1.0
- **Problem (critical, live):** transient signal-delivery failure permanently marks the `batch_jobs` row FAILED while the provider still holds the paid result (batch_poll.py:178-197); rows >24h go MISSING with **no signal**, so the waiter burns the full 25h timeout (`_BATCH_WAIT_TIMEOUT`, workflow_blocks.py:59).
- **Scope (minimal — the subsystem dies in Phase 4):** on signal-delivery failure leave the row SUBMITTED (next cycle retries; safe under T1.2 dedup); MISSING sends the waiter an error-payload signal so it fails fast. Do **not** build the attempts-cap/two-column machinery.
- **Acceptance:** tests for both transitions, including the previously untested lossy path.

### T1.4 — Unblock the worker event loop

- **Repos:** forge
- **Depends on:** —
- **Problem:** blocking `subprocess.run` + file I/O inside async activities freezes workflow task processing and defeats heartbeats (validate.py `_run_command`, git.py via git_activities, providers.py run_tests/ruff/git handlers, code_intel discovery).
- **Scope:** route every such call through `await asyncio.to_thread(...)`; restore `_VALIDATE_HEARTBEAT` to 60s as a real crash detector.
- **Acceptance:** test that heartbeats fire during a slow (sleeping) validation command.

### T1.5 — Nested fan-out propagation fix

- **Repos:** forge
- **Depends on:** —
- **Problem (live behavior bug at default settings):** the nested gather ignores `--no-resolve-conflicts` (violating D71) and silently drops `thinking` and `model_routing` — divergence bred by the duplicated gather code.
- **Scope:** add `resolve_conflicts`, `thinking`, **and `model_routing`** to SubTaskInput; propagate from the parent in `_run_fan_out_step`; honor all three in `_run_nested_fan_out` (flag check + D27 terminal fallback; no hardcoded `ThinkingConfig()`); delete the dead `SubTaskInput.worktree_path` field.
- **Acceptance:** depth-2 regression test — fan-out with `--no-resolve-conflicts` and a real conflict terminates instead of resolving; thinking/model_routing asserted at depth ≥1.

### T1.6a — Idempotency rekey

- **Repos:** forge
- **Depends on:** —
- **Problem:** workflow-ID reuse + `insert_or_ignore` silently swallows every persisted record when the same `forge-task-{id}` is re-run (the most common solo-operator action); `_persist_seq` makes interaction keys positional.
- **Scope:** interaction keys become `{workflow_id}:{run_id}:{role}:{occurrence}` with **per-role occurrence counters held in workflow state** (replay-deterministic; immune to the repeated-sanity-check collision); runs table rekeyed `(workflow_id, run_id)` — Alembic migration + all reader updates (`forge status` etc.).
- **Acceptance:** test — rerunning the same task_id records a second complete run; readers display both.

### T1.6b — Batch-wait failure symmetry

- **Repos:** forge
- **Depends on:** —
- **Problem:** when the 25h batch wait times out or errors, forge leaves no run record and an orphaned worktree (ocr handles the same condition correctly via `_mark_failed`).
- **Scope:** wrap the wait in `try/except TimeoutError|ApplicationError` → persist FAILURE_TERMINAL run row + clean the worktree in `finally`; share the shape via workflow_blocks.
- **Acceptance:** test — batch failure leaves a run row and no orphaned worktree.

### T1.7 — Env scrub at model-influenced subprocess seams

- **Repos:** forge
- **Depends on:** —
- **Problem (critical):** `validate.py` test commands and the `run_tests` exploration provider execute model-influenced commands with the worker's full env (API keys, DB URLs, TLS paths).
- **Scope:** explicit allowlist env dict (PATH, HOME, VIRTUAL_ENV, LANG, TMPDIR) at those two seams.
- **Acceptance:** test asserts `ANTHROPIC_API_KEY` (and `FORGE_DB_URL`) absent from the child env.

### T1.8 — Small dedup batch + kill runs-extraction

- **Repos:** forge, ocr, sax-llm
- **Depends on:** —
- **Scope:** forge imports `FORGE_TASK_QUEUE` from forge_contracts.constants (delete the local copy at workflows.py:87); remove ocr's unused sax-llm dependency + fix its `__init__.py` docstring; delete sax-llm's dead sync-OCR protocol surface (`supports_sync_ocr`/`call_ocr` in protocol + both adapters); remove the forge-extraction Schedule creation from worker.py and delete `ForgeExtractionWorkflow` + its CLI command (the scheduled re-extraction loop re-extracts the same runs forever; the playbooks table and retrieval stay until Phase 6).
- **Acceptance:** all three suites green; rg confirms each deletion has zero remaining references.

---

## Phase 2 — Monorepo `sax`

### T2.1 — Workspace creation

- **Depends on:** Phase 1
- **Scope:** create the `sax` repo; import all five repos via **git-filter-repo subdirectory rewrite** (full history preserved — verified total history is only ~8.7MB; subtree's savings are immaterial) into `libs/sax-platform` (start from forge-contracts + sax-llm merged side-by-side; real consolidation happens in Phase 3) and `apps/{forge,ocr,pbook}`; root `pyproject.toml` with `[tool.uv.workspace]`; all sibling deps `{ workspace = true }`; single `uv.lock`; **`.python-version` = 3.14 standard GIL** (wheels verified for temporalio/psycopg-binary/pgvector/pymupdf/grimp/anthropic/openai; do NOT use 3.14t); archive the old repos with pointer READMEs; rewrite CLAUDE.md's cross-project section.
- **Acceptance:** all five package suites green from the workspace; **the skill's uvx invocation verified end-to-end with a pinned tag** (`uvx --from "git+…@vX.Y#subdirectory=apps/pbook" pbook --help`) — an unpinned branch ref re-resolves and can break mid-flight.

### T2.2 — Root gates

- **Depends on:** T2.1
- **Scope:** root justfile + GitHub Actions running ruff check/format, mypy, `lint-imports`, pytest with per-package 85% coverage gates; import-linter contracts: apps never import apps; libs never import apps; **`sax_platform.contracts` may not import SDK/shell modules** (forbidden-externals: boto3, sqlalchemy, anthropic, openai, mistralai + internal shell siblings).
- **Acceptance:** CI green on a no-op PR; a deliberate DAG-violating import fails `lint-imports`.

### T2.3a–d — mypy strict per package (four tasks)

- **Depends on:** T2.2
- **Scope:** one task each — (a) sax-platform/contracts side, (b) sax-platform/llm+rest, (c) ocr, (d) pbook (targeted per-module deferrals allowed, recorded). Forge already has strict mypy.
- **Acceptance:** each lands green with its own gate flipped on in CI; DECISIONS entry for the monorepo (rationale, DAG, gates) written with (d).

---

## Phase 3 — `sax_platform` consolidation + structured outputs

### T3.1 — Platform LLM client (both lanes)

- **Depends on:** Phase 2
- **Scope:** `sax_platform.llm`: `AnthropicLLM(client)` with `complete[T: BaseModel](..., output_type: type[T]) -> Completion[T]` on `client.messages.parse`, `complete_schema(...)` (JSON-schema dict path), `complete_text(...)`; frozen `Completion[T]` with token/cache telemetry; **plus the batch lane Plan B forgot**: pure batch request-body builder (model, system, messages, `output_config.format` schema → request dict), batch submit/status/results helpers. Verified: structured outputs are GA, work in the Batch API, compose with caching (use 1h-TTL cache_control in batches; hits are best-effort 30–98%).
- **Acceptance:** mocked-transport tests both lanes; mypy-strict from day one.

### T3.2 — One tier registry + thinking migration

- **Depends on:** T3.1
- **Scope:** `sax_platform.llm.tiers` — CapabilityTier, ModelConfig, resolve_model, one default map. Drift adjudicated: generation/summarization default **`anthropic:claude-sonnet-4-6`** (forge's `claude-sonnet-4-5-20250929` is stale legacy; pbook's worker default `claude-3-5-sonnet-20241022` is retired). Migrate ThinkingConfig: `budget_tokens` is deprecated on 4.6 → adaptive thinking; set `effort` explicitly (4.6 defaults to high — a latency/cost behavior change). Delete forge/models.py:82-105 copy and all forge imports; pbook's copy dies in T6.4.
- **Acceptance:** one registry, both apps resolve through it; D10/D58 amendment drafted.

### T3.3 — MistralOcr; Mistral chat deleted

- **Depends on:** T3.1
- **Scope:** port from sax-llm: OCR sync call + OCR batch (file-based upload for `/v1/ocr`, poll with error-file merging, `_extract_images_from_response`). **Delete Mistral chat support** — verified zero production users (all forge tier defaults are Anthropic; ocr never imports mistralai/sax_llm directly; only mocked unit tests exercise it). ocr receives `MistralOcr` via DI.
- **Acceptance:** ocr e2e (env-gated) still passes; the only deleted-code references are the removed mock tests.

### T3.4 — Platform plumbing modules

- **Depends on:** Phase 2
- **Scope:** `sax_platform.temporal` (connect_temporal with mTLS, run_worker scaffold with sandbox passthrough + pydantic converter + graceful shutdown, LLM_RETRY/DB_RETRY/IO_RETRY presets, `classify_llm_error` (typed SDK exceptions first, message markers fallback, unit-tested), `heartbeat_during`, persist_block + **public** retry presets); `sax_platform.db` (engine factory with pooler detection — **set `prepare_threshold=None` when port 6543**; run_migrations; insert_or_ignore; UTCDateTime; s3_blobs); `sax_platform.embeddings` (Embedder protocol + OpenAIEmbeddings, frozen EmbeddingResult, base64-float32 codec); `sax_platform.config` (frozen pydantic-settings groups — the only env readers; existing env names kept as aliases); `sax_platform.logging`. Retire the forge-contracts and sax-llm package identities; delete ocr/persist.py + the three `_LOCAL_RETRY` copies.
- **Acceptance:** forge-contracts/sax-llm names gone from all imports; contracts-layer import-linter contract passes; per-module tests.

### T3.5 — Forced-tool-use retirement

- **Depends on:** T3.1, T3.2
- **Scope:** forge's sync lane onto `complete[T]`; batch call sites onto the builder (+ fetch-time `model_validate` with the call site's class); delete both string-keyed registries (sax-llm `_output_type_registry` + pbook's fork) — replaced by frozen OUTPUT_TYPES mappings at composition roots; delete forge's session-autouse conftest registry mirror (already drifted from worker.py).
- **Acceptance:** rg shows zero `tool_choice` forcing for output shaping and zero `register_output_type`; supersession of D75's mechanism drafted.

### T3.6 — Composition roots everywhere

- **Depends on:** T3.4
- **Scope:** frozen `ForgeSettings`/`OcrSettings`/`PbookSettings` constructed once in worker/CLI mains, fail-fast; class-based activities — forge `StoreActivities(engine)`, `LlmActivities(llm, output_types)`, `BatchActivities(llm, output_types, engine, blob_store)`, `ContextActivities(engine)`; ocr/pbook mirrored; one engine per process; delete `batch_poll.set_temporal_client`, every per-call `get_store_engine()`, sax-llm `_provider_cache`/`_client` + all `reset_*()`, pbook `set_provider`/`_engines`/`_client`, the `dispose_store_engines` global-sqlalchemy monkeypatch fixtures; pbook's runtime `os.environ` write → `cfg.attributes`; tracing cleanup (delete the `_TRACER_PROVIDER_SET_ONCE` private-API reset; fix the three false "pure" docstrings); `sax_platform.testing` (temporal_env fixture + provider fakes) imported explicitly by app conftests — **no pytest11 plugin**.
- **Acceptance:** rg shows zero module-level mutable clients/engines/registries across all packages; tests construct classes with fakes (no monkeypatched globals).

---

## Phase 4 — Batch transport simplification (timer-loop)

### T4.1 — forge: submit → poll-loop → fetch

- **Depends on:** Phase 3
- **Scope:** replace the signal wait: `submit` activity (platform builder; `output_config.format`; **request_id/custom_id minted in the workflow via `workflow.uuid4()`** — closes the submit-retry orphan window) → `workflow.sleep(poll_interval)` + thin `batch_status(batch_id)` activity loop (**default 300–600s, configurable; never below 300s** — verified history math: ~11 events/poll; 25h at 600s ≈ 1,650 events; 30-wait worst case ≈ 24k of 51.2k) → `fetch_results` activity (download; validate with the call site's output class; provider threaded through — no more `"anthropic"` default; claim-check return: inline ≤256KB else S3 pointer). Keep the 25h ceiling + T1.6b failure symmetry. Delete: BatchPollerWorkflow + its Schedule, `BATCH_RESULT_SIGNAL` consumption in forge, all signal handlers + the T1.2/T1.3 interims, the BatchResult cross-workflow envelope. `batch_jobs` reduced to forge-internal audit/spend ledger (written at submit + final fetch; provider-lifecycle statuses only).
- **Acceptance:** all workflow tests pass on the new transport (no signal stubs); mistral-model-routes-to-mistral-parse test; duplicate-submit-retry test shows one paid batch; documented tradeoff in the task file: a dead waiter orphans its batch (reconciliation deferred).

### T4.2 — ocr: own polling + gather restructure

- **Depends on:** T4.1, T3.3
- **Scope:** ocr polls its own Mistral batches via `MistralOcr` timer-loop (same pattern); delete its `BATCH_RESULT_SIGNAL` handling and the SPI wire models from contracts; **restructure gather to parent-awaited children** (today both gather and store are ABANDON-children of a fire-and-forget parent, so completion travels by signal and a failed store child hangs the gather for 26h — verified); typed activity inputs (the seven `input_json: str` activities → pydantic models through the installed converter); `OcrJobEntry.status` enum; `_derive_status(OcrProcessingStatus, BatchJobStatus|None)` + docstring fixes.
- **Acceptance:** failed-chunk test propagates failure immediately (no 26h hang); zero signals remain platform-wide; ocr e2e green.

### T4.3 — Transport decisions sweep

- **Depends on:** T4.1, T4.2
- **Scope:** DECISIONS entries — supersede D77/D78 (signal wait → timer-loop polling, with the verified arithmetic recorded); restore D80 (batch_jobs audit-only, now true); reaffirm D76/D79/D82 (per-token economics of an unattended orchestrator — explicitly NOT a realized-volume argument, and explicitly not eroded by pbook's sync exception); correct D81 (poller Schedule deleted; record the 600s-vs-60s doc drift). Amend pbook's TEMPORAL_PATTERNS.md rule 8 with the scoping note (interval ≥300s; history budget; continue-as-new escape if wait counts grow).
- **Acceptance:** markdownlint clean; cross-references resolve.

---

## Phase 5 — Workflow consolidation (forge)

### T5.1 — Pure step logic

- **Depends on:** Phase 4
- **Scope:** `forge/step_logic.py` (zero temporalio imports): `determine_transition` moves here, called inline — **the evaluate_transition activity, its registration, and its test mocks are deleted** (deterministic work stays deterministic); `failure_summary` (replaces five join copies); result builders aggregating per-call LLMStats into run totals + a one-field `failure_kind` Literal on Task/Step/SubTaskResult; persist-key builder (per-role occurrence counters); `merge_resolution(...) -> MergedFiles | MissingResolutions` union; child-ID/timeout helpers. Update CLAUDE.md + ARCHITECTURE.md's universal-step text; record the D3-clause supersession.
- **Acceptance:** full decision-matrix unit tests without Temporal (parametrized, microsecond-fast).

### T5.2 — Single step block

- **Depends on:** T5.1
- **Scope:** `blocks/step.py` — `run_step_attempts(spec, settings)`; StepSpec carries `mode: Literal["single_step","planned_step","sub_task"]` mapped to the (assemble activity, worktree recreate-vs-reset, commit task/step/never) triple by one pure table (3 legal states, not 27); attempt loop rebuilds assemble input with `prior_errors` via `model_copy`; **exploration runs as a per-attempt hook inside the loop** (it needs the per-attempt worktree — ordering pinned by a test).
- **Acceptance:** one copy of the pipeline; per-mode behavior tests via the harness.

### T5.3 — Single gather + dispatch

- **Depends on:** T5.2
- **Scope:** `blocks/gather.py` — `run_fan_out_gather(children, fan_out, settings, *, commit)` for parent (commit=True) and nested (commit=False, D16) — the T1.5 fix now lives in exactly one place; `blocks/dispatch.py` — `typed_batch_dispatch` consolidating the planner/sanity/exploration/conflict arms onto the Phase-4 transport.
- **Acceptance:** depth-2 regression still green; the four dispatch arms share one implementation.

### T5.4 — Split the monolith

- **Depends on:** T5.3
- **Scope:** `workflows.py` (1,861 LOC) → package: `task.py` (~150 LOC driver), `subtask.py` (~60 LOC), `blocks/`, exploration block as a free function; RunSettings (frozen) threaded as a parameter; workflow classes end with **zero** signal state and no duplicated init/persist/dispatch blocks.
- **Acceptance:** ~1,100 LOC of source duplication gone (measure); all behavioral tests green.

### T5.5 — Harness rebuild + replay tests

- **Depends on:** T5.4
- **Scope:** `tests/support/workflow_harness.py` — per-test mutable `ScenarioState` captured by closures producing the named `@activity.defn` stubs (`build_stub_activities(scenario)`), `run_task(env, input, scenario) -> HarnessResult` (frozen); split the 4,134-LOC test_workflows.py into per-mode files; delete 30 `global` statements, 11 `_reset_*` functions, ~1.5k LOC of quadruplicated stubs. Keep behavioral Temporal coverage (retry, sanity abort/revise, fan-out merge/conflict/child-failure, S3-pointer fetch path, depth-2 regression). Add the platform replay-test scaffold: committed histories under `tests/replay/` + one-command regeneration.
- **Acceptance:** pytest-xdist now possible (no shared globals); behavioral scenario count preserved or justified.

---

## Phase 6 — Knowledge: pbook product + forge consumption

### T6.1 — pbook library-first

- **Depends on:** Phase 3 (serialized after Phase 5 — both touch forge worker registration and OUTPUT_TYPES)
- **Scope:** record golden `--json` envelope characterization tests for every CLI command **first**; build `pbook/service.py` sync functions over frozen `AppContext(settings, engine, embedder|None)`; delete the 16 cli_ops wrapper workflows + 4 other RPC workflows (verified: nothing outside pbook references them); CLI rewired to direct service calls; missing `PBOOK_DATABASE_URL` → `ConfigError` → `db_disabled` envelope.
- **Acceptance:** golden envelope tests stay green through the refactor; `pbook search` needs no worker; p50 < 1s on the dev DB.

### T6.2 — Judge calibration (BEFORE the migration sweep)

- **Depends on:** T3.1
- **Scope:** Suite B — ~40 hand-graded cases including 10 generic-advice traps (~2h human work; the one irreducible manual task); judge = CLASSIFICATION tier, four binary checks (grounded / specific / non-generic / actionable).
- **Acceptance:** ≥85% agreement, 100% trap rejection — **gates the T6.3 sweep**. (Sequencing fix from the cross-review: never run an uncalibrated judge over the corpus.)

### T6.3 — Destructive schema migration

- **Depends on:** T6.1, T6.2
- **Scope:** **one-time JSON dump of pbk_entries first**; single destructive Alembic migration: `status ∈ {probation, active, stale, rejected, superseded}` + `status_reason`/`status_changed_at`/`rejected_by`/`superseded_by_id`/`last_validated_at` (CHECK-constrained; backfill: rejected wins → rejected; needs_review → probation; else active); `origin_hash` UNIQUE (sha256(session_id + experience_hash + normalized_title), NULL for manual/legacy); `search_tsv` generated tsvector + GIN; embeddings → `halfvec(1536)` + HNSW rebuilt (`halfvec_cosine_ops`) + `embedding_model`/`embedding_dim` columns; `pbk_retrieval_events` + `pbk_feedback_events` (UNIQUE entry/session/polarity); `knowledge.approved_entries` view (**non-vector columns + search_tsv**); migration preflights pgvector ≥0.7 (halfvec) and ≥0.8 (iterative scans) and refuses otherwise. Then the **report-only** judge sweep of backfilled actives; apply demotions (→ probation) only after T6.2 gates pass. First migration test.
- **Acceptance:** migration test green against real Postgres (podman fixture); dump file exists; sweep report reviewed before apply.

### T6.4 — IngestWorkflow + CurationWorkflow

- **Depends on:** T6.3
- **Scope:** `IngestWorkflow` (id `pbook-ingest-{session_id}`; union input TranscriptSource|InlineExperiences; first activity writes the `running` session row — CLI never seeds; claim-check transcripts by path; one sync structured-output analyze activity with heartbeat; per-experience try/except isolation; extract GENERATION/sonnet with cached static system prompt → judge CLASSIFICATION/haiku → embed+save `ON CONFLICT (origin_hash) DO NOTHING`; survivors → probation, judge failures → rejected_by='validator'; workflow error handler records `error`). `CurationWorkflow` (weekly Schedule `pbook-curation-weekly`, idempotent creation): staleness sweep (pure `compute_staleness`, provisional constants 270/540d documented as provisional), consolidation per-cluster isolated (same judge gate; one-transaction apply: survivor → probation with zeroed counters, parents → superseded, sources reparented), abandoned-session sweep (>48h running → error). pbook composition root from T3.6; delete pbook's tier-registry copy (use `sax_platform.llm.tiers`); **delete forge's ingestion side in the same change** (ingestion_workflow.py, activities/ingestion.py, pbook imports, `_INGESTION_AVAILABLE` guards, cli.py:1217 direct DB read, `forge ingest`); `pbook ingest` is the only frontend; replay tests; cross-queue worker count drops to zero cross-calls.
- **Acceptance:** per-item isolation test (one bad experience doesn't kill the batch); retry-duplication test (origin_hash holds); session-row lifecycle test; forge suite green with ingestion gone.

### T6.5 — Hybrid retrieval + feedback

- **Depends on:** T6.3
- **Scope:** `ranking.score_candidates` (pure; frozen ScoringWeights): candidates = union of lexical/semantic/tag top-50 (status-filtered in SQL, `hnsw.iterative_scan` on); RRF k=60 over lexical+semantic ranks × capped tag boost × mode alignment (CREATE/FIX) × feedback ratio (gated ≥3 retrievals) × probation ×0.7 (provisional); pack to token budget; record retrieval events. Degraded mode without `OPENAI_API_KEY`: lexical+tag, `"degraded": "no_embedding_key"`. Feedback: `record_feedback` event + same-transaction counter bump; CLI additions `invalidate`, `review --stale`, `purge` (the only hard delete); skill-pbook update (pinned uvx, probation annotation, degraded mode).
- **Acceptance:** zero-tag-overlap entries surface via lexical/semantic lists (test); NULL-embedding entries surface lexically (test); per-session feedback idempotency test.

### T6.6 — Eval suites A & C

- **Depends on:** T6.4, T6.5
- **Scope:** `tests/evals/` marked `eval`, excluded from default runs; `make evals`. Suite A extraction goldens (30→100 cases; 10 must-extract-zero negatives gated 100%, positives ≥80%; grown by mining validator rejections). Suite C retrieval goldens (~80-entry frozen corpus, embeddings pre-computed by a PEP-723 helper and committed — zero API calls at test time; ~50 labeled queries across modes). **Gates set from a measured baseline run minus tolerance** (not a priori); model pins move only on green paired-delta runs.
- **Acceptance:** `make evals` green; baseline numbers recorded in the commit.

### T6.7 — Forge consumption + playbooks deletion

- **Depends on:** T6.3 (view), Phase 5 (forge side)
- **Scope:** `sax_platform.contracts.knowledge` — read-only `sa.Table` for `knowledge.approved_entries` + frozen `KnowledgeEntry` + query helper (lexical `websearch_to_tsquery`/`ts_rank_cd` UNION tag candidates); forge `assemble_context` + the exploration playbook provider switch to it: deterministic SQL read + small pure fused scorer (capped tag boost — tags never gate) + token-budget slice, active-only; **no embeddings/OpenAI on forge's hot path**. Explicit degraded-mode test (knowledge_db_url unset → empty + log). pbook-side schema-sync test asserting the view shape after every pbook migration. **Forge playbooks deletion**: one-time JSON dump → manual triage via `pbook add` (**no blanket-approve migration** — the table is polluted by re-extraction duplicates); drop table + store helpers + Alembic migration; delete manual/export playbook workflows, playbook_review/export activities, `forge playbooks` CLI, forge tag inferrers; OUTPUT_TYPES becomes a true module constant. DECISIONS supersessions D13/D43–D47.
- **Acceptance:** forge context assembly shows pbook entries in a live test (podman Postgres); degraded-mode test; forge suite green with the subsystem gone; dump file exists.

---

## Phase 7 — Context engine (forge)

### T7.1 — ProjectDescriptor

- **Depends on:** Phase 5
- **Scope:** frozen ProjectDescriptor (project_root, src_root, package_name, python_cmd, test_cmd, lint_cmd), detected once per task (pyproject-first heuristic chain), carried on AssembleContextInput/FulfillContextInput; delete every hardcoded `"forge"`/`"src"`/bare-`python` literal in providers.py; **PROVIDER_SPECS derived from each provider's params_model** (menu drift impossible by construction).
- **Acceptance:** providers run against a non-forge repo in a test worktree.

### T7.2 — Worktree-accurate graph

- **Depends on:** T7.1
- **Scope:** build the grimp graph in a subprocess with `PYTHONPATH={worktree}/{src_root}` (the ~20-line fix; fixes the critical worktree-drift finding and makes discovery work off-repo); `degraded: bool + degradation_reason` on ContextStats (logged, not silent); repo map survives graph failure via file-walk fallback; delete `module_to_file_path`'s CWD-relative checks. D31 amendment.
- **Acceptance:** test — graph reflects a worktree-only change; degradation test.

### T7.3 — Honest token accounting

- **Depends on:** T7.1
- **Scope:** calibrated chars-per-token (~3.4 for code) injected as an estimator parameter on build_context_items/pack_context; `effective_budget = min(token_budget, model_window − scaffolding − output_reserve)` per task, scaffolding re-estimated on retries (error sections grow). D33 amendment.
- **Acceptance:** output_reserve provably live (test that packing shrinks when reserve grows).

### T7.4 — Exploration budget

- **Depends on:** T7.1
- **Scope:** ExplorationBudget on ContextConfig — **mode-aware max_rounds default (2–3 batch / 10 sync; the single highest-ROI change of the review)**, per-round request cap, dedup by (provider, sorted params) answered with an already-retrieved stub, deterministic total-estimated-token cap, `exploration_exhausted` surfaced and logged. D48/D49 amendments.
- **Acceptance:** round/dedup/cap tests; batch-mode default verified.

### T7.5 — One prompt builder

- **Depends on:** T7.1
- **Scope:** the four copy-pasted builder grammars collapse into one parameterized builder; `TaskFacts(task_id, description, target_files, domain)` replaces the four `task_mock` TaskDefinition fabrications; `build_error_section`'s file reads move to the shell (pre-read snippets passed in).
- **Acceptance:** golden-prompt tests for each mode; cache-friendly stable-first ordering preserved.

### T7.6 — Fuzzy-edit governance residue

- **Depends on:** Phase 5
- **Scope:** `allow_fuzzy`/`fuzzy_threshold` on validation/domain config; write_output surfaces fuzzy-applied edits (path, score) into ValidationResult/run records so branch reviewers see them. D55–D57 reaffirmed. (The ast.parse gate stays rejected — a valid wrong-location edit passes it by definition.)
- **Acceptance:** fuzzy application visible in `forge status` output for a test run.

---

## Phase 8 — Docs, decisions, honesty

### T8.1 — Review doc + DECISIONS completion

- **Depends on:** Phases 1–7
- **Scope:** consolidated review doc under `docs/reviews/` (both plans' findings → verdicts → the ten adjudications → rejected-ideas dispositions; mine the wave JSONs in `review-artifacts-2026-06-10/`); remaining DECISIONS entries per the sweep checklist (D3, D9 deferred-with-rationale, D10/D58, D13, D31–33, D43–47, D75, D76–D82, monorepo, composition roots, knowledge contract); WORKERS.md ops section: drain/revert-and-replay default, `workflow.patched()` only with paid batches in flight, **workflow-RESET-cannot-recover-batch-results warning**.
- **Acceptance:** every superseded Dxx carries a forward pointer; markdownlint clean.

### T8.2 — Test-tier honesty + status-of-record rewrite

- **Scope:** rename forge `tests/test_e2e.py` → `tests/test_pipeline.py` (it is a default-run integration suite); fix the stale `uv run pytest -m e2e` line in CLAUDE.md; document the marker scheme (e2e = env-gated real APIs, lives in ocr; postgres = opt-in, mandatory for pgvector/knowledge); rewrite docs/OVERVIEW.md (status, module map, tech-debt table — items 1/2/3/5 closed with dispositions; transition-vocabulary row → deferred-by-decision).
- **Acceptance:** markers match reality (`pytest -m e2e` collects only env-gated tests).

### T8.3 — pbook design-docs truth pass

- **Scope:** update the eight `pbook/design/` docs to the merged reality: TEMPORAL_PATTERNS rule 8 scoping amendment; INTEGRATION.md gains the forge-consumption section (view contract); DECISIONS.md batch-threshold entry rewritten with the measured cost arithmetic (~110 sessions/month ≈ $2–5/month) + the explicit forge non-applicability boundary; psycopg pooler wording fixed (`prepare_threshold=None` is the app's job); skill-pbook SKILL.md (worker requirements, degraded mode, probation annotation, pinned uvx).
- **Acceptance:** no design doc contradicts the implemented system.

### T8.4 — Final sweep

- **Scope:** replace string-patched CLI tests with injected fakes in commands touched by Phases 5–6; verify 85% + mypy strict + ruff + import-linter green in every package via the root justfile; tag `sax` v1.0.
- **Acceptance:** one command (`just check`) green from the workspace root.

---

## Task count: 47

| Phase | Tasks | Theme |
|---|---|---|
| 1 | 10 | live bugs + deletions, current repos |
| 2 | 6 | monorepo + gates |
| 3 | 6 | platform lib + structured outputs |
| 4 | 3 | timer-loop transport |
| 5 | 5 | workflow consolidation |
| 6 | 7 | knowledge product + consumption |
| 7 | 6 | context engine |
| 8 | 4 | docs + honesty |
