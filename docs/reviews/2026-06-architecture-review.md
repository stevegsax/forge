# Platform Architecture Review — 2026-06

Consolidated record of the adversarial architecture review of the five-repo
platform (forge, forge-contracts, ocr, sax-llm, pbook), run 2026-06-09/10,
cross-reviewed against an independent peer plan, and approved 2026-06-10.
The umbrella decision is [D86](../DECISIONS.md); the individual decisions are
D87–D97; the migration is the 47-task, 8-phase list under
[development-plans/tasks/](../../development-plans/tasks/).

## 1. Scope & method

Two independently produced redesign plans were reviewed and merged:

- **Plan A** — this review's multi-agent adversarial pass over all five
  repos, deep on forge internals.
- **Plan B** — the pbook design docs
  (`~/repos-sax/pbook/design/{OVERVIEW,REVIEW-2026-06,DECISIONS,DATA_MODEL,WORKFLOWS,CLI,INTEGRATION,TEMPORAL_PATTERNS}.md`),
  written the same nights, deep on knowledge quality and Temporal
  minimalism.

Plan A's wave structure (~67 subagents, ≈6.2M subagent tokens, five
workflow runs):

1. **Wave 1** — 13 read-only subsystem mappers across all five repos:
   structural maps plus 140 raw findings.
2. **Wave 2** — 10 adversarial dimension reviews (repo-decomposition,
   temporal-design, fcis-conformance, domain-model, workflow-structure,
   batch-pipeline, knowledge-architecture, context-engine,
   safety-governance, test-architecture) producing 83 findings with
   file:line evidence, each paired with a target proposal.
3. **Waves 2–3** — 20 independent verifier panels (a fact-check lens and a
   pragmatist "worth-it" lens per dimension) re-verifying every finding
   and critiquing every proposal against the code.
4. **Wave 4** — 3 integrator architectures (deletion-first /
   contract-first / runtime-first) judged by 2 independent panels; both
   judges picked **deletion-first** (fidelity/coherence/sequencing/
   simplicity 9/8/9/9 and 9/9/8/9); grafts and flagged flaws were folded
   into the plan.
5. **Wave 5** — the cross-review against Plan B: 4 fact-check panels on
   the decisive claims (API capabilities, Temporal history arithmetic,
   codebase usage, build/infra) plus each plan attacked from the other's
   perspective.

Durable artifacts (wave outputs with full evidence):
`~/.claude/projects/-Users-stevengreenberg-repos-sax-forge/review-artifacts-2026-06-10/`
(`wave1-subsystem-maps.json`, `wave2-dimension-reviews-and-7-verifications.json`,
`wave3-remaining-13-verifications.json`, `wave4-three-designs-and-judges.json`,
`wave4-winner-deletion-first-extracted.md`,
`wave5-cross-review-vs-pbook-plan.json`, `merged-plan.md`). Method and cost
summary:
[HANDOFF-architecture-review-2026-06-10.md](../../development-plans/HANDOFF-architecture-review-2026-06-10.md).

Precepts held throughout: Temporal owns orchestration; deterministic work
stays deterministic; clarity/testability over performance; Functional Core /
Imperative Shell; no deployed base, so no backward-compatibility constraint.

## 2. Verdict summary

| Measure | Count |
| --- | --- |
| Plan A findings filed (10 dimensions) | 83 (10 critical / 48 major / 25 minor as filed) |
| Verifier panel verdicts (2 lenses × 10 dimensions) | 164 |
| — confirmed (evidence and severity verified) | 117 |
| — overstated (evidence verified, severity reduced) | 43 |
| — refuted (evidence real but fix or harm rejected) | 4 |
| Findings with fabricated evidence | 0 |
| Criticals standing after dedup and verification | 5 (§3) |
| Plan B pbook-product findings (spot-verified by cross-review) | ~20 majors (§5) |
| Cross-review fact-check panels / questions | 4 / 21 |
| Independent-convergence points between the plans | 10 (§6) |
| Genuine conflicts adjudicated | 10 (§7) |
| Ideas rejected with recorded dispositions | 29 (§8) |

Several filed criticals were merged or reduced by verification:
`dead-llm-provider-copy` was downgraded to major (no runtime path executes
the dead code); `batch-result-no-correlation` and
`uncorrelated-result-consumption` are one defect found by two dimensions;
`no-composition-root`, `triplicated-step-pipeline`,
`scheduled-reextraction-loop`, and `blocking-io-async-activities` each drew
a critical/major split and land as majors below; `poller-state-machine-
loses-results` was filed major but promoted to critical in the merged plan
(paid-result data loss). The five that stand follow.

## 3. Critical findings (5)

All five confirmed by verifier panels; all addressed by the merged plan.

### C1 — Batch results consumed without correlation

`workflow_blocks.py:119-123`: `await workflow.wait_condition(lambda:
len(batch_results) > 0, ...); result = batch_results.pop(0)` — no check
against `submit_result.request_id` even though it is in scope (line 96) and
`BatchResult.request_id` exists (forge-contracts `models.py:60`). Signal
handlers are plain appends with no dedup (`workflows.py:276-278`, `:1462-1464`,
`ocr/workflow_store.py:39-41`, `ingestion_workflow.py:62-64`). Duplicate
delivery is reachable: `batch_poll.py:180` signals before the status write
at `:197`; if the write raises, Temporal retries the activity and signals
again. A duplicate or stale signal becomes the wrong LLM call's result.

- **Verdict:** confirmed critical by all four panels (temporal-design and
  batch-pipeline, both lenses).
- **Disposition:** interim dict + `setdefault(request_id)` stopgap in
  [T1.2](../../development-plans/tasks/T1.2-interim-batch-result-correlation.md);
  then unconstructible by construction under timer-loop polling in
  [T4.1](../../development-plans/tasks/T4.1-forge-timer-loop-transport.md)
  — the requester is the recipient (D88).

### C2 — Poller abandons paid results

`batch_poll.py:178-197`: on a transient signal-delivery failure
`job_signals` stays 0 and `final_status = PROCESSING if job_signals > 0
else FAILED` — a batch whose provider result succeeded is permanently
marked FAILED, and `get_pending_batch_jobs` (`store.py:549-556`) selects
only SUBMITTED, so it is never re-polled. The >24h MISSING path
(`batch_poll.py:122-130`) updates status but sends no signal, so the waiter
only learns via the 25h `wait_condition` timeout (`workflow_blocks.py:59`).

- **Verdict:** confirmed by all four panels (filed major; promoted to
  critical in the merged plan — paid results are silently lost).
- **Disposition:** interim minimal patch (never FAILED on delivery failure;
  MISSING signals the waiter) in
  [T1.3](../../development-plans/tasks/T1.3-interim-poller-patch.md); the
  whole poller subsystem is deleted in Phase 4
  ([T4.1](../../development-plans/tasks/T4.1-forge-timer-loop-transport.md)).

### C3 — Model-influenced subprocesses inherit worker secrets

`validate.py:124` (`sh -c test_command`) and `providers.py:187` (pytest
exploration provider) call `subprocess.run` with no `env` argument, so
LLM-influenced commands inherit the worker's full environment: API keys
(`llm_client.py:255`), DB URLs (`db.py:36`), Temporal TLS material
(`temporal.py:78-80`). The `run_tests` provider is enabled by default
(`models.py:768`).

- **Verdict:** confirmed critical (fact-check) / overstated-major
  (pragmatist: single-operator deployment narrows the threat). Kept
  critical: it is the one verified safety gap with a cheap fix.
- **Disposition:** scrubbed env allowlist (PATH/HOME/VIRTUAL_ENV/LANG/
  TMPDIR) at both seams, with an ANTHROPIC_API_KEY-absent test —
  [T1.7](../../development-plans/tasks/T1.7-env-scrub-subprocess-seams.md).

### C4 — grimp analyzes the worker's installed package, not the worktree

`code_intel/graph.py:246-248`: `grimp.build_graph(package_name)` with no
path argument; grimp resolves via `importlib.util.find_spec`, which the
venv's `_editable_impl_forge.pth` points at the main checkout — while
`activities/context.py:848-853` and `:1011-1016` pass
`project_root=input.worktree_path` and file contents are read from the
worktree. Import-graph context can be wrong for every task worktree.

- **Verdict:** confirmed critical (fact-check) / overstated-major
  (pragmatist: today's single-repo use masks it). Kept critical: it
  invalidates context discovery the moment a worktree diverges.
- **Disposition:** grimp in a subprocess with
  `PYTHONPATH={worktree}/{src_root}`, degradation flags, file-walk fallback
  (amends D31) —
  [T7.2](../../development-plans/tasks/T7.2-worktree-accurate-graph.md).

### C5 — Knowledge loop disconnect

`activities/context.py:541-548`: `get_playbooks_by_tags(engine, tags,
limit=5)` against forge's own `playbooks` table is the only retrieval
feeding context assembly; `rg "RetrievalWorkflow|pbk_entries" forge/src`
returns zero hits outside ingestion plumbing. Ingested pbook knowledge
never reaches task execution.

- **Verdict:** confirmed critical by both panels.
- **Disposition:** pbook publishes the read-only
  `knowledge.approved_entries` view (non-vector columns + `search_tsv`);
  forge's `assemble_context` switches to a deterministic lexical+tag fused
  SQL read; forge's playbooks subsystem is deleted (D92) —
  [T6.7](../../development-plans/tasks/T6.7-forge-consumption-playbooks-deletion.md).

## 4. Major findings — forge/platform

83 findings, by dimension. Verdicts column shows the fact-check / worth-it
panel outcomes (`✓` confirmed, `↓` overstated-with-downgrade, `✗` refuted).
Dispositions are task IDs under `development-plans/tasks/`.

### Repo decomposition

| Finding | Evidence | Verdicts | Disposition |
| --- | --- | --- | --- |
| Dead 1,400-LOC provider copy, actively maintained, documented as live | `src/forge/llm_providers/` 1,135 LOC + `llm_client.py` 265 LOC; zero live imports; commits #26/#27 touched it; `docs/OVERVIEW.md:31,76` cite it as live | ↓ major / ↓ major (filed critical; no runtime path executes it) | T1.1 |
| sax-llm's tests stranded in forge; sax-llm gate 25% | `forge/tests/test_llm_client.py:10-15` imports sax_llm (2,515 LOC across 4 files); `sax-llm/pyproject.toml` `--cov-fail-under=25` | ✓ / ✓ | T1.1 |
| forge↔pbook boundary incoherent: required dep with dead "optional" guards, module-level pbook imports, direct DB reads | `forge/pyproject.toml:22`; `worker.py:77-79,262`; `activities/ingestion.py:14-15`; `cli.py:1217` | ✓ major / ↓ minor | T6.4, T6.7 |
| Contracts version skew: tag-pin vs editable across the same wire contract | forge pins git tags (`pyproject.toml:27-29`, `uv.lock:496` v0.1.1); ocr uses editable paths; CLAUDE.md documents a third state | ✓ / ✓ | T1.0 (interim), T2.1 |
| `batch_jobs` schema defined twice, no sync test | `forge_contracts/batch_jobs.py:20-31` vs `forge/store.py:117-132` (indexes and server_default differ) | ✓ major / ↓ minor | T4.1 (single forge-internal definition) |
| FORGE_TASK_QUEUE duplicated despite contracts owning it | `forge_contracts/constants.py:19` vs `workflows.py:87` | ✓ / ✓ (minor) | T1.8 |
| Persist retry policy duplicated; `_LOCAL_RETRY` ×3; cross-module private imports | `ocr/persist.py:22-30` byte-identical to `forge_contracts/persist.py:33-41`; `ocr/workflow_store.py:23` imports `_PERSIST_RETRY` | ✓ / ✓ (minor) | T3.4 |
| ocr declares sax-llm dep it never imports; sync-OCR surface has zero callers | `ocr/pyproject.toml:15`, zero `sax_llm` hits in ocr; `sax_llm/protocol.py:71-83` uncalled | ✓ / ✓ (minor) | T1.8, T3.3 |
| CapabilityTier/ModelConfig copied forge↔pbook, already drifted | `pbook/models.py:121-159` "duplicated from Forge" comment; 4-5 vs 4-6 drift | ✓ / ✓ (minor) | T3.2 |

### Temporal design

| Finding | Evidence | Verdicts | Disposition |
| --- | --- | --- | --- |
| Blocking `subprocess.run` in async activities defeats heartbeats | `validate.py:35-47` called inside `async with heartbeat_during()` (`:150-158`); heartbeat task starves on the same loop; workaround comment at `workflows.py:108` | ✓ / ✓ | T1.4 |
| No workflow versioning/patching policy despite deploy-spanning waits | zero `patched`/`deprecate_patch` hits; 25h waits (`workflow_blocks.py:59`), 48h execution timeouts (`cli.py:66,394`) | ✓ major / ↓ minor | T8.1 (WORKERS.md ops section: drain/revert-and-replay default) |
| Workflow-ID reuse silently swallows re-run records | `workflows.py:264` key `{workflow_id}:{role}:{seq}`; `cli.py:392` fixed IDs, no `id_reuse_policy`; `store.py:260-262` `insert_or_ignore` drops the second run's rows | ✓ / ✓ | T1.6a |
| Full prompts transit history 3+ times per call; exploration grows history quadratically | AssembledContext ≈400KB budget (`models.py:143`) enters history as activity result, activity input, persist payload (`workflows.py:581-598`, `workflow_blocks.py:96-104`, `:249-274`) | ✓ major / ↓ minor (under payload limits; the real blowup is exploration) | T7.4 (exploration cap); context-by-pointer rejected (§8) |
| Batch-wait timeout leaves no run record, no worktree cleanup (ocr handles it) | `workflow_blocks.py:119-122` no try/except; `PersistRun` only on normal return (`workflows.py:294-299`); contrast `ocr/workflow_store.py:51-60` | ✓ / ✓ (minor) | T1.6b |
| Module-global Temporal client injected by setter | `batch_poll.py:56-70`; `worker.py:230` | ✓ / ✓ (minor) | T3.6, deleted with the poller in T4.1 |
| Queue constants and poller defaults drift across repos and docs; `result_type` lies | `pbook-task-queue` literal ×3; D81 says 60s, code is 600s; `BatchResult.result_type` set to "succeeded"/"errored", never read | ✓ / ✓ (minor) | T1.8, T4.3 |

### FCIS conformance

| Finding | Evidence | Verdicts | Disposition |
| --- | --- | --- | --- |
| No composition root anywhere: per-call engines, module-global clients, global caches | `forge_contracts/db.py:53-76` fresh Engine per call; `activities/persist.py:50` engine inside every survivable write under a 20-attempt retry; `ocr/activities.py:601-637`; `pbook/llm.py:78-103`, `embeddings.py:28`, `store.py:317` | ✓ critical / ↓ major (filed critical) | T3.6 |
| Output-type registry forked three ways; last-writer-wins forced pbook's fork | `sax_llm/registry.py:20,26`; `pbook/workflow_steps/output_types.py:21` (docstring admits the cause); `forge/llm_client.py:190` (dead third copy) | ✓ / ✓ | T3.5 |
| Dead provider stack carries a second credential read and client singleton | `llm_client.py:249` AsyncAnthropic singleton; `llm_providers/mistral.py:222` second MISTRAL_API_KEY read | ✓ / ✓ | T1.1 |
| Zero Settings classes; 14+ point-of-use env reads incl. a runtime `os.environ` write | no `BaseSettings` match in five repos; `sax_llm/mistral.py:191` empty-string API-key default; `pbook/store.py:383` env write | ✓ major / ↓ minor | T3.4, T3.6 |
| "Pure" docstrings over env reads; OTel private set-once guard reset | `tracing.py:7-9` vs `:89`; `:228-230` `_TRACER_PROVIDER_SET_ONCE` reset | ↓ minor / ↓ minor | T3.6 (tracing cleanup) |
| mypy strict only in forge; no CI, no import-linter anywhere | `forge/pyproject.toml:101-113` alone; `.github/workflows` absent ×5 | ✓ major / ↓ minor | T2.2, T2.3a–d |
| ~400 LOC of pure formatting/validation logic embedded in `cli.py` (1,837 LOC) | formatters at `cli.py:86-230` etc.; 9 except-Exception catch-alls | ✓ major / ↓ minor (importing `forge.cli` loads zero temporalio modules — payoff overstated) | opportunistic in T8.4; dedicated split rejected (§8) |
| Effectively zero frozen value types (75 mutable BaseModels in `models.py`) | only `subprocess_result.py:11` and `domains.py:53` frozen | ✓ minor / ↓ none | rejected as retrofit program (§8); new platform models frozen by default |

### Domain model

| Finding | Evidence | Verdicts | Disposition |
| --- | --- | --- | --- |
| Dead provider stack + drifted tier copies | `diff` shows formatting-only divergence; OVERVIEW.md:26 stale; pbook GENERATION pin drifted | ✓ / ✓ | T1.1, T3.2 |
| TransitionSignal poverty: 3-member enum, missing vocabulary in comments, status+error soup | `models.py:37-47` commented-out members; `activities/transition.py:43-50` returns bare `str`; SUCCESS-with-error constructible (`models.py:284-306`) | ✓ major / ↓ minor | T5.1 (`failure_kind` Literal); Outcome union rejected (§8) |
| BatchResult invariant lives in a docstring; every consumer re-checks by hand | contracts `models.py:51-65`; `workflow_blocks.py:124-127` and `ocr/workflow_store.py:64-69` duplicate checks | ✓ / ✓ | T4.1 (envelope deleted; claim-check fetch shape) |
| Deterministic plan checks exist but never run at plan acceptance | `eval/deterministic.py:52-353` checks; only gate is `Plan.model_validate` (`planner.py:221`) | ✓ / ✓ | merged target §Domain & config: plan preflight gate fed back to planner retries (Phase 5 scope; no dedicated task file) |
| OCR + cross-queue contracts bypass pydantic: `json.dumps` strings, hand-indexed dicts | ten `input_json: str` activities (`ocr/activities.py:604-744`); `ingestion_workflow.py:193` | ✓ / ✓ | T4.2 (typed inputs), T6.4 (contract dies) |
| `models.py` god-module: 1,094 LOC, ~60 classes, 34 importers, re-export shim | `models.py:8-22` shim; prompt-text Field descriptions at `:341-373` | ✓ major / ↓ minor | package split rejected (§8); module shrinks by deletion (T3.5, T5.1, T6.7) |
| ValidationConfig/LLMResponse make silent-failure states constructible | `run_tests=True` with `test_command=None` skips tests silently (`validate.py:157`); empty `files`+`edits` validates whole repo | ↓ minor / ↓ minor | cheap validators in merged target §Domain & config |
| Forty bare-str ID fields, empty-string/zero sentinels, free-form status strings | `models.py` sentinels (`:435-437`, `:504`, `:114`); `ocr/activities.py:579-587` raw literals | ✓ / ✓ (minor) | Literal status types where touched (T4.2); NewType program rejected (§8) |
| LLMStats inheritance mixin undone by `build_llm_stats` + getattr duck-typing | `models.py:313-333`; `persist_models.py:44-50` | ✓ / ✓ (minor) | T5.1 (result builders + run-total aggregation) |

### Workflow structure

| Finding | Evidence | Verdicts | Disposition |
| --- | --- | --- | --- |
| Step pipeline written ×3, gather ×2, ~100 LOC wrappers ×2 in two god-classes | `workflows.py:553-747`, `:965-1121`, `:1530-1660` identical sequences; gathers `:1169-1397` vs `:1662-1861` share 128 identical lines | ✓ critical / ↓ major (filed critical) | T5.2, T5.3, T5.4 |
| Nested fan-out silently ignores `--no-resolve-conflicts`, drops thinking config | parent honors flag (`workflows.py:1273`), nested does not (`:1753`); `:1772` hardcodes `ThinkingConfig()`; SubTaskInput lacks the fields (`models.py:857-876`) | ✓ / ✓ | T1.5 |
| Pure 6-line transition function executed as an activity 5× per pipeline | `activities/transition.py:19-50`; invoked at `workflows.py:668,1047,1348,1608,1826` | ✓ / ✓ (minor) | T5.1 (inlined; activity deleted; amends D3 clause) |
| Ambient instance state; positional persist keys (single counter across roles) | `workflows.py:241-247`, `:262-264` | ↓ minor / ↓ minor | T1.6a, T5.1 (per-role occurrence counters) |
| Monolith mirrored by a 4,134-LOC test file; policy assertions need a Temporal server | `test_workflows.py`: mock sets re-declared per section; 14 Worker instantiations | ✓ major / ↓ minor | T5.5 |
| Batch-signal protocol hand-copied into four workflow classes | `workflows.py:242/276-278`, `:1430/1462-1464`, `ingestion_workflow.py:60-64`, `ocr/workflow_store.py:36-41` | ✓ minor / ✗ (mixin fix net-negative) | moot — signal path deleted in T4.1/T4.2 |
| `SubTaskInput.worktree_path` dead field | `models.py:876`; never set, never read | ✓ / ✓ (minor) | T1.5 |

### Batch pipeline

| Finding | Evidence | Verdicts | Disposition |
| --- | --- | --- | --- |
| Delivery failure marks FAILED; delivered per-entry errors recorded as PROCESSING, contradicting the contract docs | `batch_poll.py:178-197`; contracts `models.py:37-41` documents FAILED as the per-entry-error state | ✓ / ✓ | T1.3 interim → Phase 4 deletion (C2) |
| Provider identity dropped between submit and parse: mistral results parsed by the Anthropic parser | `models.py:1074` default `"anthropic"`; construction site `workflow_blocks.py:132-139` never sets it | ✓ major / ↓ minor (no mistral chat caller today) | T4.1 (provider threaded; mistral-routes-to-mistral-parse test) |
| Envelope stringly and self-contradictory: dead `result_type` with false description; unversioned raw-dict S3 envelope | contracts `models.py:65`, `:116-133`; zero reads of `.result_type` | ✓ major / ↓ minor | T4.1 (envelope retired; claim-check shape) |
| `batch_jobs` is the routing table while D80 claims audit-only; schema ×2, no drift guard | poller routes via `job["workflow_id"]` (`batch_poll.py:109,144,179`); D78/D80 marked stale | ✓ major / ↓ minor | T4.1 + T4.3 (audit/spend ledger only; D80 restored to truth) |
| `request_id` minted inside the retried submit activity orphans paid batches | `batch_submit.py:73` `uuid.uuid4()` inside `_LLM_RETRY`; contrast ocr pre-minting (`ocr/activities.py:310`) | ✓ / ✓ (minor) | T4.1 (workflow-minted `workflow.uuid4()` custom_id) |
| Scheduled extraction — most latency-tolerant workload — runs sync-only against D76/D82 | `extraction_workflow.py:66-72` sync `call_extraction_llm` | ✓ / ✓ (minor) | T1.8 (schedule + workflow killed; subsystem superseded by pbook) |
| Poller's module-global Temporal client | `batch_poll.py:56-70` | ✓ / ✓ (minor) | deleted in T4.1 |

(The dimension's critical, `uncorrelated-result-consumption`, is C1; its
`dead-provider-stack` duplicate is covered under repo decomposition.)

### Knowledge architecture

| Finding | Evidence | Verdicts | Disposition |
| --- | --- | --- | --- |
| Scheduled re-extraction loop: coverage delegated to the LLM; 4h schedule re-extracts the same runs forever under fresh keys | `store.py:446-449` exclusion depends on LLM-attributed `source_workflow_id` (`extraction.py:213-217`); keys are extraction-scoped (`store.py:212-214`) | ✓ critical / ↓ major (host pipeline condemned; fix is the kill switch) | T1.8 (kill schedule + ForgeExtractionWorkflow); Phase 6 supersession |
| forge↔pbook wire contract is hand-rolled JSON; contracts package contains none of it | `ingestion_workflow.py:175-193` hand-built dicts; `pbook/activities/extraction.py:164,113-114` raw indexing; zero ingestion hits in forge-contracts | ✓ major / ↓ minor | T6.4 (contract dies; pbook owns ingestion end-to-end) |
| Ingestion lifecycle/idempotency smeared across two CLIs and three mechanisms | `forge/cli.py:1217-1225` opens pbook's DB; `pbook/cli.py:1103-1108` starts forge's workflow by string name and seeds `running` rows | ✓ / ✓ | T6.4 (session-row ownership: first activity writes `running`; 48h sweep) |
| Every knowledge LLM call is sync — the only subsystem violating batch-first | `extraction.py:208`; `pbook/workflow_steps/llm.py:110`; D82 exempted nothing | ✓ major / ↓ minor | resolved by adjudication #4 / R2 (D91): sync adopted deliberately at measured volume; D76 boundary recorded in T4.3 |
| Three tag-inference implementations, two incompatible vocabularies; pbook's extraction path violates its own taxonomy | `extraction.py:139-178` vs `context.py:468-502` vs `pbook/tags.py:111`; `insert_entry` never validates (`store.py:438-461`) | ✓ / ✓ | T6.4 (single namespaced inferrer enforced at write), T6.7 (forge inferrers deleted) |
| `needs_review`/`rejected` boolean pair makes illegal states representable | `pbook/store.py:83-88`, `:121-127`; PruneWorkflow overloads the boolean | ✓ / ✓ (minor) | T6.3 (status state machine) |
| Export fans out one activity per row for a pure dict conversion | `export_playbook_workflow.py:47-56` | ✓ minor / ↓ none | T6.7 (subsystem deleted) |

(The dimension's other critical, `knowledge-loop-disconnect`, is C5.)

### Context engine

| Finding | Evidence | Verdicts | Disposition |
| --- | --- | --- | --- |
| Blocking subprocess + file I/O in async activities, no activity executor | `providers.py:187-284` sync `subprocess.run`; `exploration.py:255-257` async activity calls them on-loop; `worker.py:305-312` no `activity_executor` | ✓ critical / ↓ major (same defect family as T1.4's) | T1.4 |
| Providers hardcode `package_name="forge"`, `src_root="src"`, worker's python | `providers.py:145,301,327-329,176`; ContextConfig.package_name exists but never reaches providers (`models.py:413-418`) | ✓ / ✓ | T7.1 (ProjectDescriptor threaded through) |
| Token accounting fiction: 4:1 chars/token underestimates code; budget covers only ContextItems; `output_reserve` dead | `repo_map.py:36-38`; D33; scaffolding + errors + exploration appended outside any budget (`context.py:267-359`) | ✓ major / ↓ minor | T7.3 (calibrated estimator + `effective_budget`) |
| Exploration loop unbounded: no dedup, no caps, quadratic resend, 10 sequential batch round-trips | `workflows.py:479-522` accumulate-and-resend; `max_rounds=10` default (`cli.py:360,409`); each round a full batch cycle | ✓ / ✓ (the single highest-ROI fix of the review) | T7.4 |
| `context.py` monolith: four copy-pasted prompt grammars, I/O mid-render, `task_mock` | builders at `context.py:207-253,267-359,718-800,886-957`; mocks at `:656,676,837,999` | ✓ major / ↓ minor | T7.5 (one builder + TaskFacts) |
| Stringly provider registry; hand-maintained parallel spec list | `providers.py:403-416` vs `:419-494`; `dict[str,str]` params | ✓ / ✓ (minor) | T7.1 (PROVIDER_SPECS derived from params models) |
| CWD-relative `Path.exists` inside the "pure functions" section | `graph.py:84-92` | ✓ minor / ↓ none | T7.2 |
| Graph-build failure silently drops the repo map | `code_intel/__init__.py:143-155` | ✓ / ✓ (minor) | T7.2 (degradation flags) |

(The dimension's other critical, `graph-analyzes-worker-env`, is C4.)

### Safety & governance

| Finding | Evidence | Verdicts | Disposition |
| --- | --- | --- | --- |
| No per-run cost/token/wall-clock budget or admission control | no budget checks at spend seams (`workflow_blocks.py:152,81,186`); LLMStats never accumulated (`models.py:313-321`) | ✓ major / ↓ minor (every loop already structurally capped) | BudgetLedger rejected (§8); LLMStats run-total aggregation kept (T5.1) |
| Transition vocabulary cannot express halt-when-confused (D9) | `models.py:37-47` escalation members commented out; sanity check off by default (`:776`) | ↓ minor / ↓ minor | `failure_kind` residue (T5.1); D9 stands as-is; escalation tables rejected (§8) |

(The dimension's critical, `unsandboxed-exec-env-inheritance`, is C3. Its
other two findings were refuted — see below.)

### Test architecture

| Finding | Evidence | Verdicts | Disposition |
| --- | --- | --- | --- |
| e2e tier is dead: marker defined and documented, zero tests carry it; `test_e2e.py` mislabeled | `pyproject.toml:79-81`; zero `mark.e2e` in forge/tests; `test_e2e.py:1-5` runs in the default suite | ✓ major / ↓ minor | T8.2 (marker honesty; rename to `test_pipeline.py`) |
| Workflow-test harness wired through module globals: 30 `global` statements, 11 reset functions | `test_workflows.py:102-103,175-177,135,289` | ✓ / ✓ | T5.5 (ScenarioState closures) |
| Four near-duplicate stub-activity suites in one 4,134-LOC file | `call_llm` stubbed at `:227,530,897,1230`; four `_MOCK_ACTIVITIES` lists, four run helpers | ✓ / ✓ | T5.5 |
| sax-llm tests live in forge, importing sax-llm privates; 263 LOC at home vs 2,515 stranded | `test_llm_providers_mistral.py:11-18` imports six `sax_llm.mistral` privates | ✓ / ✓ | T1.1 |
| conftest hand-mirrors worker.py's registry registration — already drifted | `conftest.py:18-44` (7 types) vs `worker.py:128-157` (7 + TranscriptAnalysisResult) | ✓ major / ↓ minor | T3.5 (registry deleted; frozen OUTPUT_TYPES shared) |
| Coverage gates incoherent: 85 / 84 / 25 / none / none | forge 85, pbook 84, sax-llm 25, ocr none, forge-contracts none | ✓ major / ↓ minor | T2.2 (85% per package in root CI) |
| Workflow scenarios re-run once per assertion; `parametrize` ×4 in 1,354 tests | `test_workflows.py:310-343` five full round-trips for one TaskResult | ✓ minor / ↓ none | T5.5 |
| CLI tests string-patch private helpers of the module under test | 16× `patch("forge.cli._submit_and_wait")`; cross-repo pbook patches ×17 | ✓ / ✓ (minor) | T8.4 (replaced opportunistically in touched commands) |
| forge/ocr conftests duplicate fixtures; `dispose_store_engines` monkeypatches sqlalchemy globally | `forge/conftest.py:127-199` vs `ocr/conftest.py:38-117` | ✓ / ✓ (minor) | T3.6 (engine-per-process removes the need) |

### Refuted findings (checked and rejected)

Four findings drew a refutation from at least one panel. The evidence in
each was real; the claimed harm or the proposed fix was not.

| Finding | Refutation |
| --- | --- |
| Fuzzy edits need an `ast.parse` edit-local gate (safety) | Already triple-mitigated: fuzzy is the last resort of the D55 chain, D57's 0.05 uniqueness margin raises `EditApplicationError` on ambiguity, and ruff lints every written file for code domains; a syntactically valid wrong-location edit passes `ast.parse` by definition. Residue kept: `allow_fuzzy` knob + surfacing in ValidationResult (T7.6). |
| Plaintext prompts need retention/redaction/TTL (safety) | Factually correct (`store.py:86-87`, no purge path) but no harm vector: a local single-user store inside the same trust boundary as the repos whose bytes it copies; redaction would corrupt the interactions table's purpose (debugging, eval, learning loop). |
| SQLite-default integration tests lack production fidelity (test) | The opt-in postgres tier already covers the dialect-branching code (`test_migrations_postgres.py:92-112`) and is demonstrably run; making the default suite require a podman daemon is net-negative. SQLite stays default; postgres stays opt-in, mandatory for pgvector/knowledge tests (T8.2 documents the scheme). |
| Batch-signal protocol copies need a contracts-level mixin (workflow) | The actual contract (signal name, BatchResult) is already centralized; the receiving idiom is five stable lines, and a cross-repo behavioral mixin resting on inspected temporalio MRO internals is worse than the duplication. Moot once the signal path is deleted (Phase 4). |

## 5. Major findings — pbook product

From Plan B's own review (`pbook/design/REVIEW-2026-06.md`, verified against
code 2026-06-09), spot-verified by the wave-5 cross-review panels.

| Finding | Evidence | Disposition |
| --- | --- | --- |
| Workflow-as-RPC everywhere: 16 one-activity wrapper workflows; every CLI command except `migrate`/`skill-prompt` requires the worker | `pbook/workflows/cli_ops.py:50-272`; cross-review verified zero external importers of all 16 | T6.1 (library-first `service.py`; 20 of 22 workflows deleted) |
| Agent hot-path latency: `pbook search` = uvx fetch + Temporal connect + four sequential activities | two-pass DB access (`fetch_candidates` → `llm_embed` → `compute_similarities_by_id` → `score_and_pack`) | T6.1 (sync path, p50 <1s target) |
| Retrieval hierarchical and gated: zero-tag-overlap entries can never surface; NULL embeddings invisible; no lexical search; feedback ignored on the query path | `score_entry` returns 0.0 and `get_entries_by_tags` gates the candidate pool | T6.5 (hybrid lexical+semantic+tag RRF; tags boost, never gate) |
| No evals: zero golden sets, no judge calibration, no retrieval metrics; published agent-memory research puts unvalidated extraction noise ~30% | "better to extract nothing than mislead" exists only as prompt text | T6.2 (judge calibration first), T6.6 (eval suites A & C) |
| No entry lifecycle: `needs_review`/`rejected` booleans + CLI hard-delete; illegal states constructible; `pbook reject` destroys audit | confirmed independently by Plan A's knowledge dimension | T6.3 (status state machine; `pbook purge` the only hard delete) |
| Idempotency by cosine thresholds (0.85 entries / 0.92 sources, hardcoded): retries after embedding drift duplicate entries | no deterministic keys anywhere in pbook | T6.3/T6.4 (`origin_hash` UNIQUE + ON CONFLICT; thresholds demoted to match-or-attach policy) |
| ExtractionWorkflow loops per experience with no error isolation — first failure kills the batch | `pbook/workflows/extraction.py:68-136` | T6.4 (per-experience try/except isolation) |
| Consolidation save/prune split non-atomic; consolidated output lands pre-approved, bypassing the quality gate | four sequential activities per cluster; 45% coverage | T6.4 (one-transaction apply; survivors re-earn trust in probation) |
| Worker registers retired `claude-3-5-sonnet-20241022` as the default provider | `pbook/worker.py:106`; verified retired against the live API | T3.2 |
| Full-precision `vector(1536)`; dimension hardcoded ×3; no model/dim metadata columns | halfvec halves storage at ~99% recall | T6.3 (halfvec + HNSW + metadata columns) |
| Feedback is bare counter increments; helpfulness affects only the no-query path; nothing ties served entries to outcomes | `pbook/store.py:564-570` | T6.5 (retrieval/feedback event tables; human-confirmed only) |
| Production path never chunks transcripts; no size guard | `chunk_transcript()` exists unused | T6.4 (claim-check transcripts by path) |
| No replay tests, no migration tests, no versioning posture | — | T5.5 (replay scaffold platform-wide), T6.3 (first migration test) |

**Plan-coverage asymmetry (both attack reports conceded):** Plan B is
silent on forge's internals — the correlation bug, poller data loss,
event-loop blocking, secret-inheriting subprocesses, the nested fan-out
flag bug, workflow triplication, grimp worktree drift, idempotency
swallowing, context/exploration budgets. Plan A's forge phases survive
nearly verbatim; Plan B's pbook product core (lifecycle, extract+judge,
hybrid retrieval, feedback events, evals, workflow-as-RPC deletion) grafts
wholesale. The merge is mostly additive; only batch transport and library
topology genuinely collided.

## 6. Where the two plans agreed (independent convergence)

1. **uv-workspace monorepo** — single lock, per-package pyprojects/tests,
   mypy strict + 85% gates everywhere, CI from zero, archived old repos.
2. **Kill all module-level singletons and both string-keyed registries**;
   frozen `Settings` per process; class-based activities (Temporal's
   sanctioned DI); tests stop monkeypatching globals.
3. **Delete forge's dead 1,400-LOC provider copy** and repatriate the
   2,515 LOC of sax-llm tests stranded in forge.
4. **Ingestion inverts to pbook end-to-end** — forge's ingestion
   workflows, `_INGESTION_AVAILABLE` guards, pbook imports, direct
   pbook-DB reads, and the raw-JSON cross-queue contract all die; session
   rows owned by the workflow, never CLI-seeded.
5. **forge's playbooks subsystem is superseded by pbook** (owner decision)
   — table, extraction/manual/export workflows, CLI, tag inferrers all
   deleted.
6. **Deterministic idempotency at every durable write** — content-hash
   natural keys + `ON CONFLICT DO NOTHING`; cosine thresholds demoted to
   match-or-attach policy, never retry protection.
7. **Entry lifecycle as a status state machine** replacing the
   `needs_review`/`rejected` boolean pair; pure transition function;
   nothing hard-deletes.
8. **One model-tier registry** — both found the forge↔pbook copies had
   drifted; current Sonnet verified as `claude-sonnet-4-6` (forge's
   `claude-sonnet-4-5-20250929` pin stale; pbook's worker default
   retired).
9. **Namespaced tags** normalized at write, never save-failing;
   per-item failure isolation in batch loops; claim-check for large
   payloads (inline ≤256KB, pointer beyond); centralized retry presets;
   replay-determinism rules.
10. **No NewType/frozen-base retrofit programs, no BudgetLedger, no
    network-denial sandbox, no retention/redaction layer** — both plans'
    maintenance-cost bars reject them independently.

## 7. Where they differed — the ten adjudications

Evidence below is from the wave-5 cross-review (4 fact-check panels: API
capabilities, Temporal arithmetic, codebase usage, build/infra).

| # | Conflict | Plan A | Plan B | Decision and evidence |
| --- | --- | --- | --- | --- |
| 1 | Batch result transport | Shared `BatchPollerWorkflow` + signals, hardened (envelope union, correlation dicts, delivery state machine) | Timer-loop polling per waiting workflow | **B wins (reversal R1, D88).** Under D79 (1 request = 1 batch = 1 waiter) a shared poller amortizes zero provider calls. Verified arithmetic: ~11 history events/poll → 25h wait at 600s = 1,650 events (~3% of the 51.2k limit); worst case ~30 waits × 6h at 300s ≈ 24k — safe. Cadence equals today's 600s poller, so latency unchanged. Both criticals C1/C2 become unconstructible — the requester is the recipient. Batches-API RPM trivial (100 waiters at 600s = 10 RPM vs Tier 1's 50). Residue from A: workflow-minted request_id/custom_id, claim-check fetch shape, `batch_jobs` as audit/spend ledger. Accepted tradeoff: a dead waiter orphans its paid batch (reconciliation deferred; symptom is a slightly higher invoice). |
| 2 | Library topology | `libs/forge-contracts` + `libs/sax-llm`, DAG-separated | One `libs/sax-platform` | **B wins, with A's enforcement grafted (D89).** With the signal SPI dead, forge-contracts has no second party: verified pbook imports none of it (zero hits, not even a dependency) and ocr needs ~390 of its 641 lines of generic plumbing. import-linter verifiably enforces layering within one package; the old repo split was only module-level discipline anyway (boto3 a hard dep). Grafted: `sax_platform.contracts` sandbox-light layer forbidden from importing SDKs, enforced in CI. |
| 3 | Structured output | Forced tool use (sax-llm); registry → explicit Mapping | `client.messages.parse`; registries deleted | **B wins, extended to the batch lane it forgot (D90).** Verified: structured outputs GA, work in the Message Batches API (`output_config.format`), compose with prompt caching (30–98% hit rates in batches; 1h TTL), supported on opus-4-8/sonnet-4-6/haiku-4-5. `messages.parse` is sync-only, so the platform lib owns both lanes; forced tool use retires platform-wide (supersedes D75's mechanism). |
| 4 | pbook ingestion LLM transport | Batch via the platform SPI | Sync structured-output activities | **B wins, rationale rewritten and scoped (reversal R2, D91).** Measured volume: 102 ingestable sessions / ~27 days ≈ 110/month — batch saves ~$2–5/month while adding two up-to-24h round-trips to a freshness-sensitive loop. With #1 decided there is no signal SPI to consume. Boundary recorded: the volume exception does not extend to forge — D76 batch-first stands on per-token economics, not realized volume. |
| 5 | Forge's knowledge consumption | Read-only `knowledge.approved_entries` view + tag-overlap ranking | Unspecified | **A's contract, upgraded with B's retrieval insight (D92).** Cross-review confirmed the gap: INTEGRATION.md names forge only to delete couplings; library/CLI paths would recreate app→app coupling or put uvx on the hot path. Fix to A's recall hole: the view exposes `search_tsv`; forge retrieval = lexical rank UNION tag-overlap candidates, fused by one pure scorer with a capped tag boost (tags boost, never gate). No embeddings on forge's hot path. |
| 6 | ocr batch waiting | Consumer of forge's poller via SPI signals | Timer-loop its own Mistral polling | **B wins** (consequence of #1). Verified: all Mistral API interaction currently runs on forge's worker, and no production code path uses Mistral chat → chat support deleted; `MistralOcr` keeps the OCR-batch pieces. Bonus fix: ocr's gather restructured to parent-awaited children (signals existed only because both sides were ABANDON-children of a fire-and-forget parent; a failed store child hung the gather for 26h). Last cross-workflow signal disappears platform-wide. |
| 7 | Monorepo import mechanics | git-filter-repo, full history | Shallow subtree, small clones | **A wins.** uvx caches the bare repo and tool environments; all five repos total ~8.7MB of history, so subtree's savings are immaterial while filter-repo preserves blame/`git log --follow`. Grafted from B: pin the skill's uvx ref to a tag; verify end-to-end early. Repo name `sax` (B's). |
| 8 | Python version | Unaddressed | 3.14 if wheels allow | **B wins, condition verified:** temporalio (abi3), psycopg-binary (cp314 incl. macOS arm64), pgvector, pymupdf, grimp, anthropic, openai all install on CPython 3.14 today. Standard GIL (3.14t wheels incomplete). |
| 9 | pbook destructive migration sequencing | n/a | Single destructive migration incl. a one-time judge sweep of backfilled actives; eval calibration later | **B with two attack-found fixes:** (a) judge calibration moves before the backfill sweep, and the sweep runs report-only first, applying demotions only after the ≥85%/100%-trap gates pass — otherwise an uncalibrated judge demotes the corpus and seeds the eval goldens with its own errors; (b) one-time JSON dump before the destructive migration. Magic constants marked provisional; eval gates set from a measured baseline. |
| 10 | Supabase posture | Unaddressed | Dedicated `pbook` schema; pooler 6543 notes; pgvector ≥0.7/≥0.8 preflight | **B verified accurate, one wording fix:** psycopg does not auto-disable prepared statements on the transaction pooler — the engine factory must set `prepare_threshold=None` on port 6543. |

## 8. Rejected ideas — dispositions

Everything proposed during the review that both plans' (or the verifiers')
maintenance bars rejected, recorded so it is not re-litigated. D97 records
the durable subset.

| Idea | Disposition |
| --- | --- |
| Shared-poller hardening (envelope union, correlation dicts, two-column delivery state machine) | Superseded by adjudication #1: the whole signal subsystem is deleted; hardening a structure that amortizes zero provider calls is wasted work. |
| Two-library topology (`forge-contracts` + `sax-llm`) | Superseded by adjudication #2: no second party left for the contracts package; layering enforced inside `sax_platform` by import-linter instead. |
| Batch pbook ingestion (the SPI's "second consumer") | Reversed (R2, D91): ~$2–5/month savings against two 24h round-trips at ~110 sessions/month; boundary clause protects forge's batch-first rationale. |
| NewType ID program + frozen-WireModel base retrofit | Pydantic erases NewType to `str` (zero runtime enforcement); `frozen=True` is shallow and does not prevent the cited list-field mutation; both are sustained-discipline programs for one operator. Literal types on free-form status strings capture the real value. |
| RunBudget/BudgetLedger admission control at every dispatch seam | Every loop is structurally capped; spend is half-price batch; wall-clock ceilings exist as Temporal timeouts; provider-side spend limits cost zero code. Kept: run-level LLMStats aggregation (T5.1). |
| `run_sandboxed` policy module with command allowlist + network denial | Network denial not cleanly implementable on macOS (sandbox-exec deprecated) and breaks documented test_command/e2e usage. The verified kernel is the scrubbed env dict at the two seams (T1.7). |
| Prompt retention/redaction/TTL layer | Refuted: local single-user store inside the same trust boundary; redaction corrupts the interactions table's purpose; purge destroys the only per-call token records. |
| `ast.parse` fuzzy-edit gate with retryable FuzzyEditRejected | Refuted: a syntactically valid wrong-location edit passes `ast.parse` by definition; ruff already guards code domains; non-code domains emit markdown. Residue: `allow_fuzzy` knob + surfacing (T7.6). |
| Escalations table + CLI + resume primitive | For one operator the FAILURE_TERMINAL run row is the escalation; a resume primitive contradicts the disposable-worktree recovery model. D9 stands as-is. |
| Outcome union (BlockedOnHuman/BudgetExhausted/PartialSuccess) | No producers exist; dead match arms are speculative generality. Adding a variant when D9 escalation is built is a small local change. Residue: `failure_kind` Literal (T5.1). |
| `models.py` dissolution into domain/wire/llm_schemas packages | 34-importer churn justified by aesthetics; a single sandbox-safe models module is a serviceable Temporal pattern; the module shrinks by deletion anyway. |
| `schema_version` on the S3 result envelope | Guards a drift window that effectively does not exist: blob life bounded by the 25h wait + bucket TTL; one operator deploys all packages in lockstep. (Envelope deleted anyway in Phase 4.) |
| BatchResultReceiver mixin in contracts | Refuted: cross-repo behavioral inheritance resting on inspected temporalio MRO internals, to save ~20 stable lines. Moot under timer-loop. |
| BatchResultInbox helper class | Fact-checker found its pop-on-consume dedup hole (duplicates re-enter after consumption), requiring tombstones — machinery creep. The interim is a dict + `setdefault`; the endgame deletes signals. |
| Multi-request batch accumulator / per-queue pollers | D79 stands: the 50% discount is size-independent; grouping-by-batch_id keeps future accumulation a pure submit-side decision. |
| `provider.list_batches` reconciliation pass for orphaned batches | Over-engineering at this scale: workflow-minted request_ids make duplicates correlatable; the orphan symptom is a slightly higher invoice. Deferred, documented in D88. |
| Context-by-pointer (blob offload for prompts end-to-end) | Full-budget contexts sit under Temporal's payload limits; the realistic blowup (exploration accumulation) is fixed by a deterministic cap (T7.4); pointers would blind Temporal history — the primary forensic surface. |
| Universal `workflow.patched()` discipline for every workflow change | In batch mode that means every change forever, accumulating deprecate_patch debt with no fleet to coordinate. Drain/revert-and-replay is the default; the valuable line is the reset-cannot-recover-batch-results warning (T8.1, WORKERS.md). |
| `count_tokens` SPI method with per-task scaffolding measurement | Cross-repo churn plus an online API dependency inside context assembly for a latent invariant with 2x headroom; undercounts exactly on retries. Calibrated constant + `effective_budget` arithmetic instead (T7.3). |
| LanguageAnalyzer protocol / hand-rolled import resolution now | Phase 13 / Release 2 scope; bespoke resolution (relative imports, re-exports, namespace packages) is an accuracy regression the operator then owns. The subprocess-PYTHONPATH fix resolves both verified failure modes (T7.2). |
| PromptSpec/facts-compose-render dissolution of `context.py` | The largest proposed refactor justified by the weakest finding. One parameterized builder + TaskFacts + shell error-reads gets ~80% of the value (T7.5). |
| WorkflowGateway protocol + dedicated `cli.py` split workstream | Verified: importing `forge.cli` loads zero temporalio modules, so the testability payoff was overstated; the CLI shrinks anyway when ingest/playbooks commands leave in Phase 6. String-patched tests replaced opportunistically (T8.4). |
| Postgres-by-default integration tests / retiring the `postgres` marker | Refuted: overturns db.py's documented SQLite dev/test contract and makes every default `uv run pytest` depend on a podman daemon. SQLite default; postgres opt-in, mandatory for pgvector/knowledge tests. |
| Real-provider e2e batch smoke test in forge | Batch turnaround is unbounded, so the smoke is slow, costly, and never run — recreating the dead-tier problem. The real e2e lives env-gated in ocr. |
| Actions-as-data step engine (workflows as interpreters) | Temporal workflow code is already the deterministic, replay-tested core; an interpreter adds indirection with no determinism or testability gain. Proposal and verifier rejected it independently. |
| Absorbing pbook into forge as `forge.knowledge` | Overridden by owner constraint (pbook stays a separate app) and independently broken by verifiers: pbook has cross-project consumers, and the pgvector requirement would force forge's SQLite dev loop onto containers. |
| sax-llm re-exporting a contracts-owned ImageBlob | Dependency inversion — entangles the generic provider library with platform contracts. Map at the existing conversion point instead. (Moot after D89.) |
| Env-var renames (`FORGE_OCR_S3_*` → `PLATFORM_S3_*`, etc.) | Breaks the single operator's working configuration for zero functional gain. |
| Blanket-approve migration of forge playbooks rows into pbook | Self-contradiction caught by the fact-checker: the table is documented as polluted by the re-extraction loop's duplicates; wholesale approval would push that junk past the review gate. Dump + manual triage via `pbook add` (T6.7). |

## 9. Outcome

The merged plan was **approved 2026-06-10** (D86), explicitly including the
two owner-decision reversals:

- **R1** — the signal-based batch SPI is replaced by per-workflow
  timer-loop polling (D88, supersedes D77/D78, restores D80).
- **R2** — pbook ingestion runs sync, not batch (D91, with the recorded
  boundary protecting forge's D76 batch-first rationale).

Decision records: [DECISIONS.md](../DECISIONS.md) D86–D97 (monorepo D87,
timer-loop D88, one platform library D89, structured outputs D90, sync
ingestion D91, knowledge view contract D92, composition roots D93, tier
registry + adaptive thinking D94, pure step logic D95, context-engine
corrections D96, rejected-hardening dispositions D97).

Migration: 47 tasks across 8 phases —

1. Stop the bleeding in the current repos (T1.0–T1.8)
2. Monorepo `sax` (T2.1–T2.3d)
3. `sax_platform` consolidation + structured outputs (T3.1–T3.6)
4. Batch transport simplification (T4.1–T4.3)
5. Workflow consolidation (T5.1–T5.5)
6. Knowledge product + forge consumption (T6.1–T6.7)
7. Context engine (T7.1–T7.6)
8. Docs, decisions, honesty (T8.1–T8.4)

Task files: [development-plans/tasks/](../../development-plans/tasks/),
sourced one-for-one from
[HANDOFF-architecture-review-2026-06-10-tasks.md](../../development-plans/HANDOFF-architecture-review-2026-06-10-tasks.md).
Phase ordering is load-bearing: Phases 5 and 6 are serialized (both touch
forge worker registration and OUTPUT_TYPES); Phase 4 needs Phase 3's
platform batch helpers; T1.2/T1.3 are explicitly interim and are deleted by
Phase 4.
