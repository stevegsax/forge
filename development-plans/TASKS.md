# Forge Task List

The single source of truth for what is done vs. not. Work the next unchecked task in priority order (see [PROCESS.md](PROCESS.md)). Per-task detail lives in task files in this directory; narrative status and the full evidence for each tech-debt item live in [../docs/OVERVIEW.md](../docs/OVERVIEW.md).

## Architecture migration (approved 2026-06-10)

The merged platform redesign plan was approved 2026-06-10, including reversals R1 (timer-loop batch transport) and R2 (sync pbook ingestion). This is the active work queue; it takes priority over the tech-debt backlog below, much of which it subsumes (reconciled at [T8.2](tasks/T8.2-test-tier-honesty-overview-rewrite.md)). Context: [HANDOFF-architecture-review-2026-06-10.md](HANDOFF-architecture-review-2026-06-10.md), decisions D86–D97 in [../docs/DECISIONS.md](../docs/DECISIONS.md), findings in [../docs/reviews/2026-06-architecture-review.md](../docs/reviews/2026-06-architecture-review.md).

**Amended 2026-07-08** per [../forge-review-2026-07-08.md](../forge-review-2026-07-08.md): dated amendment sections in T1.1, T1.7, T2.1, T3.1, T3.2, T3.5, T4.1, T5.2, T5.3, T6.3, T8.1, T8.4; T5.6 added; Phase 0 (T0.1–T0.4) extracted from T8.4's inventory. The external merged-plan/wave-JSON evidence copies are lost — the review doc + D86–D97 are the surviving record.

**Capture sweep 2026-07-08 (same day):** a 31-agent verification pass confirmed 44 of 45 outstanding findings live and mapped each to a home — adding T0.5–T0.8, capture-sweep amendments to T1.3, T1.4, T3.1, T3.2, T3.4, T3.6, T4.1, T5.1, T5.3, T5.5, T5.6, T6.4, T7.2, T7.4, T7.5, T7.6, explicit waivers in T8.4, and one restored June disposition the task files had dropped (T6.4's write-time tag enforcement). **56 tasks total.**

Phase ordering is load-bearing: 1 → 2 → 3 → 4 → 5; Phase 6 is serialized after Phase 5 (both touch forge worker registration and OUTPUT_TYPES); Phases 6 and 7 may run in parallel (disjoint files); Phase 8 closes. Within Phase 1 all tasks are independent except T1.3 (needs T1.0). Phase 0 sits outside the graph — its four tasks are independent of every phase and of each other, and can land before, during, or after any of them.

### Phase 0 — Standalone fixes (no phase dependency; added 2026-07-08)

- [ ] [T0.1 — Observability defaults](tasks/T0.1-observability-defaults.md)
- [ ] [T0.2 — CLI helper fixes](tasks/T0.2-cli-helper-fixes.md)
- [ ] [T0.3 — Dead code and stale artifact deletion](tasks/T0.3-dead-code-stale-artifacts.md)
- [ ] [T0.4 — Docs truth sweep (pre-migration)](tasks/T0.4-docs-truth-sweep.md)
- [ ] [T0.5 — Idempotent, path-safe edit application](tasks/T0.5-idempotent-edit-application.md) *(added 2026-07-08 sweep)*
- [ ] [T0.6 — Eval judge integrity](tasks/T0.6-eval-judge-integrity.md) *(added 2026-07-08 sweep)*
- [ ] [T0.7 — Deploy hardening](tasks/T0.7-deploy-hardening.md) *(added 2026-07-08 sweep)*
- [ ] [T0.8 — Record the pre-migration operating decision](tasks/T0.8-operating-decision.md) *(added 2026-07-08 sweep)*

### Phase 1 — Stop the bleeding (current repos)

- [ ] [T1.0 — Uniform editable sibling sources](tasks/T1.0-uniform-editable-sibling-sources.md)
- [ ] [T1.1 — Delete the dead provider stack; repatriate sax-llm's tests](tasks/T1.1-delete-dead-provider-stack.md)
- [ ] [T1.2 — INTERIM batch-result correlation stopgap](tasks/T1.2-interim-batch-result-correlation.md) *(deleted by Phase 4 — do not extend)*
- [ ] [T1.3 — INTERIM minimal poller patch](tasks/T1.3-interim-poller-patch.md) *(needs T1.0; deleted by Phase 4)*
- [ ] [T1.4 — Unblock the worker event loop](tasks/T1.4-unblock-worker-event-loop.md)
- [ ] [T1.5 — Nested fan-out propagation fix](tasks/T1.5-nested-fan-out-propagation.md)
- [ ] [T1.6a — Idempotency rekey](tasks/T1.6a-idempotency-rekey.md)
- [ ] [T1.6b — Batch-wait failure symmetry](tasks/T1.6b-batch-wait-failure-symmetry.md)
- [ ] [T1.7 — Env scrub at model-influenced subprocess seams](tasks/T1.7-env-scrub-subprocess-seams.md)
- [ ] [T1.8 — Small dedup batch + kill runs-extraction](tasks/T1.8-small-dedup-batch.md)

### Phase 2 — Monorepo `sax`

- [ ] [T2.1 — Workspace creation](tasks/T2.1-workspace-creation.md)
- [ ] [T2.2 — Root gates](tasks/T2.2-root-gates.md)
- [ ] [T2.3a — mypy strict: sax-platform contracts](tasks/T2.3a-mypy-strict-platform-contracts.md)
- [ ] [T2.3b — mypy strict: sax-platform llm + rest](tasks/T2.3b-mypy-strict-platform-llm.md)
- [ ] [T2.3c — mypy strict: ocr](tasks/T2.3c-mypy-strict-ocr.md)
- [ ] [T2.3d — mypy strict: pbook](tasks/T2.3d-mypy-strict-pbook.md)

### Phase 3 — `sax_platform` consolidation + structured outputs

- [ ] [T3.1 — Platform LLM client (both lanes)](tasks/T3.1-platform-llm-client.md)
- [ ] [T3.2 — One tier registry + thinking migration](tasks/T3.2-tier-registry-thinking-migration.md)
- [ ] [T3.3 — MistralOcr; Mistral chat deleted](tasks/T3.3-mistral-ocr-chat-deleted.md)
- [ ] [T3.4 — Platform plumbing modules](tasks/T3.4-platform-plumbing-modules.md)
- [ ] [T3.5 — Forced-tool-use retirement](tasks/T3.5-forced-tool-use-retirement.md)
- [ ] [T3.6 — Composition roots everywhere](tasks/T3.6-composition-roots.md)

### Phase 4 — Batch transport simplification (timer-loop)

- [ ] [T4.1 — forge: submit → poll-loop → fetch](tasks/T4.1-forge-timer-loop-transport.md)
- [ ] [T4.2 — ocr: own polling + gather restructure](tasks/T4.2-ocr-own-polling-gather-restructure.md)
- [ ] [T4.3 — Transport decisions sweep](tasks/T4.3-transport-decisions-sweep.md)

### Phase 5 — Workflow consolidation (forge)

- [ ] [T5.1 — Pure step logic](tasks/T5.1-pure-step-logic.md)
- [ ] [T5.2 — Single step block](tasks/T5.2-single-step-block.md)
- [ ] [T5.3 — Single gather + dispatch](tasks/T5.3-single-gather-dispatch.md)
- [ ] [T5.4 — Split the monolith](tasks/T5.4-split-the-monolith.md)
- [ ] [T5.5 — Harness rebuild + replay tests](tasks/T5.5-harness-rebuild-replay-tests.md)
- [ ] [T5.6 — Plan preflight gate](tasks/T5.6-plan-preflight-gate.md) *(added 2026-07-08)*

### Phase 6 — Knowledge: pbook product + forge consumption (after Phase 5)

- [ ] [T6.1 — pbook library-first](tasks/T6.1-pbook-library-first.md)
- [ ] [T6.2 — Judge calibration (BEFORE the migration sweep)](tasks/T6.2-judge-calibration.md)
- [ ] [T6.3 — Destructive schema migration](tasks/T6.3-destructive-schema-migration.md) *(gated by T6.2)*
- [ ] [T6.4 — IngestWorkflow + CurationWorkflow](tasks/T6.4-ingest-curation-workflows.md)
- [ ] [T6.5 — Hybrid retrieval + feedback](tasks/T6.5-hybrid-retrieval-feedback.md)
- [ ] [T6.6 — Eval suites A & C](tasks/T6.6-eval-suites-a-c.md)
- [ ] [T6.7 — Forge consumption + playbooks deletion](tasks/T6.7-forge-consumption-playbooks-deletion.md)

### Phase 7 — Context engine (forge; parallel with Phase 6)

- [ ] [T7.1 — ProjectDescriptor](tasks/T7.1-project-descriptor.md)
- [ ] [T7.2 — Worktree-accurate graph](tasks/T7.2-worktree-accurate-graph.md)
- [ ] [T7.3 — Honest token accounting](tasks/T7.3-honest-token-accounting.md)
- [ ] [T7.4 — Exploration budget](tasks/T7.4-exploration-budget.md)
- [ ] [T7.5 — One prompt builder](tasks/T7.5-one-prompt-builder.md)
- [ ] [T7.6 — Fuzzy-edit governance residue](tasks/T7.6-fuzzy-edit-governance.md)

### Phase 8 — Docs, decisions, honesty

- [ ] [T8.1 — Review doc + DECISIONS completion](tasks/T8.1-review-doc-decisions-completion.md)
- [ ] [T8.2 — Test-tier honesty + status-of-record rewrite](tasks/T8.2-test-tier-honesty-overview-rewrite.md)
- [ ] [T8.3 — pbook design-docs truth pass](tasks/T8.3-pbook-design-docs-truth-pass.md)
- [ ] [T8.4 — Final sweep](tasks/T8.4-final-sweep.md)

## Completed

### Phase roadmap (Release 1)

- [x] **Phases 1–12 and 14** — universal step, planning, fan-out, context assembly, observability store, knowledge extraction, exploration, error-aware retries, prompt caching, fuzzy edit matching, model routing, extended thinking, batch processing. Per-phase module map: [../docs/PHASES.md](../docs/PHASES.md).

### Beyond the roadmap

- [x] **Store externalization** — Postgres backend + S3 OCR blobs + survivable writes ([externalize-store-postgres-s3.md](externalize-store-postgres-s3.md)).
- [x] **OCR pipeline** — Mistral OCR sync + batch, S3 blobs, PDF chunking (since extracted to the sibling `ocr` repo).
- [x] **OCR separation** — OCR extracted into the sibling `ocr` repo as a `forge-contracts` consumer; Forge is an OCR-agnostic batch platform ([separate-ocr-into-its-own-repo.md](separate-ocr-into-its-own-repo.md)).
- [x] **Transcript ingestion** — `forge ingest` → pbook `ExtractionWorkflow` cross-queue.
- [x] **Planner evaluation framework** — `eval/` (corpus, deterministic checks, LLM-as-judge, baseline/candidate comparison).
- [x] **Secure remote access** — mTLS Temporal access + EC2 deploy package.

## Remaining

Priority order. OPEN tech-debt (no mechanism today) before PARTIAL (hardening) before deferred features. Evidence and code pointers for every debt item: [../docs/OVERVIEW.md](../docs/OVERVIEW.md) → "Known issues & technical debt."

### Technical debt — Open (no mechanism in code)

- [ ] Execution sandboxing / action policy — providers and `validate.py` run raw `subprocess`; no allowlist/sandbox/egress policy.
- [ ] Run-level cost / latency budget — per-run dollar/token/wall-clock cap + kill-switch + fan-out/exploration admission control.
- [ ] Fuzzy-edit semantic verification — post-edit AST/parse check; make fuzzy matching policy-gated.
- [ ] Closed-loop model routing — calibration / canary / quality-driven fallback.
- [ ] Prompt/response privacy — retention/TTL, redaction, and encryption in the observability store.

### Technical debt — Partial (harden existing mechanism)

- [ ] Expand transition vocabulary (`policy_violation`, `budget_exhausted`, `partial_success`, `blocked_on_human`).
- [ ] Plan contract — symbol-level write-sets + preflight overlap/cycle gate (today: file-level, checked only in `eval/`).
- [ ] Conflict resolution — intent/regression verification; symbol-level (not file-level) scope.
- [ ] Retry — failure-class classifier (deterministic bug / missing context / flaky / environment).
- [ ] Exploration — per-provider quota + ROI scoring + dedup.
- [ ] Halt-when-confused — escalation queue / paging / safe resume.
- [ ] Validation depth — SAST / secrets / contract / performance gates.
- [ ] Eval as CI release gate — end-to-end task success + adversarial/prompt-injection corpus.
- [ ] Domain-agnosticism — non-Python context discovery + positive validators for non-code domains.
- [ ] Human-in-the-loop — structured intervention beyond out-of-band git review + manual playbook approval.
- [ ] Multi-provider parity — cross-provider conformance suite; reduce Anthropic-default coupling.

### Remaining features / specs

- [ ] OCR Web API — specified, not built; the spec (`ocr_web_api.feature`) was removed from this repo in the OCR split (`d661f41`; recoverable from git history at `7395e65`). If built, both the spec and the API belong in the sibling `ocr` repo.
- [ ] Structured human-in-the-loop — [../docs/requirements/human_in_the_loop.feature](../docs/requirements/human_in_the_loop.feature) (specified, not built).
- [ ] Phase 13 — tree-sitter multi-language — [../docs/planning/PHASE13.md](../docs/planning/PHASE13.md).
- [ ] LSP-based context generation — [../docs/planning/LSP_INTEGRATION_PLAN.md](../docs/planning/LSP_INTEGRATION_PLAN.md).
- [ ] Multi-transform DAG planner — [../docs/planning/task-management/DECOMPOSITION.md](../docs/planning/task-management/DECOMPOSITION.md) (draft; would replace `activities/planner.py`).

## Dependencies

- **Structured human-in-the-loop** is a prerequisite for the **multi-transform DAG planner** (its clarification/approval gates depend on it).
- **OCR Web API** — OCR now lives in the sibling `ocr` repo; if built, the web API belongs there (the spec was removed in the OCR split and is recoverable from git history).
- **Phase 13 (tree-sitter)** and **LSP context generation** both rewrite context assembly (`code_intel/`) — coordinate to avoid rework.
- Sandboxing (Open #1) gates safe execution of any non-`code_generation` domain work and any multi-tenant use.
