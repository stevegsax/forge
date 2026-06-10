# Forge Task List

The single source of truth for what is done vs. not. Work the next unchecked task in priority order (see [PROCESS.md](PROCESS.md)). Per-task detail lives in task files in this directory; narrative status and the full evidence for each tech-debt item live in [../docs/OVERVIEW.md](../docs/OVERVIEW.md).

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

- [ ] OCR Web API — [../docs/requirements/ocr_web_api.feature](../docs/requirements/ocr_web_api.feature) (specified, not built).
- [ ] Structured human-in-the-loop — [../docs/requirements/human_in_the_loop.feature](../docs/requirements/human_in_the_loop.feature) (specified, not built).
- [ ] Phase 13 — tree-sitter multi-language — [../docs/planning/PHASE13.md](../docs/planning/PHASE13.md).
- [ ] LSP-based context generation — [../docs/planning/LSP_INTEGRATION_PLAN.md](../docs/planning/LSP_INTEGRATION_PLAN.md).
- [ ] Multi-transform DAG planner — [../docs/planning/task-management/DECOMPOSITION.md](../docs/planning/task-management/DECOMPOSITION.md) (draft; would replace `activities/planner.py`).

## Dependencies

- **Structured human-in-the-loop** is a prerequisite for the **multi-transform DAG planner** (its clarification/approval gates depend on it).
- **OCR Web API** — OCR now lives in the sibling `ocr` repo; if built, the web API belongs there (the spec stays here for the record).
- **Phase 13 (tree-sitter)** and **LSP context generation** both rewrite context assembly (`code_intel/`) — coordinate to avoid rework.
- Sandboxing (Open #1) gates safe execution of any non-`code_generation` domain work and any multi-tenant use.
