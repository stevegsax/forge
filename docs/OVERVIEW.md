# Forge — Project Overview

Forge is a batch-first LLM task orchestrator: it decomposes a request into independent work units, runs each as a single document-completion step in a Temporal workflow, validates, and reconciles. This file is the **status-of-record** — what is built, what remains, and what is known-broken. It points to the live tracker and roadmap rather than restating them.

- **Live task list** (what's in flight, completed vs. not): [../development-plans/TASKS.md](../development-plans/TASKS.md)
- **Phase roadmap** (the 14-phase build): [PHASES.md](PHASES.md)
- **How it works** (architecture): [ARCHITECTURE.md](ARCHITECTURE.md) · **Why** (decisions): [DECISIONS.md](DECISIONS.md)
- **Behavioral spec** (Gherkin): [requirements/](requirements/)

Completion below was established by verifying against the code on 2026-06-04, not by prior documentation. The OCR extraction was verified against the code on 2026-06-09 (merge `2556bfe`); the packages it created were later absorbed back into this repo as workspace members (D98, 2026-07-16) — the platform/consumer split survives as a package boundary, not a repo boundary.

## Status at a glance

- **Release 1 (core orchestrator + batch) is shipped.** Phases 1–12 and 14 are implemented and wired into the worker; Phase 13 (tree-sitter) is deferred to Release 2. See [PHASES.md](PHASES.md).
- **This repo is the monorepo** (D98, T2.1 complete 2026-07-16): one uv workspace, one lock, one venv — `apps/pbook`, `apps/ocr`, `libs/sax-llm`, `libs/forge-contracts`. A bare clone is self-contained; Python is pinned to **3.14** (standard GIL). The five predecessor repos are archived.
- **Deployment is local-first** (D99, T0.7 complete 2026-07-16): Temporal self-hosts in the podman stack with persistence in that stack's Postgres; the workers run under launchd on an always-on desktop; Supabase Postgres + S3 remain the managed stores. EC2, Terraform, and the mTLS remote-access gateway are deleted — remote access is out of scope until revisited.
- **Migration status:** Phase 1 done; Phase 2 has T2.2 (root gates) and T2.3a–d (mypy strict) left; Phases 3–8 ahead. Live queue: [../development-plans/TASKS.md](../development-plans/TASKS.md).
- **Work shipped outside the phase roadmap:** OCR pipeline (sync + batch — since extracted to the `ocr` app, now the `apps/ocr` workspace member), OCR separation (platform/consumer split via the shared `forge-contracts` package; merged 2026-06-05), store externalization (Postgres + S3 + survivable writes), transcript ingestion (`forge ingest` → pbook), planner evaluation framework, and secure remote access (mTLS + EC2 deploy; that infrastructure was retired by D99 in favor of the local-first deployment — client-side TLS code remains, dormant).
- **The orchestrator is code-first in practice.** Despite the "task-agnostic" framing, context discovery is Python-specific (see tech debt below).

## Implemented capabilities

Module paths are under `src/forge/`.

- **Core loop** — universal workflow step, transitions, git worktree isolation: `workflows.py` (`ForgeTaskWorkflow`), `activities/transition.py`, `git.py`.
- **Planning & fan-out** — single-pass planner producing ordered `PlanStep`s with optional parallel `sub_tasks`; LLM conflict resolution: `activities/planner.py`, `workflows.py` (`ForgeSubTaskWorkflow`), `activities/conflict_resolution.py`.
- **Context** — import-graph + PageRank + token-budget assembly, plus LLM-guided exploration: `code_intel/`, `activities/context.py`, `activities/exploration.py`.
- **Output & validation** — diff-based edits with 4-level fuzzy fallback; ruff + optional tests with error-aware retries: `activities/output.py`, `activities/validate.py`.
- **Model routing & thinking** — capability tiers, extended thinking for planning, prompt caching: `models.py` (`CapabilityTier`), `activities/planner.py`, `sax_llm.anthropic` (the `libs/sax-llm` workspace member).
- **Batch** — async submission + polling via Anthropic/Mistral Batch APIs: `batch_poller_workflow.py`, `activities/batch_*`.
- **Knowledge** — extraction → playbooks, transcript ingestion to pbook: `extraction_workflow.py`, `activities/extraction.py`, `ingestion_workflow.py`.
- **Observability** — SQLite/Postgres store, Alembic migrations, CLI inspection: `store.py`, `alembic/`, `cli.py` (`forge status`).
- **Batch SPI (OCR-agnostic)** — opaque-blob batch submit (`submit_batch_blob`) plus a domain-agnostic poller that forwards verbatim provider results to consumer workflows cross-queue: `activities/batch_submit.py`, `batch_poller_workflow.py`. OCR itself lives in the `apps/ocr` workspace member, consuming the platform via the shared `forge-contracts` package; neither package imports the other.
- **Providers** — provider protocol + Anthropic/Mistral adapters live in the `sax-llm` workspace member (`libs/sax-llm`) (`sax_llm/`), consumed via `get_provider`; Forge no longer carries its own provider layer.

## Requirements: complete vs. remaining

The behavioral spec lives in [requirements/](requirements/) (18 Gherkin feature files, frozen this pass). Sixteen feature areas are implemented; two are specified but not built.

**Remaining (specified, not implemented):**

| Feature spec | State | Evidence |
| --- | --- | --- |
| `requirements/ocr_web_api.feature` | **Not built** — no OpenAPI/paginated OCR web service; OCR now lives in the `apps/ocr` workspace member, so this API belongs there if built | No web framework (FastAPI/Flask/Starlette/uvicorn) exists anywhere in the repo |
| `requirements/human_in_the_loop.feature` | **Not built** — no structured pause/resume/approval primitive | No `HumanInputRequest`/`emit_user_prompt`; only batch-result and OCR-gather signals; human gating is out-of-band git review + manual playbook approval |

## Remaining work (beyond requirements)

- **Phase 13 — tree-sitter multi-language** (deferred): [PHASES.md](PHASES.md) → [planning/PHASE13.md](planning/PHASE13.md).
- **LSP-based context generation** (deferred, D38): [planning/LSP_INTEGRATION_PLAN.md](planning/LSP_INTEGRATION_PLAN.md).
- **Multi-transform DAG planner** — a richer planner (classify → clarify → DAG → adversarial judges) is a **draft design, not implemented**; it would replace the single-pass `activities/planner.py`: [planning/task-management/DECOMPOSITION.md](planning/task-management/DECOMPOSITION.md). The structured human-in-the-loop requirement above is part of this design.

## Known issues & technical debt

Mined from the four code reviews in `archive/to-merge/code-review/` (≈2026-02-16) and re-triaged against current code on 2026-06-04. Items already resolved by later phases were dropped; what remains is live. None of the reviews' themes are cleanly closed.

**Open** (no mechanism in code):

1. **Execution is unsandboxed.** Providers run `pytest`/`git`/`ruff` and `validate.py` runs arbitrary test commands via raw `subprocess.run` — no allowlist, sandbox profile, or network-egress policy. Only path-traversal guards exist (`activities/output.py::resolve_file_paths`).
2. **No run-level cost/latency budget.** The only budget is context-assembly token packing (`code_intel/budget.py`); there is no per-run dollar/token/wall-clock cap, kill-switch, or admission control for fan-out/exploration/retry cost.
3. **Fuzzy edits aren't semantically verified.** `activities/output.py::_fuzzy_match` applies edits at 0.6 similarity (D56) with no post-edit AST/parse check, and fuzzy matching can't be disabled per task/policy.
4. **Model routing is static.** `models.py::resolve_model` is a fixed tier→model map; no calibration, canary, or quality/cost-driven fallback.
5. **Stored prompts/outputs have no privacy controls.** `store.py::Interaction` persists prompts and results as plaintext with no retention/TTL, redaction, or encryption.

**Partial** (mechanism exists but stops short):

| Theme | Residual gap | Pointer |
| --- | --- | --- |
| Transition vocabulary | Policy-violation / budget-exhausted / partial-success / blocked-on-human collapse into retryable or terminal | `models.py::TransitionSignal` (3 members) |
| Plan contract | File-level targets only — no symbol-level write-sets, budgets, or confidence; overlap/cycle checks run in eval, not as a preflight gate | `models.py::PlanStep`, `eval/deterministic.py` |
| Conflict resolution | File-granularity merge accepted on path-completeness alone; no intent/regression verification | `activities/conflict_resolution.py` |
| Retry strategy | Errors are fed back, but one undifferentiated reset+retry; no failure-class classifier | `activities/context.py::build_error_feedback` |
| Exploration cost | Bounded only by round count + char truncation; no quota/ROI/dedup | `workflows.py::_run_exploration_loop` |
| Halt-when-confused | Only opt-in sanity-check ABORT → terminal result; no escalation queue, paging, or resume | `activities/sanity_check.py`, `models.py` (`sanity_check_interval=0`) |
| Validation depth | ruff + optional tests only; no SAST/secrets/contract/perf/behavior-diff gates | `activities/validate.py` |
| Eval as release gate | Compares plan quality (baseline vs candidate) but isn't a CI gate; no end-to-end/adversarial coverage | `eval/runner.py` |
| Domain-agnosticism | Prompts/validation parameterized per domain, but context discovery is Python/import-graph-specific; non-code domains have no positive validators | `domains.py::DomainConfig`, `code_intel/` |
| Human-in-the-loop | Only batch/OCR signals + out-of-band merge gating + manual playbook approval; no structured intervention | `manual_playbook_workflow.py` |
| Multi-provider parity | Protocol + Anthropic/Mistral adapters exist (in the `libs/sax-llm` workspace member), but defaults are Anthropic and there's no cross-provider conformance suite | `sax_llm/protocol.py`, `sax_llm/registry.py` |

**Tooling & ops debt** (surfaced 2026-07-16 during the monorepo/deployment work; T2.2 owns most of it):

| Item | Detail | Pointer |
| --- | --- | --- |
| No CI at all | Every gate is a local `uv run` by hand; nothing runs on push. The workspace now makes one root gate possible | T2.2 |
| Gates are uneven across members | forge 85%, pbook 84%, sax-llm 85%; `apps/ocr` and `libs/forge-contracts` have **no coverage gate**; mypy still covers `src/forge` only | T2.2, T2.3a–d |
| Four standing lint/type findings | 2× ruff `TC003` in `tests/test_worker.py`; 2× mypy `type-arg`/`arg-type` in `src/forge/alembic/versions/002_idempotency_rekey.py`. Pre-existing, never gated, deliberately not fixed in-flight | T2.2 |
| `batch-status` skill is dead | `.claude/skills/batch-status/batch-status.sh` still resolves a SQLite store (`FORGE_DB_PATH` → `$XDG_STATE_HOME/forge/forge.db`) — broken since the Postgres externalization. Needs a rework against `FORGE_DB_URL` or retirement (its `.agents/` Codex copy too) | `.claude/skills/batch-status/` |
| pbook skill's eval harness is dead | The `skill-pbook` repo's Makefile seeds a SQLite test DB (`sqlite3 .backup`, `PBOOK_DB_PATH`); pbook has been Postgres+pgvector-only since its cutover. Trigger evals unaffected | `~/repos-sax/skill-pbook` |
| Desktop is the availability story | Accepted by D99: batch polling (D88 timer loops) stalls while the machine sleeps. `pmset -c sleep 0` is documented but **not yet applied** | [operations/DEPLOYMENT.md](operations/DEPLOYMENT.md) |
