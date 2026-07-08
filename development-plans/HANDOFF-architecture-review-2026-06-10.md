# Handoff — Platform Architecture Review & Merged Redesign Plan

**Date:** 2026-06-10 (session ran 2026-06-09 evening → 06-10 ~02:20)
**Status:** Review complete. Merged plan written. **APPROVED 2026-06-10, including reversals R1 and R2.** The docs-only execution step is done (same day): 47 task files under `development-plans/tasks/` + the [TASKS.md](TASKS.md) index, DECISIONS D86–D97, [docs/reviews/2026-06-architecture-review.md](../docs/reviews/2026-06-architecture-review.md), and the four pbook design-doc amendments. No code has been changed; implementation proceeds task-by-task per [TASKS.md](TASKS.md), human-gated merges.

*(Original status at handoff time: merged plan written, not yet approved — plan approval was deferred in favor of this handoff. No code was changed, nothing was committed in any repo.)*

## What was asked

1. A thorough adversarial review of the platform architecture (forge, forge-contracts, ocr, sax-llm, pbook) with improvement proposals. No deployed base → backward compatibility not required. Hard precepts: python-quality guidelines (`~/.claude/python-guidelines.md`), testability over performance, Functional Core / Imperative Shell. Effort: ultracode (multi-agent workflows).
2. Mid-session, the owner pointed at a **second, independently produced plan** — the pbook design docs (`~/repos-sax/pbook/design/*.md`, written the same night) — and asked for an adversarial review of *both* plans, taking the best of each, with an explicit agree/differ analysis.
3. Final instruction: write this handoff (instead of starting execution).

## The deliverable

**The merged plan:** `~/.claude/plans/perform-a-thorough-adversarial-vectorized-barto.md`
(durable copy: `~/.claude/projects/-Users-stevengreenberg-repos-sax-forge/review-artifacts-2026-06-10/merged-plan.md`)

It contains: the agreement matrix (10 points of independent convergence), the conflict table (10 adjudications with cross-review evidence), verified findings (5 criticals, ~20 majors per side), the merged target architecture, and 8 migration phases.

**The detailed task list:** [HANDOFF-architecture-review-2026-06-10-tasks.md](HANDOFF-architecture-review-2026-06-10-tasks.md) — all 47 tasks with scope, dependencies, and acceptance criteria (including every judge/attack-flagged criterion), ready to be converted one-for-one into `development-plans/tasks/` files per PROCESS.md on approval.

## Evidence base (all durable copies in `~/.claude/projects/-Users-stevengreenberg-repos-sax-forge/review-artifacts-2026-06-10/`)

> **Correction (2026-07-08):** the "durable" copies below, and the merged
> plan at `~/.claude/plans/`, no longer exist — verified missing. The
> surviving record is
> [docs/reviews/2026-06-architecture-review.md](../docs/reviews/2026-06-architecture-review.md),
> DECISIONS D86–D97, and the task files. See
> [../forge-review-2026-07-08.md](../forge-review-2026-07-08.md) §3.2.

| File | What it is |
| --- | --- |
| `wave1-subsystem-maps.json` | 13 read-only mappers across all 5 repos: structural maps + 140 raw findings |
| `wave2-dimension-reviews-and-7-verifications.json` | 10 adversarial dimension reviews (83 findings with file:line evidence + per-dimension target proposals) + the 7 verifier panels that completed before a session-limit pause |
| `wave3-remaining-13-verifications.json` | The other 13 verifier panels (fact-check + pragmatist lenses per dimension) |
| `wave4-three-designs-and-judges.json` | 3 integrator architectures (deletion-first / contract-first / runtime-first) + 2 judge panels. Both judges picked **deletion-first**; grafts + flaws were folded into the plan |
| `wave4-winner-deletion-first-extracted.md` | The winning design extracted as readable markdown (target architecture, key decisions, phases, rejected-ideas list) |
| `wave5-cross-review-vs-pbook-plan.json` | The Plan-A-vs-Plan-B cross-review: 4 fact-check panels (API capabilities, Temporal arithmetic, codebase usage, build/infra) + 2 attack reports |
| `merged-plan.md` | Copy of the final merged plan |

Originals live under `/private/tmp/claude-501/...` (volatile — the copies above are authoritative). Workflow scripts (re-runnable/resumable) are under `~/.claude/projects/-Users-stevengreenberg-repos-sax-forge/193baec7-dc75-475e-9efd-68343f0a0ad9/workflows/scripts/`.

## Owner decisions, in order (later entries supersede earlier)

1. *(AskUserQuestion, ~22:30 06-09)* **Monorepo**: collapse the five repos into one uv-workspace.
2. *(AskUserQuestion)* **pbook stays a separate consumer app**; at the time, framed as "drives LLM calls through the forge-contracts batch SPI like ocr". **Partially superseded — see reversal R2.**
3. *(AskUserQuestion)* **Deliverable**: review doc + DECISIONS entries + task files; no code in the execution pass; human-gated implementation.
4. *(chat)* **"Forge playbooks are superseded by pbook"** — forge's playbooks table/workflows/CLI die; pbook's model is the keeper.
5. *(rejection message, ~01:00 06-10)* **The pbook design docs are a peer plan**; adversarially review both, merge the best elements.

### The two reversals the merged plan adopts (flag these at approval time)

- **R1 — The signal-based batch SPI dies.** Plan A had hardened the shared `BatchPollerWorkflow` + `BATCH_RESULT_SIGNAL` apparatus. The cross-review proved that under D79 (one request per batch) a shared poller amortizes zero provider calls, and the verified arithmetic (≈11 history events/poll; 25h wait at 600s ≈ 1,650 events, ~3% of the 51.2k cap; 30-wait worst case ≈ 24k at 300s) makes per-workflow **timer-loop polling** strictly simpler at equal latency — and both critical batch bugs become *unconstructible* (the requester is the recipient). Accepted tradeoff: a dead waiter orphans its paid batch (documented; reconciliation deferred).
- **R2 — pbook ingestion is sync, not batch.** Measured volume ≈ 110 sessions/month → batch saves ~$2–5/month while adding up-to-24h round-trips to a freshness-sensitive loop. With R1 there is no signal SPI to consume anyway. Scoped explicitly so the volume logic cannot erode forge's batch-first rationale (D76 = per-token economics of an unattended orchestrator, not realized volume).

## The ten adjudications (summary — full table with evidence in the plan)

1. Batch transport: **timer-loop polling** (B) — poller, signals, correlation dicts, delivery state machine all deleted; `batch_jobs` survives as forge-internal audit/spend ledger (restores D80's truth).
2. Library topology: **one `libs/sax-platform`** (B) absorbing sax-llm + forge-contracts (with the SPI dead, contracts had no second party left — pbook imports none of it, ocr needs ~390 lines of generic plumbing); **with Plan A's enforcement grafted**: import-linter internal layers, sandbox-light `contracts/` submodule forbidden from importing SDKs.
3. Structured outputs: **adopt everywhere** (B, extended) — verified GA, works in the Batch API (`output_config.format`), composes with caching; `messages.parse` for the sync lane; the platform lib must also own the **batch lane B forgot** (request-body builder + submit/status/results + fetch-time `model_validate`). Forced tool use retires; both string registries die.
4. pbook ingestion transport: **sync** (B) — see R2.
5. Forge knowledge consumption: **Plan A's read-only view contract** (B had a confirmed gap here — no spec for forge at all), **upgraded** with B's retrieval insight: the `knowledge.approved_entries` view exposes `search_tsv`, forge does lexical+tag fused deterministic SQL (tags boost, never gate — fixes A's recall hole). No embeddings on forge's hot path.
6. ocr: **polls its own Mistral batches** (B); gather restructured to parent-awaited children (fixes a latent 26h failed-child hang); **Mistral chat deleted** (verified: zero production users); MistralOcr keeps the OCR-batch pieces.
7. Monorepo mechanics: **git-filter-repo full history** (A) — uvx caches, total history is ~8.7MB, subtree saves nothing that matters; skill's uvx ref pinned to a tag (B's caution kept). Repo name **`sax`**.
8. Python: **3.14, standard GIL** — all seven key wheels verified available.
9. pbook destructive migration: B's plan with two attack-found fixes — **judge calibration before the backfill sweep** (sweep report-only first), and a **pre-migration JSON dump**; magic constants marked provisional, eval gates set from a measured baseline.
10. Supabase posture: B verified accurate, one fix — psycopg does **not** auto-disable prepared statements on the 6543 pooler; the engine factory must set `prepare_threshold=None`.

Coverage asymmetry: B is silent on forge internals — A's Phases 1/2/4/5/7/8 (bleeding-stoppers, monorepo, composition roots, workflow consolidation, context engine, docs) survive nearly verbatim; B's pbook product core (probation lifecycle, extract+judge, hybrid RRF retrieval, feedback events, eval suites, workflow-as-RPC deletion) grafts wholesale.

## Critical findings (all verified; how the merged plan addresses each)

| Finding | Disposition |
| --- | --- |
| Batch results consumed without `request_id` correlation (forge + ocr) | Interim dict+setdefault stopgap (T1.2), then unconstructible under timer-loop (Phase 4) |
| Poller marks deliverable results FAILED / strands MISSING waiters for 25h | Interim minimal patch (T1.3), then subsystem deleted (Phase 4) |
| Model-influenced subprocesses inherit worker secrets | Env-allowlist scrub (T1.7) |
| grimp analyzes the installed package, not the task worktree | Subprocess-grimp with worktree PYTHONPATH (T7.2) |
| Knowledge loop disconnect (pbook never feeds task execution) | View contract + forge retrieval switch (T6.7) |

## What the next session should do

1. **Get the merged plan approved** (it was never approved — re-present `~/.claude/plans/perform-a-thorough-adversarial-vectorized-barto.md`, drawing attention to reversals R1/R2).
2. Execute the plan's docs-only step:
   - `docs/reviews/2026-06-architecture-review.md` in forge — mine the wave JSONs (each finding already carries file:line evidence and verifier verdicts).
   - `docs/DECISIONS.md` D86+ entries per the sweep checklist in the plan (D3, D9, D10/D58, D13, D31–33, D43–47, D75, D76–D82, monorepo, composition roots, knowledge contract, timer-loop supersession of D77/D78).
   - `development-plans/tasks/` — one file per T-task (Problem / Acceptance Criteria per PROCESS.md; include the judge/attack-flagged criteria verbatim, e.g. T1.1's `test_llm_client.py:363` forge-import split, T6.3's calibration-before-sweep ordering); update `TASKS.md` in dependency order.
   - Amend the four pbook design docs where the merge changed decisions (TEMPORAL_PATTERNS rule 8 scoping: interval ≥300s + history arithmetic + continue-as-new escape; INTEGRATION.md forge-consumption section; DECISIONS.md batch-threshold cost arithmetic + psycopg pooler wording).
3. Then implementation proceeds task-by-task (Phase 1 first — every task is independently green and mergeable), human-gated merges per the git strategy.

## Gotchas for the executing session

- The `/private/tmp` wave outputs may be gone; use the durable copies in `review-artifacts-2026-06-10/`.
- Phase ordering is load-bearing: 5 and 6 are **serialized** (both touch forge worker registration and OUTPUT_TYPES); Phase 4 needs Phase 3's platform batch helpers; T1.2/T1.3 are explicitly interim (deleted by Phase 4 — say so in their task files so nobody "finishes" them).
- forge's `models.py` Sonnet pin (`claude-sonnet-4-5-20250929`) and pbook's worker default (`claude-3-5-sonnet-20241022`, retired) are both stale; target is `claude-sonnet-4-6` with adaptive thinking (`budget_tokens` is deprecated on 4.6 — forge's ThinkingConfig/D62 needs the migration noted in T3.2).
- sax-llm's coverage gate is 25% only because its tests live in forge (4 files, 2,515 LOC) — move them before raising the gate.
- ocr declares `sax-llm` and `mistralai` deps it never imports (all Mistral traffic currently runs on forge's worker).
- pbook is a **required** forge dependency wrapped in dead "optional" ImportError guards; forge's CLI reads pbook's Postgres directly (`cli.py:~1217`) — all of this dies in Phase 6.
- Existing forge playbooks rows: one-time JSON dump → manual triage via `pbook add` — **no blanket-approve migration** (the table is polluted by the re-extraction loop's duplicates).

## Cost/usage note

Five workflow runs, ~67 subagents total, ≈6.2M subagent tokens. One session-limit pause occurred mid-wave-2 (13 verifiers failed and were re-run after the 10:40pm reset via a follow-up workflow) — all verifications eventually completed; nothing is missing.
