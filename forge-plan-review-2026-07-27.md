# Forge plan review — 2026-07-27

**What this is.** A premise-level structural review of the 19 remaining
migration tasks (T5.5–T8.4) plus a reverse pass over the completed work
(T0.x–T5.4) hunting for recorded caveats, deferrals, and rearchitecture
candidates with no owner in the remaining queue. Commissioned by the
owner 2026-07-27 after T5.4 closed Phase 5's code work; executed as five
parallel read-only reviewers (three forward — Phase 6, Phase 7,
T5.5/T5.6/Phase 8; two reverse — Phase 5 artifacts, Phase 0–4 artifacts
plus an in-code marker sweep) with synthesis in the planning session.
Repo state: `7dd43f7`; production pinned at `7002435`.

**What this is not.** Not a line-level verification of every task file
(that stays with per-task pickup reconciliation, which is five-for-five
since T5.1), and not the Phase 8 close-out audit (T8.1–T8.4 remain the
scheduled full pass; several of their own defects are findings below).

**Method note.** Premise questions only: does the task's reason-to-exist
survive the landed work; do its dependencies and ordering hold; has it
grown or shrunk enough to change sizing; and is anything homeless.
Reverse-pass findings were verified in code where cheap, and candidate
homes were checked against every remaining task file plus TASKS.md's
parked and tech-debt lists before anything was declared homeless. Two
seeded premises **failed** verification and are recorded in §6 so they
don't get re-invented at grooming.

---

## §1 Verdict table

| Task | Verdict | One line |
| --- | --- | --- |
| T5.5 harness rebuild | **PARTIAL / GROWN** | Harness half grew ~1.5× (6,180 lines; the real blocker is 29 module-level mutable containers, not the counted `global`s); replay-scaffold half ~40% consumed; xdist AC currently unfalsifiable (dep absent) |
| T5.6 plan preflight | **INTACT** | Cheaper post-T5.3 (one `dispatch_planner` seam); REVISE-splice hazard changed shape (workflow-internal `ValidationError`, hung workflow); should precede T5.5 |
| T6.1 pbook library-first | **TRANSFORMED** | Its entire fix direction is forbidden by Architecture Principle 8 (2026-07-23, postdates spec); needs owner adjudication before execution |
| T6.2 judge calibration | **INTACT** | New trap: calibrating against pbook's tier map that T6.4 then swaps invalidates the calibration |
| T6.3 destructive migration | **INTACT** | Pooler AC obsolete (D102); pbook is a separate *database* (view can't span DBs); platform embedding-dim constant now mirrors the column |
| T6.4 ingest/curation | **INTACT / GROWN** | Now a live-system cutover (pbook worker serves prod traffic); forge-side direct pbook-DB read is a live Principle 8 violation it kills; tier-map symptom sentence stale |
| T6.5 hybrid retrieval | **INTACT** | Degraded mode ~half pre-built; its CLI additions must follow T6.1's adjudication |
| T6.6 eval suites A & C | **INTACT** | Must reconcile against forge's existing T0.6-hardened `src/forge/eval/` harness (spec doesn't know it exists) |
| T6.7 consumption + playbooks deletion | **PARTIAL** | OUTPUT_TYPES bullet consumed (T3.5); playbooks table empty in prod (dump step a formality); forge needs a **second engine** for the knowledge DB; must precede T7.1/T7.3/T7.4/T7.5 |
| T7.1 ProjectDescriptor | **GROWN** | `params_model` doesn't exist — deriving PROVIDER_SPECS means inventing eleven params models + a handler-shape migration the spec treats as bookkeeping |
| T7.2 worktree-accurate graph | **INTACT (worsened)** | T5.2 made drift per-attempt (worktree file reads + worker-package graph edges in one context); subprocess fix needs a T1.7 env-allowlist decision; ContextStats persistence offers a free stronger AC |
| T7.3 honest token accounting | **INTACT** | Not consumed by T5.3 (opposite direction of time) — but calibration is now measurable from the interactions store; `output_reserve` is a dead config field; two divergent `estimate_tokens` to not confuse |
| T7.4 exploration budget | **PARTIAL** | Enforcement untouched; T5.3's per-round actuals allow an actual-token cap (replay-deterministic); hidden dependency on T5.5 (round caps change command sequences) |
| T7.5 one prompt builder | **INTACT / GROWN** | Fabrication sites 4→5, fence pairs 3→11; T5.3/T5.4 moved *who persists* prompts, not *who builds* them |
| T7.6 fuzzy-edit governance | **INTACT (shrunk)** | Surfacing half nearly built (`apply_edits_detailed` exists, discarded); byte-round-trip AC gained a retry dimension via T0.5's staging |
| T8.1 review doc + DECISIONS | **CONSUMED ~85%** | Review doc built; 19/21 banners exist; survivors: WORKERS.md ops section (the RESET-loses-paid-batches trap), D9/D32 banners, verification read; checklist omits D95–D97 |
| T8.2 test-tier honesty + OVERVIEW | **PARTIAL** | Rename/marker half intact; OVERVIEW-predates-migration premise consumed; "items 1/2/3/5 closed" needs re-adjudication — none is closed by any remaining task |
| T8.3 pbook design-docs pass | **CONSUMED** | All four named content fixes already applied; survives as a verification read + one out-of-workspace file (skill-pbook) |
| T8.4 final sweep | **PARTIAL** | Two acceptance criteria are factually wrong (`just check` — the project chose Make at T2.2; tag `sax` v1.0 — the monorepo is `forge`); inventory part-consumed, part-moved |

---

## §2 Ordering changes (adjudications A1–A3)

1. **T6.7 before T7.1/T7.3/T7.4/T7.5** *(both forward reviewers converged
   independently)*. TASKS.md's "Phases 6 and 7 may run in parallel
   (disjoint files)" is false for exactly one task: T6.7 deletes the
   playbooks channel out of `providers.py` (which T7.1 rewrites and T7.4
   caps), `activities/context.py` (which T7.5 rewrites), and one of the
   two budget sites T7.3 rewrites (`context.py:626`). Landing T6.7's
   deletion first *shrinks* the surface those four tasks touch.
   T6.1–T6.6 are pbook/eval-local and genuinely parallel-safe.
   **Proposed amendment:** TASKS.md ordering note → "disjoint except
   T6.7, which precedes T7.1/T7.3/T7.4/T7.5."
2. **T5.6 before T5.5.** T5.6 adds a preflight retry arm and a REVISE
   cap — new behavioral scenarios the rebuilt harness must cover.
   Rebuild first and those scenarios get written twice. T5.6 also got
   cheaper (one `dispatch_planner` seam post-T5.3).
3. **Tier-map swap before (or into) T6.2.** T6.2 would calibrate the
   pbook judge at ≥85% agreement against pbook's own tier map
   (`haiku-4-5-20251001`); T6.4 then swaps in the platform registry
   (`haiku-4-5`), invalidating the calibration. Either T6.2 builds its
   harness against `sax_platform.llm.tiers` directly (one import), or
   the tier-map deletion splits out of T6.4 and lands first.

Also noted: T7.4 (and T7.2 if its graph fix becomes a new activity)
change activity command sequences, so they sit behind T5.5's replay
scaffold — no Phase 7 task mentions replay at all. Phase ordering
5 → 6 → (6∥7 as amended) → 8 otherwise stands; v1.0 still tags at T8.4.

---

## §3 Owner adjudications required

- **A1–A3** — the three ordering changes above (recommend: adopt all).
- **A4 — T6.1's direction.** The spec's fix (pbook CLI → sync library
  functions over a direct engine) is the shape Principle 8 forbids, and
  pbook's own code now encodes the opposite invariant
  (`apps/pbook/src/pbook/models.py:163-170`). Options: (a) grant
  Principle 8 a read-only-local-reads exception for pbook's CLI, or (b)
  re-scope T6.1 to "collapse the 16 one-activity wrapper workflows into
  a few multi-op workflows + bounded first-step waits," which recovers
  most of the latency win, folds in the parked bounded-wait item
  (TASKS.md:164-170), and keeps the principle intact.
  **Recommendation: (b).** T6.5's new CLI commands follow whichever way
  this goes.
- **A5 — `batch_jobs` spend columns (D100 deferral): decline or park.**
  Carried by three handoffs and two decision rationales; owned by
  nothing. Since T5.3, every dispatch arm persists an interactions row
  with token counts, so the columns would be a denormalization of an
  authoritative surface that already exists.
  **Recommendation: decline** — record the disposition as a dated note
  under D100 (per D97's spend-visibility rationale) so the pointer
  stops dangling.
- **A6 — T8.2's "OVERVIEW Open items 1/2/3/5 closed" instruction.**
  None of those items (unsandboxed execution, run-level budget,
  fuzzy-edit verification, privacy controls) is closed by any remaining
  task; all four live on the post-1.0 parked/tech-debt lists.
  **Recommendation:** amend T8.2 to record them as
  *parked-post-1.0-with-dispositions*, not closed.
- **A7 — backup verification (the review's only blocking-soon
  operational item).** The nightly `pg_dump` → S3 is the **sole**
  durable copy of the store of record (Supabase frozen), verified
  exactly once, manually, at T0.9 close; nothing consumes the backup
  script's exit status, and no restore rehearsal is documented.
  **Recommendation:** a small standalone operational task (verify-latest
  cadence + one documented restore rehearsal + a failure signal),
  rather than a parked entry — parked items don't get scheduled.
- **A8 — WORKERS.md ops section (T8.1's surviving deliverable).** It is
  decoupled from Phases 6–7, and its centerpiece — "a workflow RESET
  cannot recover already-paid batch results" — is a live operational
  trap for as long as it's unwritten. **Recommendation:** pull it
  forward (a one-session docs task any time), leaving T8.1 as banners +
  verification read.

---

## §4 Per-task amendment sketches (forward pass)

Applied only after adjudication; each lands as a dated amendment section
per PROCESS.md.

- **T5.5** — re-baseline the numbers (≈6,180 lines; 19 `global`
  statements over 9 names; 12 `_reset_*`; ~1,633 stub/helper LOC — ×7
  average duplication, ×16 worst); reframe the target as **the 29
  module-level mutable containers** (the actual xdist blocker and the
  flake's mechanism), not the `global` count; strike the consumed
  scaffold bullet ("add the platform replay-test scaffold" — it exists);
  add a replay-coverage AC naming the four promised missing histories
  (planned-step/`assemble_step_context`, nested gather, exploration,
  sanity) with conflict-resolution, sync-planner, and
  `reset_worktree`-retry as stretch; add `pytest-xdist` (and the
  `test-strategy.md` claims that depend on it) to scope; note the
  hypothesis item was consumed by T0.5 and the crash-recovery scenario
  survives; fold in the polling.py docstring truth fix (§5-9).
- **T5.6** — refresh stale refs (models.py:299-307; splice now at
  `workflows/task.py:435-443`); replace the "unbounded history" failure
  mode with the real one (over-cap splice → workflow-internal pydantic
  `ValidationError` → infinitely retrying workflow task); note
  `max_length=25` pre-consumed the step-count ceiling and the single
  `ARMS` seam replaced the two-call-site problem; the cap fix must
  *catch*, not just cap.
- **T6.2** — calibration harness resolves models via
  `sax_platform.llm.tiers` (kills the A3 trap regardless of ordering).
- **T6.3** — drop the pooler-connection AC (D102 retired it); state that
  the `knowledge` schema lands inside the **pbook database** and a view
  cannot span databases (T6.7 consumes via a second engine); add the
  `DEFAULT_EMBEDDING_DIM` mirror (platform `embeddings.py:56`) to the
  halfvec-conversion checklist; add disposition of the now-vestigial
  `0002_enable_rls.py` and the Supabase-only `pooler` setting/vocabulary;
  note the migration rehearsal can copy D102's verified dump/restore
  procedure (59 entries) and that test-vs-prod is now two env vars on
  the same podman instance, guarded by `FORGE_ENV` + D103.
- **T6.4** — rewrite the tier-map symptom (the live defect is
  a-generation-behind, not a retired model); add a prod cutover step
  (the cross-queue path serves real traffic since 2026-07-24); note the
  forge-side direct pbook-DB read it deletes is a live Principle 8
  violation (now `cli.py:1334-1346`); tag-enforcement gem confirmed live
  (`store.py:419-442` never calls `validate_tags`).
- **T6.6** — add one plan bullet reconciling against forge's
  `src/forge/eval/` (harvest or diverge deliberately); note the
  `FORGE_ENV=test` prerequisite.
- **T6.7** — restate consumption as a **second engine against
  `knowledge_db_url`** (one new composition-root dependency; the
  contracts read-only mirror precedent is `batch_jobs.py`); rewrite the
  consumed OUTPUT_TYPES bullet (what remains: the conditional
  `TranscriptAnalysisResult` import dies with T6.4, `ExtractionResult`
  with this task); mark dump-and-triage a formality (prod table empty,
  D102 counts); **delete `.claude/skills/forge-playbooks/` with the CLI
  surface it wraps**; add the §2-1 ordering note.
- **T7.1** — resize: PROVIDER_SPECS derivation requires inventing eleven
  params models + migrating handler signatures (no `params_model`
  exists); fold in the `ProviderHandler` protocol dishonesty (the 4th
  `engine` arg side-dict, which survives T6.7 for `past_runs`);
  `ContextConfig.package_name` partially absorbed already
  (`context.py:617`), `src_root` still literal.
- **T7.2** — premise worsened by design since T5.2 (per-attempt
  exploration mixes worktree reads with worker-package graph edges); the
  subprocess fix needs an explicit `allowlist_env` addition (T1.7
  artifact) + a recorded decision that the graph subprocess is not
  model-influenced; add `degraded`/`degradation_reason` to
  `ContextStats` (now persisted per interaction — queryable, free
  stronger AC); fold in the single-step discovery rooting inconsistency
  (`context.py:664` passes repo_root where step/sub-task pass the
  worktree); note replay obligations if the fix becomes a new activity.
- **T7.3** — unchanged in direction; calibration is now measurable from
  the interactions store (prompts + real `input_tokens` for all five
  arms); make `output_reserve` live or delete it (currently a config
  field nobody reads); add an explicit guard: two `estimate_tokens`
  exist with opposite safety biases (`code_intel/repo_map.py:36` packing
  vs platform `llm/cache.py:67-75` cache-eligibility) — do not "fix" the
  platform one.
- **T7.4** — re-shape the cap: per-round **actual** token counts now
  arrive via `ExplorationCallResult` (replay-deterministic), so the
  "estimated-token cap" can be an actual-token cap; mode-awareness reads
  `host.sync_mode` (one line); note `derive_execution_timeout` couples
  round counts to workflow-start options and replay histories — T7.4
  sits behind T5.5; two render-time truncators to unify
  (`activities/exploration.py:108-112`, `blocks/exploration.py:42-43`).
- **T7.5** — resize: five fabrication sites (+ `planner.py:315-321`),
  eleven fence pairs (spec counts three); scope statement stands
  (playbook channel dies with T6.7 — ordering per §2-1); record the
  adjacent-but-out-of-scope fact that nine prompt builders across five
  arms share zero infrastructure (four survive T6.7; disposition:
  post-1.0 unless T7.5 comes in cheap).
- **T7.6** — surfacing half shrunk (stop discarding
  `apply_edits_detailed`'s results; carry to WriteResult/run records);
  three 0.6 literals, not one; byte-identical round-trip AC gains a
  retry dimension (CRLF normalization must live inside T0.5's staging
  phase); `tests/test_output_properties.py` already exists — extend,
  don't duplicate.
- **T8.1** — shrink scope to: WORKERS.md ops section (or confirm pulled
  forward per A8), D9 + D32 banners, **add D95–D97 to the checklist**
  (D95 describes shapes that did not ship: `run_fan_out_gather(commit)`,
  a nine-value Literal — as-shipped is `(spec, host)`, a two-row policy
  table, a `GatherSuccess | GatherFailure` union, ten kinds), and the
  verification read of the 479-line review doc.
- **T8.2** — re-adjudicate items 1/2/3/5 per A6; add the stale
  `batch-status` OVERVIEW row (§5-8); the rename half is intact and
  pairs naturally with T5.5 (both touch the suite; the spec's stale
  `uv run pytest -m e2e` and marker-scheme bullets are already
  consumed).
- **T8.4** — **pre-emptive AC corrections** (cannot wait for T8.4
  itself): `just check` → `make gates` (T2.2 chose Make); tag `sax v1.0`
  → `forge v1.0` (D98). Inventory updates: `DEFAULT_TEMPORAL_ADDRESS`
  duplication moved (now `src/forge/cli.py:534` + ocr `cli.py:30`);
  providers subprocess item mostly consumed (truncate idiom remains);
  `int(time.time())` moved to `cli.py:1213`; add `docs/ARCHITECTURE.md`
  - `docs/PHASES.md` staleness, `sax_platform/config.py:1-10`'s stale
  T3.6 note, and `docs/reference/mistral.md:89` ("no signals" — one
  sanctioned signal exists since T4.4) to the docs sweep; resolve the
  `cli.feature` ownership fork in favor of T8.4; note the
  string-patched CLI-test fragility is real but latent (66 patches, 8
  targets, zero currently dangling).

---

## §5 Reverse-pass findings and dispositions

Grouped by proposed home. Severity in brackets; sources in the
reviewers' reports (verified file:line for every item).

**Immediate — fold into the pending T5.4→T5.5 handoff sweep** (docs-only
truth fixes; all actively misleading today):

1. [should-capture] **Prod version corrections**: production runs
   `7002435` (T5.3 deployed), not `001247c` as CLAUDE.md/OVERVIEW still
   say; only T5.4 (`7dd43f7`) is undeployed.
2. [should-capture] **`docs/ARCHITECTURE.md:331`** affirmatively claims
   transition evaluation lives in `activities/transition.py` (deleted at
   T5.1); tree also lists `_heartbeat.py` (deleted). Fix the false
   sentence + tree lines now; the fuller pass stays with T8.4.
3. [nice-to-have] **Sanity-cadence sentence**: ARCHITECTURE.md says
   sanity checks run "periodically between steps" — T5.3 verified the
   driver deliberately skips them after fan-out steps and after the last
   step.
4. [should-capture] **CLAUDE.md Principle 8 corollary tense**: the
   bounded-first-step-wait corollary is written as shipped behavior;
   zero implementations exist in any CLI (pbook's `_execute_workflow`
   passes no timeout at all and its docstring claims the opposite).
   Mark it parked/aspirational like its sibling clause.
5. [nice-to-have] **CLAUDE.md conventions wording**: "inline copies in
   `workflows.py`" → the `forge/workflows/` package; consider promoting
   the twice-applied "no compat shims — update the importers" rule from
   task-file precedent into the conventions.
6. [should-capture] **`sax_platform/temporal/polling.py:1-10`** still
   claims two consumers ("forge's Anthropic waiters, ocr's Mistral
   waiters"); ocr stopped using the loop at T4.4 (imports only the
   ceiling constant).
7. [nice-to-have] **TASKS.md hygiene**: close parked item 2
   ("deploy from a committed ref") as resolved-by-D103-for-prod with the
   dev-lane residue restated; repoint the nearly-vacuous parked "pbook
   direct-DB CLI review" at the real live violation (forge's
   `cli.py:1334-1346`, which T6.4 deletes) and the batch-status skill
   (below).

**New parked entries** (TASKS.md):

1. [should-capture] **`batch-status` skill**: OVERVIEW's debt row is
   false on both clauses (skill was reworked to Postgres 2026-07-20; the
   `.agents/` copy doesn't exist) — but the reworked skill reads
   `FORGE_DB_URL` with **no `FORGE_ENV` guard** (the one
   application-data reader bypassing D102) and its SKILL.md still
   asserts Supabase-in-production. Park the Principle-8/guard
   adjudication; the OVERVIEW row itself → T8.2.
2. [should-capture] **Worktree janitor sweep** (CancelledError-path
   leak, deliberately excluded at T5.2): price honestly — since the
   idempotent git seams, a leftover is disk debris, not a bricked task
   id. Plus one docstring sentence in `blocks/step.py` beside the
   batch-wait paragraph.
3. [should-capture] **Orphaned-batch reconciliation sweep** (deferred
    by D88/T4.1; the T4.2 cache-refresher variant that would have
    provided it for free was superseded by T4.4's broadcast tracker,
    which never diffs forge's ledger against Anthropic). D88's own
    severity framing: a slightly higher invoice, not data corruption.
4. [should-capture] **Worker containerization**: deferred behind a
    trigger that no longer exists (the deleted sax-llm absorption
    increment); D103's rationale calls it "the owner's next step."
    Re-park with an honest trigger.
5. [should-capture] **Tracker-heartbeat monitoring**: D101's
    no-fallback risk acceptance rests on "observable via the heartbeat,"
    and nothing observes it (no cron, no launchd probe, no consumer of
    `ocr tracker-status` exit codes). Escalates whenever OCR volume
    resumes.
6. [nice-to-have] **Prompt-cache breakpoint layout**: named unowned by
    the 07-08 review; T5.3 made `cache_read_input_tokens` populated for
    all five arms, so the "should be ~0" check is now runnable — still
    unowned.
7. [nice-to-have] **S3 key-namespace/TTL follow-up**
    (`contracts/s3_blobs.py:17-20` promises "a tracked follow-up" that
    doesn't exist; interacts with `deploy/s3/lifecycle.json`). The TLS
    sibling comment (`temporal/client.py:24-25`) is moot post-D99 —
    soften the wording, no entry.
8. [nice-to-have] **`sax_platform.temporal.polling` fold decision**:
    the loop has one consumer (forge transport) since T4.4 — fold it
    into `blocks/transport.py` leaving only `BATCH_WAIT_CEILING`
    platform-side, or keep the seam and say why.

**Already owned or already fixed — verified, no action** (recorded so
grooming doesn't re-litigate): claim-check corollary (parked item 6,
honestly labelled); CLI env-glue triplication (parked item 3); the
~1/8 flake and test-globals (T5.5, consistently); eval significance
backlog row; `batch_processing.feature`/`cli.feature` (T8.4 inventory,
one ownership fork resolved per §4); OVERVIEW Open/Partial rows all map
to tech-debt lists (the *closure claim* is A6's problem, not ownership);
install.sh bootout-vs-drain and dev-lane relaunch hazards (documented in
DEPLOYMENT/launchd docs); prod `-dirty` refusal (enforced in code);
T5.1 `/simplify` skips (closed with reasons); `git.py` inline porcelain
parse (two-line cleanup for whoever next touches the file); ocr
`store.py:46` "deferred to the squash" (shipped; historical comment).

---

## §6 Records corrections (including two failed seeds)

- **Two review-seed premises were verified false** — recorded so they
  are not re-invented: (a) `blocks/transport.py` carries **no**
  "composition into purpose-built workflows (OCR, research)" docstring —
  that framing died with `workflow_blocks.py`; the blocks `__init__`
  describes today's two drivers, proposes nothing. (b) The
  `polling.py → forge.workflow_blocks` dangling pointer recorded as an
  open item in T5.4's Dev Notes was already fixed in the T5.4 commit
  (`7dd43f7`); the *remaining* falsehood there is the two-consumers
  claim (§5-6).
- **Measurement reconciliation** for T5.5's premise: 6,179–6,180 lines
  (measurement-moment delta); **19** `global` statements (the "21" in
  T5.4's notes counted two comment lines; the spec's "30" predates
  deletions); **9** distinct global names; **12** `_reset_*` functions
  (spec says 11). The load-bearing number is none of these — it's the
  **29 mutable containers**.
- **Marker sweep**: zero conventional TODO/FIXME markers exist anywhere
  in src/libs/apps/deploy — the deferral surface lives entirely in task
  files, decisions, handoffs, and docstrings, which is what made this
  reverse pass necessary and is worth keeping true.

## §7 Proposed execution order

1. **Owner adjudicates A1–A8** (§2–§3).
2. **Handoff sweep** (already owed for T5.4→T5.5): fold in §5 items 1–7.
3. **One amendment change-set** applying §4's dated amendments + §5's
   new parked entries (8–15) per PROCESS.md's Specification Changes
   policy, owner-adjudicated wording where flagged.
4. **T5.6 → T5.5** proceed per §2, each with its normal pickup
   reconciliation (this review does not replace it).
5. A7's backup-verification task and A8's WORKERS.md ops section run
   whenever convenient — both are decoupled from the phase sequence.
