# Grill Session: forge-document-refresh

Started: 2026-06-04
Last updated: 2026-06-04
Status: complete
Domain: Technical-documentation strategy for an LLM-developer audience. The interviewer probes scope, accuracy-vs-code, structure, link durability, and sequencing — it does NOT rewrite any docs.

## Summary

A first-principles rewrite of Forge's documentation for an **LLM-developer audience** (optimize for "don't make it search"). Every fact must re-earn its place against the usefulness test (business-domain knowledge · write-it-right-first-time · contrary-to-LLM-knowledge · debugging aid · test context); code is the source of truth and wins every disagreement.

**Shape (Layered canon):** four status/planning docs — `docs/OVERVIEW.md` (status, complete/remaining requirements, known issues/tech debt), `development-plans/TASKS.md` (atomic completed-vs-uncompleted source; fixes PROCESS.md's broken link), `docs/PHASES.md` (14-phase roadmap: brief for done, summary+links for remaining), and the frozen `docs/requirements/` Gherkin set — sit on a kept-and-verified **technical** layer (`docs/ARCHITECTURE.md` canonical "how"; `docs/DECISIONS.md` reconciled), **operational** layer (`docs/operations/`: workers, deploy, debug, secure-remote, adding-a-domain, test-strategy, usage), and **reference** layer (`docs/reference/`: mistral). CLAUDE.md's Project Status shrinks to a one-line pointer to OVERVIEW.

**Disposition of old material:** DESIGN.md, the three research docs, the to-merge tree, and PHASE1–12+14 are mined for surviving value, then moved to a **top-level `archive/`** with a "not authoritative" banner. The four code reviews are triaged against current code and their still-valid critiques become OVERVIEW's tech-debt section. PHASE13.md and LSP_INTEGRATION_PLAN.md are kept as linked remaining-work specs.

**References:** point-don't-copy to Forge's own code (file+symbol, never bare line numbers); SHA-pinned GitHub permalinks (not local clone paths) for surgical Temporal links at non-obvious points only.

**Execution:** content first (verify completion, reconcile DECISIONS, mine, write canon), structure second (`git mv` moves, archive+banners, repoint ~9 inbound links, regenerate TOC, CLAUDE pointer). Ships as two human-gated PRs (content, then reorg) against a checklist-based definition of done.

## Decision Log

### DECIDED: Scope = first-principles rewrite, archive don't delete

- **Decision**: Treat all existing docs as potentially obsolete; discard content that doesn't earn its place; move superseded files to an `archive/` folder rather than deleting.
- **Rationale**: Docs assumed out of date; prior existence is not a reason to keep information.
- **Date**: 2026-06-04

### DECIDED: Audience = LLM code-developer; principle "don't make it search"

- **Decision**: Write for an LLM that will develop code. If the location of a fact is known, point to it directly rather than making the reader search for it.
- **Date**: 2026-06-04

### DECIDED: Usefulness test for retaining/adding information

- **Decision**: Information earns a place only if it (a) provides business-domain knowledge guiding the engineer to the right thing, (b) helps write the code correctly the first time, (c) is contrary to the LLM's inherent knowledge (non-obvious answers, fixes for recurring problems), (d) assists debugging, or (e) provides context for a test.
- **Date**: 2026-06-04

### DECIDED: Target documentation structure

- **Decision**: Produce — master project overview (complete requirements, remaining requirements, known issues/tech debt); detailed task list (completed vs uncompleted); individual product requirements (completed + remaining); development phases (brief summary of completed, detailed summary of remaining).
- **Date**: 2026-06-04

### DECIDED: Requirements frozen this pass

- **Decision**: Catalog product requirements as complete/remaining but do NOT rewrite their content now; defer requirement updates to a later pass.
- **Date**: 2026-06-04

### DECIDED: repos-temporal clones are authoritative

- **Decision**: The local Temporal repos are authoritative on technical matters (platform author). ai-cookbook examples are synchronous and do NOT model Forge's batch poller.
- **Date**: 2026-06-04

### DECIDED: Code reviews — triage-and-mine before archiving

- **Decision**: Triage each code-review critique against current code; promote surviving issues into the master overview's "Known issues / tech debt" section (and/or stub task files); archive the raw reviews only after mining.
- **Rationale**: The codex code reviews are the only existing tech-debt analysis and are load-bearing for the development-plans task workflow, but date to the 2026-02-16 snapshot, so some critiques are already resolved.
- **Date**: 2026-06-04

### DECIDED: Staleness method — verify against code before discarding

- **Decision**: Trust no doc by default; verify each against the code before discarding or rewriting. Concentrate verification on the older cohorts (to-merge=Feb; the 2026-04-08 bulk-commit planning/PHASE/research docs). Code wins every disagreement.
- **Rationale**: Mechanical "discard if not obviously useful" + confident LLM prose manufactures confidently-wrong docs.
- **Date**: 2026-06-04

### DECIDED: Canon shape = Layered canon

- **Decision**: The 4 status/planning docs (OVERVIEW, TASKS, PHASES, requirements) sit on top of a kept-and-verified technical layer (ARCHITECTURE, DECISIONS), operational layer (workers/deploy/debug/secure-remote/adding-a-domain/test-strategy/usage), and reference layer (mistral). Research, original PHASE1–14 specs, DESIGN.md, and to-merge are archived. Target tree:
  ```
  docs/
  ├── OVERVIEW.md     status: done/remaining reqs, known issues, tech debt
  ├── TASKS.md        completed vs uncompleted
  ├── PHASES.md       brief(done) + detailed(remaining)
  ├── requirements/   18 .feature (frozen) + corrected index
  ├── ARCHITECTURE.md keep+verify (canonical "how"; Module Map, Data Models)
  ├── DECISIONS.md    keep+reconcile (D1–D85, mark superseded)
  ├── operations/     workers, deploy, debug, secure-remote, adding-a-domain, test-strategy, usage
  ├── reference/      mistral
  └── archive/        DESIGN, research, to-merge, PHASE1–14
  ```
- **Rationale**: ARCHITECTURE (Module Map, Data Models) and DECISIONS (85 non-obvious calls) are the highest-value "don't make it search" docs for an LLM-dev audience.
- **Date**: 2026-06-04

### DECIDED: Status architecture & single source of completion truth

- **Decision**:
  - **TASKS.md is the atomic source of "done," located at `development-plans/TASKS.md`** (fixes PROCESS.md's broken link). The existing development-plans/ workflow (PROCESS.md, task files, CHANGELOG.md) is kept, not forked. *(Amends the canon tree: TASKS.md is in development-plans/, not docs/.)*
  - **PHASES.md** (docs/) = coarse historical roadmap rolling up TASKS — brief for done, detailed for remaining.
  - **OVERVIEW.md** (docs/) = narrative status; points to TASKS/PHASES for counts; unique payload is requirements complete/remaining + known issues/tech debt.
  - **CLAUDE.md's Project Status paragraph shrinks to a one-line pointer** to docs/OVERVIEW.md (auto-loaded → no search).
  - PHASES is the 14-phase roadmap (historical); **TASKS + OVERVIEW also capture non-phase completed work** (store externalization, OCR pipeline, ingest→pbook, planner eval, mTLS/EC2).
  - **Initial completion state is established by verifying against code and tests, NOT by transcribing CLAUDE.md.**
- **Rationale**: Four docs encoding completion would drift; one atomic source + derived/pointer views prevents confidently-wrong status.
- **Date**: 2026-06-04

### DECIDED: Linking & reference strategy

- **Decision**:
  - **Code references (Forge's own): point, don't copy.** Inline only non-derivable facts (rationale, magic-number reasons, gotchas); for code-derivable facts point to **file + symbol** (e.g., `store.py::record_run`), never bare line numbers.
  - **External Temporal links: SHA-pinned GitHub permalinks, not local clone paths** (the reading LLM is sometimes off this machine — cloud agents/CI/other checkouts). The `/Users/.../repos-temporal` clones are a research tool, not link targets. Pin to the tag/commit matching Forge's resolved `temporalio` version (uv.lock; floor `>=1.9.0`, clone is 1.27.2).
  - **Temporal links are surgical**, gated by the usefulness test — only at non-obvious points Forge depends on (signal-based wait D77, search attributes for batch routing D78, time-skipping test env, child-workflow fan-out, workflow-sandbox module-state restriction, activity heartbeating). No links for vanilla Temporal.
  - Temporal coupling is deep (36 import sites across 30 files), so surgical links pay off.
- **Rationale**: A link that rots or dead-ends is a liability; optimize for durability + portability + surgical relevance.
- **Date**: 2026-06-04

### DECIDED: Phases & requirements content

- **Decision**:
  - **Extract-before-archive** for PHASE docs. Done phases (1–12, 14) → one-line summaries in PHASES.md, then raw files archived. **Phase 13 + LSP are remaining work and are KEPT** (not archived): PHASES.md carries a tight remaining-summary and links to kept `PHASE13.md` and `LSP_INTEGRATION_PLAN.md`. *(Amends canon tree: archive holds PHASE1–12 and 14; PHASE13.md + LSP_INTEGRATION_PLAN.md stay as linked remaining specs.)*
  - **PHASES.md = the 14-phase implementation roadmap only.** Task-internal phasing (externalize-store "Phase A/B/C") stays in its task file/CHANGELOG.
  - **Requirements**: `.feature` scenarios frozen. Fix the stale `requirements/README.md` index (15→18 files; correct capability→source map) — metadata, not requirements. Derive requirement completion from existing `@phase-N` tags (no scenario edits), surfaced in OVERVIEW + the index.
- **Date**: 2026-06-04

### DECIDED: Archive mechanics & inbound-link integrity

- **Decision**:
  - **Archive location: top-level `archive/`** (sibling of docs/, src/, development-plans/). *(Amends canon tree: archive/ is top-level, not docs/archive/.)*
  - **Inbound-link blast radius is small**: CLAUDE.md (4), deploy/README.md (2), development-plans/externalize-store (3); **zero source-code refs; diataxis clean**. Moved docs → repoint links to new path; archived docs → repoint to replacement (DESIGN→ARCHITECTURE/OVERVIEW; stray "Phase N"→PHASES.md). Links inside archived files left as-is.
  - **Archived banner** on each archived file: `> ARCHIVED 2026-06-04 — superseded by <replacement>; not authoritative.`
  - **CLAUDE.md in scope**: update its 4 doc links + shrink Project Status to a one-line OVERVIEW pointer.
  - **TOC.md regenerated** wholesale: canon only + one note that archive/ exists and is non-authoritative.
  - **USAGE.md** (→ operations/): full CLI reference stays in README.md; USAGE strips duplicated option tables, points to README, keeps unique task/domain-comparison content.
- **Date**: 2026-06-04

### DECIDED: Done-criteria & PR/sequencing

- **Decision**:
  - **Execution order (content first, structure second — forced by extract-before-archive):** (1) verify completion vs code/tests; (2) reconcile DECISIONS + verify ARCHITECTURE; (3) mine DESIGN/research/code-reviews → OVERVIEW/ARCHITECTURE/DECISIONS; (4) write OVERVIEW/PHASES/TASKS + fix requirements index; (5) `git mv` to operations//reference/, create top-level archive/ + banners, repoint ~9 links, regenerate TOC, shrink CLAUDE.md status to pointer; (6) gate.
  - **`git mv`** for all moves/archives (preserve blame/history).
  - **Two PRs**: PR1 = content (canon + verify + reconcile + mine, old files in place); PR2 = reorg (moves, archive+banners, links, TOC, CLAUDE pointer). Both human-gated.
  - **Definition of done (checklist):** canon verified vs code (Feb/Apr cohorts); completion claims backed by code/tests; all DECISIONS superseded entries marked; zero broken internal links (markdownlint-cli2 + relative-link existence) + permalinks spot-checked at pinned SHA; TOC matches tree; every archive/ file banner-tagged; CLAUDE.md links valid + Project Status is a pointer; OVERVIEW remaining-work captures the OCR-separation plan (after verification).
- **Date**: 2026-06-04

## Homework Findings (2026-06-04)

- **to-merge was never merged out.** Git history shows only move/rename/add commits; `merged-batch-completion-guide.md` only consolidates the 5 completion sources *within* to-merge; no canonical doc cites any to-merge file (only TOC.md lists them). Verdict: staged-and-forgotten, not integrated.
- **The 4 code-review docs are the richest unconsumed source of "known issues / tech debt."** PROCESS.md's task workflow is explicitly designed to cite "code review sections," but no task file has consumed them. They date to ~2026-02-16 (early snapshot), so some critiques are likely already addressed by later phases — they need triage-against-code, not wholesale keep OR blind archive.
- **"Assume completely out of date" is false for a meaningful subset.** README.md is current (accurate CLI reference + architecture). The 18 `docs/requirements/*.feature` files are phase-tagged and map capability→source files (high value; user already froze them). `development-plans/externalize-store-postgres-s3.md` is done and accurate. Freshness is cohort-dependent (to-merge=Feb; bulk planning/PHASE/research=single 2026-04-08 commit, likely a move so content is older; DEBUGGING/playbooks/DEPLOYMENT/SECURE-REMOTE=May–Jun, fresh).
- **The "detailed task list" already half-exists and is broken.** PROCESS.md links to `TASKS.md` and `CHANGELOG.md` in development-plans/ — **neither file exists.** The user's task-list deliverable ≈ the missing TASKS.md. development-plans/ is the established home.
- **"Phases" is overloaded.** PHASE1–14 = implementation roadmap (docs/planning/). Phase A/B/C = sub-phases *inside* the externalize-store task. Clarify which the deliverable means.
- **Three overlapping overview docs.** CLAUDE.md (auto-loaded; has a Project Status section) + README.md (human intro + full CLI ref) + the proposed master overview. For an LLM audience, CLAUDE.md already satisfies "don't make it search."
- **requirements/README.md is itself stale** — lists 15 feature files but 18 exist (missing ocr_cli, ocr_web_api, human_in_the_loop). The index is stale even though the user froze .feature content.
- **Temporal deep-link drift is real.** Forge pins `temporalio>=1.9.0`; the sdk-python clone is 1.27.2 on a moving HEAD (commits past the tag); the server clone is ~v1.29.0. Line-number links will rot; version-specific API links may already mismatch Forge's resolved version.
- **ARCHITECTURE.md vs DESIGN.md overlap heavily** (workflow step, execution modes, context, routing, batch). ARCHITECTURE is richer/current (Module Map, Key Data Models, End-to-End Example) → canonical "how." DESIGN's unique bits (Principles, Plan Format, Transition Vocabulary, Tech Stack, Development Phases) → mine then archive.
- **DECISIONS.md = 85 decisions (D1–D85); highest value per usefulness-test (c), but append-only with no supersession markers → self-contradictory.** D54 (pydantic-ai cache) vs D75 (remove pydantic-ai); D29 (no recursive fan-out) vs D69 (recursive fan-out + `--max-fan-out-depth`). Must reconcile against code; cannot archive, cannot keep as-is.
- **"Phases" scattered across ≥4 places**: 14 PHASE*.md, DESIGN.md §Development Phases, `@phase-N` tags in .feature files, ARCHITECTURE §Execution Modes. PHASES deliverable must be the single source.
- **DECOMPOSITION.md + SCENARIOS (~1744L) may be UNBUILT design** ("Open Questions", "Relationship to Existing Planner"). Verify vs planner.py before classifying.
- **LSP_INTEGRATION_PLAN.md = genuine remaining-work plan** (ties to D38 defer-LSP) → feeds "detailed remaining phases," not archive.
- **research/ (system-prompts-claude, attractor-analysis, planner-prompt-research)** = exploratory; adopted conclusions live in code/DECISIONS → archive candidates.
- **USAGE.md overlaps README.md CLI reference** → consolidate.

## Resolved Branches (all decided)

1. ✅ DECIDED — staleness method (verify-against-code) + code-review triage-and-mine.
2. ✅ DECIDED — Layered canon.
3. ✅ DECIDED — Status architecture (TASKS.md = atomic source in development-plans/; PHASES = historical roadmap; OVERVIEW points; CLAUDE → pointer; verify completion vs code).
4. ✅ DECIDED — Linking & reference strategy (SHA-pinned permalinks; point-don't-copy file+symbol; surgical Temporal links).
5. ✅ DECIDED — Phases & requirements content (extract-before-archive; PHASE13+LSP kept & linked; PHASES = roadmap only; fix req index + derive completion from @phase tags).
6. ✅ DECIDED — Archive mechanics (top-level archive/; banners; small link blast-radius repointed; CLAUDE.md in scope; TOC regenerated; USAGE→README for CLI ref).
7. ✅ DECIDED — Done-criteria & PR/sequencing (content-first/structure-second; git mv; two PRs; done-checklist).

## Execution Watch-Items (decided in principle; resolve during the work)

- **DECOMPOSITION.md + SCENARIOS (~1744L)** — classify during the verify step: built → fold to ARCHITECTURE; planned-but-unbuilt → keep as remaining spec (link from PHASES); abandoned → archive with banner. Not yet classified.
- **DECISIONS.md reconciliation is the single largest task** — 85 entries (D1–D85), each checked against code, superseded ones marked.
- **OCR-separation plan** — verify still current (`grill-me-sessions/separate-ocr-modules.grill.md`) before capturing in OVERVIEW remaining-work and flagging in the OCR docs.
- **diataxis/** — out of scope (disposable/non-authoritative per CLAUDE.md); leave as-is, do not fix its links.
- **TOC.md** — currently missing SECURE-REMOTE-ACCESS.md and docs/inference-providers/README.md; the full regenerate fixes this.

## Deferred Items

None — every branch was decided; nothing deferred with unaddressed risk.
