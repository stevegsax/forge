---
name: handoff-sweep
description: Perform the forge handoff sweep — write the next HANDOFF-*-start.md and truth-pass the status-of-record (CLAUDE.md, docs/OVERVIEW.md, TOC.md) so every factual claim is re-derived, never copied forward. Use when the user says "update the handoff", "handoff sweep", "write the handoff", or is closing a task/session. Runs inline or dispatched to the handoff-scribe agent. Never commits.
---

# Handoff sweep

The canonical procedure for closing a unit of work (usually one task) into
the repo's standing records. This file is the single source of truth: the
`handoff-scribe` agent follows it verbatim, and inline runs follow it
verbatim. If reality and this procedure disagree, stop and report the
mismatch instead of improvising.

**Prime rule — derive, never carry.** Every factual claim written by this
sweep is re-derived from a command or a file read *during this run*.
Copying a fact forward from the previous handoff is how the record went
stale twice (the 2026-07-27 plan review caught "Production still runs
`001247c`" carried forward after prod had moved to `7002435`, and an
"open" act-later item that had already been fixed). A claim with no
derivation does not go in.

**Never commit.** The owner commits on their explicit word only. The sweep
ends with a report and a suggested commit message.

## Inputs

From the dispatcher (or ask if missing — do not guess):

- What closed since the last handoff (task ids, owner decisions,
  operational changes like deploys). Cross-check against ground truth
  below; if the dispatcher's summary and the record disagree, report it.
- The next task (the new handoff's "-start" subject).

## Step 1 — establish ground truth (commands, not memory)

```bash
git log --oneline -15                                  # what landed, real SHAs
git status --short                                     # tree must be explainable
ls development-plans/HANDOFF-*.md | sort | tail -3     # previous handoff = template + diff base
git -C ~/repos-sax/forge-prod rev-parse --short HEAD   # prod SHA — the claim that went stale
rg -n "Next up:|Next:" CLAUDE.md docs/OVERVIEW.md      # current "next" claims
rg -n "^\*\*Status" development-plans/tasks/T*.md      # task states
```

Also read: the closing task's file (its Dev Notes carry the gate numbers
and deviations — cite those recorded numbers with their source; do not
re-run gates), `development-plans/TASKS.md` checkboxes, and
`development-plans/CHANGELOG.md`'s newest rows.

## Step 2 — write the new handoff

Path: `development-plans/HANDOFF-<YYYY-MM-DD>-<next-task-slug>-start.md`
(e.g. `HANDOFF-2026-07-27-t5.5-start.md`). Structure is fixed — copy the
previous handoff's skeleton, replace every fact:

1. **Title:** `# Handoff — <what completed>; start <next task>`.
2. **Status block:** date; landed commits with SHAs; gate results *as
   recorded in the closing task file* (named as such); **prod SHA from
   the Step 1 command** and whether the new work is deployed; the next
   task.
3. **The disclaimer paragraph** (keep verbatim in spirit): a
   state-of-the-world note that does not restate TASKS.md, DECISIONS.md,
   or OVERVIEW.md — it says what changed since the previous handoff
   (link it), what to know before touching anything, and where to start.
4. **"What landed since the last handoff"** — one bullet per landed
   unit, dense house style, SHAs inline.
5. **"Owner rulings"** (when any were made) — binding decisions with the
   owner's rationale in one sentence each.
6. **"What to know before touching anything"** — numbered; each item is
   a hazard, constraint, or open question with its mechanism, not a bare
   caveat. Always include: the standing-directives paragraph,
   reproduced from `development-plans/SDLC.md` § Standing directives
   **as it reads during this run** — every item, compact prose form is
   fine. That section is the list's canonical home; never source the
   list from the previous handoff (an item added or retired in SDLC.md
   must reach the next session even when the previous handoff predates
   the change). List any open owner adjudications explicitly with a
   do-not-act instruction.
7. **"Where to start"** — the next task's read order and first moves;
   note that pickup reconciliation still applies.

## Step 3 — truth-pass the status-of-record

Edit only statements the landed work falsified; do not restyle.

- **CLAUDE.md**: (a) the Landed paragraph's `(as of …)` date; (b) append
  the landed-task sentence in house style — bold headline with date and
  SHAs, what shipped, the proof (gates/replay as recorded), deviations
  that matter forward; (c) the closing `**Phases … Next up: …**`
  sentence; (d) the Context bullet's current-handoff pointer (rotate the
  old one into the history list); (e) the Release Roadmap "Current"
  line; (f) scan the convention sections (Running the System, The
  Universal Workflow Step, Test Patterns, Architecture Principles) for
  any sentence the landed work made false — fix it in place.
- **docs/OVERVIEW.md**: the migration-status bullet gains the landed
  sentence and its `**Next: …**` pointer moves.
- **TOC.md**: point the "current state" handoff line at the handoff
  this sweep wrote (the superseded entry stays in the history list,
  keeping its one-line summary without the marker), and fix any other
  line the landed work falsified — the DECISIONS.md range is the
  recurring one. Added 2026-07-27 after TOC.md drifted three handoffs
  stale as the one standing record no procedure owned.
- **Consistency check** (all must agree): CLAUDE.md "Next up",
  OVERVIEW "Next", the first unchecked task in TASKS.md, and TOC.md's
  current-state pointer naming the handoff written in Step 2.
- **Prod claims**: anywhere the records state what production runs, the
  value must equal Step 1's prod SHA.

Out of scope for the sweep (owned elsewhere — do not do these here):
task-file amendments and Plan sections (task work / adjudicated
amendment change-sets), CHANGELOG rows and TASKS.md checkboxes (task
close-out per PROCESS.md), DECISIONS.md entries and banners (the
Specification Changes policy), deploys, and any `src/**` change. One
narrow exception: a comment or docstring the landed work made *false*
may be corrected if it is cited in the report and its package's checks
are run.

## Step 4 — verify

```bash
markdownlint-cli2 <every file touched>        # zero errors
uv run pytest tests/test_replay.py --no-cov -q  # only if anything outside docs/ was touched
```

Plus the package checks for any non-markdown exception edit (ruff + that
package's suite).

## Step 5 — report (this is the deliverable)

- Files created/edited, one line each.
- **Claims corrected**, each as: the false statement → the derived truth
  → the derivation (command or file).
- Anything found that this sweep does not own (stale task premise, unowned
  deferral) — flagged, not fixed.
- Suggested commit message in house style, first line:
  `Handoff sweep: HANDOFF-<date>-<slug>-start + status-of-record truth pass`,
  body summarizing the sweep, ending with the standard co-author line.
- The reminder that nothing was committed.

## Hazards

- Never start a wrapped prose line in `.md` with `+` (the formatter hook
  re-parses it as a list marker).
- `rg -n`, never `rg -rn` (`-rn` parses as `--replace n` and fabricates
  output).
- `gsed`, not `sed`, if stream-editing is ever needed (prefer Edit).
- House doc style: succinct, no praise adjectives, box-drawing characters
  for trees, mechanism before instruction.
