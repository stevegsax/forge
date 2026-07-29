# Development Process

This document describes the workflow for working on tasks in
`development-plans/`. It is the task-lifecycle detail; the end-to-end
map — roles, the dispatch model, records ownership, and the standing
directives — is [SDLC.md](SDLC.md).

Actors (defined in [SDLC.md](SDLC.md)): the planning session works the
task file end to end — Plan, Sub-tasks, Development Notes, close-out
records; implementation is dispatched to the `implementation-driver`
agent with hard-bounded scopes, and its reports feed Development Notes;
the owner gates every commit.

## Creating a Task File

New tasks enter by owner decision or adoption of a parked proposal. Create
`tasks/T<phase>.<n>-<kebab-slug>.md`, add its entry to
[TASKS.md](TASKS.md) (dated parenthetical noting the origin, and update the
task count), and structure the file in this order:

1. Header block: **Status** (`NOT STARTED`), **Phase**, **Repos**,
   **Depends on**, **Last updated**
2. **Problem** — the dense, precise statement with grounding data
3. **Background and Detailed Explanation** — required at creation; see below
4. **Scope** (or **Design**) — what changes where
5. **Acceptance Criteria** — checkboxes
6. **References**
7. **Plan**, **Sub-tasks**, **Development Notes** — placeholders; the
   implementing session fills them per "Before Coding"

The **Background and Detailed Explanation** section is written for a reader
with no shared context, in plain language. Rules: a problem → fix → payoff
narrative that explains *why* (which principle or cost is at stake), not
just what; call out the buried gems and gotchas the dense spec implies;
define project jargon inline at first use; complete sentences, no praise
adjectives; timeless (no references to sessions or conversations); tense
matches status (planned tasks present/future, completed tasks past); depth
matches weight (substantial tasks 3–6 paragraphs, small mechanical ones
1–2 — never pad). Reference examples at the bar:
[T5.1](tasks/T5.1-pure-step-logic.md) (planned) and
[T4.4](tasks/T4.4-mistral-status-tracker.md) (completed).

## Picking Up a Task

1. Open [TASKS.md](TASKS.md) and find the next unchecked task
2. Check the Dependencies section -- skip tasks whose dependencies are incomplete
3. Open the task file for full context
4. Reconcile the task file against the current tree before planning
   (**pickup reconciliation**): re-derive its premises — do the named
   files, symbols, line references, and counts still hold; does the
   stated problem still exist; has a landed task, a recorded amendment
   (TASKS.md preamble, plan reviews), or a decision changed the scope?
   Record drift as a dated amendment section (spec-level) or a
   Development Note (detail-level) before writing the Plan. Task specs
   age while queued: T5.4's premise was largely consumed by T5.1–T5.3
   and was re-baselined by owner adjudication at exactly this step

## Before Coding

1. Read the task file's **Problem** and **Acceptance Criteria** sections
2. Read the referenced code review sections and source files
3. Write a **Plan** section in the task file describing your approach
4. Break the plan into a **Sub-tasks** checklist
5. Update the task file's **Status** to `IN PROGRESS`

## During Coding

1. Check off sub-tasks as you complete them
2. Append to the **Development Notes** section immediately when you:

    - Discover something unexpected
    - Make a design decision and why
    - Find a gotcha or edge case
    - Change the plan from what was originally written

3. If the plan needs to change, update it and note why in Development Notes

## After Coding

1. Run the verification steps listed in the task file, then the
   workspace gates (`make gates`) for any change-set touching code;
   markdown-only change-sets run `markdownlint-cli2 --config
   <repo-root>/.markdownlint-cli2.jsonc` on the touched files instead
   (the explicit `--config` matters: config discovery stops at the
   invoking directory, so a bare run from a member directory silently
   lints against tool defaults)
2. Record the gate results in Development Notes — suite counts,
   coverage, mypy/lint outcomes. The handoff sweep cites these recorded
   numbers and never re-runs gates, so an unrecorded result is lost to
   the record
3. Record the replay outcome in Development Notes: committed histories
   passing unregenerated are the behavior-preservation proof;
   regeneration (of a named subset) is justified only by a deliberate
   workflow-logic change, and the note names it
4. Revisit the **Background and Detailed Explanation**: flip it to past
   tense and fold in what actually shipped (Development-Note surprises,
   superseded pieces) so the explanation stays true of the as-built work
5. Update the task file's **Status** to `DONE`
6. Check off the task in [TASKS.md](TASKS.md) immediately
7. Append a row to [CHANGELOG.md](CHANGELOG.md)
8. Close the unit of work with the handoff sweep
   ([.claude/skills/handoff-sweep/SKILL.md](../.claude/skills/handoff-sweep/SKILL.md),
   run inline or dispatched to `handoff-scribe`); it ends with a
   suggested commit message
9. Nothing in this list commits, and deploying is a separate owner
   decision — the owner commits on explicit word only
   ([SDLC.md](SDLC.md#task-flow))

## Specification Changes

There is no change-request form; there are three artifacts, chosen by what the
change touches. The owner adjudicates all three.

1. **A change that contradicts a recorded decision** requires a new entry in
   [../docs/DECISIONS.md](../docs/DECISIONS.md) *and* a supersession or
   amendment banner (`> Superseded by DNN` / `> Amended by DNN`) on the old
   entry — **in the same change-set** as the code. Banners applied at change
   time are what keep DECISIONS.md truthful without periodic repair sweeps.
   A change that reverses previously decided direction also gets an R-number,
   cross-referenced from TASKS.md and CLAUDE.md.
2. **A scope change to a not-yet-built task** lands as a dated amendment
   section in the task file (and a TASKS.md preamble note when it touches
   several tasks at once).
3. **A proposed design that is not adopted now** gets a parked-proposal note
   in the owning task's Development Notes: the proposal, an assessment,
   explicit revisit prerequisites, and a dated owner disposition line.

Mid-implementation discoveries that change the spec follow the same rules —
record in the Development Notes immediately (per "During Coding"), and if the
discovery contradicts a recorded decision, rule 1 applies before the task
closes.

## Principles

- **Accurate status documentation is as important as writing code.** Another engineer must be able to pick up where you left off with no shared context.
- **Capture knowledge immediately.** If you learn something during implementation, write it in Development Notes before moving on.
- **Cross-reference, don't duplicate.** Link to the code review, source files, and other task files rather than copying content.
- **Work in priority order** unless blocked by dependencies.
