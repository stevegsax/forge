# Development Process

This document describes the workflow for working on tasks in `development-plans/`.

## Picking Up a Task

1. Open [TASKS.md](TASKS.md) and find the next unchecked task
2. Check the Dependencies section -- skip tasks whose dependencies are incomplete
3. Open the task file for full context

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

1. Run the verification steps listed in the task file
2. Update the task file's **Status** to `DONE`
3. Check off the task in [TASKS.md](TASKS.md) immediately
4. Append a row to [CHANGELOG.md](CHANGELOG.md)

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
