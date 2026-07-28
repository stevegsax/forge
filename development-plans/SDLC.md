# Development Lifecycle (SDLC)

How forge itself is built: the roles, the task flow from queue to
production, the record system and the procedure that keeps each record
true, and the standing directives that bind every session. This file is
the map — each linked document owns its own detail, and nothing here
duplicates it. [PROCESS.md](PROCESS.md) is the task-lifecycle
mechanics; this file is everything around it.

Scope: this is the meta-process for building forge. The product-facing
standard for requirement packages fed *to* forge for autonomous
implementation is separate —
[docs/requirements/STANDARD.md](../docs/requirements/STANDARD.md). The
directives here govern how sessions work; the product's Architecture
Principles live in [CLAUDE.md](../CLAUDE.md).

## Documentation theory

Four rules keep the records true. Every practice below is an
application of one of them.

1. **One source of truth per fact; cross-reference, don't duplicate.**
   Where an operative copy must exist anyway (agent prompts are
   self-contained; each handoff carries the standing directives), the
   copy is refreshed from its source, and a change to the source lands
   in the same change-set as the copies.
2. **Derive, never carry.** A fact is re-derived from a command or a
   file read at write time, never copied forward from the previous
   version of a record. Adopted as the handoff sweep's prime rule after
   carried-forward claims went stale twice (a prod SHA and an
   already-fixed "open" item, both caught by the 2026-07-27 plan
   review).
3. **Banners at change time.** A change that contradicts a recorded
   decision lands the supersession banner in the same change-set
   ([PROCESS.md — Specification Changes](PROCESS.md#specification-changes)),
   so truthfulness never depends on periodic repair sweeps.
4. **Every standing record has an owning procedure** (see the records
   map). A record nothing re-derives goes stale silently: TOC.md
   drifted three handoffs behind by 2026-07-27 precisely because it was
   the one standing record no procedure owned. It is owned by the
   handoff sweep now.

## Roles

The agent model split (owner directive, 2026-07-24):

| Role | Runs as | Owns |
| --- | --- | --- |
| Owner | — | Every commit and merge; spec-change adjudication; deploys; commissioning premise reviews; rulings on parked proposals and A-numbered adjudications |
| Planning session | Fable, main session | Premise work and design; the task file end to end (Plan, Sub-tasks, Development Notes, close-out records); dispatch scopes; review of driver reports; sweep dispatch |
| `implementation-driver` | Opus subagent | Landing one scoped task: writes non-trivial code itself, delegates only obviously-easy mechanical code, reviews everything it accepts, runs the scoped verification |
| `code-writer` | Sonnet subagent | Exact-spec mechanical changes only; stops and reports on any ambiguity |
| `handoff-scribe` | Opus subagent | The handoff sweep, per the skill, verbatim |

The bias is explicit: if it is not obviously easy, Opus writes it.
Temporal workflows, the environment guard, concurrency, and migrations
are never delegated to Sonnet. Drivers report; the planning session
records — task files are written by the planning session throughout.

The operative role prompts are [.claude/agents/](../.claude/agents/),
deliberately self-contained, so they embed copies of the rules that
bind them; a rule change updates the agent file and this document in
the same change-set (theory rule 1).

## Task flow

Queue to production, with the owning document for each step:

1. **Queue.** [TASKS.md](TASKS.md) in priority order. Entry is by owner
   decision or adoption of a parked proposal
   ([PROCESS.md — Creating a Task File](PROCESS.md#creating-a-task-file)).
2. **Pickup + reconciliation.** Re-derive the task file's premises
   against the current tree before planning
   ([PROCESS.md — Picking Up a Task](PROCESS.md#picking-up-a-task)).
3. **Plan.** The planning session writes the Plan and Sub-tasks
   sections in the task file
   ([PROCESS.md — Before Coding](PROCESS.md#before-coding)).
4. **Dispatch.** Implementation goes to `implementation-driver` with a
   hard-bounded scope: file list, expected behavior, verify commands,
   boundaries. The planning session reviews the report and folds
   deviations into Development Notes.
5. **Verify.** Scoped package checks during implementation; the
   workspace gates and the replay decision at close, with the numbers
   recorded in Development Notes
   ([PROCESS.md — After Coding](PROCESS.md#after-coding); conventions
   in [CLAUDE.md — Test Patterns](../CLAUDE.md) and
   [docs/operations/test-strategy.md](../docs/operations/test-strategy.md)).
6. **Close-out records.** Status flip, Background revisit, TASKS.md
   checkbox, CHANGELOG row (PROCESS.md — After Coding).
7. **Handoff sweep.** The canonical close of a unit of work:
   [.claude/skills/handoff-sweep/SKILL.md](../.claude/skills/handoff-sweep/SKILL.md),
   run inline or dispatched to `handoff-scribe`. Ends with a suggested
   commit message.
8. **Commit.** On the owner's explicit word only, directly on `main` —
   trunk-based, no pull requests (the only merge commits in history are
   T2.1's monorepo history grafts). CLAUDE.md's "Git Strategy" section
   is the *product's* worktree model for forge-driven tasks, not how
   this repo is developed — the two are easy to conflate.
9. **Deploy — decoupled.** Owner-triggered, never implied by task
   close: `make prod-deploy REF=<ref>` to the pinned checkout (D103),
   then verify every poller reports `prod-<role>@<sha>` with no
   `-dirty`
   ([docs/operations/DEPLOYMENT.md](../docs/operations/DEPLOYMENT.md),
   [docs/operations/WORKERS.md](../docs/operations/WORKERS.md)).

A `/simplify` pass over the landed diff (reuse, simplification,
altitude) is owner-triggered and optional, not a standing step — T5.1
got one, most tasks do not.

## Premise reviews and adjudications

A premise-level review re-grooms the remaining queue: does each task's
reason-to-exist survive the landed work, do its dependencies and
ordering hold, has it grown or shrunk enough to change sizing, and is
anything homeless. Owner-commissioned at phase boundaries or when the
queue's assumptions have aged — deliberately no standing cadence. The
pattern so far:
[forge-review-2026-07-08.md](../forge-review-2026-07-08.md) (findings →
dated amendments plus a capture sweep) and
[forge-plan-review-2026-07-27.md](../forge-plan-review-2026-07-27.md)
(the premise pass proper); each artifact records its own method.

Findings that need an owner ruling get A-numbers in the review
artifact. Open adjudications are carried in every handoff's "What to
know" with a do-not-act instruction until ruled; rulings land as dated
amendments per the Specification Changes policy.

## Records map

Every standing record and the procedure that keeps it true. A record
with no row here has no owner — that is a defect (theory rule 4): add
the row and the owning procedure together.

| Record | Holds | Kept true by |
| --- | --- | --- |
| [TASKS.md](TASKS.md) | Done vs not; queue order; preamble amendments | Task close-out (PROCESS.md); premise reviews |
| `tasks/T*.md` | Per-task spec, plan, notes, as-shipped truth | The planning session per PROCESS.md; pickup reconciliation |
| [CHANGELOG.md](CHANGELOG.md) | One row per landed task | Task close-out (PROCESS.md) |
| [docs/DECISIONS.md](../docs/DECISIONS.md) | Numbered decisions + banners | Specification Changes policy, same change-set |
| [CLAUDE.md](../CLAUDE.md) | Status narrative + working conventions | Handoff sweep truth pass; spec changes same change-set |
| [docs/OVERVIEW.md](../docs/OVERVIEW.md) | Status-of-record | Handoff sweep truth pass |
| [TOC.md](../TOC.md) | Documentation index + current-handoff pointer | Handoff sweep truth pass (added 2026-07-27) |
| `HANDOFF-*.md` | Session-boundary state; open adjudications; the standing-directives copy | Written fresh by each sweep; history once committed — corrections go in the next handoff |
| [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md), `docs/operations/*` | How the system works and runs | The change-set that falsifies a statement fixes it (no sweep coverage) |
| Review artifacts (`forge-*review*.md`, [docs/reviews/](../docs/reviews/)) | Point-in-time findings + adjudications | Immutable after publication; live consequences move into task files and TASKS.md |
| This file, [PROCESS.md](PROCESS.md), the sweep skill, `.claude/agents/*` | The process itself | Owner-gated amendment, same change-set as the practice change |

## Standing directives

The canonical list. The handoff sweep reproduces it into each handoff
from this section as it reads at sweep time — never from the previous
handoff — so additions and retirements happen here (owner-gated), and
in the agent prompts where operative copies exist, in the same
change-set.

### Authority and scope

- Commits and merges are the owner's, on explicit word only. No agent,
  skill, or sweep commits.
- Implementation is dispatched to `implementation-driver` with
  hard-bounded scopes; Temporal workflows, the environment guard,
  concurrency, and migrations are never delegated to Sonnet.
- Pre-existing uncommitted changes in the tree belong to other work;
  leave them untouched.

### Engineering

- No compatibility shims. Forge has no external consumers (true as of
  2026-07-27): fix the seam in place and update all callers; never add
  parallel compat paths.
- Activity timeout/retry presets come from `forge/presets.py` only —
  the values are ScheduleActivityTask command attributes, so changing
  one is a replay-breaking change.
- No inline copies of the step/gather/dispatch pipelines in the
  `forge/workflows/` drivers; the blocks are the single home.
- New workflow tests pass `sync_mode=True` and script outcomes keyed by
  identity, never arrival order; no module-level mutable containers in
  test files (the pytest-xdist blocker, removed by T5.5).
- Read `~/.claude/python-guidelines.md` before writing or reviewing
  Python; require the same of any spawned code-writer.

### Working style (owner agreements)

- Mechanism-level explanations: the underlying cause, act-now versus
  act-later versus no-action, concrete command sequences. A safety
  caveat is never compressed into a bare clause.
- Challenge weak arguments and spot missing data; do not praise.

### Tool hazards

- `rg -n`, never `rg -rn` — `-rn` parses as `--replace n` and
  fabricates output.
- Never start a wrapped `.md` prose line with `+` — the formatter hook
  re-parses it as a list marker and rewrites the list.
- `gsed`, not `sed`, when stream-editing is unavoidable; prefer Edit.
