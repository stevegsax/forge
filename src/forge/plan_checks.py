"""Pure plan-structure checks (T5.6) — one implementation, two consumers.

Every rule about what makes a :class:`~forge.models.Plan` structurally sound
lives here as a *finder*: a pure function that returns a tuple of offending
descriptors, empty when the plan is clean. Two callers dress those findings up
differently and neither owns the algorithm:

* :mod:`forge.eval.deterministic` wraps each finder in a
  ``DeterministicCheckResult`` with its pass/fail wording (Layer 1 of the
  planner eval harness);
* :func:`preflight_plan` runs the :data:`PREFLIGHT_CHECKS` subset at plan
  acceptance in ``blocks.dispatch.dispatch_planner``, so a malformed plan is
  rejected before any step runs and the planner gets the specific violations
  back in its retry context.

Before T5.6 the checks existed only in the eval harness, so nothing ran them on
real planner output; the sole acceptance gate was ``Plan.model_validate``, whose
only structural rule is ``1 <= len(steps) <= MAX_PLAN_STEPS``. Duplicate ids,
overlapping fan-out targets, and absolute paths all executed happily — and two
sub-tasks declaring the same file bought an LLM conflict-resolution call to
untangle a collision this module rejects for free (Principle 2).

**Recursion.** The pre-T5.6 checks iterated exactly one nesting level, so a
violation inside a nested sub-task was invisible even to eval. Every finder here
recurses. Two scoping rules are deliberate:

* ids are unique among *siblings*, not globally — a child's identity is the
  compound ``<parent>.sub.<child>`` path, so two ``gc1`` leaves under different
  parents are distinct;
* target overlap compares each child's *effective* target set (its own targets
  plus every descendant's), because a grandchild's file surfaces in its parent's
  merged output — a grandchild/uncle collision is a real collision.

**Zero ``temporalio`` imports** by design (the :mod:`forge.step_logic` pattern):
workflow code calls :func:`preflight_plan` and :func:`splice_revision` inline,
Temporal replay reproduces the same answer, and the whole decision surface gets
microsecond unit tests without a Temporal server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from forge.models import MAX_PLAN_REVISIONS, MAX_PLAN_STEPS, Plan, ThinkingPolicy

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from forge.models import PlanStep, SubTask

__all__ = [
    "PREFLIGHT_CHECKS",
    "PlanViolation",
    "RevisedPlan",
    "RevisionRejected",
    "duplicate_step_ids",
    "duplicate_sub_task_ids",
    "escalate_thinking",
    "forward_references",
    "implausible_context_files",
    "nodes_without_targets",
    "overlapping_sub_task_targets",
    "preflight_plan",
    "retry_prompt_section",
    "splice_revision",
    "uncovered_task_targets",
    "undersized_fan_outs",
    "unsafe_target_paths",
]


# ---------------------------------------------------------------------------
# Node walking (pure)
# ---------------------------------------------------------------------------

type PlanNode = PlanStep | SubTask
"""A step or a sub-task: both carry ``target_files``, ``context_files``, and
optional ``sub_tasks``, which is every field these checks read."""


def _children(node: PlanNode) -> tuple[SubTask, ...]:
    """The node's sub-tasks (empty when it is a leaf)."""
    return tuple(node.sub_tasks or ())


def _walk(plan: Plan) -> Iterator[tuple[str, PlanNode]]:
    """Yield ``(path, node)`` for every node in the plan, parents before children.

    ``path`` is the slash-joined id chain (``"step-1/st1/gc1"``), which is the
    detail format the pre-T5.6 one-level checks already produced for depth 1.
    """
    for step in plan.steps:
        yield from _walk_node(step.step_id, step)


def _walk_node(path: str, node: PlanNode) -> Iterator[tuple[str, PlanNode]]:
    yield path, node
    for child in _children(node):
        yield from _walk_node(f"{path}/{child.sub_task_id}", child)


def _subtree(node: PlanNode) -> Iterator[PlanNode]:
    """The node and every descendant."""
    yield node
    for child in _children(node):
        yield from _subtree(child)


def _effective_targets(node: PlanNode) -> set[str]:
    """Every file this node or any descendant declares as a target.

    A fan-out parent's merged output is the union of its children's output, so
    this — not ``node.target_files`` alone — is what a sibling can collide with.
    """
    return {f for descendant in _subtree(node) for f in descendant.target_files}


# ---------------------------------------------------------------------------
# Finders (pure) — empty tuple means the rule holds
# ---------------------------------------------------------------------------


def unsafe_target_paths(plan: Plan) -> tuple[str, ...]:
    """Target files that are absolute or traverse out of the worktree."""
    return tuple(
        f"{path}: {f}"
        for path, node in _walk(plan)
        for f in node.target_files
        if f.startswith("/") or ".." in f.split("/")
    )


def duplicate_step_ids(plan: Plan) -> tuple[str, ...]:
    """Step ids appearing more than once in the plan."""
    ids = [step.step_id for step in plan.steps]
    return tuple(sorted({sid for sid in ids if ids.count(sid) > 1}))


def duplicate_sub_task_ids(plan: Plan) -> tuple[str, ...]:
    """Sub-task ids repeated among the children of one parent (any depth).

    Sibling-scoped on purpose: a child's runtime identity is the compound
    ``<parent>.sub.<child>`` id, so the same leaf id under two different parents
    is unambiguous — and the fan-out gather's own duplicate-id guard
    (``failure_kind="duplicate_sub_task_ids"``) uses exactly this scope.
    """
    dupes: list[str] = []
    for path, node in _walk(plan):
        ids = [child.sub_task_id for child in _children(node)]
        dupes.extend(f"{path}/{sid}" for sid in sorted({s for s in ids if ids.count(s) > 1}))
    return tuple(dupes)


def overlapping_sub_task_targets(plan: Plan) -> tuple[str, ...]:
    """Sibling sub-tasks whose effective target sets intersect (D27).

    Two children claiming one file is the expensive defect: it runs both
    generations, then buys a conflict-resolution LLM call to merge them.
    """
    overlaps: list[str] = []
    for path, node in _walk(plan):
        claimed: dict[str, str] = {}
        for child in _children(node):
            child_id = child.sub_task_id
            for f in sorted(_effective_targets(child)):
                owner = claimed.get(f)
                if owner is not None:
                    overlaps.append(f"{path}: {f} claimed by {owner} and {child_id}")
                else:
                    claimed[f] = child_id
    return tuple(overlaps)


def nodes_without_targets(plan: Plan) -> tuple[str, ...]:
    """Leaf nodes (no sub-tasks) that declare no target files.

    A leaf with no declared output tells the generation call nothing about what
    it is supposed to produce, and whatever it writes is outside the plan's
    conflict accounting. Fan-out parents are exempt: their output is their
    children's.
    """
    return tuple(
        path for path, node in _walk(plan) if not _children(node) and not node.target_files
    )


def undersized_fan_outs(plan: Plan) -> tuple[str, ...]:
    """Fan-out nodes with fewer than two sub-tasks.

    Eval-only (see :data:`PREFLIGHT_CHECKS`): a one-child fan-out is wasteful —
    a whole child workflow for a single unit of work — but it executes
    correctly, so it is a quality signal rather than a defect worth halting for.
    """
    return tuple(
        f"{path}: {len(node.sub_tasks or ())} sub-task(s)"
        for path, node in _walk(plan)
        if node.sub_tasks is not None and len(node.sub_tasks) < 2
    )


def implausible_context_files(plan: Plan, known_repo_files: set[str]) -> tuple[str, ...]:
    """Context files that neither exist in the repo nor are produced by an earlier step."""
    produced: set[str] = set()
    implausible: list[str] = []
    for step in plan.steps:
        for path, node in _walk_node(step.step_id, step):
            implausible.extend(
                f"{path}: {f}"
                for f in node.context_files
                if f not in known_repo_files and f not in produced
            )
        produced |= _effective_targets(step)
    return tuple(implausible)


def forward_references(plan: Plan, known_repo_files: set[str]) -> tuple[str, ...]:
    """Context files a step reads that only a later step produces.

    Requires ``known_repo_files``: without it, "step 1 reads config.py, step 3
    rewrites config.py" — an everyday plan — is indistinguishable from a genuine
    forward reference, so this check cannot run where the repo file set is not
    available (see :data:`PREFLIGHT_CHECKS`).
    """
    outputs = [_effective_targets(step) for step in plan.steps]
    refs: list[str] = []
    for index, step in enumerate(plan.steps):
        available = (
            set(known_repo_files).union(*outputs[:index]) if index else set(known_repo_files)
        )
        for path, node in _walk_node(step.step_id, step):
            refs.extend(
                f"{path}: {f}"
                for f in node.context_files
                if f not in available and any(f in later for later in outputs[index:])
            )
    return tuple(refs)


def uncovered_task_targets(plan: Plan, task_targets: Sequence[str]) -> tuple[str, ...]:
    """Declared task target files no step in the plan produces."""
    planned = {f for step in plan.steps for f in _effective_targets(step)}
    return tuple(f for f in task_targets if f not in planned)


# ---------------------------------------------------------------------------
# The preflight gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanViolation:
    """One structural defect found in a plan: which rule, and what tripped it."""

    check: str
    detail: str

    def __str__(self) -> str:
        return f"{self.check}: {self.detail}"


PREFLIGHT_CHECKS: Final[tuple[tuple[str, Callable[[Plan], tuple[str, ...]]], ...]] = (
    ("duplicate_step_ids", duplicate_step_ids),
    ("duplicate_sub_task_ids", duplicate_sub_task_ids),
    ("overlapping_sub_task_targets", overlapping_sub_task_targets),
    ("unsafe_target_paths", unsafe_target_paths),
    ("nodes_without_targets", nodes_without_targets),
)
"""The structural checks the live gate enforces, in report order.

Four of the nine eval checks are deliberately absent, and the reasons are
different in kind:

* :func:`implausible_context_files` and :func:`forward_references` need the repo
  file set. Reading it is I/O, and the workflow can only do I/O by scheduling an
  activity — a command-sequence change this gate is specifically designed to
  avoid (it must emit nothing on a clean plan so the committed replay histories
  stay valid). They remain eval-only, where the corpus supplies the file set.
* :func:`undersized_fan_outs` and :func:`uncovered_task_targets` are quality
  judgments rather than defects. A one-child fan-out runs correctly, and an
  uncovered declared target is a call about intent whose false positive costs
  the entire run (three expensive REASONING calls, then a halt). The eval
  harness scores them; the gate does not veto on them.
"""


def preflight_plan(plan: Plan) -> tuple[PlanViolation, ...]:
    """Run the gate checks over a plan; empty means the plan is accepted."""
    return tuple(
        PlanViolation(check=name, detail=detail)
        for name, find in PREFLIGHT_CHECKS
        for detail in find(plan)
    )


def violation_summary(violations: Sequence[PlanViolation]) -> str:
    """One-line rendering of a violation list, for a terminal result's ``error``."""
    return "; ".join(str(v) for v in violations)


def retry_prompt_section(
    violations: Sequence[PlanViolation], *, attempt: int, max_attempts: int
) -> str:
    """The violations, worded for the planner's next attempt.

    Appended to the planner's ``user_prompt`` — the same "here is what was wrong
    last time" shape the step block threads through ``prior_errors``, but pure
    string work, so a retry costs one more planner call and no extra activity.
    """
    lines = "\n".join(f"- {v}" for v in violations)
    return (
        f"\n\n## Plan rejected by structural validation "
        f"(attempt {attempt} of {max_attempts})\n\n"
        "The previous plan failed these deterministic checks. Return a complete, "
        "corrected plan that fixes every item below.\n\n"
        f"{lines}\n"
    )


def escalate_thinking(policy: ThinkingPolicy) -> ThinkingPolicy:
    """Thinking policy for a final planner attempt: enabled, at maximum effort."""
    return policy.model_copy(update={"enabled": True, "effort": "max"})


# ---------------------------------------------------------------------------
# Sanity-check revision splicing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class RevisedPlan:
    """The revision was accepted; ``plan`` is the spliced replacement."""

    plan: Plan


@dataclass(frozen=True, slots=True, kw_only=True)
class RevisionRejected:
    """The revision cannot be spliced; ``reason`` is the terminal-result wording."""

    reason: str


def splice_revision(
    plan: Plan,
    *,
    completed_through: int,
    revised_steps: Sequence[PlanStep],
    revision_count: int,
) -> RevisedPlan | RevisionRejected:
    """Replace the plan's remaining steps with a sanity check's revision.

    ``completed_through`` is the index of the last step that has run; steps up to
    and including it are kept, everything after is replaced by ``revised_steps``.

    Rejection *catches* the two ways the splice can go wrong instead of letting
    them happen. An over-cap splice would raise a pydantic ``ValidationError``
    from ``Plan(steps=...)`` — inside workflow code, where an ordinary exception
    is a workflow *task* failure that Temporal retries forever, i.e. a hung
    workflow, not a clean failure. And a structurally invalid revision would
    execute exactly the defects :func:`preflight_plan` exists to keep out. Both
    return a reason for the caller to turn into a terminal result.
    """
    if revision_count >= MAX_PLAN_REVISIONS:
        return RevisionRejected(
            reason=(
                f"Plan revision cap exceeded: {revision_count} revisions already applied "
                f"(max {MAX_PLAN_REVISIONS})"
            )
        )

    kept = plan.steps[: completed_through + 1]
    total = len(kept) + len(revised_steps)
    if total > MAX_PLAN_STEPS:
        return RevisionRejected(
            reason=(
                f"Revised plan would exceed the step cap: {len(kept)} completed + "
                f"{len(revised_steps)} revised = {total} steps (max {MAX_PLAN_STEPS})"
            )
        )

    revised = Plan(
        task_id=plan.task_id,
        steps=[*kept, *revised_steps],
        explanation=plan.explanation,
    )
    violations = preflight_plan(revised)
    if violations:
        return RevisionRejected(
            reason=f"Revised plan failed structural validation: {violation_summary(violations)}"
        )
    return RevisedPlan(plan=revised)
