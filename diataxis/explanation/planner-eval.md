+++
title = "Planner Evaluation"
weight = 161
description = "How to evaluate planner output quality using deterministic structural checks and LLM-as-judge scoring."
topic = "planner-eval"
covers = [
    "Why planner evaluation matters — plan quality bounds everything downstream",
    "The two evaluation modes: deterministic structural checks and LLM-as-judge",
    "What deterministic checks verify: file coverage, step ordering, constraint adherence",
    "How LLM-as-judge scoring works: scoring criteria, judge prompt, score aggregation",
    "The eval corpus: how test cases are defined and organized",
]
detail = "Connect planner evaluation to the principle 'planning is the hard part.' Explain the two-mode approach and why both are needed."
+++
Forge's core principle — "planning is the hard part" — has a direct implication for
testing: if plan quality bounds everything that runs downstream, then measuring plan
quality is the highest-leverage form of testing available. A plan that misorders
steps, omits target files, or constructs invalid file references will cause failures
at execution time that are expensive to diagnose. The planner evaluation framework
exists to catch these problems before execution, and to provide a stable signal for
comparing planner behavior across model versions or prompt changes.

For background on how planning works and why it matters, see
[Task Decomposition and Execution](task-decomposition/).


## Two Evaluation Modes

The framework uses two complementary approaches: deterministic structural checks
and LLM-as-judge scoring. These are not alternatives — they address different
questions.

**Deterministic checks** ask: is the plan structurally valid? They are fast, free,
and produce binary pass/fail results. A plan that fails a deterministic check is
broken in a way that would cause a predictable failure at execution time. These
checks run on every evaluation, unconditionally.

**LLM-as-judge scoring** asks: is the plan good? A plan can be structurally valid
but still poor — all steps present, but ordered incorrectly; all target files
covered, but with steps that are too coarse to execute cleanly. The LLM judge
evaluates qualities that require semantic understanding: whether step granularity
is appropriate, whether context files are relevant, whether fan-out is used where
it would help. Judge scoring is optional because it costs tokens and takes time.

The two modes compose: run deterministic checks to gate on structural validity,
then optionally run the judge to assess plan quality.


## What Deterministic Checks Verify

Deterministic checks are pure functions over a `Plan` and its `TaskDefinition`. They fall into a few categories, each corresponding to a class of planner error that would cause predictable execution failures. Path-safety checks reject absolute paths and `..` traversal. Structural-invariant checks catch duplicate IDs and sub-task target overlap. Coverage checks ensure every file the task declared as a target ends up in some step. Ordering checks prevent a step from depending on a file that a later step produces. File-plausibility checks verify that context references exist in the repository or are produced by an earlier step — this is the one category that requires the check runner to know the repository's file set, and it skips when that set is not available.

These checks encode execution invariants, not style preferences. A plan that fails a deterministic check is not merely inelegant — it will fail in a specific, foreseeable way when run. For the exact check names, what each one verifies, and the `known_repo_files` dependency, see the [planner evaluation reference](../reference/planner-eval/).


## How LLM-as-Judge Scoring Works

The judge prompt presents the full plan — step descriptions, target files, context files, sub-task structure — alongside the original task definition and a rubric. The judge scores the plan along several dimensions that deterministic checks cannot reach: whether the decomposition is complete, whether step sizes are appropriate, whether the ordering lets each step build on its predecessors, whether context choices are plausible, whether fan-out is used where it helps, and whether the plan's own explanation is coherent. Each dimension gets a 1–5 score and a written rationale; the judge also emits an overall assessment. For the canonical rubric and the full criterion definitions, see the [planner evaluation reference](../reference/planner-eval/).

The judge defaults to `claude-sonnet-4-5-20250929` and runs synchronously rather than via the batch API — evaluation is an interactive workflow, not a pipeline stage. Any model the Forge LLM provider can route to may be substituted via `--judge-model`.


## The Eval Corpus

An eval corpus is a directory of JSON files, each defining one `EvalCase`. A case
contains a `TaskDefinition` (the task the planner was given), a `repo_root` path
(for repository file discovery), optional tags (for filtering), and an optional
`reference_plan` (a known-good plan for reference, not used by the current checks
but available for future comparison logic).

Cases are discovered by scanning the corpus directory for `*.json` files. Each
file is parsed against the `EvalCase` schema; invalid files are logged and skipped.
Cases are processed in sorted order by `case_id` for deterministic output.

Corpus organization is left to the team. A reasonable convention is to group cases
by task domain or complexity level. Tags can be used to filter subsets.

Evaluation results are stored as `EvalRunRecord` JSON files under
`$XDG_DATA_HOME/forge/eval/` (default: `~/.local/share/forge/eval/`), enabling
comparison between runs.


## Comparing Runs

The `compare_runs` function compares two `EvalRunRecord` objects and identifies
regressions and improvements. Deterministic pass/fail is the primary comparison
signal: a case that passes in the baseline but fails in the candidate is a
regression. When both runs include judge verdicts, runs are also compared by
average score across criteria, with a threshold of 0.5 score points to distinguish
meaningful change from noise.

For technical details, see the [Planner Evaluation reference](../reference/planner-eval/).
For step-by-step instructions, see [How to Run Evaluations](../howto/run-evaluations/).
