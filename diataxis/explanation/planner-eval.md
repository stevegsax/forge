# Planner Evaluation

Forge's core principle — "planning is the hard part" — has a direct implication for
testing: if plan quality bounds everything that runs downstream, then measuring plan
quality is the highest-leverage form of testing available. A plan that misorders
steps, omits target files, or constructs invalid file references will cause failures
at execution time that are expensive to diagnose. The planner evaluation framework
exists to catch these problems before execution, and to provide a stable signal for
comparing planner behavior across model versions or prompt changes.

For background on how planning works and why it matters, see
[Task Decomposition and Execution](task-decomposition.md).


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

Each check is a pure function that takes a `Plan` and a `TaskDefinition` and
returns a `DeterministicCheckResult`. Some checks also accept a set of known
repository files to verify that context file references are plausible.

The checks address the most common classes of planner error:

- **Path safety**: all target and context file references must be relative paths
  without `..` traversal segments.
- **ID uniqueness**: step IDs must be unique across the plan; sub-task IDs must be
  unique within each step.
- **Fan-out correctness**: sub-tasks within a fan-out step must not share target
  files (overlapping targets would cause merge conflicts), and fan-out steps must
  have at least two sub-tasks.
- **Coverage**: every file listed in the task's `target_files` must appear as a
  target in at least one plan step.
- **Target presence**: non-fan-out steps must have non-empty `target_files`.
- **File plausibility**: context files must either exist in the repository or be
  produced by an earlier step.
- **Step ordering**: no step may reference a context file that is only produced by
  a later step (forward references).

These checks encode execution invariants: violations are not style issues but
conditions that would cause predictable failures.


## How LLM-as-Judge Scoring Works

The judge prompt presents the full plan — step descriptions, target files, context
files, sub-task structure — alongside the original task definition and scoring
criteria. The judge scores the plan on six criteria, each on a 1–5 scale:

- **completeness**: does the plan cover all required targets and requirements?
- **granularity**: are steps appropriately sized?
- **ordering**: are steps in a logical sequence where each can build on prior ones?
- **context_quality**: do steps reference appropriate context files?
- **fan_out_appropriateness**: is fan-out used where it would help, and not used
  where it would not?
- **explanation_quality**: does the plan explanation clearly describe the
  decomposition strategy?

The judge returns one `JudgeScore` per criterion with a numeric score and a
rationale, plus an `overall_assessment` string.

The default judge model is `claude-sonnet-4-5-20250929`. Any model accessible via
the Forge LLM provider can be substituted. The judge call is a synchronous LLM
call, not a batch call, because evaluation is an interactive workflow.


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

For technical details, see the [Planner Evaluation reference](../reference/planner-eval.md).
For step-by-step instructions, see [How to Run Evaluations](../howto/run-evaluations.md).
