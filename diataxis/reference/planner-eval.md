# Planner Evaluation Reference

## Eval Corpus Format

### Directory Structure

A corpus is a flat directory of JSON files. Each file defines one `EvalCase`.

```
corpus/
    case-001.json
    case-002.json
    case-003.json
```

Files are discovered by glob pattern `*.json`. Subdirectories are not scanned.
Files that fail validation are logged and skipped without aborting the run.
Cases are processed in sorted order by `case_id`.

### EvalCase Schema

Each JSON file must validate against the `EvalCase` model.

| Field | Type | Required | Description |
|---|---|---|---|
| `case_id` | `str` | yes | Unique identifier for the case. Used in result output and comparisons. |
| `task` | `TaskDefinition` | yes | The task definition passed to the planner. |
| `repo_root` | `str` | yes | Absolute path to the repository. Used by `list_repo_files` for file-existence checks. |
| `reference_plan` | `Plan \| null` | no | Known-good plan for reference. Not used by current evaluation checks. |
| `tags` | `list[str]` | no | Arbitrary tags for filtering cases. Default: `[]`. |

### TaskDefinition (within EvalCase)

| Field | Type | Required | Description |
|---|---|---|---|
| `task_id` | `str` | yes | Unique task identifier. |
| `description` | `str` | yes | Human-readable task description. |
| `target_files` | `list[str]` | no | Files the task must produce or modify. |
| `context_files` | `list[str]` | no | Files provided as context. |

### Example EvalCase JSON

```json
{
  "case_id": "case-001",
  "task": {
    "task_id": "add-cli-command",
    "description": "Add a new CLI command 'forge export' that exports a workflow result to JSON.",
    "target_files": ["src/forge/cli.py"],
    "context_files": ["src/forge/models.py"]
  },
  "repo_root": "/path/to/forge",
  "tags": ["cli", "code-generation"]
}
```


## Deterministic Checks

All checks are pure functions with the signature:

```python
check(plan: Plan, task: TaskDefinition, known_repo_files: set[str] | None) -> DeterministicCheckResult
```

| Check Name | What It Verifies | Requires `known_repo_files` | PASS Condition |
|---|---|---|---|
| `check_target_files_are_relative_paths` | No target or context file uses an absolute path or `..` traversal | no | All file references are relative with no `..` components |
| `check_step_ids_unique` | All step IDs in the plan are unique | no | No duplicate `step_id` values |
| `check_sub_task_ids_unique` | Sub-task IDs are unique within each step | no | No duplicate `sub_task_id` within any step |
| `check_sub_task_targets_non_overlapping` | Sub-tasks in a fan-out step do not share target files | no | No file appears in two sub-task `target_files` within the same step |
| `check_context_files_plausible` | Context files exist in the repo or are produced by an earlier step | yes | All context files are in `known_repo_files` or in a prior step's targets |
| `check_no_forward_references` | No step references a file produced only by a later step | yes | No context file is exclusively produced by a step with a higher index |
| `check_all_task_targets_covered` | Every task target file appears in at least one plan step | no | All `task.target_files` appear in step or sub-task `target_files` |
| `check_non_fanout_steps_have_targets` | Non-fan-out steps have non-empty `target_files` | no | Every step without `sub_tasks` has `target_files` |
| `check_fanout_steps_have_min_subtasks` | Fan-out steps have at least 2 sub-tasks | no | Every step with `sub_tasks` has `len(sub_tasks) >= 2` |

Checks that require `known_repo_files` return `CheckStatus.SKIP` when no file set
is provided.


## Data Models

### DeterministicCheckResult

| Field | Type | Description |
|---|---|---|
| `check_name` | `str` | Name of the check function. |
| `status` | `CheckStatus` | `"pass"`, `"fail"`, or `"skip"`. |
| `message` | `str` | Human-readable summary. |
| `details` | `list[str]` | Specific items that caused failure (e.g. file paths, step IDs). Empty on PASS. |

### DeterministicResult

| Field | Type | Description |
|---|---|---|
| `checks` | `list[DeterministicCheckResult]` | Results for all checks. |
| `all_passed` | `bool` | True if no check returned `CheckStatus.FAIL`. SKIP does not count as failure. |

### CheckStatus

| Value | Meaning |
|---|---|
| `"pass"` | Check succeeded. |
| `"fail"` | Check found a violation. |
| `"skip"` | Check was not run (missing `known_repo_files`). |

### JudgeCriterion

| Value | Description |
|---|---|
| `"completeness"` | Does the plan cover all required target files and task requirements? |
| `"granularity"` | Are steps appropriately sized — not too coarse or too fine? |
| `"ordering"` | Are steps in a logical order where each can build on prior steps? |
| `"context_quality"` | Do steps reference appropriate context files? |
| `"fan_out_appropriateness"` | Is fan-out used for genuinely independent work? If not used, would it have been appropriate? |
| `"explanation_quality"` | Does the plan explanation clearly describe the decomposition strategy? |

### JudgeScore

| Field | Type | Description |
|---|---|---|
| `criterion` | `JudgeCriterion` | The criterion being scored. |
| `score` | `int` | 1–5 scale. 1 = poor, 5 = excellent. |
| `rationale` | `str` | Explanation of the score. |

### JudgeVerdict

| Field | Type | Description |
|---|---|---|
| `scores` | `list[JudgeScore]` | One score per criterion. |
| `overall_assessment` | `str` | Summary assessment of plan quality. |

### EvalCase

See [EvalCase Schema](#evalcase-schema) above.

### PlanEvalResult

| Field | Type | Description |
|---|---|---|
| `case_id` | `str` | The evaluated case ID. |
| `plan` | `Plan` | The plan that was evaluated. |
| `deterministic` | `DeterministicResult` | Results of all deterministic checks. |
| `judge` | `JudgeVerdict \| null` | LLM judge verdict. `None` if judge was not run. |
| `timestamp` | `datetime` | UTC timestamp of evaluation. |

### EvalRunRecord

| Field | Type | Description |
|---|---|---|
| `run_id` | `str` | Unique run identifier. |
| `timestamp` | `datetime` | UTC timestamp of the run. |
| `model_name` | `str` | Model that produced the plans. |
| `judge_model` | `str \| null` | Model used as judge, if any. |
| `results` | `list[PlanEvalResult]` | Evaluation results for each case. |

### EvalComparison

| Field | Type | Description |
|---|---|---|
| `baseline_run_id` | `str` | Run ID of the baseline. |
| `candidate_run_id` | `str` | Run ID of the candidate being compared. |
| `regressions` | `list[str]` | Case IDs where the candidate scored worse. |
| `improvements` | `list[str]` | Case IDs where the candidate scored better. |
| `summary` | `str` | Human-readable comparison summary. |


## LLM-as-Judge Configuration

| Parameter | Default | Description |
|---|---|---|
| Default judge model | `claude-sonnet-4-5-20250929` | Overridable via `--judge-model`. |
| Max tokens | `4096` | Fixed; not user-configurable. |
| Score scale | 1–5 per criterion | Integer values only. |
| Criteria count | 6 | See `JudgeCriterion` table above. |

Run comparison threshold: a change of more than 0.5 in average score across
criteria is considered a meaningful regression or improvement when comparing runs.


## Result Storage

Evaluation run records are saved as JSON files in
`$XDG_DATA_HOME/forge/eval/` (default: `~/.local/share/forge/eval/`).

File naming: `{run_id}.json`

The directory is created automatically on first save.


## CLI: forge eval-planner

```
forge eval-planner --corpus-dir PATH [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--corpus-dir PATH` | required | — | Directory containing eval case JSON files. |
| `--plans-dir PATH` | optional | — | Directory containing plan JSON files. If omitted, plans are generated by running the planner against each case. |
| `--judge / --no-judge` | flag | `--no-judge` | Run LLM-as-judge scoring. |
| `--judge-model TEXT` | optional | `claude-sonnet-4-5-20250929` | Model to use as judge. |
| `--dry-run` | flag | off | List cases without evaluating. |
| `--output-dir PATH` | optional | — | Directory to save the `EvalRunRecord` JSON file. |
| `--json` | flag | off | Output results as JSON instead of formatted text. |

**Exit codes:**

- `0`: all deterministic checks passed (or `--dry-run` used).
- Non-zero: at least one deterministic check failed, or no results were produced.

See [How to Run Evaluations](../howto/run-evaluations.md) for usage recipes. See
[Planner Evaluation](../explanation/planner-eval.md) for design background.
