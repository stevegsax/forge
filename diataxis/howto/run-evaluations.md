# How to Run Evaluations

This guide shows you how to run planner evaluations against an eval corpus, create
new eval cases, enable LLM-as-judge scoring, interpret results, and compare runs.

For background on how the evaluation framework works, see
[Planner Evaluation](../explanation/planner-eval.md). For the full schema and CLI
reference, see the [Planner Evaluation reference](../reference/planner-eval.md).


## Run evaluations against a corpus

1. Run deterministic checks against all cases in a corpus directory:

    ```bash
    forge eval-planner --corpus-dir /path/to/corpus
    ```

    Output shows one result block per case:

    ```
    case-001: add-cli-command
      deterministic: PASS (8 checks)

    case-002: refactor-context-assembly
      deterministic: FAIL
        check_all_task_targets_covered: 1 task target(s) not covered by plan.
          - src/forge/context/ranking.py
    ```

    The exit code is non-zero if any deterministic check fails.

2. To list cases without evaluating (useful for verifying the corpus loads
   correctly):

    ```bash
    forge eval-planner --corpus-dir /path/to/corpus --dry-run
    ```

    Output:

    ```
    Found 3 eval case(s):
      case-001: Add a new CLI command 'forge export' [cli, code-generation]
      case-002: Refactor context assembly to support lazy loading [context]
      case-003: Add SHA-256 duplicate detection to OCR pipeline [ocr]
    ```


## Create a new eval case

1. Create a JSON file in your corpus directory. The `case_id` should be unique
   across the corpus. Match `repo_root` to the repository where the planner will
   run.

    ```json
    {
      "case_id": "case-004",
      "task": {
        "task_id": "add-export-command",
        "description": "Add a 'forge export' CLI command that writes a workflow result to a JSON file.",
        "target_files": ["src/forge/cli.py"],
        "context_files": ["src/forge/models.py", "src/forge/store.py"]
      },
      "repo_root": "/path/to/forge",
      "tags": ["cli"]
    }
    ```

2. Verify it loads correctly:

    ```bash
    forge eval-planner --corpus-dir /path/to/corpus --dry-run
    ```

3. If you have a known-good plan for the case, add it as `reference_plan`. This
   is optional and not used by current checks, but preserves reference plans for
   future comparison tooling.


## Run evaluations with LLM-as-judge scoring

LLM-as-judge scoring requires a configured Anthropic API key. Each judge call
costs tokens.

1. Run with the default judge model:

    ```bash
    forge eval-planner \
      --corpus-dir /path/to/corpus \
      --judge
    ```

    Judge output appears below each deterministic result:

    ```
    case-001: add-cli-command
      deterministic: PASS (8 checks)
      judge:
        completeness:           5  All target files covered.
        granularity:            4  Steps are well-sized; could split CLI and tests.
        ordering:               5  Dependencies respected.
        context_quality:        4  Good context selection.
        fan_out_appropriateness: 3  Fan-out not used; would not help here.
        explanation_quality:    5  Clear decomposition rationale.
      overall: Solid plan. Minor improvement possible by separating CLI from test step.
    ```

2. To use a different judge model:

    ```bash
    forge eval-planner \
      --corpus-dir /path/to/corpus \
      --judge \
      --judge-model claude-opus-4-5-20251201
    ```


## Save results to a file

To compare runs over time, save results to a directory.

1. Run and save:

    ```bash
    forge eval-planner \
      --corpus-dir /path/to/corpus \
      --judge \
      --output-dir /path/to/results
    ```

    Output includes the saved path:

    ```
    Results saved to /path/to/results/a1b2c3d4.json
    ```

2. Results are also saved automatically to `~/.local/share/forge/eval/` when
   `--output-dir` is specified.


## Get results as JSON

Use `--json` to get machine-readable output for scripting or further processing.

```bash
forge eval-planner \
  --corpus-dir /path/to/corpus \
  --judge \
  --json
```

The output is a JSON array of `PlanEvalResult` objects. See the
[Planner Evaluation reference](../reference/planner-eval.md#planevalresult) for
the full schema.


## Interpret evaluation results

**Deterministic PASS**: the plan is structurally valid. All file references are
relative, IDs are unique, coverage is complete, and no forward references exist.
This is a necessary condition for execution success.

**Deterministic FAIL**: the plan has a structural problem. The `details` list shows
the specific violations (e.g. the file paths or step IDs that caused the failure).
Fix the planner prompt or planner configuration before running the task.

**SKIP checks**: checks that require repository file information (`check_context_files_plausible`,
`check_no_forward_references`) are skipped when no `repo_root` produces a valid
git file list. This is not a failure — it means those two checks did not run.

**Judge scores**: scores below 3 on any criterion indicate a meaningful quality
problem. A score of 1 or 2 on `completeness` or `ordering` is likely to cause
execution failures even when deterministic checks pass. Judge scores are advisory;
they do not affect the exit code.


## Evaluate plans from a directory

If you have pre-generated plan JSON files (e.g. from a batch planner run), you
can evaluate them without re-running the planner:

```bash
forge eval-planner \
  --corpus-dir /path/to/corpus \
  --plans-dir /path/to/plans
```

Plan files must be named `{case_id}.json` to match cases. Cases without a
matching plan file are skipped with a warning.
