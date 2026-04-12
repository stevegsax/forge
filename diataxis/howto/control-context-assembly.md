# How to Control Context Assembly

This guide shows you how to control what context Forge includes in the prompt sent to the LLM. Each section addresses a specific task.

For background on how context assembly works, see [Context Assembly (explanation)](../explanation/context-assembly.md). For the full list of flags, data models, and provider specifications, see [Context Assembly Reference](../reference/context-assembly.md).

## How to Include Specific Files as Context

Use `--context-file` to add files that are not reachable through the import graph -- configuration files, documentation, test fixtures, data samples, or non-Python files.

```bash
forge run \
    --task-id add-api-endpoint \
    --description "Add a /health endpoint to the API server" \
    --target-file src/myapp/server.py \
    --context-file docs/api-spec.yaml \
    --context-file tests/fixtures/sample_response.json
```

The flag is repeatable. Each file is included at priority 6 (lowest), packed into the token budget after target files, dependencies, and the repo map. If the budget is full, manually specified files may be truncated.

To verify that the files were included, use `forge status --verbose` after the run completes (see [How to Check Token Budget Utilization](#how-to-check-token-budget-utilization)).

## How to Include Dependency Contents Upfront

By default, Forge includes only target file contents and the repo map in the initial prompt. Dependency contents are omitted -- the LLM can pull them on demand through the exploration loop.

To include direct import contents and transitive import signatures upfront, use `--include-deps`:

```bash
forge run \
    --task-id refactor-validation \
    --description "Extract validation logic into a separate module" \
    --target-file src/forge/activities/validate.py \
    --include-deps
```

This changes the initial prompt to include:

- Full content of files directly imported by `validate.py` (priority 3)
- Extracted signatures of transitively imported files, ranked by PageRank (priority 4)

Use `--include-deps` when the task involves cross-module changes where the LLM needs to see dependency implementations, not just interfaces. For tasks that modify a single file, the default (no deps) is typically sufficient.

## How to Disable the Exploration Loop

The exploration loop lets the LLM request additional context before generating output. To disable it entirely:

```bash
forge run \
    --task-id simple-change \
    --description "Fix the typo in the docstring" \
    --target-file src/myapp/utils.py \
    --no-explore
```

To keep exploration enabled but limit the number of rounds:

```bash
forge run \
    --task-id add-feature \
    --description "Add retry logic to the HTTP client" \
    --target-file src/myapp/http.py \
    --max-exploration-rounds 3
```

Setting `--max-exploration-rounds 0` has the same effect as `--no-explore`.

Disabling or limiting exploration is useful for simple tasks where the target file contents and repo map provide sufficient context, or when you want faster execution with fewer LLM calls.

## How to Diagnose Why a File Was or Was Not Included

Run the task with `--log-messages` to capture the full prompt sent to the LLM:

```bash
forge run \
    --task-id my-task \
    --description "Update the data model" \
    --target-file src/myapp/models.py \
    --log-messages
```

After the run, inspect the logged messages directory. The system prompt is written as a file in `~/.local/state/forge/messages/`. Open the system prompt file and search for the file path in question.

A file can be absent from the prompt for several reasons:

- **Not in the import graph.** The file is not imported (directly or transitively) by any target file. Use `--context-file` to include it manually.
- **Below the token budget cutoff.** The file was discovered but did not fit within the token budget. Check budget utilization (see next section). Consider increasing `--token-budget` or reducing the number of target files.
- **Excluded by progressive disclosure.** Dependencies are not included upfront by default. Use `--include-deps` to include them, or rely on the exploration loop to pull them on demand.
- **Not a Python file.** Automatic discovery only traces Python imports. Non-Python files must be specified with `--context-file`.

To see what the exploration loop requested, check the exploration results section in the logged system prompt. Each provider response is labeled with the provider name and parameters.

## How to Check Token Budget Utilization

After a run completes, use `forge status` with the `--verbose` flag to see context assembly statistics:

```bash
forge status --verbose --task-id my-task
```

The verbose output includes a context stats section:

```
Context Assembly:
  Files discovered:          42
  Files included (full):      5
  Files included (sigs):     12
  Files truncated:            8
  Estimated tokens:       67,234
  Budget utilization:      67.2%
  Repo map tokens:         1,847
```

Key indicators:

- **Budget utilization above 90%** suggests the prompt is crowded. Important files may have been reduced to signatures or dropped. Consider increasing `--token-budget` or narrowing the task scope.
- **Budget utilization below 30%** with a task that involves many files suggests the import graph did not discover enough context. Check whether target files have the expected import relationships. Use `--context-file` or `--include-deps` to supplement.
- **Files truncated > 0** means some discovered files did not fit. These files were either reduced to signatures (if possible) or excluded entirely. The files are ranked by PageRank importance, so the most structurally central files are included first.
