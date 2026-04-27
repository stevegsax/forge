+++
title = "The Golden Path: A Planned Task End-to-End"
weight = 22
description = "Step-by-step walkthrough of a planned, multi-step code generation task from CLI submission through committed output."
topic = "golden-path"
covers = [
    "Submitting a planned task via the CLI",
    "Observing the planner decompose the task into steps",
    "Watching context assembly discover relevant files",
    "Seeing the exploration loop request additional context",
    "Examining the assembled prompt sent to the LLM",
    "Tracing structured output through edit application",
    "Watching validation run and a retry with error feedback",
    "Seeing the final commit in the worktree",
    "Inspecting the run with forge status --verbose",
]
detail = "End-to-end narrative walkthrough using a concrete, realistic example (e.g., adding a new CLI command to an existing Python project). Each step shows the actual CLI command or observable output, then briefly states what happened internally. Show results at every step — never more than 2-3 actions without visible output. The tutorial should take ~20 minutes to read and follow."
+++
In this tutorial, we will submit a planned task to Forge and follow it through planning, context assembly, exploration, code generation, validation, and commit. The task is adding a new `forge stats` CLI command to an existing Python project. Along the way, we will see:

- The planner decompose the task into ordered steps
- Context assembly discover relevant files automatically
- The exploration loop pull additional context on demand
- The LLM produce structured edits applied to source files
- Validation catch a lint error and trigger a retry with error feedback
- The successful commit and post-run inspection

By the end, we will have a committed feature branch with a working CLI command, built entirely through Forge's orchestration pipeline. For background on what Forge is and how its components fit together, see [System Overview](../explanation/system-overview/).

## Prerequisites

Before we start, two things must be running:

**1. Temporal server.** Forge uses Temporal for workflow orchestration. Start it if it is not already running:

```bash
temporal server start-dev
```

**2. Forge worker.** The worker polls Temporal for queued workflows and executes them. Start it in a separate terminal:

```bash
forge worker
```

You should see output confirming the worker is connected:

```text
14:02:11 INFO     forge.worker — Worker started on task queue 'forge-task-queue'
14:02:11 INFO     forge.worker — Registered workflows: ForgeTaskWorkflow, BatchPollerWorkflow, ...
14:02:11 INFO     forge.worker — Registered activities: 34 activities
```

We also need a Python project with a `src/` layout, a `pyproject.toml`, and an existing CLI module. For this tutorial, assume we are working in a project called `myapp` with this structure:

```text
myapp/
├── pyproject.toml
├── src/
│   └── myapp/
│       ├── __init__.py
│       ├── cli.py          # Existing CLI entry point using click
│       ├── models.py       # Data models
│       └── store.py        # SQLite data access layer
└── tests/
    ├── conftest.py
    └── test_cli.py
```

## Step 1: Submit the task

We submit a planned task with `forge run --plan`. The `--plan` flag tells Forge to invoke a planning LLM before execution, decomposing the task into ordered steps:

```bash
forge run \
    --task-id add-stats-command \
    --description "Add a 'stats' CLI command that queries the SQLite store for
        run counts, average token usage, and success rate, then prints a
        formatted summary table to the console" \
    --plan
```

Notice that we did not specify `--target-file`. When planning is enabled, the planner determines which files each step should create or modify. Forge prints the workflow ID and begins execution:

```text
14:03:45 INFO     forge.cli — Submitted workflow: forge-task-add-stats-command
14:03:45 INFO     forge.cli — Waiting for result...
```

## Step 2: Observe the planner decompose the task

The planner receives the full task description, a PageRank-ranked repository map, and project instructions. It uses a reasoning-tier model with extended thinking to produce a plan.

After a few seconds, the planner output appears:

```text
14:04:02 INFO     forge.workflows — Plan created: 3 steps
14:04:02 INFO     forge.workflows — step-1: Add stats query functions to store.py
14:04:02 INFO     forge.workflows — step-2: Add stats CLI command to cli.py
14:04:02 INFO     forge.workflows — step-3: Add tests for the stats command
```

The planner decided on three sequential steps. Each step has a description, target files, and explicit boundaries (what not to touch). Steps execute in order, and each commits on success before the next step begins.

For more on how the planner decomposes tasks, see [Task Decomposition](../explanation/task-decomposition/).

## Step 3: Watch context assembly discover files

Execution begins with step-1. The first activity is `assemble_context`, which builds the prompt the LLM will receive.

Forge discovers relevant files automatically using import graph analysis. It traces imports from the target file (`src/myapp/store.py`), ranks all discovered files by structural importance using PageRank, and packs them into a token budget:

```text
14:04:05 INFO     forge.activities.context — Step step-1: assembling context
14:04:05 INFO     forge.code_intel.graph — Import graph: 12 modules, 18 edges
14:04:05 INFO     forge.code_intel.graph — PageRank top-5: models.py (0.19), store.py (0.15),
                  cli.py (0.12), __init__.py (0.08), conftest.py (0.06)
14:04:05 INFO     forge.code_intel.budget — Token budget: 100,000
14:04:05 INFO     forge.code_intel.budget — Packed: 3 files, 4,218 tokens (4.2% of budget)
14:04:05 INFO     forge.activities.context — Context assembled: system=6,842 tokens,
                  user=127 tokens
```

The system prompt now contains the role statement, output format requirements, project instructions, the ranked repository map, the task description, and the full contents of `store.py`. The remaining budget is available for the exploration loop.

For more on how context assembly selects and ranks files, see [Context Assembly](../explanation/context-assembly/).

## Step 4: See the exploration loop request context

Before calling the generation LLM, Forge runs an exploration loop. A lightweight classification-tier LLM examines the task and assembled context, then requests additional files it needs to see:

```text
14:04:08 INFO     forge.activities.exploration — Exploration round 1/10
14:04:09 INFO     forge.activities.exploration — LLM requested 2 providers:
14:04:09 INFO     forge.activities.exploration —   read_file: src/myapp/models.py
14:04:09 INFO     forge.activities.exploration —   symbol_list: src/myapp/cli.py
14:04:09 INFO     forge.activities.exploration — Fulfilled 2 requests (1,847 tokens)
14:04:10 INFO     forge.activities.exploration — Exploration round 2/10
14:04:11 INFO     forge.activities.exploration — LLM returned empty request list — exploration complete
14:04:11 INFO     forge.activities.exploration — Exploration total: 2 rounds, 1,847 tokens
```

The exploration LLM read the data models (to understand what structures exist) and the CLI module's public API (to see the existing command pattern). After one round of context gathering, it signaled readiness by returning an empty request list. These results are appended to the system prompt as an "Exploration Results" section.

For more on how the exploration loop works and when it terminates, see [Context Assembly](../explanation/context-assembly/).

## Step 5: Examine the assembled prompt structure

Forge assembles the final system prompt from eleven sections ordered for cache efficiency — stable content first, volatile content last. The generation LLM receives everything it needs in a single call.

For the section-by-section table showing what each section contains and where cache breakpoints are placed, see the [Prompt Construction Reference](../reference/prompt-construction/). For the design rationale behind that ordering, see [Prompt Construction](../explanation/prompt-construction/).

## Step 6: Trace the LLM's structured response

Forge calls the generation-tier LLM. Rather than parsing free-form text, it uses Anthropic's tool-use feature to receive a structured `LLMResponse`:

```text
14:04:14 INFO     forge.activities.llm — Calling generation LLM (claude-sonnet-4-5-20250929)
14:04:22 INFO     forge.activities.llm — Response received: 1 edit, 0 new files
14:04:22 INFO     forge.activities.llm — Tokens: input=8,689 output=1,243
                  cache_read=5,104 cache_write=3,585 latency=7.8s
```

The response contains one edit to `store.py` -- three new query functions added to the existing module:

```text
14:04:22 INFO     forge.activities.output — Edits to apply:
14:04:22 INFO     forge.activities.output —   store.py: 3 edit operations
```

The LLM produced search/replace pairs rather than rewriting the entire file. Each pair identifies an exact location in the existing file and provides the replacement text.

For more on why Forge uses search/replace edits, see [Output Processing](../explanation/output-processing/).

## Step 7: Watch edit application

The `write_output` activity applies the edits sequentially. Each edit uses a four-level matching fallback chain to tolerate minor formatting differences:

```text
14:04:22 INFO     forge.activities.output — Applying edit 1/3 to store.py: exact match
14:04:22 INFO     forge.activities.output — Applying edit 2/3 to store.py: exact match
14:04:22 INFO     forge.activities.output — Applying edit 3/3 to store.py: exact match
14:04:22 INFO     forge.activities.output — All edits applied successfully
```

All three edits matched exactly on the first level. When the LLM's output differs slightly in whitespace or indentation, the fallback chain tries whitespace-normalized, indentation-normalized, and fuzzy matching before giving up.

For the full fallback chain and its thresholds, see [Output Processing](../explanation/output-processing/).

## Step 8: See validation catch an error

After edits are applied, the `validate_output` activity runs deterministic checks. For code generation tasks, this means `ruff` lint and `ruff` format:

```text
14:04:23 INFO     forge.activities.validate — Running validation: ruff lint, ruff format
14:04:23 ERROR    forge.activities.validate — ruff lint: FAILED
14:04:23 INFO     forge.activities.validate —   src/myapp/store.py:47:5 F811 Redefinition
                  of unused `total_runs` from line 38
14:04:23 INFO     forge.activities.validate — ruff format: passed
14:04:23 INFO     forge.activities.validate — Validation result: 1 passed, 1 failed
```

The LLM introduced a variable name collision. The `evaluate_transition` activity maps this to `FAILURE_RETRYABLE` because the step has retry attempts remaining:

```text
14:04:23 INFO     forge.activities.transition — Transition: FAILURE_RETRYABLE (attempt 1/2)
14:04:23 INFO     forge.workflows — Resetting uncommitted changes for retry
```

For more on validation checks and transition signals, see [Validation and Retries](../explanation/validation-and-retries/).

## Step 9: Observe the retry with error feedback

Forge does not retry blind. It builds an error section with the lint output and AST-derived code context showing exactly where the problem is:

```text
14:04:24 INFO     forge.activities.context — Injecting error context from previous attempt
14:04:24 INFO     forge.activities.context — Error section: 1 error, 847 tokens
14:04:24 INFO     forge.activities.context — AST context: function 'get_run_statistics'
                  enclosing line 47
```

The error section the LLM sees includes the lint error and the enclosing function with the offending line marked:

    Previous Attempt Errors

    The previous attempt failed validation. Fix these errors:

    ruff lint errors:
    src/myapp/store.py:47:5 F811 Redefinition of unused `total_runs` from line 38

    Context around error:
    def get_run_statistics(self, since: datetime | None = None) -> dict:
        """Query aggregate run statistics."""
        total_runs = self._count_runs(since)       # line 38
        ...
        total_runs = self._count_all_runs(since)   # <-- ERROR (line 47)

For more on how error context is injected, see [Validation and Retries](../explanation/validation-and-retries/).

The generation LLM now sees the exact error, the enclosing function, and the offending line. It produces a corrected edit:

```text
14:04:27 INFO     forge.activities.llm — Calling generation LLM (retry attempt 2/2)
14:04:32 INFO     forge.activities.llm — Response received: 1 edit, 0 new files
14:04:32 INFO     forge.activities.llm — Tokens: input=9,536 output=487
                  cache_read=8,689 cache_write=847 latency=4.9s
```

The retry was faster (4.9s vs 7.8s): 8,689 input tokens were served from cache and only 847 new tokens were written. For more on prompt caching, see [Context Assembly](../explanation/context-assembly/).

## Step 10: See the successful validation and commit

The corrected edit is applied and validation runs again:

```text
14:04:33 INFO     forge.activities.output — Applying edit 1/1 to store.py: exact match
14:04:33 INFO     forge.activities.validate — Running validation: ruff lint, ruff format
14:04:33 INFO     forge.activities.validate — ruff lint: passed
14:04:33 INFO     forge.activities.validate — ruff format: passed
14:04:33 INFO     forge.activities.validate — Validation result: 2 passed, 0 failed
14:04:34 INFO     forge.activities.transition — Transition: SUCCESS
14:04:34 INFO     forge.activities.git — Committed step-1: Add stats query functions to store.py
```

Step-1 is committed to the worktree branch. Forge moves on to step-2 (adding the CLI command) and step-3 (adding tests), each following the same construct-send-receive-validate-transition cycle. Steps 2 and 3 succeed on their first attempt.

```text
14:04:48 INFO     forge.activities.transition — step-2: SUCCESS
14:04:48 INFO     forge.activities.git — Committed step-2: Add stats CLI command to cli.py
14:05:03 INFO     forge.activities.transition — step-3: SUCCESS
14:05:03 INFO     forge.activities.git — Committed step-3: Add tests for the stats command
14:05:03 INFO     forge.workflows — Task complete: 3/3 steps succeeded
```

For more on how the universal workflow step drives each phase, see [The Universal Workflow Step](../explanation/workflow-step/).

## Step 11: Inspect the run

The final output shows a summary:

```text
Task: add-stats-command
Status: SUCCESS
Steps: 3/3 completed
Worktree: /tmp/forge-worktrees/add-stats-command

Commits:
  a1b2c3d step-1: Add stats query functions to store.py
  d4e5f6a step-2: Add stats CLI command to cli.py
  b7c8d9e step-3: Add tests for the stats command

Validation: 6 checks passed, 1 retry (step-1: ruff lint)
```

For a deeper look, use `forge status --verbose`:

```bash
forge status --workflow-id forge-task-add-stats-command --verbose
```

This queries the observability store and prints the full interaction history:

```text
Workflow: forge-task-add-stats-command
Status:   SUCCESS
Duration: 78.2s
Steps:    3

Step 1: Add stats query functions to store.py
  Attempts:  2 (1 retry)
  Model:     claude-sonnet-4-5-20250929
  Tokens:    input=18,225 output=1,730 cache_read=13,793 cache_write=4,432
  Context:   12 modules discovered, 3 packed, 4,218 tokens (4.2% budget)
  Explore:   2 rounds, 2 providers, 1,847 tokens
  Latency:   12.7s (attempt 1) + 4.9s (attempt 2)
  Retry:     ruff lint F811 — fixed on attempt 2

Step 2: Add stats CLI command to cli.py
  Attempts:  1
  Model:     claude-sonnet-4-5-20250929
  Tokens:    input=9,104 output=1,892 cache_read=7,842 cache_write=1,262
  Context:   12 modules discovered, 4 packed, 5,631 tokens (5.6% budget)
  Explore:   1 round, 1 provider, 923 tokens
  Latency:   6.4s

Step 3: Add tests for the stats command
  Attempts:  1
  Model:     claude-sonnet-4-5-20250929
  Tokens:    input=11,847 output=2,156 cache_read=9,104 cache_write=2,743
  Context:   12 modules discovered, 5 packed, 7,204 tokens (7.2% budget)
  Explore:   2 rounds, 3 providers, 2,412 tokens
  Latency:   8.1s

Totals:
  Input tokens:  39,176
  Output tokens: 5,778
  Cache reads:   30,739 (78.5% cache hit rate)
  Wall time:     78.2s
```

The cache hit rate across steps is 78.5%. For more on prompt caching, see [Context Assembly](../explanation/context-assembly/).

For more on observability and debugging, see [How to Debug a Workflow](../howto/debug-workflow/).

## What you built

We submitted a single task description and Forge produced:

- A three-step plan decomposing the task into store queries, CLI command, and tests
- Automatic context discovery via import graph analysis and PageRank ranking
- LLM-guided exploration that pulled in data models and CLI patterns on demand
- Structured search/replace edits applied to existing files
- A validation failure caught by `ruff` lint, retried with error feedback that pinpointed the exact function and line
- Three committed steps on a feature branch, ready for human review

The worktree branch contains the full commit history. From here, you review the changes and merge to `main` when ready -- Forge never auto-merges.

## Where to go next

- [How to Submit Tasks](../howto/submit-tasks/) -- single-step execution, JSON task files, fan-out, and other submission modes
- [Context Assembly](../explanation/context-assembly/) -- how file discovery, ranking, and token budgets work
- [The Universal Workflow Step](../explanation/workflow-step/) -- the five-phase pattern that drives every operation
- [Task Decomposition](../explanation/task-decomposition/) -- how the planner breaks tasks into steps and sub-tasks
- [Output Processing](../explanation/output-processing/) -- edit application and the four-level matching fallback chain
- [Validation and Retries](../explanation/validation-and-retries/) -- error-aware retries and AST-derived context
- [How to Debug a Workflow](../howto/debug-workflow/) -- inspecting prompts, tokens, and validation failures
