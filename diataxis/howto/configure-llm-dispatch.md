# How to Configure LLM Dispatch

This guide shows how to control model routing and batch processing behavior in Forge. For background on how these mechanisms work, see [Model Routing and Batch Processing](../explanation/llm-dispatch.md). For a full list of flags and configuration fields, see the [reference](../reference/llm-dispatch.md).

---

## Run a task in sync mode

By default, `forge run` uses batch mode — it submits requests to the Anthropic Batch API and waits for results via Temporal signals. To use the synchronous Messages API instead:

```
forge run --sync \
  --task-id my-task \
  --description "Add a docstring to parse_config" \
  --target-file src/forge/config.py
```

The `--sync` flag causes `call_llm` to call `client.messages.create()` directly and block until the response arrives. The workflow completes as soon as the API call returns rather than waiting for a batch poller cycle.

Use `--sync` when:

- You need a result immediately (e.g., interactive debugging)
- The batch poller has not been started or is not running
- The task involves a single short call where the cost reduction is not material

---

## Check batch job status

Batch jobs are recorded in the `batch_jobs` table in the observability store. Use `forge status` to inspect recent runs and their associated interactions:

```
forge status --limit 5
```

Example output:

```
workflow_id                          task_id        status     created_at
------------------------------------  -------------  ---------  -------------------
forge-abc123                         add-docstring  completed  2026-03-31 14:02:11
forge-def456                         refactor-api   running    2026-03-31 13:58:44
```

To see token usage and model details for a specific workflow, including cache hit counts:

```
forge status --workflow-id forge-abc123 --verbose
```

Example output:

```
Workflow: forge-abc123
Task: add-docstring
Status: completed
Created: 2026-03-31 14:02:11

Interactions (3):
  [generation] anthropic:claude-sonnet-4-5-20250929 1243in/412out 2104ms
    System prompt: You are a code generation assistant...
    User prompt: Generate the requested code changes...
  [classification] anthropic:claude-haiku-4-5-20251001 512in/88out 341ms
    System prompt: You are a context exploration assistant...
    User prompt: What context do you need...
```

To query the `batch_jobs` table directly, use any SQLite client with the database at `$FORGE_DB_PATH` (default: `~/.local/share/forge/forge.db`):

```sql
SELECT id, batch_id, workflow_id, status, created_at
FROM batch_jobs
ORDER BY created_at DESC
LIMIT 10;
```

---

## Configure the batch poll interval

The batch poller schedule interval is set when starting the Forge worker. The default is 600 seconds.

To start the worker with a shorter poll interval:

```
forge worker --batch-poll-interval 120
```

This sets the Temporal Schedule for `forge-batch-poller` to trigger every 120 seconds. Shorter intervals reduce the time workflows spend waiting for batch results, at the cost of more frequent Anthropic API polls.

If the schedule already exists (from a previous worker start), the worker updates it to the new interval on startup.

---

## Override the model tier for a specific run

To use a different model for one or more capability tiers on a single `forge run` invocation:

```
forge run \
  --reasoning-model anthropic:claude-opus-4-6 \
  --generation-model anthropic:claude-sonnet-4-5-20250929 \
  --task-id refactor-pipeline \
  --description "Refactor the context assembly pipeline" \
  --target-file src/forge/activities/context.py \
  --plan
```

Individual tier overrides can be combined. Tiers not specified use their defaults.

To route classification calls to a more capable model for debugging purposes:

```
forge run \
  --classification-model anthropic:claude-sonnet-4-5-20250929 \
  --task-id debug-exploration \
  --description "..." \
  --target-file src/forge/activities/exploration.py
```

Model strings must use the format `provider:model-name`. The `anthropic:` prefix is required for Anthropic models.

---

## Override the model tier for a specific plan step

To override the capability tier for a specific step within a plan, set the `capability_tier` field in the plan step. This requires providing the task via a JSON task file:

```json
{
  "task_id": "refactor-pipeline",
  "description": "Refactor the context assembly pipeline",
  "target_files": ["src/forge/activities/context.py"],
  "plan": true,
  "steps": [
    {
      "id": "step-1",
      "description": "Analyze the current assembly structure",
      "target_files": ["src/forge/activities/context.py"],
      "capability_tier": "reasoning"
    },
    {
      "id": "step-2",
      "description": "Rewrite the token budget packing logic",
      "target_files": ["src/forge/activities/context.py"]
    }
  ]
}
```

Submit with:

```
forge run --task-file refactor-pipeline.json
```

Step 1 uses the `REASONING` tier model. Step 2 uses the `GENERATION` tier (the default). Steps without a `capability_tier` field always use `GENERATION`.

Refer to the [reference](../reference/llm-dispatch.md) for the full list of tier values and the per-step field definition.
