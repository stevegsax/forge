# How to Debug a Workflow

This guide shows how to diagnose common Forge workflow problems using the observability store, log files, CLI commands, and the Temporal CLI. Each section starts with a symptom and walks through the diagnostic steps.

For reference material on CLI flags, schema details, and environment variables, see the [Observability Reference](../reference/observability.md). For background on why these tools exist, see [About Observability and Debugging](../explanation/observability.md).

---

## 1. Inspect a Completed Workflow's History

**Symptom:** A workflow ran — successfully or not — and you want to see what it did at each step.

1. List recent runs to find the workflow ID:

    ```bash
    forge status
    ```

    Output:

    ```
    workflow-id            task-id          status            created
    forge-abc123           my-task-001      success           2026-03-31 14:22:01
    forge-def456           my-task-002      failure_terminal  2026-03-31 13:05:44
    forge-ghi789           my-task-003      success           2026-03-31 11:30:10
    ```

2. Show details for a specific workflow:

    ```bash
    forge status --workflow-id forge-def456
    ```

    This shows the final `TaskResult` summary: transition signal, output files written, validation results, and `LLMStats` (model, tokens, latency).

3. To include the full step-by-step interaction history from the observability store — assembled prompts, model responses, context stats — add `--verbose`:

    ```bash
    forge status --workflow-id forge-def456 --verbose
    ```

    The verbose output adds one block per interaction row, including the system prompt, user prompt, model name, token counts, and latency.

4. If you need the raw data for scripting:

    ```bash
    forge status --workflow-id forge-def456 --json
    ```

If you see `no store available` instead of interaction data, the observability store is disabled (`FORGE_DB_PATH` is set to an empty string). Set it to a valid path and rerun the workflow to capture data.

---

## 2. Find What Prompt Was Sent for a Specific Step

**Symptom:** The LLM produced unexpected output for step 3. You want to see the exact prompt it received.

1. Get the workflow ID using `forge status` (see section 1).

2. Run with `--verbose` to see all interactions:

    ```bash
    forge status --workflow-id forge-abc123 --verbose
    ```

    Each interaction block includes:

    - `step_id` — which step this call corresponds to (e.g., `step-3`)
    - `role` — `llm` or `planner`
    - `system_prompt` — the full assembled system prompt
    - `user_prompt` — the task description and any appended error context
    - `model_name` — the concrete model used
    - `input_tokens` / `output_tokens` / `latency_ms`
    - `context_stats_json` — what files were assembled, token counts per file

3. If the prompt data is too long to read in the terminal, use `--json` and pipe to `jq`:

    ```bash
    forge status --workflow-id forge-abc123 --json \
      | jq '.interactions[] | select(.step_id == "step-3") | .system_prompt'
    ```

4. To inspect context stats for that step:

    ```bash
    forge status --workflow-id forge-abc123 --json \
      | jq '.interactions[] | select(.step_id == "step-3") | .context_stats_json | fromjson'
    ```

    This shows which files were included, how many tokens each consumed, and what the overall budget utilization was.

---

## 3. Diagnose Why a Step Failed Validation

**Symptom:** A step returned `failure_retryable` or `failure_terminal`. You want to know what the validation error was and what the LLM produced.

1. Get the workflow details:

    ```bash
    forge status --workflow-id forge-def456 --verbose
    ```

    Look for the interaction with the failed step's `step_id`. Check:

    - The `explanation` field — the LLM's own summary of what it attempted
    - The `output_tokens` — unusually low output tokens may indicate a truncated or malformed response

2. Get the full `TaskResult` which includes validation results:

    ```bash
    forge status --workflow-id forge-def456 --json \
      | jq '.result | .validation_result'
    ```

    The `validation_result` includes the checks that ran, which failed, and the raw error output (e.g., ruff error messages, test failure output).

3. If the workflow retried before reaching `failure_terminal`, look for multiple interaction rows with the same `step_id`. The second row's `user_prompt` includes the injected error context from the first failure:

    ```bash
    forge status --workflow-id forge-def456 --json \
      | jq '.interactions[] | select(.step_id == "step-2") | {attempt: .id, user_prompt_len: (.user_prompt | length)}'
    ```

    A longer `user_prompt` on the retry indicates error context was injected (Phase 8 error-aware retries).

4. Check the Temporal workflow event history for the raw activity failure payload:

    ```bash
    temporal workflow show --workflow-id forge-def456 -o json \
      | jq '.events[] | select(.eventType == "ActivityTaskFailed")'
    ```

---

## 4. Check Token Budget Utilization

**Symptom:** You suspect the LLM is producing lower-quality output because the context was too large, or you want to verify that the right files were included.

1. Get context stats for a specific step:

    ```bash
    forge status --workflow-id forge-abc123 --json \
      | jq '.interactions[] | select(.step_id == "step-1") | .context_stats_json | fromjson'
    ```

    The `ContextStats` object includes:

    - `files_discovered` — total files found by import graph analysis
    - `files_included` — files actually packed into the prompt
    - `tokens_used` — total tokens consumed by context
    - `token_budget` — total token budget for the model
    - `utilization` — `tokens_used / token_budget` as a fraction
    - A per-file breakdown of token costs

2. High utilization (above 0.90) means the budget was nearly full and lower-priority files were dropped. If a file you expected to be present was cut, check its priority level and consider using `--include-deps` or `--context-file` to force its inclusion.

3. Low utilization (below 0.30) means plenty of headroom was available. If the output quality was still poor, the problem is more likely in the prompt or model routing than in context selection.

4. To compare utilization across steps in a planned workflow:

    ```bash
    forge status --workflow-id forge-abc123 --json \
      | jq '.interactions[] | {step: .step_id, utilization: (.context_stats_json | fromjson | .utilization)}'
    ```

---

## 5. Read Log Files for a Stuck or Slow Workflow

**Symptom:** A workflow is running slowly or appears stuck. The terminal shows no output.

1. Tail the worker log:

    ```bash
    tail -f ~/.local/state/forge/worker.log
    ```

    Look for:

    - Activity retry messages — a stuck workflow is often retrying the same activity indefinitely due to an unregistered activity name, a timeout, or a persistent error
    - `WARNING` lines from store writes — these indicate the database had a problem during execution
    - `ERROR` lines from the Temporal worker — these are unexpected activity failures

2. Tail the CLI log for any command-side problems:

    ```bash
    tail -f ~/.local/state/forge/forge.log
    ```

3. If the worker log shows Java stack traces at shutdown but no Python errors, the Temporal test server may have shut down while activities were retrying. Look for a line like `CommitError("Nothing to commit")` in the activity retries — this causes infinite retry loops with no visible Python error.

4. If the worker was recently restarted and the log shows `activity not registered`, a new activity was added but the worker was not restarted after the code change. Restart the worker:

    ```bash
    forge worker
    ```

5. Check log rotation if no recent entries appear — the active log file may have rotated:

    ```bash
    ls -lh ~/.local/state/forge/worker.log*
    tail -f ~/.local/state/forge/worker.log.1
    ```

6. The file handler always writes at DEBUG level. If you see a `WARNING` line in the log, there is a corresponding DEBUG trace before it that shows additional context.

---

## 6. Enable API Message Logs

**Symptom:** You need the raw Anthropic API request and response payloads — for example, to verify cache control headers, inspect structured output tool calls, or reproduce an API error.

1. Add `--log-messages` to your `forge run` command:

    ```bash
    forge run --task-id my-task --log-messages ...
    ```

2. After the run, find the messages directory in the worktree:

    ```bash
    ls ~/.local/share/forge/worktrees/my-task/messages/
    ```

    Output:

    ```
    request-2026-03-31-14-22-01.json
    response-2026-03-31-14-22-01.json
    request-2026-03-31-14-22-15.json
    response-2026-03-31-14-22-15.json
    ```

    Each LLM call produces a timestamped request/response pair. For a planned multi-step workflow, there is one pair per step plus one pair for the planner call.

3. Inspect the request to verify context ordering and cache control headers:

    ```bash
    jq '.system | map({type, cache_control})' \
      ~/.local/share/forge/worktrees/my-task/messages/request-2026-03-31-14-22-01.json
    ```

4. Inspect the response for token usage including cache fields:

    ```bash
    jq '.usage' \
      ~/.local/share/forge/worktrees/my-task/messages/response-2026-03-31-14-22-01.json
    ```

    The `usage` object includes `cache_read_input_tokens` and `cache_creation_input_tokens` when prompt caching is active.

5. The `messages/` directory is automatically git-ignored. Writes are best-effort and never fail the workflow.

---

## 7. Use Verbose Output for Real-Time Debugging

**Symptom:** You are running a task and want to see what is happening as it executes, rather than inspecting logs after the fact.

1. Add `-v` for INFO-level console output:

    ```bash
    forge -v run --task-id my-task ...
    ```

    This shows activity start/stop events, context assembly progress, and validation results as they occur.

2. Add `-vv` for DEBUG-level console output:

    ```bash
    forge -vv run --task-id my-task ...
    ```

    DEBUG output includes internal state transitions, store write attempts, OTel span events, and full validation error text.

3. Add `--verbose` to the `run` subcommand to display full LLM stats and interaction history after completion:

    ```bash
    forge run --verbose --task-id my-task ...
    ```

    The `-v` / `-vv` flags and `--verbose` are independent. Use them together for maximum output:

    ```bash
    forge -vv run --verbose --task-id my-task ...
    ```

4. The console log level set by `-v` / `-vv` applies only to the current process. The log files always record at DEBUG level regardless of this flag.

5. To inspect the Temporal workflow state in real-time while a workflow is running:

    ```bash
    temporal workflow describe --workflow-id forge-abc123
    ```

    This shows the current workflow status, running activities, and any pending signals.
