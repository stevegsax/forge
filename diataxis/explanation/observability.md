+++
title = "About Observability and Debugging in Forge"
weight = 101
description = "How to inspect, diagnose, and debug Forge workflow execution using the observability store, logs, tracing, and CLI commands."
topic = "observability"
covers = [
    "The observability strategy: SQLite store for heavyweight data, Temporal for lightweight stats",
    "What is stored and why: full prompts, token usage, latency, context stats",
    "Best-effort writes: why store failures never block workflow execution",
    "OpenTelemetry tracing: span hierarchy from pipeline run to individual LLM request",
    "The execution journal: recording decisions, not just events",
]
detail = "Explain the design decisions behind the observability approach. Why SQLite? Why separate from Temporal results? Why best-effort?"
+++
Forge produces a large amount of intermediate data during workflow execution: the full assembled prompt sent to each LLM call, the model's raw response, token counts, latency measurements, context assembly statistics, and the transition signal that drove the next state. Understanding how this data is captured, where it lives, and why it is organized the way it is helps when diagnosing problems and when reasoning about workflow behavior.

## Two Layers of Observability

Forge separates observability data into two layers: a lightweight layer carried in Temporal workflow results, and a heavyweight layer persisted to a local SQLite database.

Temporal workflow results are small. They carry a `TaskResult` (or `StepResult`, `SubTaskResult`) containing the transition signal, output files, validation results, and an `LLMStats` summary — model name, input tokens, output tokens, and latency. These travel through Temporal's payload machinery, which has a practical limit near 2 MB. A planned workflow with five steps can easily produce 2 MB of prompts alone, since each step assembles up to 100,000 tokens of context. Storing full prompts in Temporal results would breach this limit. The lightweight layer keeps Temporal fast and its payloads lean.

The heavyweight layer stores what Temporal cannot: the assembled system prompt, user prompt, full model response, context statistics, and raw token counts for every LLM call. This data lives in a local SQLite database at `$XDG_STATE_HOME/forge/forge.db` (default `~/.local/state/forge/forge.db`). SQLite imposes no practical row-size limit — a 400 KB assembled prompt is stored as-is. The CLI queries this database when asked for full interaction history.

The separation is intentional. Temporal's job is workflow state management and retry coordination. The observability store's job is answering "what exactly happened during that LLM call?" These are different concerns with different retention, size, and query requirements.

## What Is Stored and Why

Every invocation of the `call_llm` and `call_planner` activities writes a row to the `interactions` table. Each row captures:

- The `task_id`, `step_id`, and `sub_task_id` that identify which workflow step produced the call
- The role — whether this was a code generation call (`llm`) or a planning call (`planner`)
- The full `system_prompt` and `user_prompt` as sent to the API
- The model name, input and output token counts, and latency in milliseconds
- The LLM's explanation field from its structured response
- A JSON-serialized `ContextStats` blob recording what was included in the assembled context: which files, how many tokens each contributed, and overall utilization

The `task_id` / `step_id` / `sub_task_id` triple lets you correlate a row with its position in the workflow DAG. Given a step that failed validation, you can retrieve the exact prompt that produced the faulty output and the exact response the model returned.

After a workflow completes, the `runs` table records the outcome — the `workflow_id`, the final status, and the JSON-serialized `TaskResult`. This gives a lightweight summary for listing recent runs without loading the full interaction history.

The `batch_jobs` table tracks asynchronous Anthropic Batch API submissions. When a workflow submits a batch request and enters a polling loop, the batch job ID, status, file path, and submission timestamp are stored here. This enables the CLI to show batch job status independently from the Temporal workflow state.

## Best-Effort Writes: Store Failures Never Block Workflows

Store writes in activities are wrapped in try/except. If the database is unavailable — disk full, bad permissions, a migration in progress — the activity logs a warning and returns its result normally. The workflow continues.

This is decision D42. Observability is a side effect of task execution, not a precondition for it. Making store writes blocking would introduce a failure mode in which a database problem prevents any code from being generated. That trade-off is wrong. The store is there to help diagnose problems; it should not be the source of them.

The practical consequence is that the observability store can be disabled entirely for testing by setting `FORGE_DB_PATH` to an empty string. Activities skip the persistence step, the CLI warns that no store is available when verbose output is requested, and everything else behaves identically.

## OpenTelemetry Tracing

Alongside the SQLite store, Forge emits OpenTelemetry spans across the activity execution hierarchy. The span tree mirrors the workflow structure:

- A root span for the pipeline run
- Child spans for each workflow instance within the run
- Activity-level spans for `call_llm`, `call_planner`, `assemble_context`, and `validate_output`
- Individual LLM request/response spans nested within the activity span

OTel tracing is initialized in the worker at startup. The exporter is configured via `FORGE_OTEL_EXPORTER` — options are `console` (default), `otlp_grpc`, `otlp_http`, and `none`. For local debugging, `console` writes span summaries to stdout. For integration with an observability backend, configure an OTLP endpoint via `FORGE_OTEL_ENDPOINT`.

OTel spans capture what happened and when. They answer timing questions: how long did context assembly take relative to the LLM call? Did a particular activity dominate the wall-clock time? These questions are difficult to answer from the SQLite store alone because the store records data at the level of completed LLM interactions, not at the level of individual Temporal activity execution.

## The Execution Journal: Decisions, Not Just Events

The design intention behind the observability store is to record *decisions* — why a particular transition signal was chosen, what context was available when the LLM produced its output, why the planner assigned a capability tier — alongside the events (timing, status codes, token counts).

The `explanation` column on the `interactions` table holds the LLM's own explanation field from its structured response. This is the model's articulation of what it did and why. Combined with the `context_stats_json` column (which shows what files were included at what token cost), you have enough information to reconstruct the reasoning chain for any step.

When the system escalates — halting because a validation failure exceeded the retry limit — it produces a structured escalation report using this stored data. The report includes completed step summaries, the triggering task and failure reason, and the orchestrator's interpretation of what went wrong. This report is itself produced by an LLM call, but it draws on the execution journal rather than requiring the human to read raw logs.

## Log Files

Alongside the structured observability data, two log files provide narrative debugging context:

- `~/.local/state/forge/forge.log` — output from CLI commands (`forge run`, `forge status`)
- `~/.local/state/forge/worker.log` — output from the Temporal worker process

Both use rotating file handlers (10 MB maximum, 5 backups). The file handler always writes at DEBUG level, regardless of the console verbosity setting. When investigating a problem that occurred hours or days ago, the log files often capture warning and error messages that scrolled off the terminal during execution.

For details on the store schema, CLI commands, environment variables, and span names, see the [Observability Reference](../reference/observability/). For practical debugging workflows — starting with a symptom and walking through the diagnostic steps — see [How to Debug a Workflow](../howto/debug-workflow/).
