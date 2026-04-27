+++
title = "Transcript Ingestion Reference"
weight = 124
description = "The pipeline that reads Claude Code session transcripts, analyzes them with forge's batch LLM path, and hands extracted experiences to pbook's ExtractionWorkflow cross-queue."
topic = "transcript-ingestion"
covers = [
    "TranscriptIngestionWorkflow: input JSON shape, output dict, signals, task queue",
    "BatchIngestionWorkflow: input JSON shape, output dict, fan-out behavior, task queue",
    "prepare_transcript activity: input and output JSON shapes",
    "Cross-queue activities called on pbook-task-queue: record_ingested_session (forge calls pbook)",
    "Cross-queue child workflow: ExtractionWorkflow (forge calls pbook)",
    "forge ingest CLI: all flags, argument, environment variables, exit codes",
    "Model tier and batch behavior: SUMMARIZATION tier default, max_tokens, retry policy",
]
detail = "Tabular. One table per subject: workflows, activities, CLI flags, env vars, exit codes. Brief intro sentence per section, no narrative."
+++
Tabular reference for the transcript ingestion workflows, the `prepare_transcript` activity, the cross-queue calls into pbook, and the `forge ingest` CLI. This reference covers only the forge side of the pipeline. For the pbook side — the `ExtractionWorkflow` input/output schemas, the `entries` table, and the `ingested_sessions` tracking table — consult pbook's documentation.

For background on why the pipeline is designed this way, see [Transcript Ingestion](../explanation/transcript-ingestion/). For recipes, see [How to Ingest Transcripts](../howto/ingest-transcripts/). For the cross-cutting discussion of how this pipeline relates to forge's self-learning loop, see [Learning Loops](../explanation/learning-loops/).

## Workflows

### `TranscriptIngestionWorkflow`

Processes a single Claude Code session end to end: parse, analyze via batch API, hand off to pbook's extraction, record the session as ingested.

| Property | Value |
|---|---|
| Module | `src/forge/ingestion_workflow.py` |
| Task queue | `forge-task-queue` |
| Input type | JSON string |
| Output type | `dict` |

**Input JSON fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `path` | str | yes | Absolute path to the JSONL session file |
| `project` | str | no | Project name override; falls back to inferred name if empty |
| `session_id` | str | no | Session identifier override; falls back to the file stem |

**Output dict fields:**

| Field | Type | Description |
|---|---|---|
| `experiences_found` | int | Number of experiences returned by the analysis LLM |
| `entries_created` | int | Number of pbook entries created by the cross-queue `ExtractionWorkflow` call |
| `session_id` | str | Echo of the input session identifier |
| `error` | str | Present only on malformed LLM response; value is `"malformed_llm_response"` |

**Signals:**

| Name | Argument type | Purpose |
|---|---|---|
| `batch_result_received` | `BatchResult` | Delivers the Anthropic Batch API completion result; mirrors the pattern used by `ForgeTaskWorkflow` |

**Early-exit conditions:**

| Condition | Return shape | Cross-queue calls made |
|---|---|---|
| Transcript file missing or empty | `{experiences_found: 0, entries_created: 0, session_id}` | None |
| Fewer than 3 messages | Same as above | None |
| Analysis returned zero experiences | Same as above | `record_ingested_session` only |
| Analysis returned malformed JSON | Above plus `error: "malformed_llm_response"` | None |

### `BatchIngestionWorkflow`

Fan-out parent that processes multiple sessions in parallel by starting one `TranscriptIngestionWorkflow` child per session.

| Property | Value |
|---|---|
| Module | `src/forge/ingestion_workflow.py` |
| Task queue | `forge-task-queue` |
| Input type | JSON string |
| Output type | `dict` |

**Input JSON fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `sessions` | list[dict] | yes | Each dict has the same shape as `TranscriptIngestionWorkflow` input |

**Output dict fields:**

| Field | Type | Description |
|---|---|---|
| `sessions_processed` | int | Total number of children that ran, including failed ones |
| `total_experiences` | int | Sum of `experiences_found` across all children |
| `total_entries_created` | int | Sum of `entries_created` across all children |
| `per_session` | list[dict] | Per-child result dicts; failed children include an `error` field |

**Child workflow IDs:** `ingest-session-{session_id}-{uuid8}`, where `{uuid8}` is the first 8 hex digits of a workflow-scoped UUID. The suffix ensures that re-ingesting the same session never collides with a prior run.

## Activities

### `prepare_transcript`

Reads and renders a Claude Code JSONL session file. Runs on `forge-task-queue`.

| Property | Value |
|---|---|
| Module | `src/forge/activities/ingestion.py` |
| Task queue | `forge-task-queue` |
| Input type | JSON string |
| Output type | JSON string |
| Timeout | 120 seconds |
| Retry policy | max 2 attempts |

**Input JSON fields:** Same as `TranscriptIngestionWorkflow` input (`path`, `project`, `session_id`).

**Output JSON fields:**

| Field | Type | Present when | Description |
|---|---|---|---|
| `transcript_text` | str | always | Rendered plaintext of the transcript; empty string if file missing |
| `system_prompt` | str | file exists | Analysis prompt from `pbook.ingestion_prompts.build_analysis_system_prompt()` |
| `user_prompt` | str | file exists | Analysis prompt from `pbook.ingestion_prompts.build_analysis_user_prompt()` |
| `project` | str | always | Final project name (caller-supplied or inferred from transcript metadata) |
| `session_id` | str | always | Session identifier (caller-supplied or derived from the file stem) |
| `message_count` | int | always | Number of parsed messages; 0 if file missing |
| `char_count` | int | always | Character count of `transcript_text` |

## Cross-queue calls into pbook

`TranscriptIngestionWorkflow` invokes the following on `pbook-task-queue`:

| Call | Type | When invoked | Arguments | Result |
|---|---|---|---|---|
| `ExtractionWorkflow` | Child workflow | When the analysis LLM returns ≥1 experiences | JSON: `{experiences: list[dict], project: str}` | `{entries_created: int}` |
| `record_ingested_session` | Activity | After extraction completes, or after an empty-result early exit | JSON: `{session_id, project_name, experiences_found, entries_created}` | `None` |

Each experience dict inside the `experiences` list has the following shape:

| Field | Type | Source |
|---|---|---|
| `project` | str | Passed through from the input |
| `problem` | str | From the analysis LLM output |
| `resolution` | str | From the analysis LLM output |
| `context` | str | From the analysis LLM output (optional, defaults to empty string) |
| `metadata` | dict | `{source: "claude-code-transcript", session_id: <id>}` |

Cross-queue child workflow IDs follow the pattern `pbook-extract-ingest-{session_id}-{uuid8}`.

## Model routing and batch configuration

The analysis LLM call uses the following settings:

| Setting | Value | Source |
|---|---|---|
| Capability tier | `SUMMARIZATION` | `resolve_model(CapabilityTier.SUMMARIZATION, ModelConfig())` |
| Default model | `claude-sonnet-4-5-20250929` | `_DEFAULT_TIER_MODELS[SUMMARIZATION]` in `forge/models.py` |
| Max tokens | 4096 | Hardcoded in `TranscriptIngestionWorkflow.run` |
| Extended thinking | Default (disabled) | `ThinkingConfig()` with `budget_tokens=0` |
| Output type name | `TranscriptAnalysisResult` | Registered via `register_output_type()` at worker startup |
| Batch dispatch | Via `batch_submit_and_wait()` | `forge/workflow_blocks.py` |

## `forge ingest` CLI

Invokes either `TranscriptIngestionWorkflow` (single path) or `BatchIngestionWorkflow` (--all) depending on arguments.

| Argument / option | Type | Default | Description |
|---|---|---|---|
| `TRANSCRIPT_PATH` | path | — | Positional. Path to a single JSONL file. Required unless `--all` is given. |
| `--all` | flag | off | Discover all sessions from `~/.claude/projects/`. Mutually relevant with `TRANSCRIPT_PATH`. |
| `--project TEXT` | str | `""` | With `--all`, filters discovered sessions to those matching the project name. With a single path, overrides the inferred project. |
| `--min-size INTEGER` | int | `10240` | Minimum session file size in bytes. Discovery only; ignored for single-path. |
| `--dry-run` | flag | off | Print the sessions that would be ingested and exit without submitting. |
| `--force` | flag | off | Skip the "already ingested" filter that queries pbook's `ingested_sessions` table. |
| `--json` | flag | off | Emit the aggregated result dict as indented JSON instead of the default human-readable summary. |
| `--temporal-address TEXT` | str | `localhost:7233` | Temporal server address. Also reads `FORGE_TEMPORAL_ADDRESS` environment variable. |

**Environment variables:**

| Name | Effect |
|---|---|
| `FORGE_TEMPORAL_ADDRESS` | Overrides the default Temporal server address. Takes precedence over the built-in default but is overridden by an explicit `--temporal-address` flag. |

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | Success, or dry-run displayed successfully, or no sessions to ingest |
| 1 | Missing argument (neither path nor `--all`), pbook not installed, or other usage error |
| 3 | Infrastructure error: Temporal unreachable, workflow failed, or cross-queue call failed |

**Workflow ID construction:**

| Invocation | Workflow ID |
|---|---|
| Single session or `--all` | `forge-batch-ingest-{unix_timestamp}` (always `BatchIngestionWorkflow`, even for a single session) |

## Module layout

| Path | Purpose |
|---|---|
| `src/forge/ingestion_workflow.py` | `TranscriptIngestionWorkflow`, `BatchIngestionWorkflow` |
| `src/forge/activities/ingestion.py` | `prepare_transcript` activity |
| `src/forge/cli.py` (the `ingest` command) | `forge ingest` entry point, `_submit_ingestion` helper, `format_ingest_dry_run`, `format_ingest_result` |
| `src/forge/worker.py` | Conditional registration of ingestion workflows/activities based on `_INGESTION_AVAILABLE` |

For guidance on managing forge's own playbook store (a separate pipeline), see the [Forge Run Extraction Reference](forge-run-extraction/). For the conceptual story of how the two stores relate, see [Learning Loops](../explanation/learning-loops/).
