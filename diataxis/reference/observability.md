# Observability Reference

Reference for the Forge observability store, CLI inspection commands, log files, environment variables, and OpenTelemetry integration.

For background on why the store is structured this way, see [About Observability and Debugging](../explanation/observability.md). For practical debugging procedures, see [How to Debug a Workflow](../howto/debug-workflow.md).

---

## SQLite Observability Store

**Default path:** `~/.local/state/forge/forge.db`
**Override:** `FORGE_DB_PATH` environment variable (empty string disables the store)
**Engine mode:** WAL (Write-Ahead Logging) for concurrent activity writes
**Migration management:** Alembic — migrations run automatically on worker startup

---

### `interactions` Table

Stores one row per LLM API call, written by `call_llm` and `call_planner` activities.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `id` | INTEGER | NOT NULL | Primary key, auto-increment |
| `task_id` | TEXT | NOT NULL | Workflow task identifier; indexed |
| `step_id` | TEXT | NULL | Step identifier within a planned workflow |
| `sub_task_id` | TEXT | NULL | Sub-task identifier for fan-out steps |
| `role` | TEXT | NOT NULL | `"llm"` for generation calls; `"planner"` for planning calls |
| `system_prompt` | TEXT | NOT NULL | Full assembled system prompt as sent to the API |
| `user_prompt` | TEXT | NOT NULL | Full user prompt as sent to the API |
| `model_name` | TEXT | NOT NULL | Concrete model identifier (e.g., `claude-sonnet-4-5`) |
| `input_tokens` | INTEGER | NOT NULL | Total input tokens billed for this call |
| `output_tokens` | INTEGER | NOT NULL | Total output tokens billed for this call |
| `latency_ms` | REAL | NOT NULL | Wall-clock time from request to response, in milliseconds |
| `explanation` | TEXT | NOT NULL | LLM's explanation field from its structured response (default `""`) |
| `context_stats_json` | TEXT | NULL | JSON-serialized `ContextStats` — files included, token counts, utilization |
| `created_at` | DATETIME | NOT NULL | UTC timestamp, server default |

---

### `runs` Table

Stores one row per completed workflow run.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `id` | INTEGER | NOT NULL | Primary key, auto-increment |
| `task_id` | TEXT | NOT NULL | Workflow task identifier; indexed |
| `workflow_id` | TEXT | NOT NULL | Temporal workflow ID; unique |
| `status` | TEXT | NOT NULL | Final workflow status (e.g., `"success"`, `"failure_terminal"`) |
| `result_json` | TEXT | NOT NULL | JSON-serialized `TaskResult` (without full prompts) |
| `created_at` | DATETIME | NOT NULL | UTC timestamp, server default |

---

### `batch_jobs` Table

Tracks Anthropic Batch API submissions. Written when a batch workflow submits a request and enters polling.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `id` | INTEGER | NOT NULL | Primary key, auto-increment |
| `batch_id` | TEXT | NOT NULL | Anthropic batch job identifier; unique |
| `status` | TEXT | NOT NULL | Current batch status (`"submitted"`, `"in_progress"`, `"ended"`, `"error"`) |
| `file_path` | TEXT | NULL | Absolute path to the source document (OCR pipeline) |
| `workflow_id` | TEXT | NULL | Associated Temporal workflow ID |
| `submitted_at` | DATETIME | NOT NULL | UTC timestamp when the batch was submitted |
| `result_json` | TEXT | NULL | JSON-serialized result, populated when the batch completes |

---

### Alembic Migration Management

Alembic manages schema versioning. Migration files live in `src/forge/alembic/versions/`.

| File | Description |
|---|---|
| `src/forge/alembic/alembic.ini` | Alembic configuration; resolves DB URL from `get_db_path()` |
| `src/forge/alembic/env.py` | Migration environment |
| `src/forge/alembic/versions/001_initial.py` | Initial schema: `interactions` and `runs` tables |

Migrations run automatically when the worker starts (`alembic upgrade head`). Running multiple workers concurrently is safe — Alembic uses migration locking. To inspect migration status manually:

```bash
alembic current
alembic history
```

---

## CLI Commands

### `forge status`

Lists recent workflow runs and shows details for a specific run.

| Flag | Type | Default | Description |
|---|---|---|---|
| _(no flags)_ | — | — | Lists the 20 most recent runs from `runs` table |
| `--limit N` | integer | `20` | Maximum number of runs to list |
| `--workflow-id ID` | string | — | Show details for a specific workflow run |
| `--verbose` | flag | off | Include full interaction history (prompts, tokens, latency per step) from `interactions` table |
| `--json` | flag | off | Emit output as machine-readable JSON |

**Examples:**

```bash
forge status
forge status --limit 5
forge status --workflow-id forge-abc123
forge status --workflow-id forge-abc123 --verbose
forge status --json
```

---

### `forge run` (observability-related flags)

| Flag | Type | Default | Description |
|---|---|---|---|
| `--verbose` | flag | off | After completion, show LLM stats, context stats, and full interaction history |
| `--json` | flag | off | Emit the full `TaskResult` as JSON |
| `--log-messages` | flag | off | Save raw Anthropic API request and response JSON to `<worktree>/messages/` |
| `-v` | flag | off | Set console log level to INFO |
| `-vv` | flag | off | Set console log level to DEBUG |

---

### API Message Log Files

When `--log-messages` is passed, raw API payloads are written to the worktree:

| File | Contents |
|---|---|
| `<worktree>/messages/request-YYYY-MM-DD-HH-MM-SS.json` | Full Anthropic API call parameters |
| `<worktree>/messages/response-YYYY-MM-DD-HH-MM-SS.json` | Full API response including usage fields and structured output |

The `messages/` directory is automatically added to `.gitignore`. Writes are best-effort and never disrupt workflow execution.

---

## Log Files

| File | Written by | Contents |
|---|---|---|
| `~/.local/state/forge/forge.log` | CLI process | Output from `forge run`, `forge status`, and other commands |
| `~/.local/state/forge/worker.log` | Worker process | Output from `forge worker` — activity execution, retry events, OTel initialization |

**Rotation policy:** `RotatingFileHandler`, 10 MB maximum file size, 5 backups (e.g., `worker.log`, `worker.log.1`, ... `worker.log.5`).

**Log level:** File handlers always write at DEBUG level, regardless of the console verbosity flag (`-v` / `-vv`). Console output is controlled separately.

**Log format:** `HH:MM:SS LEVEL    logger — message`

**Console default:** WARNING (no `-v` flag)

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FORGE_DB_PATH` | `~/.local/state/forge/forge.db` | Override observability store path. Set to empty string to disable the store entirely. |
| `FORGE_LOG_DIR` | `~/.local/state/forge/` | Override log file directory. Set to empty string to disable file logging. |
| `FORGE_OTEL_EXPORTER` | `console` | OTel trace exporter type. Values: `console`, `otlp_grpc`, `otlp_http`, `none`. |
| `FORGE_OTEL_ENDPOINT` | _(exporter default)_ | OTel exporter endpoint URL. Applies to `otlp_grpc` and `otlp_http` exporters. |
| `XDG_STATE_HOME` | `~/.local/state` | XDG base directory for logs and database. Affects all XDG-derived paths. |

---

## OpenTelemetry Span Names and Attributes

OTel tracing is initialized in the worker at startup. Spans are emitted by activity code.

### Span Hierarchy

```
forge.pipeline_run
└── forge.workflow_instance
    ├── forge.assemble_context
    ├── forge.call_llm
    │   └── forge.llm_request
    ├── forge.call_planner
    │   └── forge.llm_request
    └── forge.validate_output
```

### Span Names

| Span Name | Activity | Description |
|---|---|---|
| `forge.assemble_context` | `assemble_context` | Full context assembly: import graph, ranking, token packing |
| `forge.call_llm` | `call_llm` | LLM code generation or analysis call |
| `forge.call_planner` | `call_planner` | LLM planning call to decompose a task |
| `forge.llm_request` | (child of `call_llm` / `call_planner`) | Individual HTTP request/response to the Anthropic API |
| `forge.validate_output` | `validate_output` | Deterministic validation (ruff lint, format, test execution) |

### `forge.call_llm` Span Attributes

| Attribute | Type | Description |
|---|---|---|
| `llm.model_name` | string | Concrete model identifier |
| `llm.input_tokens` | integer | Input tokens billed |
| `llm.output_tokens` | integer | Output tokens billed |
| `llm.latency_ms` | float | Request latency in milliseconds |
| `llm.cache_read_tokens` | integer | Tokens read from Anthropic prompt cache |
| `llm.cache_write_tokens` | integer | Tokens written to Anthropic prompt cache |

### `forge.validate_output` Span Attributes

| Attribute | Type | Description |
|---|---|---|
| `validation.passed` | boolean | Whether all checks passed |
| `validation.checks_run` | string | Comma-separated list of checks executed |
| `validation.transition` | string | Resulting transition signal (`success`, `failure_retryable`, `failure_terminal`) |

---

## `LLMStats` Model (Temporal Payloads)

`LLMStats` is the lightweight statistics object carried in Temporal result payloads.

| Field | Type | Description |
|---|---|---|
| `model_name` | `str` | Concrete model identifier |
| `input_tokens` | `int` | Total input tokens |
| `output_tokens` | `int` | Total output tokens |
| `latency_ms` | `float` | Request latency in milliseconds |

`LLMStats` is attached to `TaskResult.llm_stats`, `StepResult.llm_stats`, `SubTaskResult.llm_stats`, and `TaskResult.planner_stats`. All fields have `None` as the default for backward compatibility with results produced before Phase 5.
