# Debugging

Forge provides several layers of logging and inspection for diagnosing issues.

## Console Verbosity

Control console log level with `-v` flags on any command:

```bash
forge -v run ...      # INFO
forge -vv run ...     # DEBUG
```

Default (no flag) is `WARNING`. Log format: `HH:MM:SS LEVEL    logger — message`.

## Log Files

Application logs are written to `$XDG_STATE_HOME/forge/` (default `~/.local/state/forge/`). Console output is ephemeral; the filesystem logs persist for post-hoc debugging.

- `forge.log` — logs from CLI commands (`forge run`, `forge status`, etc.)
- `worker.log` — logs from the Temporal worker (`forge worker`)

Both files use `RotatingFileHandler` with 10 MB max size and 5 backups (e.g. `worker.log`, `worker.log.1`, ... `worker.log.5`). The file handler always logs at DEBUG level regardless of the console verbosity setting. Override the log directory with `FORGE_LOG_DIR`; set it to an empty string to disable file logging.

## Observability Store

Full LLM interaction data (prompts, tokens, latency, context stats) is persisted to a SQLite database at `$XDG_STATE_HOME/forge/forge.db` (default `~/.local/state/forge/forge.db`).

Override the path with `FORGE_DB_PATH`. Set it to an empty string to disable the store entirely.

### Inspecting runs

```bash
forge status                              # List recent runs (default: 20)
forge status --limit 5                    # Limit to 5 runs
forge status --workflow-id <id>           # Details for a specific workflow
forge status --workflow-id <id> --verbose  # Full interaction history (prompts, tokens, latency per step)
forge status --json                       # Machine-readable JSON output
```

### Verbose run output

```bash
forge run --verbose ...
```

Adds to the default output:

- LLM stats: model, tokens, latency, cache hit/miss
- Context stats: files discovered, token utilization
- Full interaction history from the observability store

### JSON output

```bash
forge run --json ...
```

Emits the full `TaskResult` as JSON for programmatic consumption.

## API Message Logs

Save the raw Anthropic API request and response JSON to the worktree:

```bash
forge run --log-messages ...
```

Files are written to `<worktree>/messages/`:

- `request-YYYY-MM-DD-HH-MM-SS.json` — Full API call parameters
- `response-YYYY-MM-DD-HH-MM-SS.json` — Full API response including usage and tool calls

The `messages/` directory is automatically git-ignored. Logging is best-effort and never disrupts the workflow.

## OpenTelemetry Tracing

Distributed tracing across Temporal activities is available via OpenTelemetry. Configure with environment variables:

| Variable | Values | Default |
|----------|--------|---------|
| `FORGE_OTEL_EXPORTER` | `console`, `otlp_grpc`, `otlp_http`, `none` | `console` |

## Knowledge Base

Inspect extracted playbooks from completed runs:

```bash
forge playbooks                    # List all playbooks
forge playbooks --tag <tag>        # Filter by tag
forge playbooks --task-id <id>     # Filter by source task
forge playbooks --json             # JSON output
```

## Temporal CLI

The `temporal` CLI can inspect workflow execution history directly from the Temporal server, including error messages and stack traces that may not appear in worker logs.

### List recent workflows

```bash
temporal workflow list --limit 10
```

### Show workflow event history

```bash
temporal workflow show --workflow-id <workflow-id>
```

If you have the run ID (useful when a workflow has been retried):

```bash
temporal workflow show --workflow-id <workflow-id> --run-id <run-id>
```

### Full event detail as JSON

The JSON output includes complete failure messages, stack traces, and input payloads:

```bash
temporal workflow show --workflow-id <workflow-id> -o json
```

### Describe workflow status

```bash
temporal workflow describe --workflow-id <workflow-id>
```

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `FORGE_DB_PATH` | Override observability store path (empty string disables) | `~/.local/state/forge/forge.db` |
| `FORGE_LOG_DIR` | Override log file directory (empty string disables file logging) | `~/.local/state/forge/` |
| `FORGE_OTEL_EXPORTER` | OTel trace exporter type | `console` |
| `FORGE_OTEL_ENDPOINT` | OTel exporter endpoint URL | Per exporter default |
| `XDG_STATE_HOME` | Base directory for logs and database | `~/.local/state` |
