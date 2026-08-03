# Debugging

Forge provides several layers of logging and inspection for diagnosing issues. Every command and every worker first passes the explicit-environment guard, so nothing here reaches a database or a Temporal server without a declared `FORGE_ENV` — see [WORKERS.md](WORKERS.md#environment-guard).

## Console verbosity

```bash
uv run forge -v run ...      # INFO
uv run forge -vv run ...     # DEBUG
```

With **no `-v` flag there is no console handler at all** when file logging is available: the worker stays silent on stdout and everything goes to the log file at DEBUG. The console handler is added only when you ask for verbosity, or as a fallback when file logging could not be configured — in which case its level is `WARNING`. Console format: `HH:MM:SS LEVEL    logger — message`.

## Log files

`forge.logging_config` attaches a `RotatingFileHandler` at **DEBUG level regardless of console verbosity**, 10 MB per file with 5 backups (`worker.log`, `worker.log.1`, … `worker.log.5`). Console output is ephemeral; these persist for post-hoc debugging.

The directory resolves in this order, from values the composition root read once via `LogSettings`:

1. `FORGE_LOG_DIR`, if set. An **empty string disables file logging**.
2. `$XDG_STATE_HOME/forge/`
3. `~/.local/state/forge/`

The file name is the log name the entry point passes: `forge.log` for CLI commands, `worker.log` for `forge worker`.

The supervised workers set `FORGE_LOG_DIR` in their env profile, so their application logs land together in one directory. Separately, the supervisors capture each process's stdout/stderr:

```text
$XDG_STATE_HOME/forge/logs/
├── forge-worker-1.log        launchd StandardOut/StandardError (prod)
├── forge-worker-2.log
├── ocr-worker.log
├── pbook-worker.log
└── dev-<worker>-worker.log   tmux pane, tee'd by `make dev-worker`
```

A production worker that refuses to start — guard failure, dirty checkout, `SchemaVersionError` — writes the reason to **stderr before logging is configured**, so the launchd `<agent>.log` file is where that message lands, not `worker.log`.

## Observability store

Full LLM interaction data (prompts, tokens, latency, context stats) is persisted to the store named by `FORGE_DB_URL`. The store is **mandatory**: if `FORGE_DB_URL` is unset, `ForgeSettings()` raises and the worker refuses to start; CLI store commands exit with an error. There is no disable-store mode and no runtime failover.

Both lanes point at Postgres on the shared sax-datastores stacks — `forge_dev` on `:5432`, `forge_prod` on `:5442`. The engine factory still accepts a `sqlite:///<path>` URL, which is what the test suite uses; nothing deployed runs on it.

### Inspecting runs

```bash
uv run forge status --env dev                              # recent runs (default limit 20)
uv run forge status --env dev --limit 5
uv run forge status --env dev --workflow-id <id>           # details for one workflow
uv run forge status --env dev --workflow-id <id> --verbose # full interaction history
uv run forge status --env dev --json                       # machine-readable
```

### Verbose run output

```bash
uv run forge run --verbose ...
```

Adds to the default output:

- LLM stats: model, tokens, latency, cache hit/miss
- Context stats: files discovered, token utilization
- Full interaction history from the observability store

### JSON output

```bash
uv run forge run --json ...
```

Emits the full `TaskResult` as JSON for programmatic consumption.

## API message logs

Save the raw Anthropic API request and response JSON to the worktree:

```bash
uv run forge run --log-messages ...
```

Files are written to `<worktree>/messages/`:

- `request-YYYY-MM-DD-HH-MM-SS.json` — full API call parameters
- `response-YYYY-MM-DD-HH-MM-SS.json` — full API response including usage

Timestamps are UTC. `forge.message_log` creates the directory and drops a `messages/.gitignore` containing `*`, so the payloads never reach a commit. Logging is best-effort — every failure is swallowed and never disrupts the workflow.

## OpenTelemetry tracing

Distributed tracing across Temporal activities is available via OpenTelemetry, and is **opt-in — the default is off** (T0.1). `forge.tracing.init_tracing` receives the exporter name from `TracingSettings` and resolves an unset value to the `none` exporter, so a bare worker run emits no spans.

| Variable | Values | Default |
| --- | --- | --- |
| `FORGE_OTEL_EXPORTER` | `console`, `otlp_grpc`, `otlp_http`, `none` | unset ⇒ `none` (off) |
| `FORGE_OTEL_ENDPOINT` | OTLP collector endpoint, e.g. `http://localhost:4317` | unset ⇒ the exporter's own default |

An unrecognized `FORGE_OTEL_EXPORTER` value raises at startup with the valid options listed. `FORGE_OTEL_ENDPOINT` is read only when the exporter is `otlp_grpc` or `otlp_http`.

## Knowledge base

Inspect extracted playbooks from completed runs (forge's own `playbooks` table — separate from pbook's store):

```bash
uv run forge playbooks --env dev                 # list (default limit 20)
uv run forge playbooks --env dev --tag <tag>     # filter by tag (repeatable)
uv run forge playbooks --env dev --task-id <id>  # filter by source task
uv run forge playbooks --env dev --json
```

Subcommands: `add` (with LLM review) and `export`.

## Temporal CLI

The `temporal` CLI reads workflow execution history straight from the server, including failure messages and stack traces that never reach worker logs.

**Every invocation needs an explicit `--address` and `--namespace`.** The CLI defaults to `127.0.0.1:7233` in namespace `default`; neither exists here, so a bare command connects to nothing or asks the wrong server. The examples below use the **dev** pair; for production substitute `--address 127.0.0.1:7243 --namespace forge-prod`.

### List recent workflows

```bash
temporal workflow list --address 127.0.0.1:7236 --namespace forge-dev --limit 10
```

### Show workflow event history

```bash
temporal workflow show --address 127.0.0.1:7236 --namespace forge-dev --workflow-id <workflow-id>
```

With a run ID, when a workflow has been retried:

```bash
temporal workflow show --address 127.0.0.1:7236 --namespace forge-dev \
  --workflow-id <workflow-id> --run-id <run-id>
```

### Full event detail as JSON

Includes complete failure messages, stack traces, and input payloads:

```bash
temporal workflow show --address 127.0.0.1:7236 --namespace forge-dev \
  --workflow-id <workflow-id> -o json
```

### Describe workflow status

```bash
temporal workflow describe --address 127.0.0.1:7236 --namespace forge-dev --workflow-id <workflow-id>
```

### Who is polling, and what is queued

```bash
temporal task-queue describe --address 127.0.0.1:7236 --namespace forge-dev --task-queue forge-task-queue
```

Poller identities carry the launch-time commit (`prod-forge-worker-1@104c14b`), and the statistics rows show the backlog. A queue with a growing backlog and an empty `Pollers` list means no worker is serving it — see [WORKERS.md](WORKERS.md#checking-whether-workers-are-running).

### Schedules

```bash
temporal schedule list --address 127.0.0.1:7236 --namespace forge-dev
```

`ocr-batch-tracker` should be present and unpaused wherever an ocr worker runs. Its runs keep firing on the server even when no worker is polling `ocr-task-queue` — they simply time out, which is the signature of a lane whose ocr worker is down.

The dev stack also has a Web UI at `http://localhost:8236`. The prod stack ships none.

## Environment variables

Secrets (`ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, `OPENAI_API_KEY`, database passwords, S3 credentials) live only in the chmod-600 profiles under `$XDG_CONFIG_HOME/forge/envs/` and are never recorded here or in the repo.

| Variable | Purpose | Default |
| --- | --- | --- |
| `FORGE_ENV` | **Required, no default.** `prod` / `dev` / `test`. Derives the Temporal address and namespace and selects the profile. Unset or invalid ⇒ exit 78 | _unset_ |
| `FORGE_PROD_ACK` | Must be `yes` for `FORGE_ENV=prod`. Set by the launchd plists or an interactive shell — **never** by a profile file | _unset_ |
| `FORGE_ENV_TAG` | Declared _inside_ a profile; must equal `FORGE_ENV` or the loader exits 78 | _unset_ |
| `FORGE_DB_URL` | **Required.** Store URL — `postgresql+psycopg2://…` on both lanes; `sqlite:///<path>` for tests. Unset ⇒ hard error at startup | _unset_ |
| `FORGE_TEMPORAL_ADDRESS` | Override only. `dev`/`prod` reject any value that is not their own server; `test` **requires** one (ephemeral container) | derived from `FORGE_ENV` |
| `FORGE_TEMPORAL_NAMESPACE` | **Retired.** No longer a setting; a line for it is silently ignored. The namespace is `<slug>-<env>`, derived | — |
| `FORGE_TEMPORAL_TLS` | Enable TLS for the Temporal connection (plus `FORGE_TEMPORAL_TLS_SERVER_CA`, `_CLIENT_CERT`, `_CLIENT_KEY`, `_SERVER_NAME`) | `false` |
| `FORGE_WORKER_IDENTITY` | Base worker identity; the launch-time git version is appended. Also settable as `--worker-identity` | SDK `{pid}@{hostname}` |
| `FORGE_OCR_S3_BUCKET` | Blob bucket. Required by the ocr worker; when unset, forge builds no blob client | _unset_ |
| `FORGE_OCR_S3_PREFIX` | Key prefix within the bucket | `""` |
| `FORGE_LOG_DIR` | Override the log directory; empty string disables file logging | `$XDG_STATE_HOME/forge/` |
| `FORGE_OTEL_EXPORTER` | OTel trace exporter: `console`, `otlp_grpc`, `otlp_http`, `none` | unset ⇒ off |
| `FORGE_OTEL_ENDPOINT` | OTLP collector endpoint; read only for the `otlp_*` exporters | exporter default |
| `XDG_STATE_HOME` | Base directory for logs | `~/.local/state` |
| `XDG_CONFIG_HOME` | Base directory for env profiles (`forge/envs/<env>.env`) | `~/.config` |
| `PBOOK_DATABASE_URL` | pbook's store. Unset ⇒ the pbook worker disables its store and skips schema verification | _unset_ |
| `ANTHROPIC_API_KEY` | Read by the Anthropic SDK; required by the forge and pbook workers | _unset_ |
| `MISTRAL_API_KEY` | **Required by the ocr worker** (fail-fast at startup). The forge worker never reads it | _unset_ |
| `OPENAI_API_KEY` | pbook embeddings; unset ⇒ embedding activities fail fast rather than hang | _unset_ |
