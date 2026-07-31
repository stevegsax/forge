# Forge

Forge is an LLM task orchestrator built around batch mode with document completion rather than iterative streaming. It invests heavily in upfront planning to identify parallelizable work units, then submits them as independent batch requests. Each request is a single step in a state machine, not a turn in a conversation.

Forge is suitable for any task that benefits from structured decomposition, parallel execution, and deterministic validation: code generation, research, analysis, content production, data processing, and more. The architecture is task-agnostic — the differentiation between use cases lives entirely in prompts, context, and validation criteria.

Git and worktrees serve as the general-purpose data store and isolation mechanism. Just as worktrees isolate parallel code branches, they equally isolate parallel research threads, analysis tracks, or any body of work that benefits from independent progress with controlled reconciliation.

## Prerequisites

- [Temporal](https://temporal.io/) and Postgres reachable: both come from shared stacks that forge does not own or start (`~/repos-sax/sax-temporal`, `~/repos-sax/sax-datastores`). Which server and namespace a command uses is derived from `FORGE_ENV`, not configured per command — dev `127.0.0.1:7236` / `forge-dev`, prod `127.0.0.1:7243` / `forge-prod`.
- The workspace synced: `uv sync --all-packages` from the repo root. The root is a uv workspace (`apps/pbook`, `apps/ocr`, `libs/sax-platform`) and is self-contained — a bare clone resolves with no sibling checkouts.
- **A declared environment.** Every command is fronted by the environment guard: `FORGE_ENV` must be one of `prod` / `dev` / `test` and has **no default**, so a command run in a bare shell exits **78** instead of guessing which database to touch. Declare it either way:

  ```bash
  forge --env dev status                       # loads ~/.config/forge/envs/dev.env
  forge status --env dev                       # identical — either position works

  set -a; source ~/.config/forge/envs/dev.env; set +a   # or export it yourself
  export FORGE_ENV=dev
  ```

  A bare `--env NAME` resolves to `$XDG_CONFIG_HOME/forge/envs/<NAME>.env` and sets `FORGE_ENV`; a path (or any value ending in `.env`) is read verbatim and takes `FORGE_ENV` from the file's `FORGE_ENV_TAG`. The profile is also where the rest of the runtime configuration comes from — `FORGE_DB_URL` (required by every command that reads the store), `ANTHROPIC_API_KEY`, and the S3 settings. `--env` never supplies `FORGE_PROD_ACK`, so it cannot by itself grant production access; production is a separate ceremony documented in [docs/operations/DEPLOYMENT.md](docs/operations/DEPLOYMENT.md) and CLAUDE.md's "Running the System".

  `--help` and `--version` are exempt by design (the guard runs at the command seam, after parsing short-circuits), so `forge run --help` works in any shell.

## Architecture

Forge uses Temporal for workflow orchestration. The client (`forge run`) and worker (`forge worker`) are separate processes, with the Temporal server acting as a durable queue and state machine between them.

```text
forge run  ──►  Temporal Server  ◄──  forge worker
(submits)        (queues work)        (executes)
```

`forge run` submits a workflow to the Temporal server and optionally waits for the result. `forge worker` polls the server for queued workflows and executes the activities (LLM calls, context assembly, validation, git operations). This separation provides Temporal's durability guarantees — if the worker crashes mid-task, the server retains workflow state, and a restarted worker resumes where it left off.

## Documentation

- [Table of Contents](TOC.md) — Full index of design docs, phase specs, user guides, and research.

## Usage

Start the worker in one terminal:

```bash
forge worker --env dev
```

Submit a task in another:

```bash
forge run --env dev \
    --task-id my-task \
    --description "Add error handling to the API client" \
    --target-file src/forge/api/client.py
```

Both examples assume the `dev` profile; with `FORGE_ENV` already exported, drop the `--env` flag.

## Commands

Every command accepts the global options below in addition to its own:

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--env` | — | Load an env profile (`NAME` or path) before the guard runs; sets `FORGE_ENV`. Valid before or after the subcommand — given at both, the subcommand's value wins. The bare `forge playbooks` group takes it only in the leading position |
| `-v` / `-vv` | off | Increase log verbosity (`-v` INFO, `-vv` DEBUG); leading position only |
| `--version` | — | Print the installed version and exit (no environment needed); leading position only |

Examples below show `--env dev`; substitute your own profile, or drop the flag once `FORGE_ENV` is exported.

### `forge run`

Submit a task and wait for the result.

Tasks can be defined inline via CLI options or loaded from a JSON file. The two modes are mutually exclusive.

**Inline task definition:**

```bash
forge run \
    --task-id my-task \
    --description "Refactor the validation module" \
    --target-file src/forge/activities/validate.py \
    --context-file src/forge/models.py
```

**JSON task file:**

```bash
forge run --task-file task.json
```

**Options:**

| Option | Description |
| -------- | ------------- |
| `--task-id` | Unique task identifier (required for inline) |
| `--description` | What the task should produce (required for inline) |
| `--target-file` | File to create or modify, repeatable (required for inline unless `--plan`) |
| `--context-file` | File to include as context, repeatable |
| `--task-file` | JSON file with a full `TaskDefinition` |
| `--json` | Output `TaskResult` as JSON |
| `--no-wait` | Submit and print workflow ID without waiting for completion |

**Planning options:**

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--plan` | off | Enable planning mode (decompose into ordered steps) |
| `--max-attempts` | `2` | Task-level retry limit |
| `--max-step-attempts` | `2` | Retry limit per step in planning mode |
| `--max-sub-task-attempts` | `2` | Retry limit per sub-task in fan-out steps |
| `--max-fan-out-depth` | `1` | Maximum recursive fan-out depth (1 = flat fan-out only) |
| `--sanity-check-interval` | `0` | Run sanity check every N steps in planning mode (0 = disabled) |
| `--no-resolve-conflicts` | off | Disable LLM-based conflict resolution for fan-out file conflicts |

**Validation options:**

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--no-lint` | off | Disable ruff lint check |
| `--no-format` | off | Disable ruff format check |
| `--run-tests` | off | Enable test validation |
| `--test-command` | — | Custom test command; pass `--run-tests` as well, since it does not enable tests by itself |

The domain (`--domain`) supplies the validation baseline and these flags adjust it: `code_generation` runs ruff lint and format, every other domain runs neither, and none run tests unless `--run-tests` is passed.

**Context discovery options:**

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--no-auto-discover` | off | Disable automatic context discovery via import graph |
| `--token-budget` | `100000` | Token budget for context assembly |
| `--max-import-depth` | `2` | How deep to trace imports |
| `--include-deps` | off | Include dependency file contents in upfront context |

**Exploration options:**

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--max-exploration-rounds` | `10` | Max rounds of LLM-guided context exploration (0 disables) |
| `--no-explore` | off | Disable LLM-guided context exploration |

**Model options:**

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--reasoning-model` | — | Override model for REASONING tier (planning) |
| `--generation-model` | — | Override model for GENERATION tier (code gen) |
| `--summarization-model` | — | Override model for SUMMARIZATION tier (extraction) |
| `--classification-model` | — | Override model for CLASSIFICATION tier (exploration) |
| `--effort` | `high` | Extended-thinking effort for planner/sanity-check/conflict-resolution calls in planning mode (`--plan`): `low`, `medium`, `high`, `xhigh`, `max`. No effect in single-step mode. |
| `--no-thinking` | off | Disable extended thinking for planner/sanity-check/conflict-resolution calls in planning mode. No effect in single-step mode. |
| `--domain` | `code_generation` | Task domain: `code_generation`, `research`, `code_review`, `documentation`, `generic` |

**API and debug options:**

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--sync/--no-sync` | `--no-sync` | Use synchronous Messages API (`--sync`) or batch mode (`--no-sync`, default) |
| `--batch-poll-interval` | `600` | Seconds between batch status polls, min 300 (D88). Batch mode only |
| `--verbose` | off | Show detailed LLM stats and interactions |
| `--log-messages` | off | Save full API request/response JSON to `messages/` in the worktree |

**Common options:**

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--base-branch` | `main` | Branch to create worktree from |
| `--temporal-address` | `localhost:7233` | Temporal server address (env: `FORGE_TEMPORAL_ADDRESS`) |
| `--env` | — | Env profile to load before the guard (see [Prerequisites](#prerequisites)) |

### `forge worker`

Start the Temporal worker. The worker polls for queued workflows and executes activities. It must be running for `forge run` tasks to execute.

```bash
forge worker --env dev
forge worker --env dev --temporal-address temporal.example.com:7233
```

**Options:**

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--temporal-address` | `localhost:7233` | Temporal server address (env: `FORGE_TEMPORAL_ADDRESS`) |
| `--worker-identity` | `{pid}@{hostname}` | *Base* worker identity reported to Temporal — the launch-time git version is appended, e.g. `prod-forge-worker-1@bb64d88` (env: `FORGE_WORKER_IDENTITY`) |
| `--env` | — | Env profile to load before the guard (see [Prerequisites](#prerequisites)) |

On `FORGE_ENV=prod` the worker additionally refuses to start unless its checkout is a clean commit (exit 78) — production is deployed from a pinned checkout, see [docs/operations/DEPLOYMENT.md](docs/operations/DEPLOYMENT.md).

### `forge status`

List recent workflow runs or show details for a specific workflow. Reads the store directly (`FORGE_DB_URL`), so it works whether or not a worker is running.

```bash
forge status --env dev                              # List recent runs
forge status --env dev --workflow-id <id>           # Details for a specific run
forge status --env dev --workflow-id <id> --verbose # Full prompts and interaction details
```

**Options:**

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--workflow-id` | — | Show details for a specific workflow run |
| `--verbose` | off | Show full interaction details (prompts, tokens, latency) |
| `--limit` | `20` | Number of recent runs to show |
| `--json` | off | Machine-readable JSON output |
| `--env` | — | Env profile to load before the guard (see [Prerequisites](#prerequisites)) |

### `forge eval-planner`

Evaluate planner output against an eval corpus. Runs deterministic checks and optionally LLM-as-judge scoring. No corpus ships in the repo — point `--corpus-dir` at your own directory of eval-case JSON files (`tests/fixtures/eval/cases` and `.../plans` show the shape).

```bash
forge eval-planner --env dev --corpus-dir <corpus-dir>
forge eval-planner --env dev --corpus-dir <corpus-dir> --plans-dir <plans-dir> --judge
```

**Options:**

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--corpus-dir` | — | Directory containing eval case JSON files (required) |
| `--plans-dir` | — | Directory containing plan JSON files |
| `--judge / --no-judge` | `--no-judge` | Run LLM judge scoring |
| `--judge-model` | REASONING tier pin | Model to use as judge (defaults to the REASONING tier's registry pin) |
| `--dry-run` | off | List cases without evaluating |
| `--output-dir` | — | Directory to save run results JSON |
| `--json` | off | Output results as JSON |

### `forge playbooks`

List, add, and export playbook entries. `playbooks` is a command group; invoked bare it lists entries (reading the store directly).

`--env` is the one exception to the either-position rule: the bare listing form is the group itself, which takes the flag only in the leading position. Its subcommands (`add`, `export`) accept it in either.

```bash
forge --env dev playbooks                        # List recent playbooks
forge --env dev playbooks --tag python           # Filter by tag
forge --env dev playbooks --task-id my-task      # Filter by source task
```

**Options (listing):**

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--tag` | — | Filter by tag (repeatable) |
| `--task-id` | — | Filter by source task ID |
| `--limit` | `20` | Max entries to show |
| `--json` | off | Machine-readable JSON output |

#### `forge playbooks add`

Add a playbook entry with LLM review (runs `ManualPlaybookWorkflow`, so a worker must be polling `forge-task-queue`). `--schema` prints locally but still passes the environment guard first, like every command.

```bash
forge playbooks add --env dev --schema           # Print the PlaybookEntry JSON schema
forge playbooks add --env dev --file entry.json  # Submit an entry from a JSON file
```

**Options:**

| Option | Description |
| -------- | ------------- |
| `--file`, `-f` | Path to a JSON file matching the `PlaybookEntry` schema |
| `--schema` | Print the `PlaybookEntry` JSON schema and exit |
| `--temporal-address` | Temporal server address (env: `FORGE_TEMPORAL_ADDRESS`) |

#### `forge playbooks export`

Export entries as `PlaybookEntry`-compatible JSON for backup or sharing. Runs `ExportPlaybookWorkflow`, so a worker must be polling `forge-task-queue` (unlike the bare `forge playbooks` listing, which reads the store directly).

```bash
forge playbooks export --env dev                            # All entries to stdout
forge playbooks export --env dev --tag python -o out.json   # Filtered, to a file
```

**Options:**

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--tag` | — | Filter by tag (repeatable, OR match) |
| `--task-id` | — | Filter by source task ID |
| `--limit` | `0` | Max entries to export (0 = all) |
| `--output`, `-o` | stdout | Write to a file instead of stdout |
| `--temporal-address` | `localhost:7233` | Temporal server address (env: `FORGE_TEMPORAL_ADDRESS`) |

### `forge ingest`

Ingest Claude Code conversation transcripts into pbook's knowledge store. Reads JSONL session files, analyzes them via the batch API, and hands extracted experiences to pbook's ExtractionWorkflow cross-queue.

```bash
forge ingest --env dev ~/.claude/projects/<id>/session.jsonl  # Single session
forge ingest --env dev --all                                  # All sessions from ~/.claude/projects/
forge ingest --env dev --all --project forge                  # Filter by project name
forge ingest --env dev --all --dry-run                        # Preview without submitting
forge ingest --env dev --all --force                          # Reprocess already-ingested sessions
```

**Options:**

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--all` | off | Discover and ingest all sessions from `~/.claude/projects/` |
| `--project` | — | Filter by project name (with `--all`) or override (with path) |
| `--min-size` | `10240` | Minimum session file size in bytes (discovery only) |
| `--dry-run` | off | Preview sessions without submitting |
| `--force` | off | Reprocess already-ingested sessions |
| `--json` | off | Machine-readable JSON output |
| `--temporal-address` | `localhost:7233` | Temporal server address (env: `FORGE_TEMPORAL_ADDRESS`) |

Sessions already recorded in pbook's `pbk_ingested_sessions` table are skipped unless `--force` is passed; when pbook's store is unavailable the filter is skipped rather than failing. pbook is a workspace member and is always installed, but the command still fails with a clear message if its import fails.

### `forge start`

Start an arbitrary registered Temporal workflow by name, without a Python script. Workflows registered on `forge-task-queue`: `ForgeTaskWorkflow`, `ForgeSubTaskWorkflow`, `ExportPlaybookWorkflow`, `ManualPlaybookWorkflow`, plus `TranscriptIngestionWorkflow` and `BatchIngestionWorkflow` (registered whenever pbook imports, which is the normal case — the worker logs a warning and skips them otherwise). (OCR workflows live in the `apps/ocr` workspace member, on `ocr-task-queue` with its own CLI: `uv run --package ocr ocr <cmd>`.)

```bash
forge start --env dev <WorkflowName> '{"field": "value"}' --wait # any registered workflow taking JSON input
forge start --env dev <WorkflowName> --input-file input.json     # read JSON input from a file instead
```

**Options:**

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--input-file` | — | Read JSON input from a file instead of the argument |
| `--id` | auto-generated | Custom workflow ID |
| `--task-queue` | `forge-task-queue` | Temporal task queue |
| `--wait` | off | Wait for completion and print result as JSON |
| `--timeout` | `48` | Execution timeout in hours |
| `--temporal-address` | `localhost:7233` | Temporal server address (env: `FORGE_TEMPORAL_ADDRESS`) |
