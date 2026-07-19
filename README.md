# Forge

Forge is an LLM task orchestrator built around batch mode with document completion rather than iterative streaming. It invests heavily in upfront planning to identify parallelizable work units, then submits them as independent batch requests. Each request is a single step in a state machine, not a turn in a conversation.

Forge is suitable for any task that benefits from structured decomposition, parallel execution, and deterministic validation: code generation, research, analysis, content production, data processing, and more. The architecture is task-agnostic — the differentiation between use cases lives entirely in prompts, context, and validation criteria.

Git and worktrees serve as the general-purpose data store and isolation mechanism. Just as worktrees isolate parallel code branches, they equally isolate parallel research threads, analysis tracks, or any body of work that benefits from independent progress with controlled reconciliation.

## Prerequisites

- The local stack running: `make stack-up` brings up [Temporal](https://temporal.io/), Postgres, and MinIO under podman (see [deploy/local-stack/](deploy/local-stack/)). Any Temporal server reachable at `FORGE_TEMPORAL_ADDRESS` works; the default is `localhost:7233`.
- The workspace synced: `uv sync --all-packages` from the repo root. The root is a uv workspace (`apps/pbook`, `apps/ocr`, `libs/sax-platform`) and is self-contained — a bare clone resolves with no sibling checkouts.

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
forge worker
```

Submit a task in another:

```bash
forge run \
    --task-id my-task \
    --description "Add error handling to the API client" \
    --target-file src/forge/api/client.py
```

## Commands

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
| `--test-command` | — | Custom test command (implies `--run-tests`) |

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

### `forge worker`

Start the Temporal worker. The worker polls for queued workflows and executes activities. It must be running for `forge run` tasks to execute.

```bash
forge worker
forge worker --temporal-address temporal.example.com:7233
```

**Options:**

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--temporal-address` | `localhost:7233` | Temporal server address (env: `FORGE_TEMPORAL_ADDRESS`) |
| `--worker-identity` | `{pid}@{hostname}` | Custom worker identity reported to Temporal (env: `FORGE_WORKER_IDENTITY`) |

### `forge status`

List recent workflow runs or show details for a specific workflow.

```bash
forge status                              # List recent runs
forge status --workflow-id <id>           # Details for a specific run
forge status --workflow-id <id> --verbose # Full prompts and interaction details
```

**Options:**

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--workflow-id` | — | Show details for a specific workflow run |
| `--verbose` | off | Show full interaction details (prompts, tokens, latency) |
| `--limit` | `20` | Number of recent runs to show |
| `--json` | off | Machine-readable JSON output |

### `forge eval-planner`

Evaluate planner output against an eval corpus. Runs deterministic checks and optionally LLM-as-judge scoring.

```bash
forge eval-planner --corpus-dir eval/corpus
forge eval-planner --corpus-dir eval/corpus --plans-dir eval/plans --judge
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

List, add, and export playbook entries. `playbooks` is a command group; invoked bare it lists entries.

```bash
forge playbooks                        # List recent playbooks
forge playbooks --tag python           # Filter by tag
forge playbooks --task-id my-task      # Filter by source task
```

**Options (listing):**

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--tag` | — | Filter by tag (repeatable) |
| `--task-id` | — | Filter by source task ID |
| `--limit` | `20` | Max entries to show |
| `--json` | off | Machine-readable JSON output |

#### `forge playbooks add`

Add a playbook entry with LLM review (runs `ManualPlaybookWorkflow`).

```bash
forge playbooks add --schema           # Print the PlaybookEntry JSON schema
forge playbooks add --file entry.json  # Submit an entry from a JSON file
```

**Options:**

| Option | Description |
| -------- | ------------- |
| `--file`, `-f` | Path to a JSON file matching the `PlaybookEntry` schema |
| `--schema` | Print the `PlaybookEntry` JSON schema and exit |
| `--temporal-address` | Temporal server address (env: `FORGE_TEMPORAL_ADDRESS`) |

#### `forge playbooks export`

Export entries as `PlaybookEntry`-compatible JSON for backup or sharing.

```bash
forge playbooks export                            # All entries to stdout
forge playbooks export --tag python -o out.json   # Filtered, to a file
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
forge ingest ~/.claude/projects/<id>/session.jsonl          # Single session
forge ingest --all                                          # All sessions from ~/.claude/projects/
forge ingest --all --project forge                          # Filter by project name
forge ingest --all --dry-run                                # Preview without submitting
forge ingest --all --force                                  # Reprocess already-ingested sessions
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

Requires pbook to be installed. Sessions already recorded in pbook's `ingested_sessions` table are skipped unless `--force` is passed.

### `forge start`

Start an arbitrary registered Temporal workflow by name, without a Python script. Workflows registered on `forge-task-queue`: `ForgeTaskWorkflow`, `ForgeSubTaskWorkflow`, `ExportPlaybookWorkflow`, and `ManualPlaybookWorkflow`. (OCR workflows live in the `apps/ocr` workspace member, on `ocr-task-queue` with its own CLI: `uv run --package ocr ocr <cmd>`.)

```bash
forge start <WorkflowName> '{"field": "value"}' --wait # any registered workflow taking JSON input
forge start <WorkflowName> --input-file input.json      # read JSON input from a file instead
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
