# Forge

Forge is an LLM task orchestrator built around batch mode with document completion rather than iterative streaming. It invests heavily in upfront planning to identify parallelizable work units, then submits them as independent batch requests. Each request is a single step in a state machine, not a turn in a conversation.

Forge is suitable for any task that benefits from structured decomposition, parallel execution, and deterministic validation: code generation, research, analysis, content production, data processing, and more. The architecture is task-agnostic — the differentiation between use cases lives entirely in prompts, context, and validation criteria.

Git and worktrees serve as the general-purpose data store and isolation mechanism. Just as worktrees isolate parallel code branches, they equally isolate parallel research threads, analysis tracks, or any body of work that benefits from independent progress with controlled reconciliation.

## Prerequisites

- A running [Temporal](https://temporal.io/) server (default `localhost:7233`)
- Python with the `forge` package installed (`uv sync`)

## Architecture

Forge uses Temporal for workflow orchestration. The client (`forge run`) and worker (`forge worker`) are separate processes, with the Temporal server acting as a durable queue and state machine between them.

```
forge run  ──►  Temporal Server  ◄──  forge worker
(submits)        (queues work)        (executes)
```

`forge run` submits a workflow to the Temporal server and optionally waits for the result. `forge worker` polls the server for queued workflows and executes the activities (LLM calls, context assembly, validation, git operations). This separation provides Temporal's durability guarantees — if the worker crashes mid-task, the server retains workflow state, and a restarted worker resumes where it left off.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Changelog](CHANGELOG.md)

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
|--------|-------------|
| `--task-id` | Unique task identifier (required for inline) |
| `--description` | What the task should produce (required for inline) |
| `--target-file` | File to create or modify, repeatable (required for inline unless `--plan`) |
| `--context-file` | File to include as context, repeatable |
| `--task-file` | JSON file with a full `TaskDefinition` |
| `--json` | Output `TaskResult` as JSON |
| `--no-wait` | Submit and print workflow ID without waiting for completion |

**Planning options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--plan` | off | Enable planning mode (decompose into ordered steps) |
| `--max-attempts` | `2` | Task-level retry limit |
| `--max-step-attempts` | `2` | Retry limit per step in planning mode |
| `--max-sub-task-attempts` | `2` | Retry limit per sub-task in fan-out steps |

**Validation options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--no-lint` | off | Disable ruff lint check |
| `--no-format` | off | Disable ruff format check |
| `--run-tests` | off | Enable test validation |
| `--test-command` | — | Custom test command (implies `--run-tests`) |

**Context discovery options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--no-auto-discover` | off | Disable automatic context discovery via import graph |
| `--token-budget` | `100000` | Token budget for context assembly |
| `--max-import-depth` | `2` | How deep to trace imports |

**Exploration options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--max-exploration-rounds` | `10` | Max rounds of LLM-guided context exploration (0 disables) |
| `--no-explore` | off | Disable LLM-guided context exploration |

**Common options:**

| Option | Default | Description |
|--------|---------|-------------|
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
|--------|---------|-------------|
| `--temporal-address` | `localhost:7233` | Temporal server address (env: `FORGE_TEMPORAL_ADDRESS`) |

### `forge status`

List recent workflow runs or show details for a specific workflow.

```bash
forge status                              # List recent runs
forge status --workflow-id <id>           # Details for a specific run
forge status --workflow-id <id> --verbose # Full prompts and interaction details
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
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
|--------|---------|-------------|
| `--corpus-dir` | — | Directory containing eval case JSON files (required) |
| `--plans-dir` | — | Directory containing plan JSON files |
| `--judge / --no-judge` | `--no-judge` | Run LLM judge scoring |
| `--judge-model` | `claude-sonnet-4-5-20250929` | Model to use as judge |
| `--dry-run` | off | List cases without evaluating |
| `--output-dir` | — | Directory to save run results JSON |
| `--json` | off | Output results as JSON |

### `forge extract`

Extract knowledge from completed workflow runs into playbook entries.

```bash
forge extract                          # Extract from last 24h, up to 10 runs
forge extract --dry-run                # List unextracted runs without processing
forge extract --limit 50 --since-hours 168  # Last week, up to 50 runs
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--limit` | `10` | Max runs to process |
| `--since-hours` | `24` | Look-back window in hours |
| `--dry-run` | off | List unextracted runs without running extraction |
| `--json` | off | Machine-readable JSON output |
| `--temporal-address` | `localhost:7233` | Temporal server address (env: `FORGE_TEMPORAL_ADDRESS`) |

### `forge playbooks`

List and inspect playbook entries.

```bash
forge playbooks                        # List recent playbooks
forge playbooks --tag python           # Filter by tag
forge playbooks --task-id my-task      # Filter by source task
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--tag` | — | Filter by tag (repeatable) |
| `--task-id` | — | Filter by source task ID |
| `--limit` | `20` | Max entries to show |
| `--json` | off | Machine-readable JSON output |

## Documentation

- `docs/DESIGN.md` — Architecture and design document
- `docs/DECISIONS.md` — Design decisions and rationale
- `docs/PHASE1.md` — Phase 1 spec: the minimal loop
- `docs/PHASE2.md` — Phase 2 spec: planning and multi-step
- `docs/PHASE3.md` — Phase 3 spec: fan-out / gather
- `docs/PHASE4.md` — Phase 4 spec: intelligent context assembly
- `docs/PHASE5.md` — Phase 5 spec: observability store
- `docs/PHASE6.md` — Phase 6 spec: knowledge extraction
- `docs/PHASE7.md` — Phase 7 spec: LLM-guided context exploration
