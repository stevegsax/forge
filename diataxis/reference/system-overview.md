+++
title = "System Overview Reference"
weight = 14
description = "What Forge is, the core principles that drive its design, and how the major components fit together."
topic = "system-overview"
covers = [
    "Architecture principles table (principle, consequence, enforcement point)",
    "Module map — every package and top-level module with a one-line description",
    "Technology stack table (component, technology, version/source)",
    "Environment variables table (all FORGE_* and XDG vars, defaults, effects)",
    "File system layout (config, state, logs, database)",
]
detail = "Tabular reference. One table per section. No narrative prose beyond brief introductions to each table."
+++
Prerequisite: Basic familiarity with [Temporal](https://docs.temporal.io/) and LLM APIs.

Tabular reference for Forge's architecture principles, module layout, technology stack, environment variables, and file system paths. For the design rationale behind these choices, see the [System Overview Explanation](../explanation/system-overview/).

## Architecture Principles

| # | Principle | Consequence | Enforcement Point |
|---|-----------|-------------|-------------------|
| 1 | Batch-first | Every LLM call is a self-contained document completion; no multi-turn conversations | Prompt construction in `activities/context.py`; batch submission in `activities/batch_submit.py` |
| 2 | Deterministic work should be deterministic | Pre-compute facts (import graphs, PageRank, token budgets) and include them in context | `code_intel/` package; `activities/context.py` |
| 3 | Context isolation is a feature | Each task gets a fresh context with no shared conversation history | `activities/context.py` assembles context per call |
| 4 | Planning is the hard part | Planner uses reasoning-tier model and highest token budget | `activities/planner.py`; `domains.py` capability tier defaults |
| 5 | Halt when confused | Unclassifiable results produce `FAILURE_TERMINAL` and structured escalation reports | `activities/transition.py`; `workflows.py` |
| 6 | The LLM call is the universal primitive | Every task type follows construct-send-receive-serialize-transition | `workflows.py`; `workflow_blocks.py` |
| 7 | Follow Temporal best practices | Workflows are deterministic; all I/O lives in activities; fan-out uses child workflows | `workflows.py`; `worker.py` |

## Module Map

The module layout mirrors `src/forge/`. For detailed reference on individual modules, see topic-specific reference docs.

### Top-Level Modules

| Module | Description |
|--------|-------------|
| `__init__.py` | Package initialization |
| `workflows.py` | Temporal workflow definitions (ForgeTaskWorkflow, ForgeSubTaskWorkflow) |
| `workflow_blocks.py` | Shared workflow building blocks reused across workflow types |
| `batch_poller_workflow.py` | Batch status polling workflow; delivers results via Temporal signals |
| `extraction_workflow.py` | Forge run extraction workflow; processes completed forge runs into forge playbooks |
| `ingestion_workflow.py` | Transcript ingestion workflows (`TranscriptIngestionWorkflow`, `BatchIngestionWorkflow`); reads Claude Code sessions and hands experiences to pbook cross-queue |
| `export_playbook_workflow.py` | Playbook export workflow |
| `manual_playbook_workflow.py` | Manual playbook entry workflow |
| `models.py` | Pydantic data models (TaskDefinition, ForgeTaskInput, LLMResponse, Plan, TaskResult, etc.) |
| `llm_client.py` | Anthropic API request construction and response parsing |
| `providers.py` | Context provider registry (12 providers for exploration loop) |
| `domains.py` | Domain configurations (code_generation, research, code_review, documentation, generic) |
| `cli.py` | CLI entry point (forge run, forge worker, forge status, forge extract, etc.) |
| `worker.py` | Temporal worker process setup and registration |
| `git.py` | Git operations: worktree create/remove, branch management, commit, reset |
| `store.py` | SQLite observability store (interactions, runs, playbooks, batch_jobs tables) |
| `tracing.py` | OpenTelemetry instrumentation and span management |
| `logging_config.py` | Log file paths, rotation policy, verbosity configuration |
| `message_log.py` | API message logging utilities (request/response JSON to worktree) |
| `subprocess_result.py` | Subprocess result models for validation and git operations |

### `activities/` -- Temporal Activity Implementations

| Module | Description |
|--------|-------------|
| `__init__.py` | Activity registration and exports |
| `_heartbeat.py` | Heartbeat management for long-running activities |
| `context.py` | Prompt assembly (system prompt + user prompt construction) |
| `llm.py` | LLM call execution (sync and batch paths) |
| `output.py` | File writing and edit application with four-level fallback chain |
| `planner.py` | Planning LLM call (task decomposition into ordered steps) |
| `exploration.py` | Exploration loop LLM calls (context request/fulfillment rounds) |
| `extraction.py` | Forge run extraction activities (fetch unextracted runs, call LLM, save entries to forge's playbooks table) |
| `ingestion.py` | Transcript ingestion activity (`prepare_transcript`); parses Claude Code JSONL and builds analysis prompts |
| `playbook_export.py` | Playbook export activity |
| `playbook_review.py` | Playbook review activity |
| `sanity_check.py` | Mid-plan sanity checks (CONTINUE, REVISE, or ABORT) |
| `conflict_resolution.py` | Fan-out file conflict resolution via reasoning-tier LLM |
| `git_activities.py` | Worktree create/remove/reset/commit activities |
| `batch_submit.py` | Batch API submission activity |
| `batch_parse.py` | Batch response parsing activity |
| `batch_poll.py` | Batch status polling activity |
| `validate.py` | Deterministic validation (ruff lint, ruff format, test execution) |
| `transition.py` | Outcome signal evaluation (SUCCESS, FAILURE_RETRYABLE, FAILURE_TERMINAL) |

### `code_intel/` -- Code Intelligence and Analysis

| Module | Description |
|--------|-------------|
| `__init__.py` | Package exports and convenience functions |
| `graph.py` | Import graph analysis via grimp and PageRank ranking via networkx |
| `parser.py` | Symbol extraction from Python files via the ast module |
| `budget.py` | Token budget packing (knapsack-style priority algorithm) |
| `repo_map.py` | Repository structure mapping (PageRank-ranked file tree with signatures) |

### `eval/` -- Planner Evaluation Framework

| Module | Description |
|--------|-------------|
| `__init__.py` | Package exports |
| `runner.py` | Evaluation harness (runs eval cases, aggregates results) |
| `deterministic.py` | Deterministic structural checks for planner output |
| `judge.py` | LLM-as-judge scoring for plan quality |
| `models.py` | Evaluation data models (EvalCase, EvalResult, DeterministicCheckResult) |
| `corpus.py` | Test corpus management (loading and validating eval cases) |

### `llm_providers/` -- LLM Provider Abstraction

| Module | Description |
|--------|-------------|
| `__init__.py` | Package exports |
| `protocol.py` | Provider protocol definition (interface contract) |
| `registry.py` | Provider registry and resolution (tier-to-model mapping) |
| `models.py` | Provider data models (ProviderConfig, ModelSpec) |
| `anthropic.py` | Anthropic provider implementation |
| `mistral.py` | Mistral provider implementation |

### `ocr/` -- OCR Pipeline

| Module | Description |
|--------|-------------|
| `__init__.py` | Package exports |
| `activities.py` | OCR pipeline activities (API calls, image extraction, storage) |
| `models.py` | OCR data models (OcrInput, OcrResult, OcrImage) |
| `workflow_sync.py` | Synchronous OCR workflow (direct Mistral API call) |
| `workflow_submit.py` | Batch OCR submission workflow (Mistral Batch API) |
| `workflow_store.py` | Batch result storage workflow |
| `workflow_gather.py` | Multi-chunk document gathering workflow |
| `workflow_export.py` | Document export workflow |
| `workflow_mark_removal.py` | Mark-for-removal workflow |

### `alembic/` -- Database Migrations

| Path | Description |
|------|-------------|
| `alembic.ini` | Alembic configuration |
| `env.py` | Migration environment setup |
| `versions/` | Versioned migration scripts |

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Workflow orchestration | [Temporal](https://docs.temporal.io/) | Durable execution, retry semantics, child workflows, signal handling |
| LLM client library | `anthropic` (direct SDK) | Anthropic API request/response handling |
| LLM providers | Anthropic, Mistral | Model inference (Anthropic for core tasks, Mistral for OCR) |
| Data models | `pydantic` | Input/output validation, structured LLM responses |
| Observability store | SQLite + SQLAlchemy | Persistent LLM interaction data, playbooks, batch job tracking |
| Schema migrations | Alembic | Versioned database schema changes |
| Distributed tracing | OpenTelemetry | Span hierarchy from pipeline run to individual LLM request |
| Code analysis | Python `ast`, grimp, networkx | Import graph, symbol extraction, PageRank ranking |
| Data store / isolation | git, git worktrees | Task isolation, conflict detection, audit trail |
| Python linting/formatting | ruff | Deterministic validation of generated code |
| Package management | uv | Dependency resolution and virtual environment management |

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `FORGE_DB_PATH` | Override observability store path; empty string disables the store | `~/.local/state/forge/forge.db` |
| `FORGE_LOG_DIR` | Override log file directory; empty string disables file logging | `~/.local/state/forge/` |
| `FORGE_OTEL_EXPORTER` | OTel trace exporter type (`console`, `otlp_grpc`, `otlp_http`, `none`) | `console` |
| `FORGE_OTEL_ENDPOINT` | OTel exporter endpoint URL | Per-exporter default |
| `XDG_STATE_HOME` | Base directory for logs and database (per XDG Base Directory Specification) | `~/.local/state` |
| `ANTHROPIC_API_KEY` | Anthropic API authentication key | None (required) |
| `MISTRAL_API_KEY` | Mistral API authentication key (required for OCR) | None |

## File System Layout

All paths follow the [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/latest/).

| Path | Contents |
|------|----------|
| `$XDG_STATE_HOME/forge/forge.db` | SQLite observability store (interactions, runs, playbooks, batch_jobs) |
| `$XDG_STATE_HOME/forge/forge.log` | Application log (CLI commands) |
| `$XDG_STATE_HOME/forge/worker.log` | Worker log (Temporal worker process) |
| `<worktree>/messages/` | API message logs (request/response JSON, created with `--log-messages`) |
| `tool-config/ruff.toml` | Ruff configuration shared across all worktrees |

Log rotation: 10 MB maximum file size, 5 backup files retained.

## See Also

- [System Overview Explanation](../explanation/system-overview/) -- design rationale and principles
- [The Universal Workflow Step](workflow-step/) -- the five-phase execution pattern
- [Golden Path Tutorial](../tutorials/golden-path/) -- end-to-end walkthrough of a planned task
