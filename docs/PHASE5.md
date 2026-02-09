# Phase 5: Observability Store

## Goal

Persist full LLM interaction data to a local SQLite database so operators can inspect prompts, context, token usage, and results for every step of every workflow. Add lightweight statistics to Temporal result payloads. Provide CLI commands for inspecting workflow history and step details.

The deliverable: run a multi-step planned workflow, then use `forge status --workflow-id <id> --verbose` to see the full prompt, assembled context, model name, token usage, and LLM response for each step.

## Problem Statement

Phases 1-4 produce workflow results that contain only output files, validation results, and status signals. The rich intermediate data — assembled prompts, context statistics, LLM responses, token counts, model names, latency — is computed during activity execution and discarded. An operator cannot answer basic questions:

- What prompt did the LLM receive for step 3?
- How much of the token budget was utilized?
- Which model was used and how long did it take?
- What context files were included?

## Scope

**In scope:**

- SQLite observability store with SQLAlchemy ORM and Alembic migrations
- Interaction persistence from `call_llm` and `call_planner` activities
- `LLMStats` model on `TaskResult`, `StepResult`, `SubTaskResult`
- `step_id` and `sub_task_id` on `AssembledContext` for store correlation
- OTel span instrumentation in activities (using existing tracing infrastructure)
- `forge run --verbose` to show interaction details from SQLite
- `forge status` command to list/inspect workflow runs
- Run result persistence to SQLite

**Out of scope:**

- Remote/shared database (PostgreSQL). SQLite is sufficient for single-machine deployment.
- Web UI for browsing interactions. CLI is the interface.
- Tool/function call tracking. The LLM uses structured output, not tool calling.
- Cost estimation from token counts. Can be added later as a store query.
- Streaming/real-time updates. Results are available after workflow completion.

## Architecture

### Observability Store (`src/forge/store.py`)

**Database location:** `$XDG_STATE_HOME/forge/forge.db` (default `~/.local/state/forge/forge.db`). Overridable via `FORGE_DB_PATH` env var. Empty string disables the store.

**SQLAlchemy models:**

```
class Interaction(Base):
    __tablename__ = "interactions"

    id: int (PK, autoincrement)
    task_id: str (NOT NULL, indexed)
    step_id: str | None
    sub_task_id: str | None
    role: str (NOT NULL)          # "llm" or "planner"
    system_prompt: str (NOT NULL)
    user_prompt: str (NOT NULL)
    model_name: str (NOT NULL)
    input_tokens: int (NOT NULL)
    output_tokens: int (NOT NULL)
    latency_ms: float (NOT NULL)
    explanation: str (default "")
    context_stats_json: str | None  # JSON-serialized ContextStats
    created_at: datetime (server default: now UTC)

class Run(Base):
    __tablename__ = "runs"

    id: int (PK, autoincrement)
    task_id: str (NOT NULL, indexed)
    workflow_id: str (NOT NULL, unique)
    status: str (NOT NULL)
    result_json: str (NOT NULL)   # JSON-serialized TaskResult (without prompts)
    created_at: datetime (server default: now UTC)
```

**Function Core / Imperative Shell:**

Pure functions:

- `get_db_path() -> Path | None` — XDG resolution
- `build_interaction_dict(task_id, step_id, sub_task_id, role, context, llm_result) -> dict` — assembles a dict from activity data

Imperative shell:

- `get_engine(db_path: Path) -> Engine` — create SQLAlchemy engine with WAL mode
- `save_interaction(engine, ...)` — insert into interactions table
- `save_run(engine, result: TaskResult, workflow_id: str)` — insert into runs table
- `get_interactions(engine, task_id, step_id=None) -> list[Interaction]` — query
- `list_recent_runs(engine, limit=20) -> list[Run]` — query

### Alembic Configuration

```
src/forge/
├── store.py                    # SQLAlchemy models and store functions
└── alembic/
    ├── alembic.ini             # Config (references get_db_path for URL)
    ├── env.py                  # Migration environment
    └── versions/
        └── 001_initial.py      # Initial schema: interactions + runs tables
```

Alembic config resolves the database URL from `get_db_path()`. Migrations run automatically on worker startup (or on first store access). `forge worker` calls `alembic.command.upgrade("head")` during initialization.

### Model Enrichment

Add to `src/forge/models.py`:

```
class LLMStats(BaseModel):
    """Lightweight LLM call statistics for Temporal payloads."""
    model_name: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
```

Add optional fields (backward compatible defaults):

- `StepResult.llm_stats: LLMStats | None = None`
- `SubTaskResult.llm_stats: LLMStats | None = None`
- `TaskResult.llm_stats: LLMStats | None = None` (single-step)
- `TaskResult.planner_stats: LLMStats | None = None`
- `TaskResult.context_stats: ContextStats | None = None`

Add pure builder: `build_llm_stats(llm_result: LLMCallResult) -> LLMStats`

### Activity Changes

**`AssembledContext`:** Add `step_id: str | None = None` and `sub_task_id: str | None = None` for store correlation.

**`activities/context.py`:** Populate `step_id` in `assemble_step_context`, `sub_task_id` in `assemble_sub_task_context`.

**`activities/llm.py`:** In `call_llm` (activity shell only), after `execute_llm_call`:

```python
_persist_interaction(context, result, role="llm")
```

Where `_persist_interaction` resolves the store engine, calls `save_interaction`, and catches all exceptions with a warning log.

**`activities/planner.py`:** Same pattern in `call_planner`.

The testable functions (`execute_llm_call`, `execute_planner_call`) are unchanged.

### Workflow Changes

**`workflows.py`:** After LLM/planner calls, build `LLMStats` and pass to result constructors. No store access in workflows (determinism).

### OTel Instrumentation

**`worker.py`:** Call `init_tracing()` / `shutdown_tracing()` around `worker.run()`.

**Activities:** Wrap activity bodies in spans using `get_tracer()`:

- `call_llm` -> `forge.call_llm` span with `llm_call_attributes()`
- `call_planner` -> `forge.call_planner` span
- `assemble_context` -> `forge.assemble_context` span
- `validate_output` -> `forge.validate_output` span with `validation_attributes()`

### CLI Enhancements

**`forge run --verbose`:** After getting the result, query SQLite for interactions and display full prompts, context stats, token usage per step.

**`forge status`:** New command:

```
forge status                              # list recent runs from SQLite
forge status --workflow-id <id>           # details for a specific run
forge status --workflow-id <id> --verbose # full prompts from SQLite
forge status --json                       # machine-readable output
```

## Project Structure

New and modified files:

```
src/forge/
├── store.py                         # New: SQLAlchemy models, store functions
├── alembic/                         # New: Alembic migration configuration
│   ├── alembic.ini
│   ├── env.py
│   └── versions/
│       └── 001_initial.py
├── models.py                        # Modified: LLMStats, AssembledContext fields
├── worker.py                        # Modified: init_tracing, auto-migrate
├── cli.py                           # Modified: --verbose, forge status
├── tracing.py                       # Unchanged (reuse existing functions)
└── activities/
    ├── llm.py                       # Modified: store write after LLM call
    ├── planner.py                   # Modified: store write after planner call
    ├── context.py                   # Modified: step_id/sub_task_id on AssembledContext
    └── validate.py                  # Modified: OTel span
```

## Dependencies

New runtime dependencies:

- `sqlalchemy>=2.0` — Database access and ORM
- `alembic>=1.14` — Schema migration management

## Key Design Decisions

### D39: SQLite Observability Store Outside Temporal Payloads

**Decision:** Full LLM interaction data (prompts, context, responses) is persisted to a local SQLite database rather than stored in Temporal workflow results. Temporal payloads carry only lightweight statistics (model name, token counts, latency).

**Rationale:** Temporal has a ~2MB payload limit for workflow results. A planned workflow with 5 steps could produce 2MB+ of prompts alone (each step assembles up to 100k tokens of context). Storing prompts in the Temporal result would hit this limit. A local SQLite database has no such constraint and provides rich queryability (filter by task, step, date range). The Temporal result remains lean and fast to retrieve, while the full observability data is available via CLI queries against SQLite.

### D40: SQLAlchemy for Database Access

**Decision:** Use SQLAlchemy (Core + ORM) for all database access rather than raw `sqlite3`.

**Rationale:** SQLAlchemy provides a well-tested abstraction over database connections, transactions, and schema definition. It handles connection pooling, thread safety, and migration support (via Alembic) that would require manual implementation with raw `sqlite3`. The ORM layer maps directly to Pydantic models, reducing boilerplate. SQLAlchemy also enables future migration to PostgreSQL or another backend without changing application code — important if Forge eventually runs with a shared database in a multi-worker deployment.

### D41: Alembic for Schema Management

**Decision:** Use Alembic for database schema migrations rather than ad-hoc `CREATE TABLE IF NOT EXISTS` statements.

**Rationale:** The observability schema will evolve as Forge adds features (tool calling, cost tracking, evaluation scores). Alembic provides versioned migrations, rollback support, and autogeneration from SQLAlchemy models. This prevents schema drift between development and production databases and makes schema changes reviewable in version control. The initial migration creates the baseline schema; subsequent features add migrations incrementally.

### D42: Best-Effort Store Writes in Activities

**Decision:** Store writes in LLM activities are wrapped in try/except and log warnings on failure. The store never blocks or fails the main workflow.

**Rationale:** Observability is secondary to task execution. If the database is unavailable (disk full, permissions, corruption), the LLM call should still succeed and return its result to the workflow. The store is a side effect, not a dependency. This also simplifies testing — activities can be tested without a database by setting `FORGE_DB_PATH` to empty string.

## Implementation Order

1. `store.py` — SQLAlchemy models, engine creation, CRUD functions
2. `alembic/` — Migration config and initial migration
3. `models.py` — `LLMStats`, `build_llm_stats`, `AssembledContext` fields
4. `activities/context.py` — Populate `step_id`/`sub_task_id`
5. `activities/llm.py` — Store write after LLM call
6. `activities/planner.py` — Store write after planner call
7. `workflows.py` — Wire up `LLMStats` in all result constructors
8. `worker.py` — `init_tracing()`, `shutdown_tracing()`, auto-migrate
9. `activities/` — OTel spans in llm, planner, context, validate
10. `cli.py` — `--verbose` flag and `forge status` command

## Edge Cases

- **Database doesn't exist:** `get_engine()` creates it. Alembic `upgrade("head")` creates tables.
- **Store write fails:** Caught with warning log. LLM call result is returned normally.
- **`FORGE_DB_PATH=""`:** Store is disabled. Activities skip persistence. CLI verbose mode warns "no store available."
- **Concurrent writes:** SQLite WAL mode handles concurrent writers from async activities. SQLAlchemy manages connection pooling.
- **Large prompts:** SQLite has no practical row size limit (default 1GB). A 400KB prompt is fine.
- **Schema upgrade needed:** Worker runs `alembic upgrade head` on startup. Safe for concurrent workers (Alembic uses migration locking).

## Definition of Done

Phase 5 is complete when:

- Every `call_llm` and `call_planner` activity writes full interaction data to SQLite
- `TaskResult`, `StepResult`, `SubTaskResult` carry `LLMStats` (model, tokens, latency)
- `forge run --verbose` displays full prompts, context stats, and token usage per step
- `forge status` lists recent runs and shows details for a specific workflow
- OTel spans are emitted for LLM, planner, context, and validation activities
- All Phase 1-4 tests continue to pass (store is a side effect, not a dependency)
- New unit tests cover: store CRUD, `LLMStats` model, CLI verbose formatting, CLI status command
- The store can be disabled via `FORGE_DB_PATH=""` for testing
