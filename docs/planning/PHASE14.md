# Phase 14: Batch Processing

## Goal

Replace synchronous LLM calls with asynchronous batch processing via the Anthropic Message Batches API. The workflow submits a request, yields control, and resumes when the result arrives — cutting token costs by 50% and decoupling workflow execution from LLM latency.

The deliverable: all LLM calls (generation, planning, exploration, sanity check, conflict resolution) go through the Batch API by default. A `--sync` flag preserves the current synchronous path. A Temporal Schedule polls for completed batches and delivers results via Temporal signals.

## Prerequisite: Remove pydantic-ai (D75)

The current codebase routes all LLM calls through pydantic-ai's `Agent` class, which bundles request construction, API call, and response parsing into a single `agent.run()` call. Batch mode requires splitting this into submit (construct + send) and resume (receive + parse), which pydantic-ai does not support.

**Before this phase begins**, replace pydantic-ai with direct Anthropic SDK calls + plain Pydantic:

- **Request construction**: Build Anthropic Messages API requests directly. Generate tool definitions from Pydantic models via `Model.model_json_schema()`.
- **Response parsing**: Extract tool call arguments from the Anthropic `Message` response. Validate with `Model.model_validate_json()`.
- **Structured output**: The same Pydantic models (`LLMResponse`, `Plan`, `ExplorationResponse`, `SanityCheckResponse`, `ConflictResolutionResponse`) define the tool schemas and validate responses. Both sync and batch paths share these models.

This prerequisite is a refactoring — behavior is unchanged, only the call mechanism changes.

## Problem Statement

Every LLM call in Forge is synchronous: the workflow blocks an activity slot while waiting for the Anthropic Messages API to respond. This has three costs:

1. **Token cost**: The synchronous Messages API charges full price. The Batch API charges 50% for the same work.
2. **Worker slot consumption**: Each in-flight LLM call holds an activity slot for 30-300 seconds. Fan-out with 10 sub-tasks holds 10 slots simultaneously.
3. **Tight coupling**: The workflow's progress is bound to API response latency. Network hiccups, rate limits, and API slowdowns directly stall the workflow.

The Batch API decouples submission from completion. The workflow submits a request, yields its activity slot, and resumes only when the result is ready.

## Prior Art

- **Anthropic Message Batches API**: Accepts up to 100K requests per batch, processes asynchronously, returns results within 24 hours (most within 1 hour). 50% cost reduction. Results available for 29 days. Supports tool use, system messages, and prompt caching (cache hits are best-effort, 30-98%).
- **Temporal signals**: External events delivered to a running workflow. The workflow registers a signal handler, then calls `workflow.wait_condition()` to block until the signal arrives. Temporal durably persists the workflow state — if the worker crashes, the workflow resumes waiting on restart.
- **Temporal Search Attributes**: Custom indexed attributes on workflows, queryable via Temporal's visibility API. The batch poller uses these to find workflows waiting for specific batch results.
- **Temporal Schedules**: Cron-like scheduling managed by the Temporal server. The batch poller runs on a Schedule rather than a custom polling loop.

## Scope

**In scope:**

- Batch submission activity for all five LLM call sites.
- Signal-based wait in workflow (submit → wait → resume).
- Temporal Schedule for batch polling (configurable interval, default 60s).
- Temporal Search Attribute (`forge_batch_id`) for poller→workflow routing.
- `batch_jobs` audit table in the existing SQLite store.
- `--sync` flag to bypass batch mode and call the Messages API directly.
- `--batch-poll-interval` configuration (default 60s).
- Missing job detection (jobs in our DB but absent from server).
- Unknown job logging (jobs on server but not in our DB).
- Knowledge extraction migration to Temporal Schedule.

**Out of scope (deferred to Release 2):**

- Multi-request batches (grouping multiple LLM calls into a single batch submission).
- Prompt caching optimization for batch requests.
- Per-call routing (some calls to batch, some to sync, some to local LLMs).
- Batch cancellation on workflow failure/timeout.
- Batch cost tracking and reporting.

## Architecture

### Overview

Three components work together:

1. **The workflow** submits a batch request, sets a search attribute, and waits for a signal.
2. **The batch poller** (Temporal Schedule) periodically checks the Anthropic API, finds completed batches, and sends signals to waiting workflows.
3. **The batch_jobs table** records submissions and outcomes for observability and anomaly detection.

```
┌─────────────────┐     submit      ┌─────────────────┐
│  Forge Workflow  │───────────────→│  Anthropic Batch │
│  (Temporal)      │                │  API             │
│                  │←───signal──────│                  │
│  wait_condition  │     │          └─────────────────┘
└─────────────────┘     │                    ↑
        ↑               │                   │ poll
        │           ┌───┴────────┐          │
        │           │ Batch      │──────────┘
        └──signal───│ Poller     │
                    │ (Schedule) │
                    └────────────┘
```

### Modified Workflow: Signal-Based Wait

The workflow gains a signal handler and a helper method that replaces direct LLM activity calls:

```python
@workflow.defn
class ForgeTaskWorkflow:
    def __init__(self):
        self._batch_result: BatchResult | None = None

    @workflow.signal
    async def batch_result_received(self, result: BatchResult):
        self._batch_result = result

    async def _call_llm(
        self,
        context: AssembledContext,
        output_type_name: str,
    ) -> Message:
        """Submit an LLM call via batch or sync, return raw Anthropic Message."""
        if self._sync_mode:
            return await workflow.execute_activity(
                "call_llm_sync",
                CallLLMInput(context=context, output_type_name=output_type_name),
                start_to_close_timeout=_LLM_TIMEOUT,
                result_type=Message,
            )

        # --- Batch path ---
        batch_info = await workflow.execute_activity(
            "submit_batch_request",
            BatchSubmitInput(
                context=context,
                output_type_name=output_type_name,
                workflow_id=workflow.info().workflow_id,
            ),
            start_to_close_timeout=_SUBMIT_TIMEOUT,
            result_type=BatchSubmitResult,
        )

        # Set search attribute so the poller can find us
        workflow.upsert_search_attributes({"forge_batch_id": batch_info.batch_id})

        # Wait for signal (Temporal durably persists this state)
        await workflow.wait_condition(lambda: self._batch_result is not None)
        result = self._batch_result
        self._batch_result = None

        # Clear search attribute
        workflow.upsert_search_attributes({"forge_batch_id": ""})

        if result.error:
            raise BatchError(result.error)

        return result.raw_response
```

Each call site uses this helper, then parses the raw `Message` into its expected output type via a parse activity:

```python
# In _run_single_step:
raw_response = await self._call_llm(context, output_type_name="LLMResponse")
llm_result = await workflow.execute_activity(
    "parse_llm_response",
    ParseResponseInput(raw_response=raw_response, context=context),
    start_to_close_timeout=_PARSE_TIMEOUT,
    result_type=LLMCallResult,
)
```

### Batch Submission Activity

```python
@activity.defn
async def submit_batch_request(input: BatchSubmitInput) -> BatchSubmitResult:
    """Submit a single-request batch to the Anthropic Batch API.

    Retries submission up to 3 times. Records the submission in the
    batch_jobs audit table. On permanent failure, marks the record as
    FAILED and raises.
    """
    request_id = str(uuid4())
    api_request = build_anthropic_request(
        context=input.context,
        output_type_name=input.output_type_name,
    )

    batch = submit_with_retries(
        client=get_anthropic_client(),
        request_id=request_id,
        api_request=api_request,
        max_retries=3,
    )

    record_batch_submission(
        request_id=request_id,
        batch_id=batch.id,
        workflow_id=input.workflow_id,
    )

    return BatchSubmitResult(request_id=request_id, batch_id=batch.id)
```

The `build_anthropic_request` function constructs the Anthropic Messages API parameters (system, messages, tools, model, max_tokens) from the `AssembledContext` and the output type's JSON schema. This function is shared between sync and batch paths.

### Response Parsing Activity

A single `parse_llm_response` activity handles all output types:

```python
@activity.defn
async def parse_llm_response(input: ParseResponseInput) -> LLMCallResult:
    """Parse a raw Anthropic Message into a typed LLM result.

    Extracts tool call arguments, validates against the expected Pydantic
    model, and builds the LLMCallResult with usage statistics.
    """
    output_type = OUTPUT_TYPE_REGISTRY[input.output_type_name]
    parsed = extract_tool_result(input.raw_response, output_type)
    usage = input.raw_response.usage

    return LLMCallResult(
        task_id=input.context.task_id,
        response=parsed,
        model_name=input.raw_response.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        latency_ms=0,  # Not meaningful for batch
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0),
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0),
    )
```

The `OUTPUT_TYPE_REGISTRY` maps type names to Pydantic model classes:

```python
OUTPUT_TYPE_REGISTRY: dict[str, type[BaseModel]] = {
    "LLMResponse": LLMResponse,
    "Plan": Plan,
    "ExplorationResponse": ExplorationResponse,
    "SanityCheckResponse": SanityCheckResponse,
    "ConflictResolutionResponse": ConflictResolutionResponse,
}
```

Each LLM call site (planner, exploration, sanity check, conflict resolution) has its own parse activity that returns the appropriate result type. All share the same `extract_tool_result` core function.

### Batch Poller (Temporal Schedule)

A Temporal Schedule triggers `BatchPollerWorkflow` at a configurable interval (default 60 seconds):

```python
@workflow.defn
class BatchPollerWorkflow:
    @workflow.run
    async def run(self, input: BatchPollerInput) -> BatchPollerResult:
        return await workflow.execute_activity(
            "poll_batch_results",
            input,
            start_to_close_timeout=timedelta(minutes=5),
            result_type=BatchPollerResult,
        )
```

The `poll_batch_results` activity:

1. **List all batches** from the Anthropic API.
2. **For each completed batch** (`processing_status == "ended"`):

    a. Retrieve results via `client.messages.batches.results(batch_id)`.
    b. For each result, extract the `custom_id` (our request UUID).
    c. Query the `batch_jobs` table for the corresponding `workflow_id`.
    d. Query Temporal visibility for the workflow with search attribute `forge_batch_id == batch_id`.
    e. If the workflow is found and still waiting, send the `batch_result_received` signal with the raw `Message` response (or error details).
    f. Update the `batch_jobs` record with the final status.

3. **Detect anomalies**:

    - **Unknown server batch**: A batch on the server whose `custom_id` does not match any record in `batch_jobs`. Log an INFO note; take no action.
    - **Missing server batch**: A `batch_jobs` record with status SUBMITTED whose `batch_id` does not appear in the server's batch list and was submitted more than 24 hours ago. Log an ERROR (indicates data corruption or loss).
    - **Status regression**: A `batch_jobs` record previously marked SUCCEEDED that now shows a different status on the server. Log an ERROR.
    - **Server-side error**: A batch result with type `errored`. Log an ERROR. Send a signal to the workflow with the error details so it can transition appropriately.
    - **Expired request**: A batch result with type `expired`. Log a WARNING. Send a signal with an expiration error so the workflow can retry or fail.

### Batch Jobs Table (Audit Log)

Added to the existing SQLite store. This table is for **observability and anomaly detection only** — all coordination uses Temporal signals and search attributes.

```python
class BatchJob(Base):
    __tablename__ = "batch_jobs"

    id: Mapped[str] = mapped_column(sa.String, primary_key=True)  # Request UUID (= Anthropic custom_id)
    batch_id: Mapped[str] = mapped_column(sa.String, nullable=False, index=True)
    workflow_id: Mapped[str] = mapped_column(sa.String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(sa.String, nullable=False)  # SUBMITTED / SUCCEEDED / ERRORED / EXPIRED / CANCELED / MISSING
    error_message: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=...)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime, default=..., onupdate=...)
```

Follows D42 (best-effort store writes): recording failures are logged but never block workflow execution.

### Temporal Search Attribute

A custom search attribute `forge_batch_id` (type: Keyword) is registered on the Temporal namespace. Workflows set this attribute after batch submission and clear it after receiving the signal.

The poller queries Temporal visibility:

```
forge_batch_id = "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d"
```

For Release 1 (single-request batches), the batch_id uniquely identifies both the batch and the single request within it, so one search attribute suffices. Release 2 (multi-request batches) will add a `forge_batch_request_id` attribute for per-request routing.

### Knowledge Extraction Schedule

Migrate knowledge extraction from manual CLI invocation to a Temporal Schedule:

- **Schedule**: Runs `ExtractionWorkflow` at a configurable interval (default: every 4 hours).
- **Logic**: Queries `get_unextracted_runs()` (already implemented in `store.py`). If unextracted runs exist, processes them. If none, completes immediately.
- **CLI**: The manual `forge extract` command remains available for on-demand extraction.

### Sync Mode

When `--sync` is passed (or configured in settings), the workflow calls `call_llm_sync` directly — identical to the current behavior. No batch submission, no signals, no poller dependency.

The `_call_llm` helper method checks `self._sync_mode` (set from `ForgeTaskInput.sync_mode`) and dispatches accordingly. Both paths use the same request construction and response parsing functions (post pydantic-ai removal).

### Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--sync` | `False` | Use synchronous Messages API instead of Batch API |
| `--batch-poll-interval` | `60` | Seconds between batch polling runs |
| `--extraction-interval` | `14400` | Seconds between knowledge extraction schedule runs (4 hours) |

## Data Models

```python
class BatchSubmitInput(BaseModel):
    context: AssembledContext
    output_type_name: str
    workflow_id: str

class BatchSubmitResult(BaseModel):
    request_id: str  # UUID, used as Anthropic custom_id
    batch_id: str    # Anthropic msgbatch_... ID

class BatchResult(BaseModel):
    """Delivered via Temporal signal."""
    request_id: str
    batch_id: str
    raw_response_json: str | None = None  # Serialized Anthropic Message
    error: str | None = None
    result_type: str  # "succeeded" / "errored" / "expired" / "canceled"

class ParseResponseInput(BaseModel):
    raw_response_json: str
    output_type_name: str
    context: AssembledContext

class BatchPollerInput(BaseModel):
    pass  # Poller needs no input; reads from Anthropic API and Temporal visibility

class BatchPollerResult(BaseModel):
    batches_checked: int
    signals_sent: int
    errors_found: int
```

## Project Structure

New files:

```
src/forge/
├── batch.py                    # build_anthropic_request, extract_tool_result, OUTPUT_TYPE_REGISTRY
├── activities/
│   ├── batch_submit.py         # submit_batch_request activity
│   └── batch_poll.py           # poll_batch_results activity
└── batch_poller_workflow.py    # BatchPollerWorkflow (Temporal Schedule target)
```

Modified files:

```
src/forge/
├── models.py                   # New data models (BatchSubmitInput, BatchResult, etc.)
├── store.py                    # New BatchJob table, migration
├── workflows.py                # Signal handler, _call_llm helper, sync_mode dispatch
├── worker.py                   # Register new activities, Schedule setup, search attribute registration
├── cli.py                      # --sync, --batch-poll-interval, --extraction-interval flags
└── activities/
    ├── llm.py                  # Refactored to use batch.py (post pydantic-ai removal)
    ├── planner.py              # Route through _call_llm
    ├── exploration.py          # Route through _call_llm
    ├── sanity_check.py         # Route through _call_llm
    └── conflict_resolution.py  # Route through _call_llm
```

## Dependencies

- `anthropic` (already installed, SDK version 0.78.0) — `client.messages.batches` for batch operations.
- No new dependencies.

## Key Design Decisions

Decisions D75–D82 are recorded in `DECISIONS.md`.

- **D75**: Remove pydantic-ai (prerequisite).
- **D76**: Batch mode as default execution path.
- **D77**: Signal-based wait over terminate-and-restart.
- **D78**: Temporal Search Attributes for poller→workflow routing.
- **D79**: Single-request batches for Release 1.
- **D80**: Batch jobs table as audit log (not coordination).
- **D81**: Temporal Schedules for batch polling and knowledge extraction.
- **D82**: All LLM call sites go through batch.
- **D83**: Decompose Phase 14 into sub-phases 14a/14b/14c.

## Implementation Order

Phase 14 is decomposed into three sub-phases (D83), each independently deliverable with all tests passing.

### Prerequisite (complete)

Remove pydantic-ai. Replace with direct Anthropic SDK + Pydantic. All existing tests pass.

### Phase 14a — Batch Infrastructure

Foundation layer. No workflow changes — everything testable in isolation.

1. Add `BatchSubmitInput`, `BatchSubmitResult`, `BatchResult`, `ParseResponseInput` to `models.py`.
2. Add `BatchJob` table to `store.py` with Alembic migration.
3. Add `batch.py` with `build_anthropic_request`, `extract_tool_result`, and `OUTPUT_TYPE_REGISTRY`.
4. Add `submit_batch_request` activity in `activities/batch_submit.py`.
5. Add `parse_llm_response` activity (and per-call-type variants) in the respective activity modules.
6. Tests for all of the above.

### Phase 14b — Workflow Batch Integration

Signal plumbing and call site wiring. Defaults to sync mode until the poller exists in 14c.

7. Add signal handler and `_call_llm` helper to `ForgeTaskWorkflow` and `ForgeSubTaskWorkflow`.
8. Register `forge_batch_id` custom search attribute on Temporal namespace.
9. Add `--sync` CLI flag (default: sync enabled, batch opt-in until 14c).
10. Update all five call sites (generation, planner, exploration, sanity check, conflict resolution) to use `_call_llm`.
11. Workflow and E2E tests.

### Phase 14c — Batch Poller + Scheduling

Completes the loop. Flips default to batch mode.

12. Add `poll_batch_results` activity in `activities/batch_poll.py`.
13. Add `BatchPollerWorkflow` in `batch_poller_workflow.py`.
14. Register the Temporal Schedule for batch polling in `worker.py`.
15. Add `--batch-poll-interval` CLI flag.
16. Anomaly detection (missing/unknown/expired batches).
17. Migrate knowledge extraction to Temporal Schedule with `--extraction-interval` flag.
18. Flip default from sync to batch mode.
19. Tests for poller, scheduling, and anomaly detection.

## Edge Cases

- **Worker crash while waiting for signal**: Temporal durably persists workflow state. On worker restart, the workflow resumes waiting. The poller eventually delivers the signal.
- **Batch expires (24-hour limit)**: The poller sends a signal with `result_type="expired"`. The workflow treats this as a transient failure and retries (creates a new batch submission).
- **Submission fails after 3 retries**: The activity raises an exception. Temporal's activity retry policy handles transient network errors. If all retries fail, the workflow transitions to FAILURE_TERMINAL.
- **Duplicate signals**: If the poller sends a signal for an already-completed workflow (race condition), Temporal discards the signal. No harm.
- **Multiple pollers running**: The Temporal Schedule guarantees at most one concurrent execution by default. If configured otherwise, signal delivery is idempotent.
- **Batch result contains `errored`**: The poller sends a signal with the error. The workflow logs the error and decides retry vs. terminal based on the error type (invalid_request → terminal, server_error → retryable).
- **Exploration loop latency**: Each exploration round submits a batch and waits for the poller. At a 60-second poll interval, a 10-round exploration adds up to ~10 minutes of polling latency. This is acceptable for Release 1 given the 50% cost savings. Release 2 can optimize with shorter intervals for small requests or sync fallback for exploration.
- **Sync mode during batch outage**: The `--sync` flag provides an escape hatch if the Batch API is unavailable.
- **Anthropic workspace shared with other tools**: Unknown batches (not in our batch_jobs table) are logged at INFO level and ignored.

## Definition of Done

Phase 14 is complete when:

- All five LLM call sites (generation, planner, exploration, sanity check, conflict resolution) submit requests via the Anthropic Batch API by default.
- Workflows wait for batch results via Temporal signals and resume correctly when results arrive.
- A Temporal Schedule polls for completed batches at a configurable interval (default 60s).
- The poller routes results to waiting workflows via Temporal Search Attributes.
- The `batch_jobs` table records all submissions and outcomes.
- Missing and anomalous batches are detected and logged.
- `--sync` bypasses batch mode and calls the Messages API directly.
- Knowledge extraction runs on a Temporal Schedule.
- All existing tests pass.
- New tests cover: batch submission, signal handling, poll+dispatch cycle, sync fallback, error handling, missing job detection.
