+++
title = "Model Routing and Batch Processing Reference"
weight = 94
description = "How Forge routes LLM calls to appropriate models via capability tiers, and how batch processing decouples workflow execution from LLM latency."
topic = "llm-dispatch"
covers = [
    "Capability tier table: tier name, default model, use cases, LLM call sites",
    "Model override mechanism: per-step capability_tier in plans",
    "Batch processing: batch_jobs table schema, batch states, signal names",
    "Batch poller: schedule configuration, poll interval, search attributes",
    "Prompt caching: cache control header placement, cache token fields in LLMStats",
    "CLI flags: --sync, --batch-poll-interval, model tier overrides",
]
detail = "Tabular reference for model configuration and batch mechanics."
+++
For design rationale and conceptual background, see [Model Routing and Batch Processing](../explanation/llm-dispatch/). For step-by-step recipes, see [How to Configure LLM Dispatch](../howto/configure-llm-dispatch/).

---

## Capability Tiers

### CapabilityTier enum

Defined in `src/forge/models.py`.

| Value | String | Description |
|-------|--------|-------------|
| `CapabilityTier.REASONING` | `"reasoning"` | Planning, conflict resolution, complex architectural decisions |
| `CapabilityTier.GENERATION` | `"generation"` | Code generation, test writing, documentation |
| `CapabilityTier.SUMMARIZATION` | `"summarization"` | Knowledge extraction, progress digests |
| `CapabilityTier.CLASSIFICATION` | `"classification"` | Exploration loop, transition evaluation |

### Default tier-to-model mapping

Defined in `_DEFAULT_TIER_MODELS` in `src/forge/models.py`.

| Tier | Default model |
|------|--------------|
| `REASONING` | `anthropic:claude-opus-4-6` |
| `GENERATION` | `anthropic:claude-sonnet-4-5-20250929` |
| `SUMMARIZATION` | `anthropic:claude-sonnet-4-5-20250929` |
| `CLASSIFICATION` | `anthropic:claude-haiku-4-5-20251001` |

### Call site assignments

| Activity / call site | Default tier | Location |
|---------------------|-------------|----------|
| `call_planner` | `REASONING` | `src/forge/activities/planner.py` |
| `call_llm` (code generation) | `GENERATION` | `src/forge/activities/llm.py` |
| `call_exploration` | `CLASSIFICATION` | `src/forge/activities/exploration.py` |
| `call_extraction_llm` | `SUMMARIZATION` | `src/forge/activities/extraction.py` |
| `create_judge_agent` | `REASONING` | `src/forge/eval/judge.py` |
| Sanity check | `REASONING` | `src/forge/activities/sanity_check.py` |
| Conflict resolution | `REASONING` | `src/forge/activities/conflict_resolution.py` |

### ModelConfig

Defined in `src/forge/models.py`. Passed as `model_routing` in `ForgeTaskInput`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `reasoning` | `str` | `"anthropic:claude-opus-4-6"` | Model for the REASONING tier |
| `generation` | `str` | `"anthropic:claude-sonnet-4-5-20250929"` | Model for the GENERATION tier |
| `summarization` | `str` | `"anthropic:claude-sonnet-4-5-20250929"` | Model for the SUMMARIZATION tier |
| `classification` | `str` | `"anthropic:claude-haiku-4-5-20251001"` | Model for the CLASSIFICATION tier |

### Per-step tier override

The `capability_tier` field on `PlanStep` overrides the default `GENERATION` tier for that step.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `capability_tier` | `CapabilityTier \| None` | `None` (resolves to `GENERATION`) | Tier override for this step's LLM call |

Example plan step with tier override:

```json
{
  "id": "step-3",
  "description": "Resolve architectural conflict between modules",
  "target_files": ["src/forge/models.py"],
  "capability_tier": "reasoning"
}
```

### resolve_model function

```python
def resolve_model(tier: CapabilityTier, config: ModelConfig) -> str
```

Returns the concrete model string for the given tier from the provided config. Used by all call sites.

---

## Batch Processing

### batch_jobs table schema

SQLAlchemy model: `BatchJob` in `src/forge/store.py`.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | `String` | No (PK) | Internal job UUID |
| `batch_id` | `String` | Yes | Anthropic Batch API job ID (set after submission) |
| `workflow_id` | `String` | No | Temporal workflow ID of the submitting workflow |
| `status` | `String` | No | Current job status (see states below) |
| `provider` | `String` | No | Provider name, default `"anthropic"` |
| `file_path` | `String` | Yes | Source document path (OCR jobs only) |
| `error_message` | `Text` | Yes | Error detail if status is `failed` |
| `created_at` | `DateTime` | No | UTC timestamp of job creation |
| `updated_at` | `DateTime` | No | UTC timestamp of last status change |

### Batch job states

Defined as `BatchJobStatus` in `src/forge/models.py`. Happy path:
`submitted` → `storing` → `succeeded`.

| Status | Description |
|--------|-------------|
| `submitted` | Row created; batch in flight at the provider, waiting for completion |
| `storing` | Provider reported the batch complete; signal delivered to the waiting workflow; parse + write to the downstream store (e.g. `ocr_results`) is in progress |
| `succeeded` | End-to-end complete. Only the downstream store workflow writes this value, and only after its write succeeds |
| `errored` | Per-entry failure from the provider, or the parse/store step raised after signal delivery |
| `failed` | Provider API refused the submission before returning a batch_id. Written by `record_batch_failure` |
| `expired` | Provider marked the batch as expired. Written from `BatchPollStatus` in the poller's terminal-failure branch |
| `canceled` | Provider marked the batch as canceled. Written from `BatchPollStatus` in the poller's terminal-failure branch |
| `missing` | Batch unretrievable after 24h. The poller gave up and marked the row |

The poller never writes `succeeded` directly — it advances a row to `storing` on signal delivery, and the downstream store workflow promotes to `succeeded` once its write commits. This preserves the invariant that `succeeded` means the stored output actually exists.

### Signal names

| Signal name | Payload type | Description |
|-------------|-------------|-------------|
| `batch_result_received` | `BatchResult` | Delivered by the batch poller to a waiting workflow when its batch job completes |

### forge_batch_id search attribute

Type: `str`. Set on the running Temporal workflow after batch submission. Used by the batch poller to route completed results back to the correct workflow.

---

## Batch Poller

### BatchPollerWorkflow

Defined in `src/forge/batch_poller_workflow.py`.

| Property | Value |
|----------|-------|
| Temporal workflow name | `BatchPollerWorkflow` |
| Temporal schedule ID | `forge-batch-poller` |
| Default poll interval | 600 seconds (configured via `--batch-poll-interval` on `forge worker`) |
| Activity timeout | 5 minutes start-to-close |
| Activity heartbeat timeout | 60 seconds |

The workflow executes a single activity (`poll_batch_results`) on each schedule tick. The activity queries the Anthropic API for completed batches, matches results to waiting workflows via the `forge_batch_id` search attribute, and sends `batch_result_received` signals.

### BatchPollerInput / BatchPollerResult

Both models are defined in `src/forge/models.py`. Both are empty (no fields); the poller does not require configuration input beyond what is in the database.

---

## Prompt Caching

### Cache control placement

Cache breakpoints are applied to Anthropic API requests in the `call_llm` activity. Breakpoints use the `cache_control: {"type": "ephemeral"}` header on message content blocks.

Breakpoint positions (stable to volatile):

1. After the stable preamble (role, output requirements, project instructions, repository structure, playbooks)
2. After the assembled context section (target file contents, direct dependencies, interface context)
3. After exploration results (if any)

Section ⑪ (previous errors, retry-only) is never cached; it appears after all breakpoints.

### Cache token fields in LLMStats

Tracked per-interaction in the `interactions` table and in the `LLMStats` model.

| Field | Type | Description |
|-------|------|-------------|
| `cache_creation_input_tokens` | `int` | Tokens written to cache on this call (write surcharge applies) |
| `cache_read_input_tokens` | `int` | Tokens read from cache on this call (discounted rate applies) |

OTel span attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `forge.llm.cache_creation_input_tokens` | `int` | Cache write tokens for this call |
| `forge.llm.cache_read_input_tokens` | `int` | Cache read tokens for this call |

---

## CLI Flags

### forge run flags

| Flag | Default | Description |
|------|---------|-------------|
| `--sync` / `--no-sync` | `--no-sync` (batch mode) | Use synchronous Messages API. `--no-sync` enables batch mode. |
| `--reasoning-model MODEL` | (tier default) | Override the model used for the REASONING tier |
| `--generation-model MODEL` | (tier default) | Override the model used for the GENERATION tier |
| `--summarization-model MODEL` | (tier default) | Override the model used for the SUMMARIZATION tier |
| `--classification-model MODEL` | (tier default) | Override the model used for the CLASSIFICATION tier |

All model override flags accept a string in the format `provider:model-name` (e.g., `anthropic:claude-opus-4-6`).

### forge worker flags

| Flag | Default | Description |
|------|---------|-------------|
| `--batch-poll-interval SECONDS` | `600` | Seconds between batch poller schedule runs |
