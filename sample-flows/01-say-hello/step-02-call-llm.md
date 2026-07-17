# Step 2 — Call LLM (Batch Path)

Forge is batch-first (`sync_mode=False` is the default). Instead of calling `messages.create` directly, the workflow submits the request to the Anthropic Message Batches API, then suspends until a poller delivers the result via a Temporal signal.

## Request construction

Request parameters are built by the provider's `build_request_params()` method — sax_llm's `AnthropicProvider` (`libs/sax-llm/src/sax_llm/anthropic.py`), called from `execute_batch_submit()` in `src/forge/activities/batch_submit.py`. The same provider method serves both the sync and batch paths:

```json
{
  "model": "claude-sonnet-5",
  "max_tokens": 4096,
  "system": [
    {
      "type": "text",
      "text": "You are a helpful assistant.\n\n## Output Requirements\n\nYou MUST respond with a valid LLMResponse containing an `explanation` string and a `files` list.\n\nWrite your complete response as one or more markdown files using the `files` list. Each entry needs `file_path` and `content` (complete file content). Use the `explanation` field for a brief summary of what you produced.\n\nDo NOT return an empty object.\n\n## Task\nSay hello\n\n## Target Files\n- hello.md",
      "cache_control": { "type": "ephemeral" }
    }
  ],
  "messages": [
    {
      "role": "user",
      "content": "Respond to the task described above. Write your response as markdown files using the `files` list."
    }
  ],
  "tools": [
    {
      "name": "llm_response",
      "description": "Structured output from the LLM call.",
      "input_schema": {
        "type": "object",
        "properties": {
          "files": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "file_path": { "type": "string" },
                "content": { "type": "string" }
              },
              "required": ["file_path", "content"]
            },
            "default": []
          },
          "edits": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "file_path": { "type": "string" },
                "edits": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "properties": {
                      "search": { "type": "string" },
                      "replace": { "type": "string" }
                    },
                    "required": ["search", "replace"]
                  }
                }
              },
              "required": ["file_path", "edits"]
            },
            "default": []
          },
          "explanation": { "type": "string" }
        },
        "required": ["explanation"]
      },
      "cache_control": { "type": "ephemeral" }
    }
  ],
  "tool_choice": { "type": "tool", "name": "llm_response" }
}
```

## Batch submission

The `submit_batch_request` activity in `src/forge/activities/batch_submit.py` wraps these parameters into a batch request and submits them:

```python
# 1. Generate a unique request ID
request_id = str(uuid.uuid4())

# 2. Wrap as a batch request item
batch_request = {
    "custom_id": request_id,
    "params": { ... }  # the messages.create params above
}

# 3. Submit to the Batches API
batch = await client.messages.batches.create(requests=[batch_request])

# Returns: BatchSubmitResult(request_id=..., batch_id="msgbatch_01ABC...")
```

The batch ID (e.g., `msgbatch_01ABC...`) identifies this batch on the Anthropic side.

## Waiting for the result

After submission, the Temporal workflow suspends:

```python
# In ForgeTaskWorkflow._call_llm_batch():
await workflow.wait_condition(lambda: len(self._batch_results) > 0)
```

A separate batch poller workflow (Phase 14c) periodically polls the Anthropic Batches API. When the batch completes, the poller sends a Temporal signal (`batch_result_received`) carrying a `BatchResult` with the raw response JSON.

## Key details

**Structured output via tool use.** The `LLMResponse` Pydantic model is converted to a JSON Schema tool definition by `build_tool_definition()`. The tool name is derived from the class name: `LLMResponse` becomes `llm_response` (snake_case conversion).

**Forced tool invocation.** `tool_choice: {"type": "tool", "name": "llm_response"}` forces the LLM to call this tool, guaranteeing a structured response that can be validated against the Pydantic model.

**Prompt caching.** Both the system prompt and tool definition include `cache_control: {"type": "ephemeral"}`. On retries or similar requests, the cached prefix avoids re-processing tokens. Cache headers are added by `build_system_param()` and `build_tool_definition()` in sax_llm (`libs/sax-llm/src/sax_llm/client.py`), which the provider's `build_request_params()` calls.

**Model selection.** The model comes from `AssembledContext.model_name`, which the workflow sets from the GENERATION tier via `ModelConfig`. When it is unset, `batch_submit.py` falls back to the GENERATION tier pin resolved from the registry (`resolve_model(CapabilityTier.GENERATION, ModelConfig())` — currently `claude-sonnet-5`).

**No thinking.** This flow's generation call runs with extended thinking disabled (a `ThinkingPolicy` with `enabled=False`), so no `thinking` parameter is included in the request.

**Sync fallback.** When `sync_mode=True`, the workflow calls the `call_llm` activity directly (`client.messages.create`), bypassing the batch/poller/signal path entirely.
