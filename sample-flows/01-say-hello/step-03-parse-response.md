# Step 3 — Parse Response

When the batch poller delivers the result, the raw Anthropic Message JSON is deserialized and the structured output is extracted by the `parse_llm_response` activity.

## Signal delivery

The batch poller sends a `BatchResult` to the workflow via a Temporal signal:

```python
BatchResult(
    request_id="a1b2c3d4-...",
    batch_id="msgbatch_01ABC...",
    raw_response_json='{"id":"msg_01XYZ...","type":"message",...}',
    error=None,
    result_type="LLMResponse",
)
```

## Raw API response

The `raw_response_json` contains the full Anthropic Message. The LLM returns a single `tool_use` content block with the structured response:

```json
{
  "id": "msg_01XYZ...",
  "type": "message",
  "role": "assistant",
  "model": "claude-sonnet-4-5-20250929",
  "content": [
    {
      "type": "tool_use",
      "id": "toolu_01ABC...",
      "name": "llm_response",
      "input": {
        "files": [
          {
            "file_path": "hello.md",
            "content": "# Hello\n\nHello! How can I help you today?"
          }
        ],
        "edits": [],
        "explanation": "Created a greeting in hello.md."
      }
    }
  ],
  "stop_reason": "tool_use",
  "usage": {
    "input_tokens": 312,
    "output_tokens": 84,
    "cache_creation_input_tokens": 290,
    "cache_read_input_tokens": 0
  }
}
```

## Parsing

The `parse_llm_response` activity in `src/forge/activities/batch_parse.py` delegates to `parse_batch_response_json()` in `src/forge/llm_client.py`:

1. Deserialize the raw JSON into an Anthropic `Message` object
2. Call `extract_tool_result(message, LLMResponse)` — iterate `message.content`, find the first `tool_use` block, validate `block.input` against the Pydantic model
3. Call `extract_usage(message)` — pull token counts from `message.usage`

```python
# extract_tool_result iterates content blocks:
for block in message.content:
    if block.type == "tool_use":
        return LLMResponse.model_validate(block.input)
# Raises ValueError if no tool_use block found
```

The result is a `ParsedLLMResponse` carrying the serialized model and usage stats:

```python
ParsedLLMResponse(
    parsed_json='{"files":[{"file_path":"hello.md","content":"# Hello\\n\\nHello! How can I help you today?"}],"edits":[],"explanation":"Created a greeting in hello.md."}',
    model_name="claude-sonnet-4-5-20250929",
    input_tokens=312,
    output_tokens=84,
    cache_creation_input_tokens=290,
    cache_read_input_tokens=0,
)
```

## Reconstruction into LLMCallResult

Back in the workflow, the `ParsedLLMResponse` is deserialized into the final `LLMCallResult`:

```python
LLMCallResult(
    task_id="say-hello",
    response=LLMResponse.model_validate_json(parsed.parsed_json),
    model_name="claude-sonnet-4-5-20250929",
    input_tokens=312,
    output_tokens=84,
    latency_ms=0.0,
    cache_creation_input_tokens=290,
    cache_read_input_tokens=0,
)
```

## Usage statistics

- `input_tokens` — total input tokens processed
- `output_tokens` — tokens generated in the response
- `cache_creation_input_tokens` — tokens written to prompt cache (first request)
- `cache_read_input_tokens` — tokens read from cache (subsequent requests)
- `latency_ms` — `0.0` for batch path (no wall-clock timing; the batch may take minutes to hours)

## What happens next

In a full Forge workflow, this result would flow to `write_output` (write `hello.md` to the worktree), then `validate_output` (all checks disabled for the generic domain), then `evaluate_transition` (returns SUCCESS), then `commit_changes`.
