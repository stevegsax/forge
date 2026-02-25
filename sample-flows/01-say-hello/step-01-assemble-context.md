# Step 1 — Assemble Context

The `assemble_context` activity builds the system prompt and user prompt from the task definition. This flow uses a minimal configuration: no exploration, no auto-discovery, no context files.

## Input

A `TaskDefinition` with a simple description and the `GENERIC` domain (no files, no validation):

```python
TaskDefinition(
    task_id="say-hello",
    description="Say hello",
    domain=TaskDomain.GENERIC,
    target_files=[],
    context_files=[],
)
```

## System prompt construction

The system prompt is assembled by `build_system_prompt()` in `src/forge/activities/context.py`. With no context files, no target files, and no prior errors, the prompt reduces to the domain role prompt, output requirements, and task description:

```
You are a helpful assistant.

## Output Requirements

You MUST respond with a valid LLMResponse containing an `explanation` string.

Put your complete response in the `explanation` field. Leave `files` and `edits` empty.

Do NOT return an empty object.

## Task
Say hello
```

The role prompt comes from the domain configuration registry (`src/forge/domains.py`). The GENERIC domain uses `"You are a helpful assistant."` and explanation-only output requirements.

## User prompt

The user prompt is a fixed template from the domain config:

```
Respond to the task described above.
```

## Output

An `AssembledContext` carrying both prompts to the next activity:

```python
AssembledContext(
    task_id="say-hello",
    system_prompt="You are a helpful assistant.\n\n## Output Requirements\n...",
    user_prompt="Respond to the task described above.",
    context_stats=None,
)
```

## Notes

- The system prompt is sent as a `system` array with `cache_control: {"type": "ephemeral"}` on the API call (see step 2). The `AssembledContext` carries the raw string; caching headers are added at call time.
- With no target files, auto-discovery is skipped (`ContextConfig.auto_discover` requires target files).
- With no prior errors, the error section is empty.
- Project instructions (CLAUDE.md) would be injected here if present in the repo root.
