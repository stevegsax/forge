# Step 1 — Assemble Context

The `assemble_context` activity builds the system prompt and user prompt from the task definition. This flow uses a minimal configuration: no exploration, no auto-discovery, no context files.

## Input

A `TaskDefinition` with a simple description, a target file, and the `GENERIC` domain (no code validation):

```python
TaskDefinition(
    task_id="say-hello",
    description="Say hello",
    domain=TaskDomain.GENERIC,
    target_files=["hello.md"],
    context_files=[],
)
```

## System prompt construction

The system prompt is assembled by `build_system_prompt()` in `src/forge/activities/context.py`. With no context files, no target files, and no prior errors, the prompt reduces to the domain role prompt, output requirements, and task description:

```
You are a helpful assistant.

## Output Requirements

You MUST respond with a valid LLMResponse containing an `explanation` string
and a `files` list.

Write your complete response as one or more markdown files using the `files` list.
Each entry needs `file_path` and `content` (complete file content). Use the
`explanation` field for a brief summary of what you produced.

Do NOT return an empty object.

## Task
Say hello

## Target Files
- hello.md
```

The role prompt comes from the domain configuration registry (`src/forge/domains.py`). The GENERIC domain uses `"You are a helpful assistant."` and prose output requirements.

## User prompt

The user prompt is a fixed template from the domain config:

```
Respond to the task described above. Write your response as markdown files using the `files` list.
```

## Output

An `AssembledContext` carrying both prompts to the next activity:

```python
AssembledContext(
    task_id="say-hello",
    system_prompt="You are a helpful assistant.\n\n## Output Requirements\n...\n\n## Target Files\n- hello.md",
    user_prompt="Respond to the task described above. Write your response as markdown files using the `files` list.",
    context_stats=None,
)
```

## Notes

- The system prompt is sent as a `system` array with `cache_control: {"type": "ephemeral"}` on the API call (see step 2). The `AssembledContext` carries the raw string; caching headers are added at call time.
- Auto-discovery runs but finds no Python imports for a `.md` target, so the packed context contains only the repo map and the target file entry.
- With no prior errors, the error section is empty.
- Project instructions (CLAUDE.md) would be injected here if present in the repo root.
