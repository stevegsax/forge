# Phase 9: Prompt Caching

## Goal

Reduce LLM input token costs by leveraging Anthropic's prompt caching. The system prompt (task description, context files, repo map, playbooks, exploration results) is large and stable across retries, across steps in a plan, and across exploration rounds. Prompt caching avoids re-processing this content on every call.

The deliverable: LLM calls that share a system prompt prefix get cache hits, reducing input token costs by up to 80% for cached content.

## Problem Statement

Forge sends the assembled system prompt to Anthropic on every LLM call via `agent.run(user_prompt, instructions=system_prompt)`. The system prompt can be large — 50k+ characters with auto-discovered context, repo map, and exploration results. Without prompt caching, Anthropic re-processes every input token on every call.

Three scenarios waste the most tokens:

1. **Retries.** The system prompt is identical (or nearly identical with Phase 8's error section appended). The LLM re-reads the entire context.
2. **Steps in a plan.** The overall task description, repo map, and much of the context is shared across steps. Each step re-sends it all.
3. **Exploration rounds.** The exploration LLM call sends the task description, available providers, and accumulated context on every round. Earlier rounds' context is a prefix of later rounds.

Anthropic's prompt caching charges 25% extra for cache writes but only 10% for cache reads. A single cache hit pays for the write cost. With Forge's retry and multi-step patterns, cache hits are virtually guaranteed.

## Prior Art

- **Aider**: Structures prompts with the most stable content first (system prompt → read-only files → repo map → editable files → chat history) to maximize the cached prefix. Uses `--cache-keepalive-pings` to prevent 5-minute cache expiry. Reports significant cost savings.
- **Claude Code**: Automatically enables prompt caching on every API call with up to 4 cache breakpoints. Places breakpoints at strategic points: after tool definitions, after system prompt, after stable context. The tool definitions (~9.4k tokens) and system prompt (~2.8k tokens) form the outermost cache layer.

## Scope

**In scope:**

- Add `cache_control` breakpoints to Anthropic API calls via pydantic-ai's model settings.
- Order prompt content for maximum cache hit rate: stable content first, volatile content last.
- Place breakpoints after: (1) system prompt preamble + task description, (2) auto-discovered context and repo map, (3) exploration results.
- Track cache hit/miss statistics in the observability store.

**Out of scope (deferred):**

- Cache keepalive pings (Anthropic's cache TTL is 5 minutes; Forge's batch mode typically completes LLM calls within seconds of each other).
- Cross-task caching (each task assembles different context; cache reuse across unrelated tasks is unlikely).
- Non-Anthropic provider caching (DeepSeek and OpenAI have different caching mechanisms).

## Architecture

### Prompt Ordering for Cache Efficiency

Reorder prompt sections so the most stable content forms the longest possible cached prefix:

**Current order** (single-step with auto-discover):

```
1. "You are a code generation assistant."
2. Task description
3. Target files list
4. Target file contents (priority 2)
5. Direct dependencies (priority 3)
6. Interface context (priority 4)
7. Repository structure (priority 5)
8. Playbooks (priority 5)
9. Additional context (priority 6)
10. Output requirements
```

**Optimized order:**

```
1. "You are a code generation assistant."
2. Output requirements (stable across all calls)           ← BREAKPOINT 1
3. Repository structure
4. Playbooks
5. Task description
6. Target files list
7. Target file contents                                    ← BREAKPOINT 2
8. Direct dependencies
9. Interface context
10. Additional context
11. Exploration results (appended by workflow)              ← BREAKPOINT 3
12. Previous attempt errors (Phase 8, only on retry)
```

Breakpoint 1 caches the role prompt and output format instructions (stable across all tasks). Breakpoint 2 caches through the task-specific context (stable across retries and exploration rounds). Breakpoint 3 caches through exploration results (stable across retries).

### pydantic-ai Integration

pydantic-ai supports Anthropic-specific model settings via `model_settings` on `Agent` or per-call via `agent.run(..., model_settings=...)`. The `AnthropicModelSettings` class supports `anthropic_prompt_caching` (a beta flag) that enables automatic cache control.

Alternatively, cache breakpoints can be placed explicitly by structuring the system prompt as a list of content blocks with `cache_control` markers, passed through pydantic-ai's message construction.

The implementation approach depends on pydantic-ai's current API surface for cache control. Two options:

**Option A: `anthropic_prompt_caching` flag.** If pydantic-ai exposes a simple flag, enable it on the Agent. pydantic-ai handles breakpoint placement automatically.

**Option B: Manual content blocks.** If more control is needed, split the system prompt into segments and pass them as structured content blocks with explicit `cache_control: {"type": "ephemeral"}` markers at breakpoints.

Option A is preferred for simplicity. Option B is the fallback if automatic placement doesn't optimize for Forge's specific prompt structure.

### Cache Statistics

Add cache token fields to `LLMCallResult` and `LLMStats`:

```
LLMCallResult:
    ...existing fields...
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

LLMStats:
    ...existing fields...
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
```

These are populated from the Anthropic API response's `usage` object. pydantic-ai exposes these via `result.usage()`.

### Step and Sub-Task Prompt Ordering

Apply the same stable-first ordering to step and sub-task prompts:

**Step prompt:**

```
1. "You are a code generation assistant."
2. Output requirements                                     ← BREAKPOINT 1
3. Overall task description
4. Plan progress (completed steps)
5. Current step details
6. Current target file contents                            ← BREAKPOINT 2
7. Context files
8. Previous attempt errors (if retry)
```

**Sub-task prompt:** Same pattern — role + requirements first, parent context, sub-task details, target contents, errors last.

## Data Models

Modified models in `models.py`:

```
LLMCallResult:
    ...existing fields...
    cache_creation_input_tokens: int = Field(default=0)
    cache_read_input_tokens: int = Field(default=0)

LLMStats:
    ...existing fields...
    cache_creation_input_tokens: int = Field(default=0)
    cache_read_input_tokens: int = Field(default=0)
```

## Project Structure

Modified files:

```
src/forge/
├── models.py                   # Modified: cache token fields on LLMCallResult, LLMStats
├── activities/
│   ├── llm.py                  # Modified: enable caching, extract cache stats
│   ├── planner.py              # Modified: enable caching for planner calls
│   └── context.py              # Modified: reorder prompt sections for cache efficiency
└── workflows.py                # No changes (caching is transparent)
```

## Dependencies

No new dependencies. Uses pydantic-ai's existing Anthropic model settings.

## Key Design Decisions

### D53: Stable-First Prompt Ordering

**Decision:** Reorder prompt sections so the most stable content (role, output requirements, repo map) comes before volatile content (target file contents, exploration results, retry errors).

**Rationale:** Anthropic's prompt caching computes cache keys cumulatively — the hash for each block depends on all preceding content. Changes at any point invalidate that point and everything after it. By placing stable content first, the longest possible prefix remains cached across retries, exploration rounds, and (partially) across steps. This is the same pattern Aider and Claude Code use.

### D54: Automatic Cache Control via pydantic-ai

**Decision:** Use pydantic-ai's Anthropic model settings to enable prompt caching rather than manually constructing Anthropic API messages.

**Rationale:** pydantic-ai abstracts the Anthropic API. Bypassing it to inject `cache_control` headers would couple Forge to Anthropic's wire format and break the model abstraction. If pydantic-ai's automatic placement is suboptimal, manual breakpoints can be added as a refinement.

## Implementation Order

1. Add `cache_creation_input_tokens` and `cache_read_input_tokens` to `LLMCallResult` and `LLMStats` in `models.py`. Update `build_llm_stats`.
2. Enable prompt caching in `create_agent` and `create_planner_agent` via model settings.
3. Extract cache statistics from `result.usage()` in `execute_llm_call` and `execute_planner_call`.
4. Reorder prompt sections in `build_system_prompt_with_context` for stable-first ordering.
5. Reorder prompt sections in `build_step_system_prompt` and `build_sub_task_system_prompt`.
6. Update the observability store schema to persist cache statistics.
7. Tests for each step.

## Edge Cases

- **Non-Anthropic models:** Cache fields default to 0. No error — caching is a no-op for providers that don't support it.
- **pydantic-ai version doesn't support caching:** Graceful degradation — caching is not enabled, no error, no cost change.
- **Cache TTL expiry:** Anthropic caches expire after 5 minutes of inactivity. For multi-step workflows where steps take >5 minutes (unlikely for LLM calls, possible for long test suites), cache misses are expected and not harmful.
- **Prompt reordering breaks existing tests:** Tests that assert specific prompt content order need updating. The output requirements section moves earlier in the prompt.

## Definition of Done

Phase 9 is complete when:

- Anthropic LLM calls use prompt caching.
- Cache hit/miss statistics are tracked in `LLMCallResult` and persisted to the store.
- Prompt sections are ordered stable-first for maximum cache hit rate.
- Retries show cache reads (not cache creation) for the shared prefix.
- Non-Anthropic models are unaffected (graceful no-op).
- All existing tests pass.
- New tests cover: cache statistics extraction, prompt ordering, backward compatibility of new fields.
