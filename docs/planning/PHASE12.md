# Phase 12: Extended Thinking for Planning

## Goal

Enable Claude's extended thinking mode for planner LLM calls so the model reasons step-by-step before producing the plan. Planning is Forge's highest-leverage LLM call — plan quality bounds everything downstream (D11). Extended thinking improves reasoning on complex decomposition tasks where the model must consider dependencies, ordering constraints, and file conflict avoidance.

The deliverable: planner calls use extended thinking with a configurable budget, producing higher-quality plans for complex tasks.

## Problem Statement

The planner receives a task description, repo map, and context files, then must produce a structured `Plan` with ordered steps, target files, context files, and optional sub-tasks. This requires multi-step reasoning:

1. Understand the task's scope and requirements.
2. Identify which files need to change and in what order.
3. Determine dependencies between steps (later steps read files created by earlier steps).
4. Decide which work can be parallelized via fan-out.
5. Specify context files that aren't reachable via import graph analysis.

Currently, the planner generates this plan in a single pass with no explicit reasoning phase. For complex tasks (multi-file refactors, architectural changes, feature additions spanning many modules), the single-pass approach produces lower-quality plans: missed dependencies, incorrect file assignments, suboptimal step ordering.

Claude's extended thinking mode gives the model a scratchpad for step-by-step reasoning before producing structured output. Anthropic reports significant improvements on complex reasoning tasks — exactly the planning use case.

## Prior Art

- **Claude Code**: Does not use explicit thinking mode but benefits from the model's internal chain-of-thought. Plans are generated iteratively through conversation.
- **Aider**: Does not use thinking mode. Uses a simpler planning approach (architect/editor split) where the architect produces free-text instructions.
- **SWE-bench top agents**: Several use chain-of-thought or multi-pass approaches for planning. RepoGraph (top scorer) uses a planning phase with explicit reasoning about file dependencies.
- **Anthropic guidance**: Extended thinking is recommended for "complex analysis, multi-step problems, and tasks requiring deep reasoning." Task decomposition is explicitly listed as a use case.

## Scope

**In scope:**

- Enable extended thinking for planner calls via pydantic-ai's `AnthropicModelSettings`.
- Configurable thinking budget (default: 10,000 tokens for Sonnet, adaptive for Opus 4.6).
- CLI flag to control thinking budget.
- Track thinking token usage in planner statistics.

**Out of scope (deferred):**

- Extended thinking for code generation calls (might help for complex steps; evaluate after planner results).
- Extended thinking for exploration calls (lightweight classification, thinking unlikely to help).
- Thinking content extraction and display (the thinking content is internal to the model; useful for debugging but not needed for Phase 12).
- Automatic thinking budget adjustment based on task complexity.

## Architecture

### pydantic-ai Integration

pydantic-ai v1.56.0 supports extended thinking via `AnthropicModelSettings`:

**For Claude Sonnet 4.5 (budget-based thinking):**

```python
from pydantic_ai.models.anthropic import AnthropicModelSettings

settings = AnthropicModelSettings(
    anthropic_thinking={'type': 'enabled', 'budget_tokens': 10000},
)
```

**For Claude Opus 4.6 (adaptive thinking):**

```python
settings = AnthropicModelSettings(
    anthropic_thinking={'type': 'adaptive'},
    anthropic_effort='high',
)
```

The implementation detects the model and applies the appropriate thinking configuration.

### Thinking Configuration

A new configuration model controls thinking behavior:

```
ThinkingConfig:
    enabled: bool = True
    budget_tokens: int = 10000
    effort: str = "high"    # For adaptive thinking: "low", "medium", "high", "max"
```

When thinking is enabled:

- If the model supports adaptive thinking (Opus 4.6+), use `{'type': 'adaptive'}` with the configured effort level.
- Otherwise, use `{'type': 'enabled', 'budget_tokens': N}` with the configured budget.
- If the model is not Anthropic, thinking configuration is silently ignored.

### Model Detection

A pure function determines thinking configuration based on the model name:

```
def build_thinking_settings(
    model_name: str,
    thinking_config: ThinkingConfig,
) -> dict | None:
    """Build AnthropicModelSettings kwargs for thinking.

    Returns None for non-Anthropic models.
    """
```

This checks for `anthropic:` prefix and known model capabilities. Opus 4.6+ gets adaptive thinking; Sonnet 4.5 gets budget-based thinking; Haiku and non-Anthropic models get no thinking.

### Planner Agent Update

`create_planner_agent` gains a `thinking_config` parameter:

```python
def create_planner_agent(
    model_name: str | None = None,
    thinking_config: ThinkingConfig | None = None,
) -> Agent[None, Plan]:
    from pydantic_ai import Agent
    from pydantic_ai.models.anthropic import AnthropicModelSettings

    model_settings = {}
    if thinking_config and thinking_config.enabled:
        thinking_settings = build_thinking_settings(model_name, thinking_config)
        if thinking_settings:
            model_settings = thinking_settings

    return Agent(
        model_name,
        output_type=Plan,
        model_settings=AnthropicModelSettings(**model_settings) if model_settings else None,
    )
```

### Structured Output Compatibility

pydantic-ai automatically handles the interaction between thinking mode and structured output. When thinking is enabled with tool-based output, pydantic-ai switches to `output_mode='prompted'` instead of `output_mode='tool'` because Anthropic does not support `tool_choice=required` with thinking enabled. This is transparent to Forge — pydantic-ai handles it internally.

### Token Tracking

The thinking budget consumes tokens that appear in the API response's `usage` object. pydantic-ai's `result.usage()` includes these in `input_tokens` (for the thinking content that feeds back into the response). No changes to `LLMStats` are needed — thinking tokens are already counted.

However, for observability, a new optional field tracks thinking-specific usage:

```
PlanCallResult:
    ...existing fields...
    thinking_tokens: int = Field(default=0)
```

## Data Models

New model in `models.py`:

```
class ThinkingConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable extended thinking for planner.")
    budget_tokens: int = Field(default=10000, description="Token budget for thinking (Sonnet).")
    effort: str = Field(
        default="high",
        description="Effort level for adaptive thinking (Opus 4.6+).",
    )
```

Modified models:

```
ForgeTaskInput:
    ...existing fields...
    thinking: ThinkingConfig = Field(default_factory=ThinkingConfig)

PlanCallResult:
    ...existing fields...
    thinking_tokens: int = Field(default=0)
```

## Project Structure

Modified files:

```
src/forge/
├── models.py                   # Modified: ThinkingConfig, PlanCallResult update
├── activities/
│   └── planner.py              # Modified: thinking settings on agent, token tracking
└── cli.py                      # Modified: --thinking-budget, --no-thinking
```

## Dependencies

No new dependencies. Uses pydantic-ai's existing `AnthropicModelSettings` support.

## Key Design Decisions

### D61: Thinking for Planning Only

**Decision:** Enable extended thinking for planner calls only, not for code generation or exploration.

**Rationale:** Planning is the highest-leverage use of extended thinking. The planner must reason about task decomposition, dependencies, and ordering — a complex multi-step reasoning task where thinking mode shows the most improvement. Code generation benefits less: the LLM already sees the target file contents and context, and produces edits in a relatively straightforward way. Exploration is classification (what context to request), which thinking mode can actually slow down. Enabling thinking selectively keeps costs bounded while maximizing quality impact.

### D62: Adaptive Thinking for Opus, Budget for Sonnet

**Decision:** Use adaptive thinking (`{'type': 'adaptive'}`) for Opus 4.6+ and budget-based thinking (`{'type': 'enabled', 'budget_tokens': N}`) for Sonnet.

**Rationale:** Opus 4.6 supports adaptive thinking where the model dynamically decides how much to think based on problem complexity. This is more efficient than a fixed budget — simple tasks get less thinking, complex tasks get more. Sonnet requires explicit budget-based thinking. The implementation detects the model and applies the appropriate configuration, so upgrading the planner model (Phase 11) automatically gets the right thinking mode.

### D63: Silent Degradation for Non-Anthropic Models

**Decision:** Thinking configuration is silently ignored for non-Anthropic models (no error, no warning).

**Rationale:** Forge's model abstraction (via pydantic-ai) supports multiple providers. If the planner is configured to use an OpenAI model, thinking configuration simply doesn't apply. This matches D54's approach to caching — provider-specific features degrade gracefully.

## Implementation Order

1. Add `ThinkingConfig` to `models.py`. Add `thinking` field to `ForgeTaskInput`. Add `thinking_tokens` to `PlanCallResult`.
2. Implement `build_thinking_settings` pure function in `planner.py`.
3. Update `create_planner_agent` to accept and apply `ThinkingConfig`.
4. Update `execute_planner_call` to extract thinking token usage.
5. Update workflow to pass `ThinkingConfig` to planner activities.
6. Add CLI options: `--thinking-budget`, `--no-thinking`.
7. Tests for thinking settings construction, model detection, and graceful degradation.

## Edge Cases

- **Non-Anthropic model:** Thinking configuration silently ignored. Planner works as before.
- **Thinking + structured output:** pydantic-ai handles the `output_mode` switch automatically. No Forge changes needed.
- **Very large thinking budget:** Anthropic enforces a maximum. Exceeding it returns an API error that surfaces as a Temporal activity failure.
- **Zero thinking budget:** Equivalent to thinking disabled. `build_thinking_settings` returns None.
- **Model name not recognized:** Falls back to budget-based thinking (safe default for any Anthropic model).
- **Backward compatibility:** `ThinkingConfig()` defaults to `enabled=True`, but `build_thinking_settings` returns None for the current default model path if thinking is not explicitly configured. Existing behavior is preserved.

## Definition of Done

Phase 12 is complete when:

- Planner calls use extended thinking with configurable budget.
- Opus 4.6+ uses adaptive thinking; Sonnet uses budget-based thinking.
- Non-Anthropic models are unaffected (graceful no-op).
- Thinking token usage is tracked in `PlanCallResult`.
- CLI flags control thinking configuration.
- All existing tests pass.
- New tests cover: thinking settings construction, model detection, adaptive vs budget modes, graceful degradation for non-Anthropic models.
