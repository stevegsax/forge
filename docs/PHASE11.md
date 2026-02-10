# Phase 11: Model Routing

## Goal

Route LLM calls to the appropriate model based on the task's capability requirements. Currently, every LLM call — planner, code generation, exploration, extraction, evaluation — uses the same model (`claude-sonnet-4-5`). This wastes expensive tokens on tasks that cheaper models handle well and prevents using more capable models where they would make a meaningful difference.

The deliverable: a capability tier system where each LLM call is routed to a model appropriate for its role, with the tier-to-model mapping configurable and separate from the workflow logic.

## Problem Statement

Forge has five distinct LLM call sites:

1. **Planner** (`create_planner_agent`): Decomposes tasks into ordered steps.
2. **Code generation** (`create_agent`): Produces code edits and new files.
3. **Exploration** (`create_exploration_agent`): Requests context before generating.
4. **Extraction** (`create_extraction_agent`): Extracts playbook entries from completed work.
5. **Evaluation judge** (`create_judge_agent`): Scores planner output quality.

All five use `DEFAULT_MODEL = "anthropic:claude-sonnet-4-5-20250929"`. This is suboptimal:

- **Planning benefits from the most capable model.** Plan quality bounds everything downstream (D11). An Opus-tier model produces better decompositions, fewer conflicts, and more accurate context_files lists.
- **Exploration is lightweight.** The exploration LLM decides which context to request — a classification task that cheaper models handle well.
- **Extraction is tolerant.** Playbook extraction from completed work is best-effort; a cheaper model reduces cost without meaningful quality loss.
- **Code generation varies.** Some steps are trivial (add an import, fix a lint error) while others are architecturally complex. A capability tier per step would allow the planner to request heavier models only where needed.

The design document (D10) already specifies capability tiers: Reasoning, Generation, Summarization, Classification. This phase implements that design.

## Prior Art

- **Aider**: Supports `--model`, `--editor-model`, `--weak-model` for three-tier routing. The weak model handles simple tasks like commit messages. Reports 60-70% cost reduction on typical sessions by routing appropriately.
- **Claude Code**: Uses a single model but with `anthropic_effort` to vary compute. The "low" effort setting reduces cost for simple operations while "high" is used for complex reasoning.
- **Cursor**: Routes between models based on task complexity. Uses a fast model for tab completion and autocomplete, a medium model for inline edits, and the most capable model for multi-file changes and planning.
- **SWE-bench agents**: Top agents use different models for different stages — typically a reasoning model for planning and a generation model for coding.

## Scope

**In scope:**

- Define a `CapabilityTier` enum with four levels: REASONING, GENERATION, SUMMARIZATION, CLASSIFICATION.
- Define a `ModelConfig` mapping tiers to concrete model names, with sensible defaults.
- Route each LLM call site to the appropriate tier.
- Allow the planner to specify a capability tier per step (optional, defaults to GENERATION).
- Pass `ModelConfig` through workflow inputs so it's configurable per-run.
- CLI flags for overriding the default tier-to-model mapping.

**Out of scope (deferred):**

- Dynamic model selection based on runtime metrics (retry rates, token usage).
- Automatic tier downgrading when a model fails (retry with a different model).
- Per-sub-task model routing (sub-tasks inherit the parent step's tier).
- Cost tracking and optimization dashboards.

## Architecture

### Capability Tiers

```
CapabilityTier: StrEnum
    REASONING       # Planning, conflict resolution, complex architectural decisions
    GENERATION      # Code generation, test writing, documentation
    SUMMARIZATION   # Extraction, progress digests, knowledge synthesis
    CLASSIFICATION  # Exploration, transition evaluation, simple validation
```

### Default Tier Assignments

| Call Site | Default Tier | Rationale |
|-----------|-------------|-----------|
| `call_planner` | REASONING | Plan quality bounds everything (D11) |
| `call_llm` (code gen) | GENERATION | Primary output production |
| `call_exploration` | CLASSIFICATION | Deciding what to request is classification |
| `call_extraction_llm` | SUMMARIZATION | Synthesizing lessons from completed work |
| `create_judge_agent` | REASONING | Quality evaluation requires strong reasoning |

### Model Configuration

```
ModelConfig:
    reasoning: str = "anthropic:claude-opus-4-6"
    generation: str = "anthropic:claude-sonnet-4-5-20250929"
    summarization: str = "anthropic:claude-sonnet-4-5-20250929"
    classification: str = "anthropic:claude-haiku-4-5-20251001"
```

The defaults use the current model for GENERATION and SUMMARIZATION (no change), upgrade REASONING to Opus, and downgrade CLASSIFICATION to Haiku. This is the configuration that maximizes quality-per-dollar: spend more where it matters (planning), spend less where it doesn't (exploration).

### Tier Resolution

A new pure function resolves tiers to model names:

```
def resolve_model(tier: CapabilityTier, config: ModelConfig) -> str:
    """Map a capability tier to a concrete model name."""
    return {
        CapabilityTier.REASONING: config.reasoning,
        CapabilityTier.GENERATION: config.generation,
        CapabilityTier.SUMMARIZATION: config.summarization,
        CapabilityTier.CLASSIFICATION: config.classification,
    }[tier]
```

### Per-Step Tier Override

`PlanStep` gains an optional `capability_tier` field:

```
class PlanStep(BaseModel):
    ...existing fields...
    capability_tier: CapabilityTier | None = Field(
        default=None,
        description="Override the default capability tier for this step.",
    )
```

When set, the step's LLM call uses the specified tier instead of the default GENERATION. The planner prompt is updated to explain this option and when to use it (e.g., architecturally complex steps should use REASONING).

### Workflow Integration

`ForgeTaskInput` gains a `model_config` field:

```
class ForgeTaskInput(BaseModel):
    ...existing fields...
    model_config_: ModelConfig = Field(
        default_factory=ModelConfig,
        alias="model_config",
    )
```

The workflow passes `model_config` to each activity that creates an agent. Each `create_*_agent` function accepts an optional `model_name` parameter (which some already do) resolved from the tier.

### Agent Factory Updates

Each agent factory function already accepts `model_name: str | None = None` and falls back to `DEFAULT_MODEL`. The change is minimal — the workflow resolves the tier to a model name and passes it:

```python
# In workflow, before calling call_planner:
model_name = resolve_model(CapabilityTier.REASONING, input.model_config_)
# Pass to activity via input model
```

## Data Models

New models in `models.py`:

```
class CapabilityTier(StrEnum):
    REASONING = "reasoning"
    GENERATION = "generation"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"

class ModelConfig(BaseModel):
    reasoning: str = Field(default="anthropic:claude-opus-4-6")
    generation: str = Field(default="anthropic:claude-sonnet-4-5-20250929")
    summarization: str = Field(default="anthropic:claude-sonnet-4-5-20250929")
    classification: str = Field(default="anthropic:claude-haiku-4-5-20251001")
```

Modified models:

```
ForgeTaskInput:
    ...existing fields...
    model_config_: ModelConfig = Field(default_factory=ModelConfig, alias="model_config")

PlanStep:
    ...existing fields...
    capability_tier: CapabilityTier | None = Field(default=None)
```

## Project Structure

Modified files:

```
src/forge/
├── models.py                   # Modified: CapabilityTier, ModelConfig, PlanStep update
├── activities/
│   ├── llm.py                  # Modified: accept model_name from workflow
│   ├── planner.py              # Modified: accept model_name from workflow
│   ├── exploration.py          # Modified: accept model_name from workflow
│   └── extraction.py           # Modified: accept model_name from workflow
├── workflows.py                # Modified: resolve tiers, pass model_name to activities
└── cli.py                      # Modified: --reasoning-model, --generation-model, etc.
```

## Dependencies

No new dependencies.

## Key Design Decisions

### D58: Capability Tiers Over Direct Model Names

**Decision:** Route LLM calls via abstract capability tiers (REASONING, GENERATION, SUMMARIZATION, CLASSIFICATION) rather than passing concrete model names through the system.

**Rationale:** This follows D10 from the design document. Abstract tiers decouple the "what capability is needed" question from the "which model provides it" question. When a cheaper model improves enough to handle GENERATION tasks, a single config change upgrades the entire system. When a new provider is added, it slots into the tier mapping without changing any workflow code. The planner specifies capability requirements, not provider-specific model IDs.

### D59: Planner Gets Premium Model

**Decision:** Default the planner to the REASONING tier (Opus-class model).

**Rationale:** D11 states "Planning is the hard part" and "invest the most expensive models in planning." Plan quality bounds everything downstream — a better decomposition reduces conflicts, improves context_files accuracy, and produces fewer retries. The planner is called once per task, so the incremental cost is bounded. Moving from Sonnet to Opus for planning is the single highest-value model routing change.

### D60: Backward-Compatible Defaults

**Decision:** All `ModelConfig` fields have defaults, and `ForgeTaskInput.model_config_` defaults to `ModelConfig()`. Existing callers and serialized payloads work unchanged.

**Rationale:** Temporal workflows may have in-flight executions when this change deploys. Default values ensure old payloads deserialize correctly. The field alias avoids collision with Pydantic's reserved `model_config` attribute.

## Implementation Order

1. Add `CapabilityTier`, `ModelConfig`, and `resolve_model` to `models.py`.
2. Add `model_config_` field to `ForgeTaskInput`.
3. Add `capability_tier` field to `PlanStep`.
4. Update `workflows.py` to resolve tiers and pass model names to activities.
5. Update each agent factory to use the resolved model name.
6. Update the planner prompt to explain capability tier specification.
7. Add CLI options for tier-to-model overrides.
8. Tests for tier resolution, default behavior, and per-step overrides.

## Edge Cases

- **Unknown model name in config:** pydantic-ai raises at agent creation time. The error surfaces as a Temporal activity failure with a clear message.
- **No model_config in existing payloads:** Defaults to `ModelConfig()` which resolves to current behavior (Sonnet for everything). Fully backward compatible.
- **Planner specifies invalid tier:** pydantic validation catches it at plan deserialization time.
- **Mixed providers in config:** Valid — e.g., `reasoning: "openai:gpt-4o"` alongside `generation: "anthropic:claude-sonnet-4-5"`. pydantic-ai handles multi-provider routing.
- **Cost tracking:** `LLMStats.model_name` already records which model was used. Existing observability infrastructure tracks this per-call.

## Definition of Done

Phase 11 is complete when:

- Each LLM call site uses the appropriate capability tier.
- The planner defaults to a REASONING-tier model.
- Exploration defaults to a CLASSIFICATION-tier model.
- The tier-to-model mapping is configurable via `ModelConfig` and CLI flags.
- The planner can specify a capability tier per step.
- All existing tests pass (backward compatible — default model behavior is unchanged for GENERATION and SUMMARIZATION tiers).
- New tests cover: tier resolution, default config, per-step overrides, serialization compatibility.
