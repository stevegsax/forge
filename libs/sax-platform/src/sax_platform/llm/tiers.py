"""Model-tier registry and thinking policy, single-sourced for the platform (D94).

Forge's copy of this registry (`forge.models.CapabilityTier` / `ModelConfig` /
`resolve_model` / `ThinkingConfig`) is retired in favor of this module (T3.2).
`sax_llm`'s wire builders and, later, `sax_platform.llm.client` are the callers
that turn a resolved model name and a `ThinkingPolicy` into an actual request.
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "CapabilityTier",
    "Effort",
    "ModelConfig",
    "ThinkingPolicy",
    "resolve_model",
    "split_provider",
]


class CapabilityTier(StrEnum):
    """Capability tier for model routing.

    Mirrors forge's `forge.models.CapabilityTier` (Phase 11); single-sourced
    here per D94 (T3.2).
    """

    REASONING = "reasoning"
    GENERATION = "generation"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"


# Default pins, live-verified 2026-07-16 against the Models API.
_DEFAULT_TIER_MODELS: dict[CapabilityTier, str] = {
    CapabilityTier.REASONING: "anthropic:claude-opus-4-8",
    CapabilityTier.GENERATION: "anthropic:claude-sonnet-5",
    CapabilityTier.SUMMARIZATION: "anthropic:claude-sonnet-5",
    CapabilityTier.CLASSIFICATION: "anthropic:claude-haiku-4-5",
}


class ModelConfig(BaseModel):
    """Maps capability tiers to concrete, provider-qualified model names.

    Defaults are the tier pins live-verified 2026-07-16 against the Models
    API (see `_DEFAULT_TIER_MODELS`).
    """

    model_config = ConfigDict(frozen=True)

    reasoning: str = Field(default=_DEFAULT_TIER_MODELS[CapabilityTier.REASONING])
    generation: str = Field(default=_DEFAULT_TIER_MODELS[CapabilityTier.GENERATION])
    summarization: str = Field(default=_DEFAULT_TIER_MODELS[CapabilityTier.SUMMARIZATION])
    classification: str = Field(default=_DEFAULT_TIER_MODELS[CapabilityTier.CLASSIFICATION])


def resolve_model(tier: CapabilityTier, config: ModelConfig) -> str:
    """Resolve a capability tier to a concrete, provider-qualified model name."""
    return {
        CapabilityTier.REASONING: config.reasoning,
        CapabilityTier.GENERATION: config.generation,
        CapabilityTier.SUMMARIZATION: config.summarization,
        CapabilityTier.CLASSIFICATION: config.classification,
    }[tier]


Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ThinkingPolicy(BaseModel):
    """Extended-thinking policy: whether thinking is enabled, and at what effort.

    Platform successor of forge's `forge.models.ThinkingConfig`. That type's
    `budget_tokens` knob is gone: on the current model generation the API
    rejects an explicit `budget_tokens` with a 400 — adaptive thinking plus
    an explicit `effort` is the only supported configuration (D94).

    On current Sonnet/Opus, omitting the `thinking` field from the request
    entirely runs adaptive thinking BY DEFAULT, so *disabling* thinking is
    not "leave the field off" — it requires sending an explicit disabled
    shape. `ThinkingPolicy` carries only the policy (enabled + effort);
    translating it into the actual wire shape — including that explicit-
    disabled case — is built in `sax_llm`'s wire builders until T3.5/T3.6,
    and by `sax_platform.llm.client` after.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    effort: Effort = "high"

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_budget_tokens(cls, data: Any) -> Any:
        """T3.2 in-flight-payload compatibility shim.

        Pre-T3.2 serialized payloads carried forge's old
        `ThinkingConfig` shape — `{"budget_tokens": N, "effort": ...}` — with
        no `enabled` key. Left alone, that shape would deserialize here with
        `budget_tokens` silently dropped (unknown field) and `enabled`
        defaulting to `True`, which flips thinking back ON for an in-flight
        workflow that had explicitly disabled it via `budget_tokens: 0`.

        When the input mapping carries `budget_tokens` and does NOT also
        carry `enabled`, derive `enabled` from whether the legacy budget was
        positive. An explicit `enabled` in the payload always wins over the
        legacy field. `budget_tokens` itself is always dropped from what
        reaches the model — this type has no such field.

        Delete this shim once no pre-T3.2 workflow history can still be in
        flight.
        """
        if not isinstance(data, Mapping):
            return data
        if "budget_tokens" not in data:
            return data
        migrated = dict(data)
        budget_tokens = migrated.pop("budget_tokens")
        if "enabled" not in migrated:
            is_numeric = isinstance(budget_tokens, int | float) and not isinstance(
                budget_tokens, bool
            )
            migrated["enabled"] = is_numeric and budget_tokens > 0
        return migrated


def split_provider(qualified: str) -> tuple[str, str]:
    """Split a provider-qualified model string into (provider, model).

    Forge's existing convention (mirrored from `sax_llm.registry.parse_model_id`):
    `"anthropic:claude-x"` -> `("anthropic", "claude-x")`. A bare name with no
    `:` defaults to the `"anthropic"` provider: `"claude-x"` -> `("anthropic", "claude-x")`.
    """
    if ":" in qualified:
        provider, model = qualified.split(":", 1)
        return provider, model
    return "anthropic", qualified
