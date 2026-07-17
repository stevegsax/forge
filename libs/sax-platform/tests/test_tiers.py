"""Tests for sax_platform.llm.tiers — the model-tier registry and thinking policy."""

import pytest
from pydantic import ValidationError

from sax_platform.llm.tiers import (
    CapabilityTier,
    ModelConfig,
    ThinkingPolicy,
    resolve_model,
    split_provider,
)


class TestResolveModelDefaults:
    @pytest.mark.parametrize(
        ("tier", "expected"),
        [
            (CapabilityTier.REASONING, "anthropic:claude-opus-4-8"),
            (CapabilityTier.GENERATION, "anthropic:claude-sonnet-5"),
            (CapabilityTier.SUMMARIZATION, "anthropic:claude-sonnet-5"),
            (CapabilityTier.CLASSIFICATION, "anthropic:claude-haiku-4-5"),
        ],
    )
    def test_resolves_default_pin_per_tier(self, tier: CapabilityTier, expected: str) -> None:
        assert resolve_model(tier, ModelConfig()) == expected


class TestModelConfigOverrides:
    def test_override_wins_over_default(self) -> None:
        config = ModelConfig(reasoning="anthropic:claude-custom-1")
        assert resolve_model(CapabilityTier.REASONING, config) == "anthropic:claude-custom-1"

    def test_overriding_one_field_leaves_others_default(self) -> None:
        config = ModelConfig(reasoning="anthropic:claude-custom-1")
        assert resolve_model(CapabilityTier.GENERATION, config) == "anthropic:claude-sonnet-5"

    def test_all_four_fields_independently_overridable(self) -> None:
        config = ModelConfig(
            reasoning="p:r",
            generation="p:g",
            summarization="p:s",
            classification="p:c",
        )
        assert resolve_model(CapabilityTier.REASONING, config) == "p:r"
        assert resolve_model(CapabilityTier.GENERATION, config) == "p:g"
        assert resolve_model(CapabilityTier.SUMMARIZATION, config) == "p:s"
        assert resolve_model(CapabilityTier.CLASSIFICATION, config) == "p:c"


class TestModelConfigFrozen:
    def test_assigning_a_field_raises(self) -> None:
        config = ModelConfig()
        with pytest.raises(ValidationError, match="frozen"):
            config.reasoning = "anthropic:claude-other"  # type: ignore[misc]


class TestThinkingPolicy:
    def test_defaults(self) -> None:
        policy = ThinkingPolicy()
        assert policy.enabled is True
        assert policy.effort == "high"

    def test_overrides(self) -> None:
        policy = ThinkingPolicy(enabled=False, effort="max")
        assert policy.enabled is False
        assert policy.effort == "max"

    def test_frozen(self) -> None:
        policy = ThinkingPolicy()
        with pytest.raises(ValidationError, match="frozen"):
            policy.enabled = False  # type: ignore[misc]

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
    def test_accepts_every_valid_effort(self, effort: str) -> None:
        policy = ThinkingPolicy(effort=effort)  # type: ignore[arg-type]
        assert policy.effort == effort

    def test_rejects_invalid_effort(self) -> None:
        with pytest.raises(ValidationError):
            ThinkingPolicy(effort="ultra")  # type: ignore[arg-type]


class TestSplitProvider:
    def test_splits_qualified_string(self) -> None:
        assert split_provider("anthropic:claude-sonnet-5") == ("anthropic", "claude-sonnet-5")

    def test_bare_name_defaults_to_anthropic(self) -> None:
        assert split_provider("claude-sonnet-5") == ("anthropic", "claude-sonnet-5")

    def test_splits_on_first_colon_only(self) -> None:
        assert split_provider("anthropic:claude:variant") == ("anthropic", "claude:variant")


class TestDefaultPinsAreProviderQualified:
    @pytest.mark.parametrize(
        "qualified",
        [
            ModelConfig().reasoning,
            ModelConfig().generation,
            ModelConfig().summarization,
            ModelConfig().classification,
        ],
    )
    def test_default_pin_parses_as_anthropic(self, qualified: str) -> None:
        provider, model = split_provider(qualified)
        assert provider == "anthropic"
        assert model
