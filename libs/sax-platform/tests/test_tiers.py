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
            config.reasoning = "anthropic:claude-other"


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
            policy.enabled = False

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
    def test_accepts_every_valid_effort(self, effort: str) -> None:
        policy = ThinkingPolicy(effort=effort)  # type: ignore[arg-type]
        assert policy.effort == effort

    def test_rejects_invalid_effort(self) -> None:
        with pytest.raises(ValidationError):
            ThinkingPolicy(effort="ultra")  # type: ignore[arg-type]


class TestThinkingPolicyLegacyBudgetTokensCompat:
    """Regression tests for the T3.2 in-flight-payload compatibility shim.

    Pre-T3.2 serialized payloads carried forge's old `ThinkingConfig` shape
    — `{"budget_tokens": N, "effort": ...}` — with no `enabled` key. Without
    the shim, that shape deserializes with `budget_tokens` silently dropped
    (unknown field to this model) and `enabled` defaulting to True — flipping
    thinking back ON for an in-flight workflow that had disabled it via
    `budget_tokens: 0`.
    """

    def test_zero_budget_tokens_disables(self) -> None:
        policy = ThinkingPolicy.model_validate({"budget_tokens": 0})
        assert policy.enabled is False

    def test_positive_budget_tokens_enables(self) -> None:
        policy = ThinkingPolicy.model_validate({"budget_tokens": 10000})
        assert policy.enabled is True
        # effort default is preserved — the legacy payload said nothing
        # about it.
        assert policy.effort == "high"

    def test_explicit_enabled_wins_over_budget_tokens(self) -> None:
        """An explicit `enabled` in the payload always wins over whatever
        the legacy `budget_tokens` value would have implied."""
        disabled_but_positive_budget = ThinkingPolicy.model_validate(
            {"budget_tokens": 10000, "enabled": False}
        )
        assert disabled_but_positive_budget.enabled is False

        enabled_but_zero_budget = ThinkingPolicy.model_validate(
            {"budget_tokens": 0, "enabled": True}
        )
        assert enabled_but_zero_budget.enabled is True

    def test_budget_tokens_key_does_not_reach_the_model(self) -> None:
        policy = ThinkingPolicy.model_validate({"budget_tokens": 5000})
        assert "budget_tokens" not in policy.model_dump()

    def test_normal_new_shape_payload_unaffected(self) -> None:
        """A payload with no `budget_tokens` at all — the current shape —
        is unaffected by the shim."""
        policy = ThinkingPolicy.model_validate({"enabled": False, "effort": "low"})
        assert policy.enabled is False
        assert policy.effort == "low"

    def test_defaults_still_apply_with_no_budget_tokens_and_no_enabled(self) -> None:
        policy = ThinkingPolicy.model_validate({})
        assert policy.enabled is True
        assert policy.effort == "high"

    def test_migrated_instance_is_still_frozen(self) -> None:
        policy = ThinkingPolicy.model_validate({"budget_tokens": 0})
        with pytest.raises(ValidationError, match="frozen"):
            policy.enabled = True


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
