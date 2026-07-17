"""Tests for llm_providers registry: parse_model_id and get_provider."""

from __future__ import annotations

import pytest

from sax_llm.registry import (
    get_provider,
    get_provider_by_name,
    parse_model_id,
    reset_provider_cache,
)

# ---------------------------------------------------------------------------
# parse_model_id
# ---------------------------------------------------------------------------


class TestParseModelId:
    """Tests for the parse_model_id pure function."""

    def test_bare_name_defaults_to_anthropic(self) -> None:
        provider, model = parse_model_id("claude-sonnet-4-5-20250929")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-5-20250929"

    def test_anthropic_prefix(self) -> None:
        provider, model = parse_model_id("anthropic:claude-sonnet-4-5-20250929")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-5-20250929"

    def test_unknown_provider(self) -> None:
        provider, model = parse_model_id("openai:gpt-4")
        assert provider == "openai"
        assert model == "gpt-4"

    def test_model_name_with_colons(self) -> None:
        """Only splits on first colon."""
        provider, model = parse_model_id("anthropic:model:variant")
        assert provider == "anthropic"
        assert model == "model:variant"

    def test_empty_string(self) -> None:
        provider, model = parse_model_id("")
        assert provider == "anthropic"
        assert model == ""


# ---------------------------------------------------------------------------
# get_provider
# ---------------------------------------------------------------------------


class TestGetProvider:
    """Tests for the get_provider factory."""

    def setup_method(self) -> None:
        reset_provider_cache()

    def test_returns_anthropic_provider_for_bare_name(self) -> None:
        from sax_llm.anthropic import AnthropicProvider

        provider = get_provider("claude-sonnet-4-5-20250929")
        assert isinstance(provider, AnthropicProvider)

    def test_returns_anthropic_provider_for_prefixed_name(self) -> None:
        from sax_llm.anthropic import AnthropicProvider

        provider = get_provider("anthropic:claude-sonnet-4-5-20250929")
        assert isinstance(provider, AnthropicProvider)

    def test_caches_provider_instances(self) -> None:
        p1 = get_provider("anthropic:model-a")
        p2 = get_provider("anthropic:model-b")
        assert p1 is p2

    def test_raises_for_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_provider("openai:gpt-4")

    def test_reset_clears_cache(self) -> None:
        p1 = get_provider("anthropic:model-a")
        reset_provider_cache()
        p2 = get_provider("anthropic:model-a")
        assert p1 is not p2

    def test_get_provider_delegates_to_get_provider_by_name(self) -> None:
        """get_provider should return the same cached instance as get_provider_by_name."""
        p1 = get_provider("anthropic:claude-sonnet-4-5-20250929")
        p2 = get_provider_by_name("anthropic")
        assert p1 is p2


# ---------------------------------------------------------------------------
# get_provider_by_name
# ---------------------------------------------------------------------------


class TestGetProviderByName:
    """Tests for the get_provider_by_name factory (bare provider names)."""

    def setup_method(self) -> None:
        reset_provider_cache()

    def test_returns_anthropic_provider(self) -> None:
        from sax_llm.anthropic import AnthropicProvider

        provider = get_provider_by_name("anthropic")
        assert isinstance(provider, AnthropicProvider)

    def test_raises_for_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_provider_by_name("openai")

    def test_caches_provider_instances(self) -> None:
        p1 = get_provider_by_name("anthropic")
        p2 = get_provider_by_name("anthropic")
        assert p1 is p2
