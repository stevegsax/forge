"""Provider registry and model ID parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.llm_providers.protocol import LLMProvider

_DEFAULT_PROVIDER = "anthropic"

_provider_cache: dict[str, LLMProvider] = {}


def parse_model_id(model_id: str) -> tuple[str, str]:
    """Parse a model ID into (provider, model) tuple.

    Supports explicit ``provider:model`` syntax. Bare names without ``:``
    default to the ``"anthropic"`` provider for backward compatibility.

    Examples:
        >>> parse_model_id("anthropic:claude-sonnet-4-5-20250929")
        ('anthropic', 'claude-sonnet-4-5-20250929')
        >>> parse_model_id("mistral:mistral-large-latest")
        ('mistral', 'mistral-large-latest')
        >>> parse_model_id("claude-sonnet-4-5-20250929")
        ('anthropic', 'claude-sonnet-4-5-20250929')
    """
    if ":" in model_id:
        provider, model = model_id.split(":", 1)
        return provider, model
    return _DEFAULT_PROVIDER, model_id


def get_provider(model_id: str) -> LLMProvider:
    """Return a cached provider instance for the given model ID.

    The model ID is parsed to extract the provider name. Provider instances
    are singletons, cached by provider name.
    """
    provider_name, _ = parse_model_id(model_id)

    if provider_name in _provider_cache:
        return _provider_cache[provider_name]

    if provider_name == "anthropic":
        from forge.llm_providers.anthropic import AnthropicProvider

        instance: LLMProvider = AnthropicProvider()
    elif provider_name == "mistral":
        from forge.llm_providers.mistral import MistralProvider

        instance = MistralProvider()
    else:
        msg = f"Unknown LLM provider: {provider_name!r}"
        raise ValueError(msg)

    _provider_cache[provider_name] = instance
    return instance


def reset_provider_cache() -> None:
    """Clear the provider cache. Intended for testing."""
    _provider_cache.clear()
