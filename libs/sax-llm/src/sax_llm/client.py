"""Shared LLM client utilities.

Pure functions for Anthropic SDK request construction and response parsing.
get_anthropic_client provides client management.

Design follows Function Core / Imperative Shell:

- Pure functions: build_tool_definition, build_system_param,
  build_messages_params, extract_tool_result, extract_usage
- Imperative shell: get_anthropic_client
- build_thinking_param is pure in its return value (same inputs always
  produce the same output) but does a shell-style side effect on the side:
  a one-time logger.warning per pre-adaptive model name it's asked to
  build a shape for. See its docstring.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from anthropic.types import Message
    from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Name fragments identifying pre-adaptive-thinking model generations. Not a
# model-generation database — a short, deliberately incomplete list of
# fragments seen in pins that predate D94's adaptive/disabled thinking
# shapes: Claude 4.5/4.6 point releases and the Claude 3 family. Extend only
# when a new pre-adaptive generation is actually seen in a pin; the current
# and future adaptive generations (e.g. "claude-sonnet-5", "claude-opus-4-8")
# must never match.
PRE_ADAPTIVE_HINTS: tuple[str, ...] = ("-4-5", "-4-6", "claude-3")

# Models we've already warned about, so a hot path calling build_thinking_param
# repeatedly for the same pin doesn't spam logs. Shell-side cache, deliberately
# module-level and mutable — mirrors the `_client` cache below.
_warned_pre_adaptive_models: set[str] = set()


def _snake_case(name: str) -> str:
    """Convert CamelCase class name to snake_case tool name."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def build_tool_definition(
    output_type: type[BaseModel],
    *,
    cache_control: bool = True,
) -> dict[str, Any]:
    """Build an Anthropic tool definition from a Pydantic model."""
    schema = output_type.model_json_schema()
    tool_name = _snake_case(output_type.__name__)
    description = (output_type.__doc__ or "").strip() or f"Structured output: {tool_name}"

    tool: dict[str, Any] = {
        "name": tool_name,
        "description": description,
        "input_schema": schema,
    }
    if cache_control:
        tool["cache_control"] = {"type": "ephemeral"}
    return tool


def build_system_param(
    system_prompt: str,
    *,
    cache_control: bool = True,
) -> list[dict[str, Any]] | str:
    """Build the system parameter for messages.create."""
    if cache_control:
        return [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    return system_prompt


def build_thinking_param(
    model_name: str,
    *,
    enabled: bool,
) -> dict[str, Any] | None:
    """Build the thinking parameter for messages.create.

    Supported-generation contract: this function only knows how to speak
    for adaptive-thinking-generation Anthropic models (D94) — it emits
    either ``{"type": "adaptive"}`` or the explicit ``{"type": "disabled"}``
    shape, and nothing else. Those are the *only* two thinking shapes it
    knows how to build. It returns None for Haiku and for non-Anthropic
    models, which support neither shape.

    For every other name containing "opus", "sonnet", or "claude", the
    function assumes the adaptive/disabled contract applies and returns one
    of those two shapes unconditionally — it does not maintain a database of
    model generations to verify that assumption. Pre-adaptive Anthropic
    pins (Claude 3.x; the "-4-5"/"-4-6" point releases) reject both shapes
    with a 400: D94 declared those pins unsupported, but a stale pin can
    still reach this function (e.g. a config that wasn't updated). For any
    model name matching a fragment in `PRE_ADAPTIVE_HINTS`, this function
    logs a one-time `logger.warning` (per distinct model name, across the
    process lifetime) flagging that the generation may reject these shapes
    and that pre-T3.2 pins are unsupported — then returns the shape
    unchanged. Behavior is otherwise identical to before this warning was
    added; this is a diagnostic, not a compatibility layer.

    For every other (adaptive-generation Anthropic) model, the "thinking"
    key is always populated in the result — never omitted. Omitting the key
    entirely causes these models to run adaptive thinking BY DEFAULT, so
    turning thinking "off" requires the explicit ``{"type": "disabled"}``
    shape rather than leaving the parameter out.
    """
    if "haiku" in model_name:
        return None

    if "opus" in model_name or "sonnet" in model_name or "claude" in model_name:
        if model_name not in _warned_pre_adaptive_models and any(
            hint in model_name for hint in PRE_ADAPTIVE_HINTS
        ):
            _warned_pre_adaptive_models.add(model_name)
            logger.warning(
                "build_thinking_param: model %r matches a pre-adaptive-generation "
                "name fragment and may reject the adaptive/disabled thinking shapes "
                "this function builds (D94). Pre-T3.2 model pins are unsupported.",
                model_name,
            )
        return {"type": "adaptive"} if enabled else {"type": "disabled"}

    return None


def build_messages_params(
    system_prompt: str,
    user_prompt: str,
    output_type: type[BaseModel],
    model: str,
    max_tokens: int,
    *,
    cache_instructions: bool = True,
    cache_tool_definitions: bool = True,
    thinking_enabled: bool = False,
    effort: str | None = None,
) -> dict[str, Any]:
    """Build the full kwargs dict for client.messages.create."""
    tool_def = build_tool_definition(output_type, cache_control=cache_tool_definitions)
    tool_name = tool_def["name"]
    system = build_system_param(system_prompt, cache_control=cache_instructions)

    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_prompt}],
        "tools": [tool_def],
        "tool_choice": {"type": "tool", "name": tool_name},
    }

    thinking = build_thinking_param(model, enabled=thinking_enabled)
    if thinking is not None:
        params["thinking"] = thinking
        if thinking_enabled:
            params["tool_choice"] = {"type": "auto"}
            if effort is not None:
                params["output_config"] = {"effort": effort}

    return params


def extract_tool_result(message: Message, output_type: type[BaseModel]) -> BaseModel:
    """Extract and validate the tool_use result from an Anthropic Message."""
    for block in message.content:
        if block.type == "tool_use":
            return output_type.model_validate(block.input)

    msg = f"No tool_use block found in message. Content types: {[b.type for b in message.content]}"
    raise ValueError(msg)


def extract_usage(message: Message) -> tuple[int, int, int, int]:
    """Extract usage statistics from an Anthropic Message.

    Returns (input_tokens, output_tokens, cache_creation, cache_read).
    """
    usage = message.usage
    return (
        usage.input_tokens,
        usage.output_tokens,
        getattr(usage, "cache_creation_input_tokens", 0) or 0,
        getattr(usage, "cache_read_input_tokens", 0) or 0,
    )


# ---------------------------------------------------------------------------
# Batch processing helpers
# ---------------------------------------------------------------------------


def build_batch_request(custom_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Wrap messages.create params into a batch request item."""
    return {"custom_id": custom_id, "params": params}


def parse_batch_response_json(
    raw_json: str,
    output_type_name: str,
) -> tuple[BaseModel, str, int, int, int, int]:
    """Deserialize a raw Anthropic Message JSON from a batch response.

    Returns (parsed_model, model_name, input_tokens, output_tokens,
             cache_creation_input_tokens, cache_read_input_tokens).

    Uses the output type registry — consumers must register their types
    via ``register_output_type()`` before calling this.
    """
    from anthropic.types import Message as AnthropicMessage

    from sax_llm.registry import get_output_type_registry

    registry = get_output_type_registry()
    if output_type_name not in registry:
        msg = f"Unknown output type: {output_type_name!r}. Registered: {list(registry)}"
        raise KeyError(msg)

    output_type = registry[output_type_name]
    data = json.loads(raw_json)
    message = AnthropicMessage.model_validate(data)

    parsed = extract_tool_result(message, output_type)
    in_tok, out_tok, cache_create, cache_read = extract_usage(message)

    return (parsed, message.model, in_tok, out_tok, cache_create, cache_read)


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------

_client: AsyncAnthropic | None = None


def get_anthropic_client() -> AsyncAnthropic:
    """Get or create a shared AsyncAnthropic client."""
    global _client
    if _client is None:
        from anthropic import AsyncAnthropic

        # Retries belong to the consumer's durable retry layer (e.g. a Temporal
        # activity RetryPolicy), not the SDK. The SDK default (2 retries) stacks
        # under a caller-side retry loop — Forge's _LLM_RETRY runs 3 attempts —
        # into up to 9 provider attempts per failing call, with 429/529 backoff
        # hidden from the orchestrator's timeouts. max_retries=0 stops the
        # client's own retry loop from stacking on top of (and hiding failures
        # from) the durable retries.
        _client = AsyncAnthropic(max_retries=0)
    return _client


def reset_client() -> None:
    """Clear the cached client. Intended for testing."""
    global _client
    _client = None
