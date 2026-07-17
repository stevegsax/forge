"""Shared LLM client utilities.

Pure functions for Anthropic SDK request construction and response parsing.
get_anthropic_client provides client management.

Design follows Function Core / Imperative Shell:

- Pure functions: build_tool_definition, build_system_param, build_thinking_param,
  build_messages_params, extract_tool_result, extract_usage
- Imperative shell: get_anthropic_client
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from anthropic.types import Message
    from pydantic import BaseModel


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

    Post-D94, the model registry pins adaptive-generation Anthropic models
    only; the pre-4.6 ``budget_tokens``-style thinking param those models
    used is unsupported by the API (a 400) and is never emitted here.
    Returns None for Haiku and non-Anthropic models, which support neither
    the adaptive nor the disabled shape.

    For every other (adaptive-generation Anthropic) model, the "thinking"
    key is always populated in the result — never omitted. Omitting the key
    entirely causes these models to run adaptive thinking BY DEFAULT, so
    turning thinking "off" requires the explicit ``{"type": "disabled"}``
    shape rather than leaving the parameter out.
    """
    if "haiku" in model_name:
        return None

    if "opus" in model_name or "sonnet" in model_name or "claude" in model_name:
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
