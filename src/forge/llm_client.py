"""Shared LLM client utilities for Forge.

Replaces pydantic-ai Agent with direct Anthropic SDK calls.
Pure functions handle request construction and response parsing;
get_anthropic_client provides client management.

Design follows Function Core / Imperative Shell:
- Pure functions: build_tool_definition, build_system_param, build_thinking_param,
  build_messages_params, extract_tool_result, extract_usage
- Imperative shell: get_anthropic_client
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
) -> dict:
    """Build an Anthropic tool definition from a Pydantic model.

    Uses model_json_schema() for the input_schema, the model's docstring
    for the description, and a snake_case version of the class name for the name.
    """
    schema = output_type.model_json_schema()
    # Remove $defs key — Anthropic tools don't use JSON Schema $ref.
    # Pydantic v2 inlines definitions when there are no circular refs,
    # but $defs may still appear. We keep it if present since the API
    # accepts full JSON Schema including $defs.
    tool_name = _snake_case(output_type.__name__)
    description = (output_type.__doc__ or "").strip() or f"Structured output: {tool_name}"

    tool: dict = {
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
) -> list[dict] | str:
    """Build the system parameter for messages.create.

    When caching, returns a list with a single text block containing
    cache_control. Otherwise returns the plain string.
    """
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
    budget_tokens: int,
) -> dict | None:
    """Build the thinking parameter for messages.create.

    Returns a thinking dict for Anthropic models, or None for non-Anthropic/Haiku.
    - Opus: adaptive thinking with effort level
    - Sonnet: budget-based thinking
    - Haiku/non-Anthropic: None (silent degradation per D63)
    """
    if budget_tokens <= 0:
        return None

    # Only Anthropic models support thinking
    if "haiku" in model_name:
        return None

    if "opus" in model_name:
        return {"type": "enabled", "budget_tokens": budget_tokens}

    if "sonnet" in model_name or "claude" in model_name:
        return {"type": "enabled", "budget_tokens": budget_tokens}

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
    thinking_budget_tokens: int = 0,
) -> dict:
    """Build the full kwargs dict for client.messages.create.

    Assembles system, messages, tools, tool_choice, model, and max_tokens.
    """
    tool_def = build_tool_definition(output_type, cache_control=cache_tool_definitions)
    tool_name = tool_def["name"]
    system = build_system_param(system_prompt, cache_control=cache_instructions)

    params: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_prompt}],
        "tools": [tool_def],
        "tool_choice": {"type": "tool", "name": tool_name},
    }

    thinking = build_thinking_param(model, thinking_budget_tokens)
    if thinking is not None:
        params["thinking"] = thinking
        # When thinking is enabled, tool_choice must be "auto" (API constraint)
        params["tool_choice"] = {"type": "auto"}
        # Thinking requires higher max_tokens to accommodate the thinking budget
        params["max_tokens"] = max(max_tokens, thinking_budget_tokens + max_tokens)

    return params


def extract_tool_result(message: Message, output_type: type[BaseModel]) -> BaseModel:
    """Extract and validate the tool_use result from an Anthropic Message.

    Iterates message.content, finds the first tool_use block, and validates
    its input against the output_type Pydantic model.

    Raises ValueError if no tool_use block is found.
    """
    for block in message.content:
        if block.type == "tool_use":
            return output_type.model_validate(block.input)

    msg = f"No tool_use block found in message. Content types: {[b.type for b in message.content]}"
    raise ValueError(msg)


def extract_usage(message: Message) -> tuple[int, int, int, int]:
    """Extract usage statistics from an Anthropic Message.

    Returns (input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens).
    """
    usage = message.usage
    return (
        usage.input_tokens,
        usage.output_tokens,
        getattr(usage, "cache_creation_input_tokens", 0) or 0,
        getattr(usage, "cache_read_input_tokens", 0) or 0,
    )


# ---------------------------------------------------------------------------
# Batch processing helpers (Phase 14)
# ---------------------------------------------------------------------------


def build_batch_request(custom_id: str, params: dict) -> dict:
    """Wrap messages.create params into a batch request item.

    Returns the dict expected by client.messages.batches.create(requests=[...]).
    """
    return {"custom_id": custom_id, "params": params}


def get_output_type_registry() -> dict[str, type[BaseModel]]:
    """Return a mapping of type names to Pydantic model classes.

    Lazy imports avoid circular dependencies between llm_client and models.
    """
    from forge.eval.models import JudgeVerdict
    from forge.models import (
        ConflictResolutionResponse,
        ExplorationResponse,
        ExtractionResult,
        LLMResponse,
        Plan,
        SanityCheckResponse,
    )

    return {
        "LLMResponse": LLMResponse,
        "Plan": Plan,
        "ExplorationResponse": ExplorationResponse,
        "SanityCheckResponse": SanityCheckResponse,
        "ConflictResolutionResponse": ConflictResolutionResponse,
        "ExtractionResult": ExtractionResult,
        "JudgeVerdict": JudgeVerdict,
    }


def parse_batch_response_json(
    raw_json: str,
    output_type_name: str,
) -> tuple:
    """Deserialize a raw Anthropic Message JSON from a batch response.

    Returns (parsed_model, model_name, input_tokens, output_tokens,
             cache_creation_input_tokens, cache_read_input_tokens).

    Raises KeyError if output_type_name is not in the registry.
    Raises ValueError if no tool_use block is found in the message.
    """
    from anthropic.types import Message as AnthropicMessage

    registry = get_output_type_registry()
    if output_type_name not in registry:
        msg = f"Unknown output type: {output_type_name!r}"
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

_client = None


def get_anthropic_client():
    """Get or create a shared AsyncAnthropic client.

    The client reads ANTHROPIC_API_KEY from the environment automatically.
    """
    global _client
    if _client is None:
        from anthropic import AsyncAnthropic

        _client = AsyncAnthropic()
    return _client
