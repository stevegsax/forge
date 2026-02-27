"""Exploration activities for LLM-guided context discovery (Phase 7).

Two activities:
1. call_exploration_llm — Calls Anthropic API with ExplorationResponse output type.
2. fulfill_context_requests — Dispatches requests to the provider registry.

Design follows Function Core / Imperative Shell:
- Testable functions: execute_exploration_call, build_exploration_prompt,
  fulfill_requests
- Imperative shell: call_exploration_llm, fulfill_context_requests
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from temporalio import activity

from forge.activities._heartbeat import heartbeat_during
from forge.domains import get_domain_config
from forge.llm_client import build_messages_params, extract_tool_result
from forge.message_log import write_message_log
from forge.models import (
    AssembledContext,
    ContextResult,
    ExplorationInput,
    ExplorationResponse,
    FulfillContextInput,
)

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

DEFAULT_EXPLORATION_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_EXPLORATION_MAX_TOKENS = 4096


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def build_exploration_prompt(
    input: ExplorationInput,
    project_instructions: str = "",
) -> tuple[str, str]:
    """Build system and user prompts for the exploration LLM call.

    Returns (system_prompt, user_prompt).
    """
    domain_config = get_domain_config(input.domain)
    parts: list[str] = []

    parts.append("You are a code exploration assistant.")
    parts.append("")
    task_noun = domain_config.exploration_task_noun
    parts.append(f"Your job is to gather the context needed to complete a {task_noun}.")
    parts.append("You have access to a set of context providers that can retrieve information")
    parts.append("about the codebase. Request context from providers until you have enough")
    parts.append("understanding to complete the task.")
    parts.append("")
    parts.append("When you have enough context, return an EMPTY requests list to signal")
    parts.append(f"that you are ready for the {domain_config.exploration_completion_noun} phase.")

    if project_instructions:
        parts.append("")
        parts.append(project_instructions)

    parts.append("")
    parts.append(f"## Round {input.round_number} of {input.max_rounds}")
    parts.append("")
    parts.append("## Task")
    parts.append(input.task_description)

    if input.target_files:
        parts.append("")
        parts.append("## Target Files")
        for f in input.target_files:
            parts.append(f"- {f}")

    parts.append("")
    parts.append("## Available Providers")
    for spec in input.available_providers:
        parts.append("")
        parts.append(f"### {spec.name}")
        parts.append(spec.description)
        if spec.parameters:
            parts.append("Parameters:")
            for param_name, param_desc in spec.parameters.items():
                parts.append(f"  - {param_name}: {param_desc}")

    if input.accumulated_context:
        parts.append("")
        parts.append("## Previously Retrieved Context")
        for ctx in input.accumulated_context:
            parts.append("")
            parts.append(f"### From: {ctx.provider}")
            # Truncate very long context to keep the prompt manageable
            content = ctx.content
            if len(content) > 8000:
                content = content[:8000] + "\n... (truncated)"
            parts.append(content)

    system_prompt = "\n".join(parts)

    user_prompt = (
        "Based on the task and any context already retrieved, decide what additional "
        "context you need. Return a list of provider requests, or an empty list if "
        f"you have enough context to proceed with {domain_config.exploration_completion_noun}."
    )

    return system_prompt, user_prompt


def fulfill_requests(
    requests: list[dict[str, object]],
    repo_root: str,
    worktree_path: str,
) -> list[ContextResult]:
    """Dispatch context requests to the provider registry.

    Args:
        requests: List of dicts with 'provider' and 'params' keys.
        repo_root: Path to the repository root.
        worktree_path: Path to the worktree.

    Returns:
        List of ContextResult with provider responses.
    """
    from forge.code_intel.repo_map import estimate_tokens
    from forge.providers import PROVIDER_REGISTRY

    results: list[ContextResult] = []

    for request in requests:
        provider_name = str(request.get("provider", ""))
        params = {str(k): str(v) for k, v in (request.get("params") or {}).items()}

        handler = PROVIDER_REGISTRY.get(provider_name)
        if handler is None:
            content = f"Error: Unknown provider '{provider_name}'."
        else:
            try:
                content = handler(params, repo_root, worktree_path)
            except Exception as e:
                logger.warning("Provider %s failed: %s", provider_name, e, exc_info=True)
                content = f"Error: Provider '{provider_name}' failed: {e}"

        results.append(
            ContextResult(
                provider=provider_name,
                content=content,
                estimated_tokens=estimate_tokens(content),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


async def execute_exploration_call(
    input: ExplorationInput,
    client: AsyncAnthropic,
    project_instructions: str = "",
) -> ExplorationResponse:
    """Call the Anthropic API for exploration and return the structured response.

    Separated from the imperative shell so tests can inject a mock client.
    """
    system_prompt, user_prompt = build_exploration_prompt(input, project_instructions)
    model = input.model_name or DEFAULT_EXPLORATION_MODEL

    params = build_messages_params(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_type=ExplorationResponse,
        model=model,
        max_tokens=DEFAULT_EXPLORATION_MAX_TOKENS,
    )
    message = await client.messages.create(**params)

    if input.log_messages and input.worktree_path:
        request_json = json.dumps(params, indent=2, default=str)
        write_message_log(input.worktree_path, "explore-request", request_json)
        write_message_log(
            input.worktree_path, "explore-response", message.model_dump_json(indent=2)
        )

    return extract_tool_result(message, ExplorationResponse)


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


@activity.defn
async def call_exploration_llm(input: ExplorationInput) -> ExplorationResponse:
    """Activity: call the exploration LLM to decide what context to request."""
    from pathlib import Path

    from forge.activities.context import (
        _read_project_instructions,
        build_project_instructions_section,
    )
    from forge.llm_client import get_anthropic_client
    from forge.tracing import get_tracer

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.call_exploration_llm") as span:
        logger.info(
            "Exploration call: task_id=%s round=%d/%d",
            input.task_id,
            input.round_number,
            input.max_rounds,
        )
        project_instructions = ""
        if input.repo_root:
            project_instructions = build_project_instructions_section(
                _read_project_instructions(Path(input.repo_root))
            )

        client = get_anthropic_client()
        start = time.monotonic()
        async with heartbeat_during():
            response = await execute_exploration_call(input, client, project_instructions)
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Exploration result: task_id=%s requests=%d", input.task_id, len(response.requests)
        )

        span.set_attributes(
            {
                "forge.exploration.round": input.round_number,
                "forge.exploration.requests_count": len(response.requests),
                "forge.exploration.latency_ms": elapsed_ms,
            }
        )

        return response


@activity.defn
async def fulfill_context_requests(input: FulfillContextInput) -> list[ContextResult]:
    """Activity: dispatch context requests to the provider registry."""
    from forge.tracing import get_tracer

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.fulfill_context_requests") as span:
        logger.info("Fulfilling %d context requests", len(input.requests))
        requests_as_dicts = [{"provider": r.provider, "params": r.params} for r in input.requests]

        results = fulfill_requests(
            requests_as_dicts,
            input.repo_root,
            input.worktree_path,
        )

        span.set_attributes(
            {
                "forge.exploration.providers_called": len(results),
                "forge.exploration.total_tokens": sum(r.estimated_tokens for r in results),
            }
        )

        return results


@activity.defn
async def assemble_exploration_context(input: ExplorationInput) -> AssembledContext:
    """Build exploration prompts as AssembledContext for batch path."""
    from pathlib import Path

    from forge.activities.context import (
        _read_project_instructions,
        build_project_instructions_section,
    )

    project_instructions = ""
    if input.repo_root:
        project_instructions = build_project_instructions_section(
            _read_project_instructions(Path(input.repo_root))
        )

    system_prompt, user_prompt = build_exploration_prompt(input, project_instructions)
    model = input.model_name or DEFAULT_EXPLORATION_MODEL

    return AssembledContext(
        task_id=input.task_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=model,
        log_messages=input.log_messages,
        worktree_path=input.worktree_path,
    )
