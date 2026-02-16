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

import logging
import time
from typing import TYPE_CHECKING

from temporalio import activity

from forge.domains import get_domain_config
from forge.llm_client import build_messages_params, extract_tool_result
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

    Phase 15: When MCP tools or Skills are present in available_providers,
    the prompt explains the two-track system: context requests (via `requests`)
    and action requests (via `actions`).
    """
    domain_config = get_domain_config(input.task.domain)
    parts: list[str] = []

    # Separate context providers from action capabilities
    context_providers = []
    action_capabilities = []
    for spec in input.available_providers:
        if spec.name.startswith("mcp_") or spec.name.startswith("skill_"):
            action_capabilities.append(spec)
        else:
            context_providers.append(spec)

    has_actions = bool(action_capabilities)

    parts.append("You are a code exploration assistant.")
    parts.append("")
    task_noun = domain_config.exploration_task_noun
    parts.append(f"Your job is to gather the context needed to complete a {task_noun}.")
    parts.append("You have access to a set of context providers that can retrieve information")
    parts.append("about the codebase. Request context from providers until you have enough")
    parts.append("understanding to complete the task.")

    if has_actions:
        parts.append("")
        parts.append("You also have access to **actions** — MCP tools and Agent Skills that")
        parts.append("can perform operations beyond information retrieval. Use the `actions`")
        parts.append("field (not `requests`) to invoke these. Each action request needs:")
        parts.append("  - `capability`: The action name (e.g. mcp_github_search, skill_deploy)")
        parts.append("  - `capability_type`: Either 'mcp_tool' or 'skill'")
        parts.append("  - `params`: Arguments for the action")
        parts.append("  - `reasoning`: Why this action is needed")

    parts.append("")
    if has_actions:
        readiness_signal = "return EMPTY `requests` and `actions` lists"
    else:
        readiness_signal = "return an EMPTY requests list"
    parts.append(f"When you have enough context, {readiness_signal} to signal")
    parts.append(f"that you are ready for the {domain_config.exploration_completion_noun} phase.")

    if project_instructions:
        parts.append("")
        parts.append(project_instructions)

    parts.append("")
    parts.append(f"## Round {input.round_number} of {input.max_rounds}")
    parts.append("")
    parts.append("## Task")
    parts.append(input.task.description)

    if input.task.target_files:
        parts.append("")
        parts.append("## Target Files")
        for f in input.task.target_files:
            parts.append(f"- {f}")

    parts.append("")
    parts.append("## Available Context Providers")
    for spec in context_providers:
        parts.append("")
        parts.append(f"### {spec.name}")
        parts.append(spec.description)
        if spec.parameters:
            parts.append("Parameters:")
            for param_name, param_desc in spec.parameters.items():
                parts.append(f"  - {param_name}: {param_desc}")

    if action_capabilities:
        parts.append("")
        parts.append("## Available Actions (MCP Tools & Skills)")
        parts.append("Use the `actions` field (not `requests`) to invoke these.")
        for spec in action_capabilities:
            parts.append("")
            cap_type = "mcp_tool" if spec.name.startswith("mcp_") else "skill"
            parts.append(f"### {spec.name} (type: {cap_type})")
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

    if has_actions:
        user_prompt = (
            "Based on the task and any context already retrieved, decide what additional "
            "context or actions you need. Return provider requests in `requests` and/or "
            "action requests in `actions`. Return both empty to proceed with "
            f"{domain_config.exploration_completion_noun}."
        )
    else:
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
        project_instructions = ""
        if input.repo_root:
            project_instructions = build_project_instructions_section(
                _read_project_instructions(Path(input.repo_root))
            )

        client = get_anthropic_client()
        start = time.monotonic()
        response = await execute_exploration_call(input, client, project_instructions)
        elapsed_ms = (time.monotonic() - start) * 1000

        span.set_attributes(
            {
                "forge.exploration.round": input.round_number,
                "forge.exploration.requests_count": len(response.requests),
                "forge.exploration.actions_count": len(response.actions),
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
        task_id=input.task.task_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=model,
    )
