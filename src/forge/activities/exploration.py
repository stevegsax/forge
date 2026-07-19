"""Exploration activities for LLM-guided context discovery (Phase 7).

Design follows Function Core / Imperative Shell:
- Pure/testable functions: execute_exploration_call, build_exploration_prompt,
  fulfill_requests. The ``call_exploration_llm`` (LlmActivities) and
  ``fulfill_context_requests`` (ContextActivities) activity shells are bound
  methods on the T3.6 composition-root classes (``forge.activities.roots``)
  that delegate to these; ``assemble_exploration_context`` stays a free
  activity (no dependency to inject).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from temporalio import activity

from forge.domains import get_domain_config
from forge.message_log import write_message_log
from forge.models import (
    AssembledContext,
    CapabilityTier,
    ContextResult,
    ExplorationInput,
    ExplorationResponse,
    ModelConfig,
    resolve_model,
)

if TYPE_CHECKING:
    from sax_platform.llm import AnthropicLLM
    from sqlalchemy import Engine

logger = logging.getLogger(__name__)

# Shadow fallback for a missing input.model_name (see forge.activities.llm).
# Matches the live wiring in workflows.py, which routes exploration through
# the CLASSIFICATION tier (cheap, high-volume calls) — the fallback must not
# silently upgrade the model when a caller forgets model_name.
DEFAULT_EXPLORATION_MODEL = resolve_model(CapabilityTier.CLASSIFICATION, ModelConfig())
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
    engine: Engine | None = None,
) -> list[ContextResult]:
    """Dispatch context requests to the provider registry.

    Args:
        requests: List of dicts with 'provider' and 'params' keys.
        repo_root: Path to the repository root.
        worktree_path: Path to the worktree.
        engine: Store engine threaded to the store-backed providers
            (``past_runs``/``playbooks``). Supplied by the
            ``ContextActivities`` composition root; ``None`` (the default,
            for callers that request no store-backed provider) makes those
            providers report the store unavailable.

    Returns:
        List of ContextResult with provider responses.
    """
    from forge.code_intel.repo_map import estimate_tokens
    from forge.providers import PROVIDER_REGISTRY, handle_past_runs, handle_playbooks

    # The two store-backed providers take the engine as a 4th argument; every
    # other handler keeps the 3-arg ProviderHandler shape.
    engine_providers = {"past_runs": handle_past_runs, "playbooks": handle_playbooks}

    results: list[ContextResult] = []

    for request in requests:
        provider_name = str(request.get("provider", ""))
        raw_params = request.get("params")
        params_source = raw_params if isinstance(raw_params, dict) else {}
        params = {str(k): str(v) for k, v in params_source.items()}

        handler = PROVIDER_REGISTRY.get(provider_name)
        if handler is None:
            content = f"Error: Unknown provider '{provider_name}'."
        else:
            try:
                engine_handler = engine_providers.get(provider_name)
                if engine_handler is not None:
                    content = engine_handler(params, repo_root, worktree_path, engine)
                else:
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
    llm: AnthropicLLM,
    project_instructions: str = "",
) -> ExplorationResponse:
    """Call the LLM for exploration and return the structured response.

    Separated from the imperative shell so tests can inject a mock client.
    """
    from sax_platform.llm.tiers import split_provider

    system_prompt, user_prompt = build_exploration_prompt(input, project_instructions)
    full_model = input.model_name or DEFAULT_EXPLORATION_MODEL
    _, model = split_provider(full_model)

    completion = await llm.complete(
        [{"role": "user", "content": user_prompt}],
        output_type=ExplorationResponse,
        model=model,
        max_tokens=DEFAULT_EXPLORATION_MAX_TOKENS,
        system=system_prompt,
    )

    if input.log_messages and input.worktree_path:
        request_json = json.dumps(
            {
                "model": model,
                "max_tokens": DEFAULT_EXPLORATION_MAX_TOKENS,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            indent=2,
            default=str,
        )
        write_message_log(input.worktree_path, "explore-request", request_json)
        write_message_log(
            input.worktree_path, "explore-response", completion.output.model_dump_json(indent=2)
        )

    return completion.output


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


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
