"""LLM call activity for Forge.

Sends the assembled context to the LLM provider and extracts the structured response.

Design follows Function Core / Imperative Shell:
- Testable function: execute_llm_call (takes the LLM client as an argument)
- Imperative shell: the ``call_llm`` bound method on ``LlmActivities``
  (forge.activities.roots), which delegates here with the composition-root
  client.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from forge.message_log import write_message_log
from forge.models import (
    AssembledContext,
    CapabilityTier,
    LLMCallResult,
    LLMResponse,
    ModelConfig,
    resolve_model,
)

if TYPE_CHECKING:
    from sax_platform.llm import AnthropicLLM

# Shadow fallback for a missing context.model_name — the workflow always sets
# it from CapabilityTier.GENERATION via ModelConfig, so this only fires if a
# caller forgets. Resolved through the registry (T3.2) rather than a
# hardcoded literal, so it tracks the tier's pinned default.
DEFAULT_MODEL = resolve_model(CapabilityTier.GENERATION, ModelConfig())
DEFAULT_MAX_TOKENS = 4096

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


async def execute_llm_call(
    context: AssembledContext,
    llm: AnthropicLLM,
) -> LLMCallResult:
    """Call the LLM and extract structured results.

    Separated from the imperative shell so tests can inject a mock client.
    """
    from sax_platform.llm.tiers import split_provider

    full_model = context.model_name or DEFAULT_MODEL
    _, model = split_provider(full_model)
    start = time.monotonic()

    completion = await llm.complete(
        [{"role": "user", "content": context.user_prompt}],
        output_type=LLMResponse,
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS,
        system=context.system_prompt,
    )

    if context.log_messages and context.worktree_path:
        request_json = json.dumps(
            {
                "model": model,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "system": context.system_prompt,
                "messages": [{"role": "user", "content": context.user_prompt}],
            },
            indent=2,
            default=str,
        )
        write_message_log(context.worktree_path, "request", request_json)
        write_message_log(
            context.worktree_path, "response", completion.output.model_dump_json(indent=2)
        )

    elapsed_ms = (time.monotonic() - start) * 1000

    return LLMCallResult(
        task_id=context.task_id,
        response=completion.output,
        model_name=completion.model,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        latency_ms=elapsed_ms,
        cache_creation_input_tokens=completion.cache_creation_input_tokens,
        cache_read_input_tokens=completion.cache_read_input_tokens,
        stop_reason=completion.stop_reason,
    )
