"""Generic structured-output chat activity.

This module provides the ``llm_chat`` activity, which any workflow can
call to make a structured-output LLM request. The activity is
intentionally agnostic about prompt construction and result parsing:
the workflow builds its prompts (typically by calling pure functions in
``pbook.prompts``), names the desired output type by string (resolved
via :mod:`pbook.workflow_steps.output_types`), and validates the
returned ``tool_input`` dict on its own side with
``OutputType.model_validate(...)``.

Why a string-keyed mapping rather than passing a class? Temporal
serializes activity inputs as JSON; a class reference can't cross that
boundary. The frozen mapping in :mod:`pbook.workflow_steps.output_types`
lets us recover the correct ``BaseModel`` subclass inside the activity to
pass as ``complete(output_type=...)``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, Field
from sax_platform.llm import LLMRefused, LLMTruncated
from sax_platform.llm.tiers import split_provider
from sax_platform.temporal.heartbeat import heartbeat_during
from temporalio import activity
from temporalio.exceptions import ApplicationError

from pbook.llm import get_provider
from pbook.workflow_steps._errors import is_nonretryable_auth_error
from pbook.workflow_steps.output_types import resolve_output_type

logger = logging.getLogger(__name__)


class LLMChatInput(BaseModel):
    """Input payload for the generic chat activity."""

    system_prompt: str
    user_prompt: str
    output_type_name: str = Field(
        description=(
            "Key into pbook.workflow_steps.output_types.OUTPUT_TYPES naming "
            "the desired structured-output class."
        ),
    )
    model: str = Field(
        description=(
            'Provider-qualified model id ("anthropic:claude-...") or '
            "bare model name. Empty string is rejected — the workflow "
            "is expected to resolve a model deliberately."
        ),
    )
    max_tokens: int = 4096


class LLMChatResult(BaseModel):
    """Telemetry-bearing result. ``tool_input`` is the raw structured-output
    dict; the workflow validates it against its own Pydantic class."""

    tool_input: dict[str, Any]
    model_name: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    latency_ms: float


@activity.defn
async def llm_chat(input: LLMChatInput) -> LLMChatResult:
    """Make a structured-output chat call against the registered provider.

    Heartbeats during the underlying network call so Temporal can detect
    a stalled worker. Telemetry (tokens, latency, cache hits) is
    returned to the caller for logging or ranking decisions.
    """
    if not input.model:
        msg = (
            "llm_chat: empty model is not accepted. The calling workflow "
            "should resolve a model via pbook.models.resolve_model() and "
            "pass it explicitly."
        )
        raise ValueError(msg)

    # Strip the `provider:` prefix if present. `resolve_model()` returns
    # the fully-qualified id (e.g. `anthropic:claude-haiku-4-5`), but the
    # client expects the bare model name. split_provider returns
    # (provider, model); we keep the model half.
    _, bare_model = split_provider(input.model)

    output_type = resolve_output_type(input.output_type_name)
    provider = get_provider()

    start = time.monotonic()
    try:
        async with heartbeat_during():
            completion = await provider.complete(
                [{"role": "user", "content": input.user_prompt}],
                output_type=output_type,
                model=bare_model,
                max_tokens=input.max_tokens,
                system=input.system_prompt,
            )
    except (LLMRefused, LLMTruncated) as exc:
        # A refusal or truncation is terminal for THIS request: retrying the
        # identical prompt reproduces it, so surface it as a non-retryable
        # ApplicationError instead of burning the bounded LLM retry budget
        # (which would otherwise leave the ingestion session stuck at
        # "running"). LLMSchemaMismatch is deliberately NOT caught here — a
        # malformed structured response can clear on retry, so it propagates
        # unchanged and stays retryable.
        category = getattr(exc, "category", None)
        raise ApplicationError(
            f"llm_chat: LLM {type(exc).__name__} "
            f"(stop_reason={exc.telemetry.stop_reason}, category={category}): {exc}",
            type=type(exc).__name__,
            non_retryable=True,
        ) from exc
    except Exception as exc:
        # A missing/invalid API key or unresolved auth method will never
        # succeed on retry — mark it non-retryable so the activity fails on
        # the first attempt instead of exhausting LLM_RETRY_POLICY's budget.
        # All other provider errors (timeouts, 429/5xx) propagate unchanged
        # and stay retryable.
        if is_nonretryable_auth_error(exc):
            raise ApplicationError(
                f"llm_chat: provider authentication/configuration error "
                f"({type(exc).__name__}): {exc}",
                type=type(exc).__name__,
                non_retryable=True,
            ) from exc
        raise
    latency_ms = (time.monotonic() - start) * 1000

    logger.info(
        "llm_chat: type=%s model=%s tokens=%d/%d latency=%.0fms",
        input.output_type_name,
        completion.model,
        completion.input_tokens,
        completion.output_tokens,
        latency_ms,
    )

    return LLMChatResult(
        # `tool_input` is a historical name from the forced-tool-use era; it
        # now carries the structured-outputs payload. Kept as-is because the
        # workflow side re-validates `chat_result.tool_input` into its own
        # Pydantic class — renaming would touch every workflow (out of scope).
        tool_input=completion.output.model_dump(),
        model_name=completion.model,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        cache_creation_input_tokens=completion.cache_creation_input_tokens,
        cache_read_input_tokens=completion.cache_read_input_tokens,
        latency_ms=latency_ms,
    )
