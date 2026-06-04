"""Classify LLM/embedding provider exceptions for Temporal retry decisions.

Pure helper (no I/O): given an exception raised while calling an LLM or
embedding provider, decide whether it is a *deterministic* authentication
or configuration failure that can never succeed on retry — a missing or
invalid API key, or an unresolved authentication method.

The shell (:func:`pbook.workflow_steps.llm.llm_chat` /
:func:`pbook.workflow_steps.embeddings.llm_embed`) uses this to re-raise
such errors as a *non-retryable* Temporal ``ApplicationError`` so the
workflow fails fast instead of climbing the bounded retry ladder for
minutes against a fault that will never clear (a misconfigured worker
otherwise leaves the ingestion session stuck at ``running`` forever).
"""

from __future__ import annotations

from typing import Final

__all__ = ["is_nonretryable_auth_error"]

# Class names both the OpenAI and Anthropic SDKs use for permanent auth
# faults (e.g. a 401 from an invalid key). Matched by name so this pure
# module need not import either SDK.
_AUTH_ERROR_TYPE_NAMES: Final = frozenset(
    {"AuthenticationError", "PermissionDeniedError"},
)

# Message substrings that mark a missing/unresolved key for errors that do
# NOT carry a distinctive type: the Anthropic client raises a plain
# ``TypeError`` ("Could not resolve authentication method..."), and the
# embeddings shell raises ``RuntimeError`` ("OPENAI_API_KEY not set..."").
_AUTH_MESSAGE_MARKERS: Final = (
    "could not resolve authentication method",
    "openai_api_key",
    "anthropic_api_key",
    "api_key",
    "auth_token",
)


def is_nonretryable_auth_error(exc: BaseException) -> bool:
    """True if ``exc`` is a deterministic auth/config failure.

    Such failures will never clear on retry, so the caller should stop
    retrying immediately rather than exhaust the retry budget.
    """
    if type(exc).__name__ in _AUTH_ERROR_TYPE_NAMES:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _AUTH_MESSAGE_MARKERS)
