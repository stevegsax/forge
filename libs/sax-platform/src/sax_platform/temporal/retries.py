"""Retry presets and error classification shared across platform/app workflows.

Pure (no I/O, no SDK imports): this module is imported inside Temporal
workflow bodies via ``workflow.unsafe.imports_passed_through()`` and inside
activities that need to re-raise a classified error, so it must stay cheap
and sandbox-safe. In particular it must NOT import ``anthropic`` — LLM error
types are matched by class *name* only (``classify_llm_error``), the same
technique pbook's ``workflow_steps/_errors.py`` uses so that module needn't
import either the OpenAI or Anthropic SDK.

Preset provenance
------------------
``LLM_RETRY``
    Ported verbatim from ``forge.workflows._LLM_RETRY`` (``src/forge/workflows.py``).
``IO_RETRY``
    Ported from the shape used by forge's three ``_LOCAL_RETRY`` copies
    (``RetryPolicy(maximum_attempts=2)``), consolidated into one shared name.
``DB_RETRY``
    New in T3.4 — no single prior copy to port verbatim. A conservative
    transient-DB-error preset: a handful of attempts with short backoff,
    non-retryable on the error types that mean "this will never succeed"
    (bad config, bad values) rather than "the database hiccuped." Treat this
    as the canonical DB retry preset going forward.
``PERSIST_RETRY``
    Re-exported from :mod:`sax_platform.contracts.persist` (its public home,
    since ``persist_block`` owns it) purely for discoverability alongside the
    other presets in this module — the retry policy itself is unchanged.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Final

from temporalio.common import RetryPolicy

from sax_platform.contracts.persist import PERSIST_RETRY

__all__ = [
    "DB_RETRY",
    "IO_RETRY",
    "LLM_RETRY",
    "PERSIST_RETRY",
    "classify_llm_error",
]

# ---------------------------------------------------------------------------
# Retry presets
# ---------------------------------------------------------------------------

LLM_RETRY: Final = RetryPolicy(
    maximum_attempts=3,
    non_retryable_error_types=[
        "BadRequestError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
    ],
)

IO_RETRY: Final = RetryPolicy(maximum_attempts=2)

# Conservative transient-DB-error preset: short bounded backoff over a few
# attempts, non-retryable on the error types that indicate a permanent
# misconfiguration or a bad value rather than a transient connection blip
# (those will never clear on retry, so failing fast beats burning the
# attempt budget).
DB_RETRY: Final = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=10),
    maximum_attempts=5,
    non_retryable_error_types=["ValueError", "ConfigError"],
)

# ---------------------------------------------------------------------------
# LLM error classification
# ---------------------------------------------------------------------------

# Class names that mark a deterministic, non-retryable LLM-call failure.
# Seeded from forge's LLM_RETRY.non_retryable_error_types (SDK HTTP-status
# exception names) plus pbook's _AUTH_ERROR_TYPE_NAMES
# (apps/pbook/src/pbook/workflow_steps/_errors.py) — matched by name only so
# this module never imports the anthropic (or openai) SDK.
_NON_RETRYABLE_ERROR_TYPE_NAMES: Final = frozenset(
    {
        "BadRequestError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
    }
)

# Message substrings that mark a missing/unresolved key for errors that do
# NOT carry a distinctive type — e.g. the Anthropic client's plain
# ``TypeError`` ("Could not resolve authentication method..."). Ported
# verbatim from pbook's _AUTH_MESSAGE_MARKERS.
_NON_RETRYABLE_MESSAGE_MARKERS: Final = (
    "could not resolve authentication method",
    "openai_api_key",
    "anthropic_api_key",
    "api_key",
    "auth_token",
)


def classify_llm_error(exc: BaseException) -> bool:
    """True when ``exc`` is a deterministic, non-retryable LLM-call failure.

    Matches ``type(exc).__name__`` against a frozenset of known permanent
    failure types first (bad request, auth, permission, not-found), then
    falls back to a lowercase message-marker scan for errors that carry a
    generic type but a distinctive message (e.g. a missing API key raised as
    a plain ``TypeError``). Such failures will never clear on retry — the
    caller should re-raise as non-retryable rather than exhaust a retry
    budget against a fault that can't resolve itself.
    """
    if type(exc).__name__ in _NON_RETRYABLE_ERROR_TYPE_NAMES:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _NON_RETRYABLE_MESSAGE_MARKERS)
