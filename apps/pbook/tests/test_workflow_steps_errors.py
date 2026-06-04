"""Tests for the auth/config error classifier used to mark LLM-activity
failures non-retryable. Pure function — no fixtures beyond parametrize."""

from __future__ import annotations

import pytest

from pbook.workflow_steps._errors import is_nonretryable_auth_error


# Stand-ins for the SDK exception classes, matched by type name. Both the
# OpenAI and Anthropic SDKs expose classes with exactly these names.
class AuthenticationError(Exception):
    pass


class PermissionDeniedError(Exception):
    pass


class APITimeoutError(Exception):
    pass


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        # Anthropic client with a missing key raises a plain TypeError.
        (
            TypeError(
                "Could not resolve authentication method. Expected either "
                "api_key or auth_token to be set."
            ),
            True,
        ),
        # The embeddings shell raises RuntimeError when OPENAI_API_KEY is unset.
        (RuntimeError("OPENAI_API_KEY not set. Embedding operations require..."), True),
        # Invalid key → SDK AuthenticationError (matched by type name).
        (AuthenticationError("401 invalid x-api-key"), True),
        (PermissionDeniedError("403 forbidden"), True),
        # Transient/unrelated failures stay retryable.
        (APITimeoutError("request timed out"), False),
        (ConnectionError("connection reset by peer"), False),
        (Exception("500 internal server error"), False),
        (ValueError("llm_chat: empty model is not accepted"), False),
        (KeyError("ExtractionResult"), False),
    ],
)
def test_is_nonretryable_auth_error(exc: BaseException, expected: bool) -> None:
    assert is_nonretryable_auth_error(exc) is expected
