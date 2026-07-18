"""Tests for the shared retry presets and LLM error classifier.

``sax_platform.temporal.retries`` is imported inside Temporal workflow and
activity sandboxes, so — beyond shape assertions — this file also verifies in
a fresh subprocess that importing it never pulls in ``anthropic`` (an SDK
import) or ``temporalio.worker`` (real worker machinery); this file's own
top-level imports below have already pulled worker/test machinery into this
process, so checking ``sys.modules`` in-process would prove nothing.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import timedelta

from sax_platform.temporal.retries import (
    DB_RETRY,
    IO_RETRY,
    LLM_RETRY,
    PERSIST_RETRY,
    classify_llm_error,
)


class TestPresetShapes:
    def test_llm_retry_shape_includes_raised_classification_outcomes(self) -> None:
        assert LLM_RETRY.maximum_attempts == 3
        assert LLM_RETRY.non_retryable_error_types == [
            "BadRequestError",
            "AuthenticationError",
            "PermissionDeniedError",
            "NotFoundError",
            # T3.5: a refusal and a truncation are deterministic, non-retryable
            # (Temporal types the ApplicationError by the raising class name).
            "LLMRefused",
            "LLMTruncated",
        ]

    def test_llm_retry_keeps_schema_mismatch_retryable(self) -> None:
        # A schema mismatch is a sampling accident a fresh call can fix, so it
        # must NOT be listed as non-retryable.
        assert LLM_RETRY.non_retryable_error_types is not None
        assert "LLMSchemaMismatch" not in LLM_RETRY.non_retryable_error_types

    def test_io_retry_matches_local_retry_shape(self) -> None:
        assert IO_RETRY.maximum_attempts == 2
        assert not IO_RETRY.non_retryable_error_types

    def test_db_retry_is_conservative_and_bounded(self) -> None:
        assert DB_RETRY.maximum_attempts == 5
        assert DB_RETRY.initial_interval == timedelta(seconds=1)
        assert DB_RETRY.backoff_coefficient == 2.0
        assert DB_RETRY.maximum_interval == timedelta(seconds=10)
        assert "ValueError" in DB_RETRY.non_retryable_error_types
        assert "ConfigError" in DB_RETRY.non_retryable_error_types

    def test_persist_retry_reexports_contracts_persist_policy(self) -> None:
        from sax_platform.contracts.persist import PERSIST_RETRY as CONTRACTS_PERSIST_RETRY

        assert PERSIST_RETRY is CONTRACTS_PERSIST_RETRY
        assert PERSIST_RETRY.maximum_attempts == 20
        assert PERSIST_RETRY.non_retryable_error_types == ["ValueError"]


class TestClassifyLlmError:
    def test_typed_bad_request_error_is_nonretryable(self) -> None:
        class BadRequestError(Exception):
            pass

        assert classify_llm_error(BadRequestError("bad")) is True

    def test_typed_authentication_error_is_nonretryable(self) -> None:
        class AuthenticationError(Exception):
            pass

        assert classify_llm_error(AuthenticationError("nope")) is True

    def test_typed_permission_denied_error_is_nonretryable(self) -> None:
        class PermissionDeniedError(Exception):
            pass

        assert classify_llm_error(PermissionDeniedError("no")) is True

    def test_typed_not_found_error_is_nonretryable(self) -> None:
        class NotFoundError(Exception):
            pass

        assert classify_llm_error(NotFoundError("missing")) is True

    def test_message_marker_fallback_for_untyped_auth_failure(self) -> None:
        # Mirrors the Anthropic client's plain TypeError on a missing key.
        assert classify_llm_error(TypeError("Could not resolve authentication method")) is True

    def test_message_marker_fallback_matches_api_key_substring(self) -> None:
        assert classify_llm_error(RuntimeError("ANTHROPIC_API_KEY not set")) is True

    def test_retryable_error_passes_through(self) -> None:
        assert classify_llm_error(TimeoutError("upstream timed out")) is False

    def test_generic_exception_with_unrelated_message_is_retryable(self) -> None:
        assert classify_llm_error(ConnectionError("connection reset")) is False


class TestSandboxSafety:
    """retries.py must stay importable from inside the Temporal sandbox: no
    anthropic (SDK), no temporalio.worker (real worker machinery)."""

    def test_import_does_not_pull_in_anthropic_or_temporalio_worker(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sax_platform.temporal.retries, sys; "
                "assert 'anthropic' not in sys.modules and 'temporalio.worker' not in sys.modules",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
