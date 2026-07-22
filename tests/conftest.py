"""Shared test fixtures for Forge."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

# Shared session Temporal fixture (T3.6, D93): the time-skipping,
# pydantic-aware environment lives in ``sax_platform.testing`` and is imported
# explicitly (no pytest11 plugin). ``env = temporal_env`` re-exports it under the
# name this suite requests; request only ``env`` (one name per session) so the
# time-skipping server starts once.
from sax_platform.testing import temporal_env

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy import Engine

env = temporal_env


def build_mock_llm(
    output=None,
    *,
    error: Exception | None = None,
    model: str = "test-model",
    stop_reason: str = "end_turn",
    input_tokens: int = 100,
    output_tokens: int = 200,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> MagicMock:
    """Build a stub AnthropicLLM whose ``complete`` returns a canned Completion.

    Pass ``output`` (a pydantic instance) for the success path, or ``error``
    (an LLMRefused/LLMTruncated/LLMSchemaMismatch instance) to make ``complete``
    raise it instead. Call args are recorded on the AsyncMock for assertions.
    """
    from unittest.mock import AsyncMock

    from sax_platform.llm import Completion

    llm = MagicMock()
    if error is not None:
        llm.complete = AsyncMock(side_effect=error)
        llm.complete_schema = AsyncMock(side_effect=error)
        llm.complete_text = AsyncMock(side_effect=error)
        return llm

    completion = Completion(
        output=output,
        model=model,
        stop_reason=stop_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        request_id=None,
    )
    llm.complete = AsyncMock(return_value=completion)
    llm.complete_schema = AsyncMock(return_value=completion)
    llm.complete_text = AsyncMock(return_value=completion)
    return llm


@pytest.fixture(autouse=True)
def forge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declare ``FORGE_ENV=test`` (and a coherent Temporal namespace) for the guard.

    Every forge CLI command and worker startup now resolves ``FORGE_ENV`` and
    refuses to run without it, and every connect additionally enforces
    env/namespace coherence: a ``test`` env must not use the ``default`` Temporal
    namespace. This one central fixture satisfies both — ``FORGE_ENV=test`` plus a
    non-default ``FORGE_TEMPORAL_NAMESPACE`` — for the whole suite; the guard's and
    coherence's own tests override them with ``monkeypatch.delenv``/``setenv``.
    """
    monkeypatch.setenv("FORGE_ENV", "test")
    monkeypatch.setenv("FORGE_TEMPORAL_NAMESPACE", "forge-test")


@pytest.fixture(autouse=True)
def forge_db_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point every test at an isolated SQLite store via ``FORGE_DB_URL``.

    The store is mandatory (no implicit default), so an unset URL raises. This
    autouse fixture gives each test its own throwaway database file; tests that
    manage their own path override ``FORGE_DB_URL``, and tests exercising the
    unset-URL hard error delete it explicitly.
    """
    db_path = tmp_path / "forge.db"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("FORGE_DB_URL", url)
    monkeypatch.delenv("FORGE_DB_PATH", raising=False)
    return url


@pytest.fixture(scope="session", autouse=True)
def _s3_backend():
    """Back OCR blob storage with an in-memory moto S3 bucket for the whole session.

    OCR blobs (file_content_blobs, ocr_images) now live in S3. Entering moto once
    per session (rather than per test) keeps overhead low; the bucket persists and
    tests use unique blob ids so accumulated objects don't collide. Fake AWS creds
    prevent any accidental real call.
    """
    import os

    for key, value in (
        ("AWS_ACCESS_KEY_ID", "testing"),
        ("AWS_SECRET_ACCESS_KEY", "testing"),
        ("AWS_SESSION_TOKEN", "testing"),
        ("AWS_DEFAULT_REGION", "us-east-1"),
    ):
        os.environ.setdefault(key, value)

    import boto3
    from moto import mock_aws

    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="forge-test-ocr")
        yield "forge-test-ocr"


@pytest.fixture(autouse=True)
def forge_ocr_s3(monkeypatch: pytest.MonkeyPatch, _s3_backend: str) -> str:
    """Point every test at the mocked OCR blob bucket via ``FORGE_OCR_S3_BUCKET``.

    Tests exercising the unset-bucket failure path delete the env var explicitly.
    """
    monkeypatch.setenv("FORGE_OCR_S3_BUCKET", _s3_backend)
    monkeypatch.delenv("FORGE_OCR_S3_PREFIX", raising=False)
    return _s3_backend


@pytest.fixture
def store_engine(forge_db_url: str) -> Iterator[Engine]:
    """A real store engine for the test's isolated SQLite DB, disposed on teardown.

    T3.6: ``StoreActivities``/``ContextActivities`` and the store-touching tests
    own their engine lifecycle explicitly (the former autouse engine-disposal
    monkeypatch of ``sa.create_engine`` is gone). Construct the engine from the
    autouse ``forge_db_url`` fixture's URL so it points at the same throwaway
    database the rest of the test uses.
    """
    from sax_platform.db import get_store_engine

    engine = get_store_engine(forge_db_url)
    try:
        yield engine
    finally:
        engine.dispose()


_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUFF_CONFIG_SRC = _REPO_ROOT / "tool-config" / "ruff.toml"


@pytest.fixture
def ruff_config(tmp_path: Path) -> Path:
    """Copy the project ruff config into tmp_path so ruff commands find it."""
    dest = tmp_path / "tool-config"
    dest.mkdir()
    shutil.copy(_RUFF_CONFIG_SRC, dest / "ruff.toml")
    return dest / "ruff.toml"


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with one initial commit.

    The repo has a committed ``README.md`` on the ``main`` branch so that
    worktrees can be created from it.

    Returns the path to the repository root.
    """
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@forge.test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Forge Test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    readme = tmp_path / "README.md"
    readme.write_text("# Test repo\n")

    # Copy ruff config so worktrees created from this repo have it.
    tool_config = tmp_path / "tool-config"
    tool_config.mkdir()
    shutil.copy(_RUFF_CONFIG_SRC, tool_config / "ruff.toml")

    subprocess.run(
        ["git", "add", "README.md", "tool-config/ruff.toml"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    return tmp_path
