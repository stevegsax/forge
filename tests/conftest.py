"""Shared test fixtures for Forge."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment


@pytest.fixture(autouse=True, scope="session")
def _register_output_types():
    """Register Forge's output types with the shared sax-llm registry.

    Required for any test that uses batch parsing or the output type registry.
    """
    from sax_llm import register_output_type
    from sax_llm.registry import reset_output_type_registry

    from forge.eval.models import JudgeVerdict
    from forge.models import (
        ConflictResolutionResponse,
        ExplorationResponse,
        ExtractionResult,
        LLMResponse,
        Plan,
        SanityCheckResponse,
    )

    reset_output_type_registry()
    register_output_type("LLMResponse", LLMResponse)
    register_output_type("Plan", Plan)
    register_output_type("ExplorationResponse", ExplorationResponse)
    register_output_type("SanityCheckResponse", SanityCheckResponse)
    register_output_type("ConflictResolutionResponse", ConflictResolutionResponse)
    register_output_type("ExtractionResult", ExtractionResult)
    register_output_type("JudgeVerdict", JudgeVerdict)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def env() -> WorkflowEnvironment:
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.testing import WorkflowEnvironment

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env:
        yield env


def build_mock_message(
    tool_name: str,
    tool_input: dict,
    *,
    input_tokens: int = 100,
    output_tokens: int = 200,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> MagicMock:
    """Build a mock Anthropic Message with a tool_use content block.

    Returns a MagicMock that mimics anthropic.types.Message structure:
    - message.content = [tool_use_block]
    - message.usage.input_tokens, .output_tokens, etc.
    """
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = tool_name
    tool_block.input = tool_input

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = cache_creation_input_tokens
    usage.cache_read_input_tokens = cache_read_input_tokens

    message = MagicMock()
    message.content = [tool_block]
    message.usage = usage
    return message


def build_mock_provider(
    tool_input: dict | None = None,
    *,
    text_content: str | None = None,
    model_name: str = "test-model",
    input_tokens: int = 100,
    output_tokens: int = 200,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    raw_response_json: str = "{}",
) -> MagicMock:
    """Build a mock LLMProvider that returns a ProviderResponse.

    Returns a MagicMock implementing the LLMProvider protocol with
    build_request_params and call methods pre-configured.
    """
    from sax_llm.models import ProviderResponse

    response = ProviderResponse(
        tool_input=tool_input or {},
        text_content=text_content,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        raw_response_json=raw_response_json,
    )

    from unittest.mock import AsyncMock

    provider = MagicMock()
    provider.build_request_params = MagicMock(return_value={"model": model_name})
    provider.call = AsyncMock(return_value=response)
    return provider


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
def _reset_mistral_ocr_cache():
    """Clear the shared lazily-cached MistralOcr between tests.

    forge.activities._mistral caches one MistralOcr/client pair at module
    scope (2026-07 Phase 3 code review, item 5a) so real workers reuse it
    across poll cycles. Tests that patch ``sax_platform.ocr.MistralOcr`` /
    ``make_mistral_client`` need a clean cache each time, or a stale instance
    from an earlier test would shadow the freshly patched constructor.
    """
    from forge.activities._mistral import reset_mistral_ocr_cache

    reset_mistral_ocr_cache()
    yield
    reset_mistral_ocr_cache()


@pytest.fixture(autouse=True)
def forge_ocr_s3(monkeypatch: pytest.MonkeyPatch, _s3_backend: str) -> str:
    """Point every test at the mocked OCR blob bucket via ``FORGE_OCR_S3_BUCKET``.

    Tests exercising the unset-bucket failure path delete the env var explicitly.
    """
    monkeypatch.setenv("FORGE_OCR_S3_BUCKET", _s3_backend)
    monkeypatch.delenv("FORGE_OCR_S3_PREFIX", raising=False)
    return _s3_backend


@pytest.fixture(autouse=True)
def dispose_store_engines(monkeypatch: pytest.MonkeyPatch):
    """Dispose SQLAlchemy engines created via forge.store.get_engine after each test."""
    import forge.store as store_module

    original_create_engine = store_module.sa.create_engine
    created_engines = []

    def tracking_create_engine(*args, **kwargs):
        engine = original_create_engine(*args, **kwargs)
        created_engines.append(engine)
        return engine

    monkeypatch.setattr(store_module.sa, "create_engine", tracking_create_engine)

    yield

    for engine in created_engines:
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
