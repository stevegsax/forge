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
    tool_input: dict,
    *,
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
    from forge.llm_providers.models import ProviderResponse

    response = ProviderResponse(
        tool_input=tool_input,
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
