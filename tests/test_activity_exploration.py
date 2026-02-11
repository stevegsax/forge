"""Tests for forge.activities.exploration — exploration activities (Phase 7)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from forge.activities.exploration import (
    build_exploration_prompt,
    execute_exploration_call,
    fulfill_requests,
)
from forge.models import (
    ContextProviderSpec,
    ContextRequest,
    ContextResult,
    ExplorationInput,
    ExplorationResponse,
    TaskDefinition,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_task() -> TaskDefinition:
    return TaskDefinition(
        task_id="explore-task",
        description="Fix the failing test in test_example.py",
        target_files=["tests/test_example.py"],
    )


def _make_providers() -> list[ContextProviderSpec]:
    return [
        ContextProviderSpec(
            name="read_file",
            description="Read file contents.",
            parameters={"path": "File path."},
        ),
        ContextProviderSpec(
            name="search_code",
            description="Search for pattern.",
            parameters={"pattern": "Regex pattern."},
        ),
    ]


# ---------------------------------------------------------------------------
# build_exploration_prompt
# ---------------------------------------------------------------------------


class TestBuildExplorationPrompt:
    def test_includes_task_description(self) -> None:
        input = ExplorationInput(
            task=_make_task(),
            available_providers=_make_providers(),
            round_number=1,
            max_rounds=5,
        )
        system, _user = build_exploration_prompt(input)
        assert "Fix the failing test" in system
        assert "Round 1 of 5" in system

    def test_includes_target_files(self) -> None:
        input = ExplorationInput(
            task=_make_task(),
            available_providers=_make_providers(),
            round_number=1,
            max_rounds=5,
        )
        system, _ = build_exploration_prompt(input)
        assert "tests/test_example.py" in system

    def test_includes_providers(self) -> None:
        input = ExplorationInput(
            task=_make_task(),
            available_providers=_make_providers(),
            round_number=1,
            max_rounds=5,
        )
        system, _ = build_exploration_prompt(input)
        assert "read_file" in system
        assert "search_code" in system

    def test_includes_accumulated_context(self) -> None:
        input = ExplorationInput(
            task=_make_task(),
            available_providers=_make_providers(),
            accumulated_context=[
                ContextResult(
                    provider="read_file",
                    content="def test_example(): pass",
                    estimated_tokens=10,
                ),
            ],
            round_number=2,
            max_rounds=5,
        )
        system, _ = build_exploration_prompt(input)
        assert "Previously Retrieved Context" in system
        assert "def test_example" in system

    def test_truncates_long_context(self) -> None:
        long_content = "x" * 10000
        input = ExplorationInput(
            task=_make_task(),
            available_providers=_make_providers(),
            accumulated_context=[
                ContextResult(
                    provider="read_file",
                    content=long_content,
                    estimated_tokens=2500,
                ),
            ],
            round_number=2,
            max_rounds=5,
        )
        system, _ = build_exploration_prompt(input)
        assert "truncated" in system

    def test_user_prompt_non_empty(self) -> None:
        input = ExplorationInput(
            task=_make_task(),
            available_providers=_make_providers(),
            round_number=1,
            max_rounds=5,
        )
        _, user = build_exploration_prompt(input)
        assert len(user) > 0


# ---------------------------------------------------------------------------
# execute_exploration_call
# ---------------------------------------------------------------------------


class TestExecuteExplorationCall:
    @pytest.mark.asyncio
    async def test_returns_exploration_response(self) -> None:
        response = ExplorationResponse(
            requests=[
                ContextRequest(
                    provider="read_file",
                    params={"path": "foo.py"},
                    reasoning="Need to see the file.",
                ),
            ]
        )

        mock_result = MagicMock()
        mock_result.output = response

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        input = ExplorationInput(
            task=_make_task(),
            available_providers=_make_providers(),
            round_number=1,
            max_rounds=5,
        )

        result = await execute_exploration_call(input, mock_agent)

        assert len(result.requests) == 1
        assert result.requests[0].provider == "read_file"

    @pytest.mark.asyncio
    async def test_empty_requests_signals_ready(self) -> None:
        response = ExplorationResponse(requests=[])

        mock_result = MagicMock()
        mock_result.output = response

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        input = ExplorationInput(
            task=_make_task(),
            available_providers=_make_providers(),
            round_number=1,
            max_rounds=5,
        )

        result = await execute_exploration_call(input, mock_agent)

        assert result.requests == []


# ---------------------------------------------------------------------------
# fulfill_requests
# ---------------------------------------------------------------------------


class TestFulfillRequests:
    def test_dispatches_to_known_provider(self, tmp_path: Path) -> None:
        (tmp_path / "test.py").write_text("hello world")

        results = fulfill_requests(
            [{"provider": "read_file", "params": {"path": "test.py"}}],
            str(tmp_path),
            str(tmp_path),
        )

        assert len(results) == 1
        assert results[0].provider == "read_file"
        assert results[0].content == "hello world"
        assert results[0].estimated_tokens > 0

    def test_unknown_provider_returns_error(self, tmp_path: Path) -> None:
        results = fulfill_requests(
            [{"provider": "nonexistent_provider", "params": {}}],
            str(tmp_path),
            str(tmp_path),
        )

        assert len(results) == 1
        assert "Error" in results[0].content
        assert "Unknown provider" in results[0].content

    def test_multiple_requests(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("aaa")
        (tmp_path / "b.py").write_text("bbb")

        results = fulfill_requests(
            [
                {"provider": "read_file", "params": {"path": "a.py"}},
                {"provider": "read_file", "params": {"path": "b.py"}},
            ],
            str(tmp_path),
            str(tmp_path),
        )

        assert len(results) == 2
        assert results[0].content == "aaa"
        assert results[1].content == "bbb"

    def test_provider_failure_returns_error(self, tmp_path: Path) -> None:
        results = fulfill_requests(
            [{"provider": "read_file", "params": {"path": "nonexistent.py"}}],
            str(tmp_path),
            str(tmp_path),
        )

        assert len(results) == 1
        assert "Error" in results[0].content


# ---------------------------------------------------------------------------
# Project instructions in exploration
# ---------------------------------------------------------------------------


class TestBuildExplorationPromptProjectInstructions:
    def test_includes_project_instructions(self) -> None:
        input = ExplorationInput(
            task=_make_task(),
            available_providers=_make_providers(),
            round_number=1,
            max_rounds=5,
        )
        instructions = "## Project Instructions\n\nUse type hints."
        system, _ = build_exploration_prompt(input, project_instructions=instructions)
        assert "## Project Instructions" in system
        assert "Use type hints." in system

    def test_instructions_before_round_info(self) -> None:
        input = ExplorationInput(
            task=_make_task(),
            available_providers=_make_providers(),
            round_number=1,
            max_rounds=5,
        )
        instructions = "## Project Instructions\n\nUse type hints."
        system, _ = build_exploration_prompt(input, project_instructions=instructions)
        instr_pos = system.index("## Project Instructions")
        round_pos = system.index("## Round 1")
        assert instr_pos < round_pos

    def test_omits_when_empty(self) -> None:
        input = ExplorationInput(
            task=_make_task(),
            available_providers=_make_providers(),
            round_number=1,
            max_rounds=5,
        )
        system, _ = build_exploration_prompt(input, project_instructions="")
        assert "## Project Instructions" not in system
