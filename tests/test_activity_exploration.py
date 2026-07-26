"""Tests for forge.activities.exploration — exploration activities (Phase 7)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from sax_platform.llm import LLMRefused, LLMTruncated, Telemetry

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
    TaskDomain,
)
from tests.conftest import build_mock_llm

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _telemetry(stop_reason: str = "end_turn") -> Telemetry:
    return Telemetry(
        model="test-model",
        stop_reason=stop_reason,
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        request_id=None,
    )


def _mock_tracer() -> MagicMock:
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span
    return mock_tracer


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
            task_id=_make_task().task_id,
            task_description=_make_task().description,
            target_files=_make_task().target_files,
            context_files=_make_task().context_files,
            context_config=_make_task().context,
            available_providers=_make_providers(),
            round_number=1,
            max_rounds=5,
        )
        system, _user = build_exploration_prompt(input)
        assert "Fix the failing test" in system
        assert "Round 1 of 5" in system

    def test_includes_target_files(self) -> None:
        input = ExplorationInput(
            task_id=_make_task().task_id,
            task_description=_make_task().description,
            target_files=_make_task().target_files,
            context_files=_make_task().context_files,
            context_config=_make_task().context,
            available_providers=_make_providers(),
            round_number=1,
            max_rounds=5,
        )
        system, _ = build_exploration_prompt(input)
        assert "tests/test_example.py" in system

    def test_includes_providers(self) -> None:
        input = ExplorationInput(
            task_id=_make_task().task_id,
            task_description=_make_task().description,
            target_files=_make_task().target_files,
            context_files=_make_task().context_files,
            context_config=_make_task().context,
            available_providers=_make_providers(),
            round_number=1,
            max_rounds=5,
        )
        system, _ = build_exploration_prompt(input)
        assert "read_file" in system
        assert "search_code" in system

    def test_includes_accumulated_context(self) -> None:
        input = ExplorationInput(
            task_id=_make_task().task_id,
            task_description=_make_task().description,
            target_files=_make_task().target_files,
            context_files=_make_task().context_files,
            context_config=_make_task().context,
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
            task_id=_make_task().task_id,
            task_description=_make_task().description,
            target_files=_make_task().target_files,
            context_files=_make_task().context_files,
            context_config=_make_task().context,
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
            task_id=_make_task().task_id,
            task_description=_make_task().description,
            target_files=_make_task().target_files,
            context_files=_make_task().context_files,
            context_config=_make_task().context,
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
    def _make_input(self) -> ExplorationInput:
        return ExplorationInput(
            task_id=_make_task().task_id,
            task_description=_make_task().description,
            target_files=_make_task().target_files,
            context_files=_make_task().context_files,
            context_config=_make_task().context,
            available_providers=_make_providers(),
            round_number=1,
            max_rounds=5,
        )

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
        llm = build_mock_llm(output=response)

        result = await execute_exploration_call(self._make_input(), llm)

        assert len(result.response.requests) == 1
        assert result.response.requests[0].provider == "read_file"

    @pytest.mark.asyncio
    async def test_empty_requests_signals_ready(self) -> None:
        response = ExplorationResponse(requests=[])
        llm = build_mock_llm(output=response)

        result = await execute_exploration_call(self._make_input(), llm)

        assert result.response.requests == []

    @pytest.mark.asyncio
    async def test_result_carries_spend_and_the_prompts_it_built(self) -> None:
        """T5.3: the counts that used to die in a trace span come home, with the
        prompts the activity assembled — an interaction row needs both."""
        llm = build_mock_llm(
            output=ExplorationResponse(requests=[]),
            model="claude-haiku-4-5",
            input_tokens=321,
            output_tokens=21,
            cache_creation_input_tokens=11,
            cache_read_input_tokens=13,
        )

        result = await execute_exploration_call(self._make_input(), llm)

        assert result.task_id == "explore-task"
        assert result.model_name == "claude-haiku-4-5"
        assert (result.input_tokens, result.output_tokens) == (321, 21)
        assert (result.cache_creation_input_tokens, result.cache_read_input_tokens) == (11, 13)
        assert result.stop_reason == "end_turn"
        assert result.latency_ms >= 0.0
        expected_system, expected_user = build_exploration_prompt(self._make_input())
        assert result.system_prompt == expected_system
        assert result.user_prompt == expected_user

    @pytest.mark.asyncio
    async def test_calls_llm_complete_with_expected_kwargs(self) -> None:
        llm = build_mock_llm(output=ExplorationResponse(requests=[]))

        await execute_exploration_call(self._make_input(), llm)

        llm.complete.assert_awaited_once()
        call = llm.complete.await_args
        assert call.kwargs["output_type"] is ExplorationResponse
        # No model_name -> the CLASSIFICATION-tier default, provider stripped.
        assert call.kwargs["model"] == "claude-haiku-4-5"
        assert call.kwargs["max_tokens"] == 4096
        assert isinstance(call.kwargs["system"], str)
        # Exploration attaches no thinking policy (matches pre-migration behavior).
        assert call.kwargs.get("thinking") is None

    @pytest.mark.asyncio
    async def test_refusal_propagates(self) -> None:
        llm = build_mock_llm(
            error=LLMRefused(category=None, telemetry=_telemetry(stop_reason="refusal"))
        )

        with pytest.raises(LLMRefused):
            await execute_exploration_call(self._make_input(), llm)

    @pytest.mark.asyncio
    async def test_truncation_propagates(self) -> None:
        llm = build_mock_llm(
            error=LLMTruncated(
                partial_text="partial",
                max_tokens=4096,
                telemetry=_telemetry(stop_reason="max_tokens"),
            )
        )

        with pytest.raises(LLMTruncated):
            await execute_exploration_call(self._make_input(), llm)


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

    def test_path_traversal_request_returns_error(self, tmp_path: Path) -> None:
        worktree = tmp_path / "worktree"
        worktree.mkdir()
        (tmp_path / "outside.py").write_text("TOP SECRET")

        results = fulfill_requests(
            [{"provider": "read_file", "params": {"path": "../outside.py"}}],
            str(worktree),
            str(worktree),
        )

        assert len(results) == 1
        assert "Error" in results[0].content
        assert "TOP SECRET" not in results[0].content


# ---------------------------------------------------------------------------
# Project instructions in exploration
# ---------------------------------------------------------------------------


class TestBuildExplorationPromptProjectInstructions:
    def test_includes_project_instructions(self) -> None:
        input = ExplorationInput(
            task_id=_make_task().task_id,
            task_description=_make_task().description,
            target_files=_make_task().target_files,
            context_files=_make_task().context_files,
            context_config=_make_task().context,
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
            task_id=_make_task().task_id,
            task_description=_make_task().description,
            target_files=_make_task().target_files,
            context_files=_make_task().context_files,
            context_config=_make_task().context,
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
            task_id=_make_task().task_id,
            task_description=_make_task().description,
            target_files=_make_task().target_files,
            context_files=_make_task().context_files,
            context_config=_make_task().context,
            available_providers=_make_providers(),
            round_number=1,
            max_rounds=5,
        )
        system, _ = build_exploration_prompt(input, project_instructions="")
        assert "## Project Instructions" not in system


# ---------------------------------------------------------------------------
# Phase 11: model_name threading via call_exploration_llm activity
# ---------------------------------------------------------------------------


class TestCallExplorationLlmModelNameThreading:
    @pytest.mark.asyncio
    async def test_threads_model_name_to_client(self) -> None:
        from forge.activities.roots import LlmActivities

        llm = build_mock_llm(
            output=ExplorationResponse(requests=[]),
            model="custom-explore",
        )

        with patch("forge.activities.roots.get_tracer", return_value=_mock_tracer()):
            input_data = ExplorationInput(
                task_id=_make_task().task_id,
                task_description=_make_task().description,
                target_files=_make_task().target_files,
                context_files=_make_task().context_files,
                context_config=_make_task().context,
                available_providers=_make_providers(),
                round_number=1,
                max_rounds=5,
                model_name="custom-explore",
            )
            await LlmActivities(llm).call_exploration_llm(input_data)

        assert llm.complete.await_args.kwargs["model"] == "custom-explore"

    @pytest.mark.asyncio
    async def test_uses_default_when_model_name_empty(self) -> None:
        from forge.activities.exploration import DEFAULT_EXPLORATION_MODEL
        from forge.activities.roots import LlmActivities

        llm = build_mock_llm(output=ExplorationResponse(requests=[]))

        with patch("forge.activities.roots.get_tracer", return_value=_mock_tracer()):
            input_data = ExplorationInput(
                task_id=_make_task().task_id,
                task_description=_make_task().description,
                target_files=_make_task().target_files,
                context_files=_make_task().context_files,
                context_config=_make_task().context,
                available_providers=_make_providers(),
                round_number=1,
                max_rounds=5,
            )
            await LlmActivities(llm).call_exploration_llm(input_data)

        _, default_model = DEFAULT_EXPLORATION_MODEL.split(":", 1)
        assert llm.complete.await_args.kwargs["model"] == default_model


# ---------------------------------------------------------------------------
# Domain-aware exploration prompts
# ---------------------------------------------------------------------------


class TestBuildExplorationPromptDomain:
    def test_research_domain_uses_research_nouns(self) -> None:
        task = TaskDefinition(
            task_id="t1",
            description="Research topic.",
            target_files=["report.md"],
            domain=TaskDomain.RESEARCH,
        )
        input_data = ExplorationInput(
            task_id=task.task_id,
            task_description=task.description,
            target_files=task.target_files,
            context_files=task.context_files,
            context_config=task.context,
            available_providers=_make_providers(),
            round_number=1,
            max_rounds=5,
            domain=task.domain,
        )
        system, user = build_exploration_prompt(input_data)
        assert "research task" in system
        assert "report writing" in system
        assert "report writing" in user

    def test_code_generation_domain_preserves_current(self) -> None:
        input_data = ExplorationInput(
            task_id=_make_task().task_id,
            task_description=_make_task().description,
            target_files=_make_task().target_files,
            context_files=_make_task().context_files,
            context_config=_make_task().context,
            available_providers=_make_providers(),
            round_number=1,
            max_rounds=5,
        )
        system, user = build_exploration_prompt(input_data)
        assert "coding task" in system
        assert "code generation" in system
        assert "code generation" in user
