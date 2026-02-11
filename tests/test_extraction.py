"""Tests for forge.activities.extraction — knowledge extraction activities."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge.activities.extraction import (
    build_extraction_system_prompt,
    build_extraction_user_prompt,
    execute_extraction_call,
    infer_tags_from_task,
)
from forge.models import (
    ExtractionInput,
    ExtractionResult,
    PlaybookEntry,
)
from tests.conftest import build_mock_message

# ---------------------------------------------------------------------------
# build_extraction_system_prompt
# ---------------------------------------------------------------------------


class TestBuildExtractionSystemPrompt:
    def test_includes_task_data(self) -> None:
        runs = [
            {
                "task_id": "t1",
                "workflow_id": "wf-1",
                "status": "success",
                "result": {},
            },
        ]
        prompt = build_extraction_system_prompt(runs)
        assert "t1" in prompt
        assert "wf-1" in prompt
        assert "success" in prompt

    def test_includes_errors(self) -> None:
        runs = [
            {
                "task_id": "t2",
                "workflow_id": "wf-2",
                "status": "failure_terminal",
                "result": {"error": "ruff_lint found errors"},
            },
        ]
        prompt = build_extraction_system_prompt(runs)
        assert "ruff_lint found errors" in prompt

    def test_includes_step_results(self) -> None:
        runs = [
            {
                "task_id": "t3",
                "workflow_id": "wf-3",
                "status": "success",
                "result": {
                    "step_results": [
                        {
                            "step_id": "s1",
                            "status": "success",
                            "validation_results": [
                                {
                                    "check_name": "ruff_lint",
                                    "passed": False,
                                    "summary": "lint failed",
                                }
                            ],
                        }
                    ]
                },
            },
        ]
        prompt = build_extraction_system_prompt(runs)
        assert "s1" in prompt
        assert "lint failed" in prompt

    def test_includes_output_files(self) -> None:
        runs = [
            {
                "task_id": "t4",
                "workflow_id": "wf-4",
                "status": "success",
                "result": {"output_files": {"a.py": "...", "b.py": "..."}},
            },
        ]
        prompt = build_extraction_system_prompt(runs)
        assert "a.py" in prompt
        assert "b.py" in prompt

    def test_handles_string_result(self) -> None:
        """result can be a raw JSON string instead of a dict."""
        runs = [
            {
                "task_id": "t5",
                "workflow_id": "wf-5",
                "status": "success",
                "result": '{"error": "test error"}',
            },
        ]
        prompt = build_extraction_system_prompt(runs)
        assert "test error" in prompt


# ---------------------------------------------------------------------------
# build_extraction_user_prompt
# ---------------------------------------------------------------------------


class TestBuildExtractionUserPrompt:
    def test_is_non_empty(self) -> None:
        prompt = build_extraction_user_prompt()
        assert len(prompt) > 0
        assert "Extract" in prompt


# ---------------------------------------------------------------------------
# infer_tags_from_task
# ---------------------------------------------------------------------------


class TestInferTagsFromTask:
    def test_python_files(self) -> None:
        tags = infer_tags_from_task("t1", "Create a module", ["src/a.py", "src/b.py"])
        assert "python" in tags

    def test_typescript_files(self) -> None:
        tags = infer_tags_from_task("t1", "Create component", ["app.tsx"])
        assert "typescript" in tags

    def test_description_keywords(self) -> None:
        tags = infer_tags_from_task("t1", "Refactor the API client", [])
        assert "refactoring" in tags
        assert "api" in tags

    def test_test_keyword(self) -> None:
        tags = infer_tags_from_task("t1", "Write tests for auth", ["tests/test_auth.py"])
        assert "test-writing" in tags
        assert "python" in tags

    def test_bug_fix_keyword(self) -> None:
        tags = infer_tags_from_task("t1", "Fix the login bug", [])
        assert "bug-fix" in tags

    def test_default_tag(self) -> None:
        tags = infer_tags_from_task("t1", "Create a module", [])
        assert tags == ["code-generation"]

    def test_sorted_and_deduplicated(self) -> None:
        tags = infer_tags_from_task("t1", "Fix the database bug", ["db.py"])
        assert tags == sorted(set(tags))


# ---------------------------------------------------------------------------
# execute_extraction_call
# ---------------------------------------------------------------------------


class TestExecuteExtractionCall:
    def _make_input(self) -> ExtractionInput:
        return ExtractionInput(
            system_prompt="Extract lessons.",
            user_prompt="Do it.",
            source_workflow_ids=["wf-1", "wf-2"],
        )

    def _make_mock_client(self) -> MagicMock:
        mock_output = ExtractionResult(
            entries=[
                PlaybookEntry(
                    title="Test lesson",
                    content="Always include type stubs.",
                    tags=["python", "test-writing"],
                    source_task_id="t1",
                    source_workflow_id="wf-1",
                ),
            ],
            summary="Extracted 1 lesson.",
        )
        mock_message = build_mock_message(
            tool_name="extraction_result",
            tool_input=mock_output.model_dump(),
            input_tokens=500,
            output_tokens=200,
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)
        return mock_client

    @pytest.mark.asyncio
    async def test_returns_extraction_call_result(self) -> None:
        input_data = self._make_input()
        client = self._make_mock_client()

        result = await execute_extraction_call(input_data, client)

        assert len(result.result.entries) == 1
        assert result.result.entries[0].title == "Test lesson"
        assert result.source_workflow_ids == ["wf-1", "wf-2"]

    @pytest.mark.asyncio
    async def test_extracts_usage(self) -> None:
        input_data = self._make_input()
        client = self._make_mock_client()

        result = await execute_extraction_call(input_data, client)

        assert result.input_tokens == 500
        assert result.output_tokens == 200

    @pytest.mark.asyncio
    async def test_latency_is_positive(self) -> None:
        input_data = self._make_input()
        client = self._make_mock_client()

        result = await execute_extraction_call(input_data, client)

        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_populates_empty_workflow_id(self) -> None:
        """If LLM doesn't populate source_workflow_id, uses first source."""
        input_data = self._make_input()
        mock_output = ExtractionResult(
            entries=[
                PlaybookEntry(
                    title="No wf",
                    content="Content.",
                    tags=["python"],
                    source_task_id="t1",
                    source_workflow_id="",
                ),
            ],
            summary="Test.",
        )
        mock_message = build_mock_message(
            tool_name="extraction_result",
            tool_input=mock_output.model_dump(),
            input_tokens=100,
            output_tokens=50,
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        result = await execute_extraction_call(input_data, mock_client)

        assert result.result.entries[0].source_workflow_id == "wf-1"


# ---------------------------------------------------------------------------
# Phase 11: model_name threading via call_extraction_llm activity
# ---------------------------------------------------------------------------


class TestCallExtractionLlmModelNameThreading:
    @pytest.mark.asyncio
    async def test_threads_model_name_to_client(self) -> None:
        from forge.activities.extraction import call_extraction_llm

        mock_output = ExtractionResult(
            entries=[],
            summary="Nothing to extract.",
        )
        mock_message = build_mock_message(
            tool_name="extraction_result",
            tool_input=mock_output.model_dump(),
            input_tokens=100,
            output_tokens=50,
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        with (
            patch("forge.llm_client.get_anthropic_client", return_value=mock_client),
            patch("forge.activities.extraction._persist_extraction_interaction"),
            patch("forge.tracing.get_tracer") as mock_get_tracer,
        ):
            mock_span = MagicMock()
            mock_span.__enter__ = MagicMock(return_value=mock_span)
            mock_span.__exit__ = MagicMock(return_value=False)
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value = mock_span
            mock_get_tracer.return_value = mock_tracer

            input_data = ExtractionInput(
                system_prompt="sys",
                user_prompt="usr",
                source_workflow_ids=["wf-1"],
                model_name="custom-extract",
            )
            await call_extraction_llm(input_data)

            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["model"] == "custom-extract"

    @pytest.mark.asyncio
    async def test_uses_default_when_model_name_empty(self) -> None:
        from forge.activities.extraction import call_extraction_llm
        from forge.activities.llm import DEFAULT_MODEL

        mock_output = ExtractionResult(
            entries=[],
            summary="Nothing.",
        )
        mock_message = build_mock_message(
            tool_name="extraction_result",
            tool_input=mock_output.model_dump(),
            input_tokens=100,
            output_tokens=50,
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        with (
            patch("forge.llm_client.get_anthropic_client", return_value=mock_client),
            patch("forge.activities.extraction._persist_extraction_interaction"),
            patch("forge.tracing.get_tracer") as mock_get_tracer,
        ):
            mock_span = MagicMock()
            mock_span.__enter__ = MagicMock(return_value=mock_span)
            mock_span.__exit__ = MagicMock(return_value=False)
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value = mock_span
            mock_get_tracer.return_value = mock_tracer

            input_data = ExtractionInput(
                system_prompt="sys",
                user_prompt="usr",
                source_workflow_ids=["wf-1"],
            )
            await call_extraction_llm(input_data)

            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["model"] == DEFAULT_MODEL
