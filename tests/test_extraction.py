"""Tests for forge.activities.extraction — knowledge extraction activities."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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

    def _make_mock_agent(self) -> MagicMock:
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
        mock_usage = MagicMock()
        mock_usage.input_tokens = 500
        mock_usage.output_tokens = 200

        mock_result = MagicMock()
        mock_result.output = mock_output
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent.model = "test-model"
        return mock_agent

    @pytest.mark.asyncio
    async def test_returns_extraction_call_result(self) -> None:
        input_data = self._make_input()
        agent = self._make_mock_agent()

        result = await execute_extraction_call(input_data, agent)

        assert result.model_name == "test-model"
        assert len(result.result.entries) == 1
        assert result.result.entries[0].title == "Test lesson"
        assert result.source_workflow_ids == ["wf-1", "wf-2"]

    @pytest.mark.asyncio
    async def test_extracts_usage(self) -> None:
        input_data = self._make_input()
        agent = self._make_mock_agent()

        result = await execute_extraction_call(input_data, agent)

        assert result.input_tokens == 500
        assert result.output_tokens == 200

    @pytest.mark.asyncio
    async def test_latency_is_positive(self) -> None:
        input_data = self._make_input()
        agent = self._make_mock_agent()

        result = await execute_extraction_call(input_data, agent)

        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_populates_empty_workflow_id(self) -> None:
        """If LLM doesn't populate source_workflow_id, uses first source."""
        input_data = self._make_input()
        agent = self._make_mock_agent()
        # Make the entry have empty source_workflow_id
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
        mock_result = MagicMock()
        mock_result.output = mock_output
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        mock_result.usage.return_value = mock_usage
        agent.run = AsyncMock(return_value=mock_result)

        result = await execute_extraction_call(input_data, agent)

        assert result.result.entries[0].source_workflow_id == "wf-1"
