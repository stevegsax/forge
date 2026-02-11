"""Tests for forge.activities.conflict_resolution — pure functions and testable function."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.activities.conflict_resolution import (
    build_conflict_resolution_system_prompt,
    build_conflict_resolution_user_prompt,
    detect_file_conflicts,
    execute_conflict_resolution_call,
)
from forge.models import (
    ConflictResolutionCallInput,
    ConflictResolutionCallResult,
    ConflictResolutionResponse,
    FileConflict,
    FileConflictVersion,
    FileOutput,
    SubTaskResult,
    TransitionSignal,
)

# ---------------------------------------------------------------------------
# TestDetectFileConflicts
# ---------------------------------------------------------------------------


class TestDetectFileConflicts:
    """detect_file_conflicts separates non-conflicting files from conflicts."""

    def test_no_conflicts(self) -> None:
        results = [
            SubTaskResult(
                sub_task_id="st1",
                status=TransitionSignal.SUCCESS,
                output_files={"a.py": "# a"},
            ),
            SubTaskResult(
                sub_task_id="st2",
                status=TransitionSignal.SUCCESS,
                output_files={"b.py": "# b"},
            ),
        ]
        non_conflicting, conflicts = detect_file_conflicts(results)
        assert non_conflicting == {"a.py": "# a", "b.py": "# b"}
        assert conflicts == []

    def test_single_conflict(self) -> None:
        results = [
            SubTaskResult(
                sub_task_id="st1",
                status=TransitionSignal.SUCCESS,
                output_files={"shared.py": "# from st1"},
            ),
            SubTaskResult(
                sub_task_id="st2",
                status=TransitionSignal.SUCCESS,
                output_files={"shared.py": "# from st2"},
            ),
        ]
        non_conflicting, conflicts = detect_file_conflicts(results)
        assert non_conflicting == {}
        assert len(conflicts) == 1
        assert conflicts[0].file_path == "shared.py"
        assert len(conflicts[0].versions) == 2
        assert conflicts[0].versions[0].source_id == "st1"
        assert conflicts[0].versions[1].source_id == "st2"

    def test_multiple_conflicts(self) -> None:
        results = [
            SubTaskResult(
                sub_task_id="st1",
                status=TransitionSignal.SUCCESS,
                output_files={"a.py": "# a from st1", "b.py": "# b from st1"},
            ),
            SubTaskResult(
                sub_task_id="st2",
                status=TransitionSignal.SUCCESS,
                output_files={"a.py": "# a from st2", "b.py": "# b from st2"},
            ),
        ]
        non_conflicting, conflicts = detect_file_conflicts(results)
        assert non_conflicting == {}
        assert len(conflicts) == 2
        conflict_paths = {c.file_path for c in conflicts}
        assert conflict_paths == {"a.py", "b.py"}

    def test_mixed_conflict_and_non_conflict(self) -> None:
        results = [
            SubTaskResult(
                sub_task_id="st1",
                status=TransitionSignal.SUCCESS,
                output_files={"shared.py": "# from st1", "unique_a.py": "# a"},
            ),
            SubTaskResult(
                sub_task_id="st2",
                status=TransitionSignal.SUCCESS,
                output_files={"shared.py": "# from st2", "unique_b.py": "# b"},
            ),
        ]
        non_conflicting, conflicts = detect_file_conflicts(results)
        assert non_conflicting == {"unique_a.py": "# a", "unique_b.py": "# b"}
        assert len(conflicts) == 1
        assert conflicts[0].file_path == "shared.py"

    def test_three_way_conflict(self) -> None:
        results = [
            SubTaskResult(
                sub_task_id="st1",
                status=TransitionSignal.SUCCESS,
                output_files={"shared.py": "# from st1"},
            ),
            SubTaskResult(
                sub_task_id="st2",
                status=TransitionSignal.SUCCESS,
                output_files={"shared.py": "# from st2"},
            ),
            SubTaskResult(
                sub_task_id="st3",
                status=TransitionSignal.SUCCESS,
                output_files={"shared.py": "# from st3"},
            ),
        ]
        non_conflicting, conflicts = detect_file_conflicts(results)
        assert non_conflicting == {}
        assert len(conflicts) == 1
        assert len(conflicts[0].versions) == 3

    def test_failed_subtasks_excluded(self) -> None:
        results = [
            SubTaskResult(
                sub_task_id="st1",
                status=TransitionSignal.SUCCESS,
                output_files={"shared.py": "# from st1"},
            ),
            SubTaskResult(
                sub_task_id="st2",
                status=TransitionSignal.FAILURE_TERMINAL,
                output_files={"shared.py": "# from st2"},
                error="validation failed",
            ),
        ]
        non_conflicting, conflicts = detect_file_conflicts(results)
        assert non_conflicting == {"shared.py": "# from st1"}
        assert conflicts == []

    def test_original_content_read_from_worktree(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            wt = Path(tmpdir)
            (wt / "existing.py").write_text("# original content")

            results = [
                SubTaskResult(
                    sub_task_id="st1",
                    status=TransitionSignal.SUCCESS,
                    output_files={"existing.py": "# from st1"},
                ),
                SubTaskResult(
                    sub_task_id="st2",
                    status=TransitionSignal.SUCCESS,
                    output_files={"existing.py": "# from st2"},
                ),
            ]
            _, conflicts = detect_file_conflicts(results, str(wt))
            assert len(conflicts) == 1
            assert conflicts[0].original_content == "# original content"

    def test_new_file_has_none_original(self) -> None:
        results = [
            SubTaskResult(
                sub_task_id="st1",
                status=TransitionSignal.SUCCESS,
                output_files={"new.py": "# from st1"},
            ),
            SubTaskResult(
                sub_task_id="st2",
                status=TransitionSignal.SUCCESS,
                output_files={"new.py": "# from st2"},
            ),
        ]
        _, conflicts = detect_file_conflicts(results)
        assert len(conflicts) == 1
        assert conflicts[0].original_content is None

    def test_original_content_none_when_no_worktree(self) -> None:
        results = [
            SubTaskResult(
                sub_task_id="st1",
                status=TransitionSignal.SUCCESS,
                output_files={"a.py": "# from st1"},
            ),
            SubTaskResult(
                sub_task_id="st2",
                status=TransitionSignal.SUCCESS,
                output_files={"a.py": "# from st2"},
            ),
        ]
        _, conflicts = detect_file_conflicts(results, None)
        assert len(conflicts) == 1
        assert conflicts[0].original_content is None


# ---------------------------------------------------------------------------
# TestBuildConflictResolutionSystemPrompt
# ---------------------------------------------------------------------------


class TestBuildConflictResolutionSystemPrompt:
    """build_conflict_resolution_system_prompt includes all relevant context."""

    def _make_conflict(
        self,
        file_path: str = "shared.py",
        original: str | None = None,
    ) -> FileConflict:
        return FileConflict(
            file_path=file_path,
            versions=[
                FileConflictVersion(source_id="st1", content="# from st1\ndef foo(): pass"),
                FileConflictVersion(source_id="st2", content="# from st2\ndef bar(): pass"),
            ],
            original_content=original,
        )

    def test_contains_all_versions(self) -> None:
        conflict = self._make_conflict()
        prompt = build_conflict_resolution_system_prompt(
            "Build API", "Create routes", [conflict], []
        )
        assert "# from st1" in prompt
        assert "# from st2" in prompt
        assert "st1" in prompt
        assert "st2" in prompt

    def test_contains_original_content(self) -> None:
        conflict = self._make_conflict(original="# original\nclass Base: pass")
        prompt = build_conflict_resolution_system_prompt(
            "Build API", "Create routes", [conflict], []
        )
        assert "# original" in prompt
        assert "class Base: pass" in prompt

    def test_new_file_marker(self) -> None:
        conflict = self._make_conflict(original=None)
        prompt = build_conflict_resolution_system_prompt(
            "Build API", "Create routes", [conflict], []
        )
        assert "(new file)" in prompt

    def test_contains_non_conflicting_paths(self) -> None:
        conflict = self._make_conflict()
        prompt = build_conflict_resolution_system_prompt(
            "Build API", "Create routes", [conflict], ["models.py", "utils.py"]
        )
        assert "models.py" in prompt
        assert "utils.py" in prompt

    def test_contains_project_instructions(self) -> None:
        conflict = self._make_conflict()
        prompt = build_conflict_resolution_system_prompt(
            "Build API",
            "Create routes",
            [conflict],
            [],
            project_instructions="## Project\nUse ruff.",
        )
        assert "Use ruff" in prompt

    def test_contains_task_and_step_context(self) -> None:
        conflict = self._make_conflict()
        prompt = build_conflict_resolution_system_prompt(
            "Build a REST API", "Create HTTP routes", [conflict], []
        )
        assert "Build a REST API" in prompt
        assert "Create HTTP routes" in prompt

    def test_multiple_conflicts(self) -> None:
        c1 = self._make_conflict("a.py")
        c2 = self._make_conflict("b.py")
        prompt = build_conflict_resolution_system_prompt("Build API", "Create routes", [c1, c2], [])
        assert "a.py" in prompt
        assert "b.py" in prompt


# ---------------------------------------------------------------------------
# TestBuildConflictResolutionUserPrompt
# ---------------------------------------------------------------------------


class TestBuildConflictResolutionUserPrompt:
    """build_conflict_resolution_user_prompt returns correct count."""

    def test_single_conflict(self) -> None:
        prompt = build_conflict_resolution_user_prompt(1)
        assert "1 file conflict" in prompt

    def test_multiple_conflicts(self) -> None:
        prompt = build_conflict_resolution_user_prompt(3)
        assert "3 file conflict" in prompt


# ---------------------------------------------------------------------------
# TestExecuteConflictResolutionCall
# ---------------------------------------------------------------------------


class TestExecuteConflictResolutionCall:
    """execute_conflict_resolution_call calls agent and extracts structured results."""

    @pytest.mark.asyncio
    async def test_returns_result_with_correct_fields(self) -> None:
        mock_response = ConflictResolutionResponse(
            resolved_files=[
                FileOutput(
                    file_path="shared.py",
                    content="# merged\ndef foo(): pass\ndef bar(): pass",
                ),
            ],
            explanation="Combined both functions.",
        )
        mock_usage = MagicMock()
        mock_usage.input_tokens = 200
        mock_usage.output_tokens = 100
        mock_usage.cache_creation_input_tokens = 10
        mock_usage.cache_read_input_tokens = 5

        mock_run_result = MagicMock()
        mock_run_result.output = mock_response
        mock_run_result.usage.return_value = mock_usage

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_run_result
        mock_agent.model = "mock-model"

        input_data = ConflictResolutionCallInput(
            task_id="test-task",
            step_id="step-1",
            system_prompt="system",
            user_prompt="user",
        )

        result = await execute_conflict_resolution_call(input_data, mock_agent)

        assert isinstance(result, ConflictResolutionCallResult)
        assert result.task_id == "test-task"
        assert result.resolved_files == {"shared.py": "# merged\ndef foo(): pass\ndef bar(): pass"}
        assert result.explanation == "Combined both functions."
        assert result.model_name == "mock-model"
        assert result.input_tokens == 200
        assert result.output_tokens == 100
        assert result.cache_creation_input_tokens == 10
        assert result.cache_read_input_tokens == 5
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_multiple_resolved_files(self) -> None:
        mock_response = ConflictResolutionResponse(
            resolved_files=[
                FileOutput(file_path="a.py", content="# merged a"),
                FileOutput(file_path="b.py", content="# merged b"),
            ],
            explanation="Merged both files.",
        )
        mock_usage = MagicMock()
        mock_usage.input_tokens = 300
        mock_usage.output_tokens = 150
        mock_usage.cache_creation_input_tokens = 0
        mock_usage.cache_read_input_tokens = 0

        mock_run_result = MagicMock()
        mock_run_result.output = mock_response
        mock_run_result.usage.return_value = mock_usage

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_run_result
        mock_agent.model = "mock-model"

        input_data = ConflictResolutionCallInput(
            task_id="test-task",
            step_id="step-1",
            system_prompt="system",
            user_prompt="user",
        )

        result = await execute_conflict_resolution_call(input_data, mock_agent)

        assert len(result.resolved_files) == 2
        assert result.resolved_files["a.py"] == "# merged a"
        assert result.resolved_files["b.py"] == "# merged b"
