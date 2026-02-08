"""Tests for forge.activities.context — context assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from forge.activities.context import (
    _read_context_files,
    assemble_context,
    assemble_step_context,
    build_step_system_prompt,
    build_step_user_prompt,
    build_system_prompt,
    build_user_prompt,
)
from forge.models import (
    AssembleContextInput,
    AssembleStepContextInput,
    PlanStep,
    StepResult,
    TaskDefinition,
    TransitionSignal,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# build_system_prompt (pure function)
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_includes_description(self) -> None:
        task = TaskDefinition(
            task_id="t1",
            description="Create a hello world module.",
            target_files=["hello.py"],
        )
        prompt = build_system_prompt(task, {})
        assert "Create a hello world module." in prompt

    def test_includes_target_files(self) -> None:
        task = TaskDefinition(
            task_id="t1",
            description="desc",
            target_files=["src/a.py", "src/b.py"],
        )
        prompt = build_system_prompt(task, {})
        assert "- src/a.py" in prompt
        assert "- src/b.py" in prompt

    def test_includes_context_file_contents(self) -> None:
        task = TaskDefinition(
            task_id="t1",
            description="desc",
            target_files=["out.py"],
        )
        context = {"ref.py": "# reference code\nprint('hi')"}
        prompt = build_system_prompt(task, context)
        assert "### ref.py" in prompt
        assert "# reference code" in prompt

    def test_empty_context_omits_section(self) -> None:
        task = TaskDefinition(
            task_id="t1",
            description="desc",
            target_files=["out.py"],
        )
        prompt = build_system_prompt(task, {})
        assert "## Context Files" not in prompt


# ---------------------------------------------------------------------------
# build_user_prompt (pure function)
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    def test_returns_nonempty(self) -> None:
        prompt = build_user_prompt()
        assert len(prompt) > 0

    def test_is_deterministic(self) -> None:
        assert build_user_prompt() == build_user_prompt()


# ---------------------------------------------------------------------------
# _read_context_files (imperative shell)
# ---------------------------------------------------------------------------


class TestReadContextFiles:
    def test_reads_existing_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("content-a")
        result = _read_context_files(tmp_path, ["a.txt"])
        assert result == {"a.txt": "content-a"}

    def test_skips_missing_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("content-a")
        result = _read_context_files(tmp_path, ["a.txt", "missing.txt"])
        assert "a.txt" in result
        assert "missing.txt" not in result


# ---------------------------------------------------------------------------
# assemble_context (activity)
# ---------------------------------------------------------------------------


class TestAssembleContext:
    @pytest.mark.asyncio
    async def test_assembles_from_files(self, tmp_path: Path) -> None:
        (tmp_path / "ref.py").write_text("# reference")
        task = TaskDefinition(
            task_id="ctx-1",
            description="Generate code.",
            target_files=["out.py"],
            context_files=["ref.py"],
        )
        input_data = AssembleContextInput(
            task=task,
            repo_root=str(tmp_path),
            worktree_path=str(tmp_path / "wt"),
        )
        result = await assemble_context(input_data)
        assert result.task_id == "ctx-1"
        assert "# reference" in result.system_prompt
        assert len(result.user_prompt) > 0

    @pytest.mark.asyncio
    async def test_skips_missing_context_files(self, tmp_path: Path) -> None:
        task = TaskDefinition(
            task_id="ctx-2",
            description="Generate code.",
            target_files=["out.py"],
            context_files=["nonexistent.py"],
        )
        input_data = AssembleContextInput(
            task=task,
            repo_root=str(tmp_path),
            worktree_path=str(tmp_path / "wt"),
        )
        result = await assemble_context(input_data)
        assert result.task_id == "ctx-2"
        assert "## Context Files" not in result.system_prompt

    @pytest.mark.asyncio
    async def test_propagates_task_id(self, tmp_path: Path) -> None:
        task = TaskDefinition(
            task_id="my-task-id",
            description="desc",
            target_files=["f.py"],
        )
        input_data = AssembleContextInput(
            task=task,
            repo_root=str(tmp_path),
            worktree_path=str(tmp_path / "wt"),
        )
        result = await assemble_context(input_data)
        assert result.task_id == "my-task-id"


# ---------------------------------------------------------------------------
# build_step_system_prompt (pure function)
# ---------------------------------------------------------------------------


class TestBuildStepSystemPrompt:
    def test_includes_task_description(self) -> None:
        task = TaskDefinition(task_id="t1", description="Build a REST API.")
        step = PlanStep(step_id="s1", description="Create models.", target_files=["models.py"])
        prompt = build_step_system_prompt(task, step, 0, 2, [], {})
        assert "Build a REST API." in prompt

    def test_includes_step_details(self) -> None:
        task = TaskDefinition(task_id="t1", description="desc")
        step = PlanStep(step_id="s1", description="Create models.", target_files=["models.py"])
        prompt = build_step_system_prompt(task, step, 0, 2, [], {})
        assert "s1" in prompt
        assert "Create models." in prompt
        assert "- models.py" in prompt

    def test_includes_progress(self) -> None:
        task = TaskDefinition(task_id="t1", description="desc")
        step = PlanStep(step_id="s2", description="step 2", target_files=["api.py"])
        prompt = build_step_system_prompt(task, step, 1, 3, [], {})
        assert "2/3" in prompt

    def test_includes_completed_steps(self) -> None:
        task = TaskDefinition(task_id="t1", description="desc")
        step = PlanStep(step_id="s2", description="step 2", target_files=["api.py"])
        completed = [StepResult(step_id="s1", status=TransitionSignal.SUCCESS)]
        prompt = build_step_system_prompt(task, step, 1, 2, completed, {})
        assert "s1: success" in prompt

    def test_includes_context_files(self) -> None:
        task = TaskDefinition(task_id="t1", description="desc")
        step = PlanStep(step_id="s1", description="step", target_files=["out.py"])
        context = {"models.py": "class Model: pass"}
        prompt = build_step_system_prompt(task, step, 0, 1, [], context)
        assert "#### models.py" in prompt
        assert "class Model: pass" in prompt


# ---------------------------------------------------------------------------
# build_step_user_prompt (pure function)
# ---------------------------------------------------------------------------


class TestBuildStepUserPrompt:
    def test_includes_step_id_and_description(self) -> None:
        step = PlanStep(step_id="s1", description="Create models.", target_files=["models.py"])
        prompt = build_step_user_prompt(step)
        assert "s1" in prompt
        assert "Create models." in prompt


# ---------------------------------------------------------------------------
# assemble_step_context (activity)
# ---------------------------------------------------------------------------


class TestAssembleStepContext:
    @pytest.mark.asyncio
    async def test_reads_from_worktree(self, tmp_path: Path) -> None:
        """Context files are read from worktree_path, not repo_root."""
        worktree = tmp_path / "worktree"
        worktree.mkdir()
        (worktree / "prev.py").write_text("# from previous step")

        task = TaskDefinition(task_id="t1", description="Build API.")
        step = PlanStep(
            step_id="s2",
            description="Create routes.",
            target_files=["routes.py"],
            context_files=["prev.py"],
        )
        input_data = AssembleStepContextInput(
            task=task,
            step=step,
            step_index=1,
            total_steps=2,
            repo_root=str(tmp_path),
            worktree_path=str(worktree),
        )
        result = await assemble_step_context(input_data)
        assert result.task_id == "t1"
        assert "# from previous step" in result.system_prompt
        assert "s2" in result.user_prompt

    @pytest.mark.asyncio
    async def test_skips_missing_context_files(self, tmp_path: Path) -> None:
        worktree = tmp_path / "worktree"
        worktree.mkdir()

        task = TaskDefinition(task_id="t1", description="desc")
        step = PlanStep(
            step_id="s1",
            description="step",
            target_files=["out.py"],
            context_files=["nonexistent.py"],
        )
        input_data = AssembleStepContextInput(
            task=task,
            step=step,
            step_index=0,
            total_steps=1,
            repo_root=str(tmp_path),
            worktree_path=str(worktree),
        )
        result = await assemble_step_context(input_data)
        assert "### Context Files" not in result.system_prompt
