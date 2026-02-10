"""Tests for forge.activities.context — context assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from forge.activities.context import (
    _build_context_stats,
    _detect_package_name,
    _read_context_files,
    assemble_context,
    assemble_step_context,
    assemble_sub_task_context,
    build_error_section,
    build_step_system_prompt,
    build_step_user_prompt,
    build_sub_task_system_prompt,
    build_sub_task_user_prompt,
    build_system_prompt,
    build_system_prompt_with_context,
    build_user_prompt,
    find_enclosing_scope,
    parse_ruff_error_lines,
)
from forge.models import (
    AssembleContextInput,
    AssembleStepContextInput,
    AssembleSubTaskContextInput,
    ContextConfig,
    PlanStep,
    StepResult,
    SubTask,
    TaskDefinition,
    TransitionSignal,
    ValidationResult,
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


# ---------------------------------------------------------------------------
# build_sub_task_system_prompt (pure function)
# ---------------------------------------------------------------------------


class TestBuildSubTaskSystemPrompt:
    def test_includes_parent_context(self) -> None:
        st = SubTask(sub_task_id="st1", description="Analyze schema.", target_files=["schema.py"])
        prompt = build_sub_task_system_prompt("parent-task", "Build an API.", st, {})
        assert "parent-task" in prompt
        assert "Build an API." in prompt

    def test_includes_sub_task_details(self) -> None:
        st = SubTask(sub_task_id="st1", description="Analyze schema.", target_files=["schema.py"])
        prompt = build_sub_task_system_prompt("t1", "desc", st, {})
        assert "st1" in prompt
        assert "Analyze schema." in prompt
        assert "- schema.py" in prompt

    def test_includes_context_files(self) -> None:
        st = SubTask(sub_task_id="st1", description="d", target_files=["out.py"])
        context = {"models.py": "class Model: pass"}
        prompt = build_sub_task_system_prompt("t1", "desc", st, context)
        assert "#### models.py" in prompt
        assert "class Model: pass" in prompt

    def test_empty_context_omits_section(self) -> None:
        st = SubTask(sub_task_id="st1", description="d", target_files=["out.py"])
        prompt = build_sub_task_system_prompt("t1", "desc", st, {})
        assert "### Context Files" not in prompt


# ---------------------------------------------------------------------------
# build_sub_task_user_prompt (pure function)
# ---------------------------------------------------------------------------


class TestBuildSubTaskUserPrompt:
    def test_includes_sub_task_id(self) -> None:
        st = SubTask(sub_task_id="st1", description="Create schema.", target_files=["schema.py"])
        prompt = build_sub_task_user_prompt(st)
        assert "st1" in prompt
        assert "Create schema." in prompt


# ---------------------------------------------------------------------------
# assemble_sub_task_context (activity)
# ---------------------------------------------------------------------------


class TestAssembleSubTaskContext:
    @pytest.mark.asyncio
    async def test_reads_from_parent_worktree(self, tmp_path: Path) -> None:
        """Context files are read from the parent worktree."""
        parent_wt = tmp_path / "parent-wt"
        parent_wt.mkdir()
        (parent_wt / "models.py").write_text("# models from parent")

        st = SubTask(
            sub_task_id="st1",
            description="Analyze.",
            target_files=["schema.py"],
            context_files=["models.py"],
        )
        input_data = AssembleSubTaskContextInput(
            parent_task_id="t1",
            parent_description="Build API.",
            sub_task=st,
            worktree_path=str(parent_wt),
        )
        result = await assemble_sub_task_context(input_data)
        assert result.task_id == "t1"
        assert "# models from parent" in result.system_prompt

    @pytest.mark.asyncio
    async def test_skips_missing_context_files(self, tmp_path: Path) -> None:
        parent_wt = tmp_path / "parent-wt"
        parent_wt.mkdir()

        st = SubTask(
            sub_task_id="st1",
            description="d",
            target_files=["out.py"],
            context_files=["nonexistent.py"],
        )
        input_data = AssembleSubTaskContextInput(
            parent_task_id="t1",
            parent_description="desc",
            sub_task=st,
            worktree_path=str(parent_wt),
        )
        result = await assemble_sub_task_context(input_data)
        assert "### Context Files" not in result.system_prompt


# ---------------------------------------------------------------------------
# Phase 4: build_system_prompt_with_context (pure function)
# ---------------------------------------------------------------------------


class TestBuildSystemPromptWithContext:
    def test_includes_task_and_targets(self) -> None:
        from forge.code_intel.budget import ContextItem, PackedContext, Representation

        task = TaskDefinition(task_id="t1", description="Build a module.", target_files=["out.py"])
        packed = PackedContext(
            items=[
                ContextItem(
                    file_path="out.py",
                    content="# existing",
                    representation=Representation.FULL,
                    priority=2,
                    estimated_tokens=5,
                ),
            ],
            total_estimated_tokens=5,
            items_included=1,
        )
        prompt = build_system_prompt_with_context(task, packed)
        assert "Build a module." in prompt
        assert "- out.py" in prompt
        assert "# existing" in prompt

    def test_includes_output_requirements(self) -> None:
        from forge.code_intel.budget import ContextItem, PackedContext, Representation

        task = TaskDefinition(task_id="t1", description="Build a module.", target_files=["out.py"])
        packed = PackedContext(
            items=[
                ContextItem(
                    file_path="out.py",
                    content="# existing",
                    representation=Representation.FULL,
                    priority=2,
                    estimated_tokens=5,
                ),
            ],
            total_estimated_tokens=5,
            items_included=1,
        )
        prompt = build_system_prompt_with_context(task, packed)
        assert "## Output Requirements" in prompt
        assert "LLMResponse" in prompt
        assert "files" in prompt
        assert "Do NOT return an empty object" in prompt

    def test_sections_by_priority(self) -> None:
        from forge.code_intel.budget import ContextItem, PackedContext, Representation

        task = TaskDefinition(task_id="t1", description="desc", target_files=["a.py"])
        packed = PackedContext(
            items=[
                ContextItem(
                    file_path="a.py",
                    content="target content",
                    representation=Representation.FULL,
                    priority=2,
                    estimated_tokens=5,
                ),
                ContextItem(
                    file_path="dep.py",
                    content="dep content",
                    representation=Representation.FULL,
                    priority=3,
                    estimated_tokens=5,
                ),
                ContextItem(
                    file_path="trans.py",
                    content="def f():",
                    representation=Representation.SIGNATURES,
                    priority=4,
                    estimated_tokens=3,
                ),
                ContextItem(
                    file_path="__repo_map__",
                    content="repo map text",
                    representation=Representation.REPO_MAP,
                    priority=5,
                    estimated_tokens=5,
                ),
                ContextItem(
                    file_path="manual.txt",
                    content="manual info",
                    representation=Representation.FULL,
                    priority=6,
                    estimated_tokens=5,
                ),
            ],
            total_estimated_tokens=23,
            items_included=5,
        )
        prompt = build_system_prompt_with_context(task, packed)
        assert "## Target File Contents" in prompt
        assert "## Direct Dependencies" in prompt
        assert "## Interface Context" in prompt
        assert "## Repository Structure" in prompt
        assert "## Additional Context" in prompt
        assert "trans.py (signatures)" in prompt


# ---------------------------------------------------------------------------
# _build_context_stats (pure function)
# ---------------------------------------------------------------------------


class TestBuildContextStats:
    def test_basic_stats(self) -> None:
        from forge.code_intel.budget import ContextItem, PackedContext, Representation

        packed = PackedContext(
            items=[
                ContextItem(
                    file_path="a.py",
                    content="code",
                    representation=Representation.FULL,
                    priority=2,
                    estimated_tokens=10,
                ),
                ContextItem(
                    file_path="b.py",
                    content="def f():",
                    representation=Representation.SIGNATURES,
                    priority=4,
                    estimated_tokens=5,
                ),
                ContextItem(
                    file_path="__repo_map__",
                    content="map",
                    representation=Representation.REPO_MAP,
                    priority=5,
                    estimated_tokens=3,
                ),
            ],
            total_estimated_tokens=18,
            budget_utilization=0.5,
            items_included=3,
            items_truncated=1,
        )
        stats = _build_context_stats(packed)
        assert stats.files_included_full == 1
        assert stats.files_included_signatures == 1
        assert stats.repo_map_tokens == 3
        assert stats.files_truncated == 1
        assert stats.total_estimated_tokens == 18
        assert stats.files_discovered == 4  # included + truncated


# ---------------------------------------------------------------------------
# _detect_package_name
# ---------------------------------------------------------------------------


class TestDetectPackageName:
    def test_detects_from_src(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        pkg = src / "myapp"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        assert _detect_package_name(str(tmp_path)) == "myapp"

    def test_fallback_to_dir_name(self, tmp_path: Path) -> None:
        assert _detect_package_name(str(tmp_path)) == tmp_path.name


# ---------------------------------------------------------------------------
# assemble_context with auto_discover disabled
# ---------------------------------------------------------------------------


class TestAssembleContextAutoDiscoverDisabled:
    @pytest.mark.asyncio
    async def test_disabled_uses_manual_context(self, tmp_path: Path) -> None:
        (tmp_path / "ref.py").write_text("# reference")
        config = ContextConfig(auto_discover=False)
        task = TaskDefinition(
            task_id="ctx-no-auto",
            description="Generate code.",
            target_files=["out.py"],
            context_files=["ref.py"],
            context=config,
        )
        input_data = AssembleContextInput(
            task=task,
            repo_root=str(tmp_path),
            worktree_path=str(tmp_path / "wt"),
        )
        result = await assemble_context(input_data)
        assert result.task_id == "ctx-no-auto"
        assert "# reference" in result.system_prompt
        assert result.context_stats is None


# ---------------------------------------------------------------------------
# D50: Target file contents in step/sub-task context
# ---------------------------------------------------------------------------


class TestBuildStepSystemPromptTargetFiles:
    def test_includes_target_file_contents(self) -> None:
        task = TaskDefinition(task_id="t1", description="desc")
        step = PlanStep(step_id="s1", description="Edit module.", target_files=["mod.py"])
        target_contents = {"mod.py": "def existing():\n    pass\n"}
        prompt = build_step_system_prompt(task, step, 0, 1, [], {}, target_contents)
        assert "### Current Target File Contents" in prompt
        assert "#### mod.py" in prompt
        assert "def existing():" in prompt

    def test_omits_section_when_no_target_contents(self) -> None:
        task = TaskDefinition(task_id="t1", description="desc")
        step = PlanStep(step_id="s1", description="Create module.", target_files=["new.py"])
        prompt = build_step_system_prompt(task, step, 0, 1, [], {})
        assert "### Current Target File Contents" not in prompt

    def test_includes_output_requirements(self) -> None:
        task = TaskDefinition(task_id="t1", description="desc")
        step = PlanStep(step_id="s1", description="Edit module.", target_files=["mod.py"])
        prompt = build_step_system_prompt(task, step, 0, 1, [], {})
        assert "### Output Requirements" in prompt
        assert "edits" in prompt
        assert "files" in prompt


class TestBuildSubTaskSystemPromptTargetFiles:
    def test_includes_target_file_contents(self) -> None:
        st = SubTask(sub_task_id="st1", description="Edit.", target_files=["mod.py"])
        target_contents = {"mod.py": "class Existing: pass"}
        prompt = build_sub_task_system_prompt("t1", "desc", st, {}, target_contents)
        assert "### Current Target File Contents" in prompt
        assert "#### mod.py" in prompt
        assert "class Existing: pass" in prompt

    def test_omits_section_when_no_target_contents(self) -> None:
        st = SubTask(sub_task_id="st1", description="Create.", target_files=["new.py"])
        prompt = build_sub_task_system_prompt("t1", "desc", st, {})
        assert "### Current Target File Contents" not in prompt

    def test_includes_output_requirements(self) -> None:
        st = SubTask(sub_task_id="st1", description="Edit.", target_files=["mod.py"])
        prompt = build_sub_task_system_prompt("t1", "desc", st, {})
        assert "### Output Requirements" in prompt
        assert "edits" in prompt
        assert "files" in prompt


class TestAssembleStepContextTargetFiles:
    @pytest.mark.asyncio
    async def test_reads_target_files_from_worktree(self, tmp_path: Path) -> None:
        """Target files are read from worktree and included in prompt."""
        worktree = tmp_path / "worktree"
        worktree.mkdir()
        (worktree / "target.py").write_text("# existing target content")

        task = TaskDefinition(task_id="t1", description="Edit target.")
        step = PlanStep(
            step_id="s1",
            description="Modify target.",
            target_files=["target.py"],
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
        assert "# existing target content" in result.system_prompt
        assert "### Current Target File Contents" in result.system_prompt

    @pytest.mark.asyncio
    async def test_missing_target_files_are_skipped(self, tmp_path: Path) -> None:
        """Non-existent target files are gracefully skipped (new files)."""
        worktree = tmp_path / "worktree"
        worktree.mkdir()

        task = TaskDefinition(task_id="t1", description="Create new.")
        step = PlanStep(
            step_id="s1",
            description="Create new file.",
            target_files=["new_file.py"],
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
        assert "### Current Target File Contents" not in result.system_prompt


class TestAssembleSubTaskContextTargetFiles:
    @pytest.mark.asyncio
    async def test_reads_target_files_from_parent_worktree(self, tmp_path: Path) -> None:
        """Target files are read from parent worktree and included in prompt."""
        parent_wt = tmp_path / "parent-wt"
        parent_wt.mkdir()
        (parent_wt / "target.py").write_text("# parent target content")

        st = SubTask(
            sub_task_id="st1",
            description="Edit target.",
            target_files=["target.py"],
        )
        input_data = AssembleSubTaskContextInput(
            parent_task_id="t1",
            parent_description="Build API.",
            sub_task=st,
            worktree_path=str(parent_wt),
        )
        result = await assemble_sub_task_context(input_data)
        assert "# parent target content" in result.system_prompt
        assert "### Current Target File Contents" in result.system_prompt


# ---------------------------------------------------------------------------
# D50: Updated output requirements in prompts
# ---------------------------------------------------------------------------


class TestOutputRequirementsInPrompts:
    def test_single_step_prompt_describes_edits(self) -> None:
        prompt = build_user_prompt()
        assert "edits" in prompt
        assert "files" in prompt

    def test_step_user_prompt_describes_edits(self) -> None:
        step = PlanStep(step_id="s1", description="Edit.", target_files=["a.py"])
        prompt = build_step_user_prompt(step)
        assert "edits" in prompt
        assert "files" in prompt

    def test_sub_task_user_prompt_describes_edits(self) -> None:
        st = SubTask(sub_task_id="st1", description="Edit.", target_files=["a.py"])
        prompt = build_sub_task_user_prompt(st)
        assert "edits" in prompt
        assert "files" in prompt

    def test_system_prompt_with_context_output_requirements(self) -> None:
        from forge.code_intel.budget import ContextItem, PackedContext, Representation

        task = TaskDefinition(task_id="t1", description="Build.", target_files=["out.py"])
        packed = PackedContext(
            items=[
                ContextItem(
                    file_path="out.py",
                    content="# existing",
                    representation=Representation.FULL,
                    priority=2,
                    estimated_tokens=5,
                ),
            ],
            total_estimated_tokens=5,
            items_included=1,
        )
        prompt = build_system_prompt_with_context(task, packed)
        assert "edits" in prompt
        assert "files" in prompt
        assert "search" in prompt
        assert "replace" in prompt


# ---------------------------------------------------------------------------
# Phase 8: parse_ruff_error_lines (pure function)
# ---------------------------------------------------------------------------


class TestParseRuffErrorLines:
    def test_standard_format(self) -> None:
        output = "src/foo.py:42:1: F401 `os` imported but unused\n"
        result = parse_ruff_error_lines(output)
        assert result == [("src/foo.py", 42, "F401 `os` imported but unused")]

    def test_multiple_lines(self) -> None:
        output = (
            "src/a.py:10:5: E501 line too long\nsrc/b.py:20:1: F811 redefinition of unused `x`\n"
        )
        result = parse_ruff_error_lines(output)
        assert len(result) == 2
        assert result[0] == ("src/a.py", 10, "E501 line too long")
        assert result[1] == ("src/b.py", 20, "F811 redefinition of unused `x`")

    def test_unparseable_lines_skipped(self) -> None:
        output = "Found 3 errors.\nsome random text\nsrc/a.py:5:1: W001 warning\n"
        result = parse_ruff_error_lines(output)
        assert len(result) == 1
        assert result[0][0] == "src/a.py"

    def test_empty_input(self) -> None:
        assert parse_ruff_error_lines("") == []


# ---------------------------------------------------------------------------
# Phase 8: find_enclosing_scope (pure function)
# ---------------------------------------------------------------------------


class TestFindEnclosingScope:
    def test_line_inside_function(self) -> None:
        source = "import os\n\ndef hello():\n    x = 1\n    y = 2\n    return x + y\n"
        result = find_enclosing_scope(source, 5)
        assert result is not None
        assert "def hello():" in result
        assert "# <-- ERROR" in result

    def test_line_inside_class_method(self) -> None:
        source = "class Foo:\n    def bar(self):\n        return 42\n"
        result = find_enclosing_scope(source, 3)
        assert result is not None
        assert "# <-- ERROR" in result

    def test_top_level_line(self) -> None:
        source = "import os\nprint('hello')\n"
        result = find_enclosing_scope(source, 1)
        assert result is None

    def test_invalid_python_source(self) -> None:
        result = find_enclosing_scope("def (broken:", 1)
        assert result is None

    def test_line_out_of_range(self) -> None:
        source = "x = 1\n"
        assert find_enclosing_scope(source, 0) is None
        assert find_enclosing_scope(source, 99) is None


# ---------------------------------------------------------------------------
# Phase 8: build_error_section
# ---------------------------------------------------------------------------


class TestBuildErrorSection:
    def test_empty_errors_returns_empty(self, tmp_path: Path) -> None:
        assert build_error_section([], 1, 2, str(tmp_path)) == ""

    def test_all_passed_returns_empty(self, tmp_path: Path) -> None:
        errors = [ValidationResult(check_name="ruff_lint", passed=True, summary="ok")]
        assert build_error_section(errors, 1, 2, str(tmp_path)) == ""

    def test_ruff_lint_with_ast_enrichment(self, tmp_path: Path) -> None:
        # Create a file for AST enrichment
        target = tmp_path / "src" / "foo.py"
        target.parent.mkdir(parents=True)
        target.write_text("import os\n\ndef greet():\n    unused_var = os\n    return 'hello'\n")
        errors = [
            ValidationResult(
                check_name="ruff_lint",
                passed=False,
                summary="ruff_lint failed",
                details="src/foo.py:4:5: F841 local variable `unused_var` assigned but unused",
            )
        ]
        result = build_error_section(errors, 2, 2, str(tmp_path))
        assert "## Previous Attempt Errors (Attempt 2 of 2)" in result
        assert "ruff_lint failed" in result
        assert "F841" in result
        assert "Context around error" in result
        assert "# <-- ERROR" in result
        assert "Do NOT repeat the same mistakes" in result

    def test_test_failure_verbatim(self, tmp_path: Path) -> None:
        errors = [
            ValidationResult(
                check_name="tests",
                passed=False,
                summary="tests failed",
                details="FAILED tests/test_foo.py::test_bar - AssertionError: expected 1, got 2",
            )
        ]
        result = build_error_section(errors, 2, 2, str(tmp_path))
        assert "tests failed" in result
        assert "FAILED tests/test_foo.py::test_bar" in result
        # No AST enrichment for test errors
        assert "Context around error" not in result

    def test_mixed_error_types(self, tmp_path: Path) -> None:
        errors = [
            ValidationResult(
                check_name="ruff_lint",
                passed=False,
                summary="ruff_lint failed",
                details="src/a.py:1:1: F401 unused import",
            ),
            ValidationResult(
                check_name="tests",
                passed=False,
                summary="tests failed",
                details="FAILED test_x.py::test_y",
            ),
        ]
        result = build_error_section(errors, 2, 3, str(tmp_path))
        assert "### ruff_lint failed" in result
        assert "### tests failed" in result
        assert "Attempt 2 of 3" in result

    def test_truncation_at_limit(self, tmp_path: Path) -> None:
        long_details = "x" * 3000
        errors = [
            ValidationResult(
                check_name="ruff_lint",
                passed=False,
                summary="ruff_lint failed",
                details=long_details,
            )
        ]
        result = build_error_section(errors, 2, 2, str(tmp_path))
        assert "... (truncated)" in result

    def test_attempt_header(self, tmp_path: Path) -> None:
        errors = [ValidationResult(check_name="ruff_lint", passed=False, summary="failed")]
        result = build_error_section(errors, 3, 5, str(tmp_path))
        assert "Attempt 3 of 5" in result


# ---------------------------------------------------------------------------
# Phase 8: Error section in prompt builders
# ---------------------------------------------------------------------------


class TestErrorSectionInPrompts:
    def test_build_system_prompt_includes_error_section(self) -> None:
        task = TaskDefinition(task_id="t1", description="desc", target_files=["a.py"])
        error_section = "## Previous Attempt Errors\nSome error"
        prompt = build_system_prompt(task, {}, error_section=error_section)
        assert "## Previous Attempt Errors" in prompt
        assert "Some error" in prompt

    def test_build_system_prompt_no_error_on_first_attempt(self) -> None:
        task = TaskDefinition(task_id="t1", description="desc", target_files=["a.py"])
        prompt = build_system_prompt(task, {})
        assert "Previous Attempt Errors" not in prompt

    def test_build_system_prompt_with_context_includes_error_section(self) -> None:
        from forge.code_intel.budget import ContextItem, PackedContext, Representation

        task = TaskDefinition(task_id="t1", description="desc", target_files=["a.py"])
        packed = PackedContext(
            items=[
                ContextItem(
                    file_path="a.py",
                    content="# code",
                    representation=Representation.FULL,
                    priority=2,
                    estimated_tokens=5,
                ),
            ],
            total_estimated_tokens=5,
            items_included=1,
        )
        error_section = "## Previous Attempt Errors\nLint error"
        prompt = build_system_prompt_with_context(task, packed, error_section=error_section)
        # Error section should appear before Output Requirements
        error_pos = prompt.index("Previous Attempt Errors")
        output_pos = prompt.index("## Output Requirements")
        assert error_pos < output_pos

    def test_build_step_system_prompt_includes_error_section(self) -> None:
        task = TaskDefinition(task_id="t1", description="desc")
        step = PlanStep(step_id="s1", description="step", target_files=["a.py"])
        error_section = "## Previous Attempt Errors\nStep error"
        prompt = build_step_system_prompt(task, step, 0, 1, [], {}, error_section=error_section)
        error_pos = prompt.index("Previous Attempt Errors")
        output_pos = prompt.index("### Output Requirements")
        assert error_pos < output_pos

    def test_build_sub_task_system_prompt_includes_error_section(self) -> None:
        st = SubTask(sub_task_id="st1", description="d", target_files=["a.py"])
        error_section = "## Previous Attempt Errors\nSub-task error"
        prompt = build_sub_task_system_prompt("t1", "desc", st, {}, error_section=error_section)
        error_pos = prompt.index("Previous Attempt Errors")
        output_pos = prompt.index("### Output Requirements")
        assert error_pos < output_pos
