"""Tests for playbook injection into context assembly (Phase 6)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    import pytest

from forge.activities.context import (
    build_playbook_context_items,
    build_system_prompt_with_context,
    infer_task_tags,
)
from forge.code_intel.budget import (
    ContextItem,
    PackedContext,
    Representation,
    pack_context,
)
from forge.models import TaskDefinition

# ---------------------------------------------------------------------------
# infer_task_tags
# ---------------------------------------------------------------------------


class TestInferTaskTags:
    def test_python_files(self) -> None:
        task = TaskDefinition(
            task_id="t",
            description="Create a module",
            target_files=["src/a.py"],
        )
        tags = infer_task_tags(task)
        assert "python" in tags

    def test_typescript_files(self) -> None:
        task = TaskDefinition(
            task_id="t",
            description="Create component",
            target_files=["app.tsx"],
        )
        tags = infer_task_tags(task)
        assert "typescript" in tags

    def test_keywords(self) -> None:
        task = TaskDefinition(
            task_id="t",
            description="Refactor the API",
            target_files=[],
        )
        tags = infer_task_tags(task)
        assert "refactoring" in tags
        assert "api" in tags

    def test_javascript_files(self) -> None:
        task = TaskDefinition(
            task_id="t",
            description="Create component",
            target_files=["app.jsx"],
        )
        tags = infer_task_tags(task)
        assert "javascript" in tags

    def test_migration_keyword(self) -> None:
        task = TaskDefinition(
            task_id="t",
            description="Add database migration",
            target_files=[],
        )
        tags = infer_task_tags(task)
        assert "database" in tags
        assert "migration" in tags

    def test_validation_keyword(self) -> None:
        task = TaskDefinition(
            task_id="t",
            description="Validate input schema",
            target_files=[],
        )
        tags = infer_task_tags(task)
        assert "validation" in tags

    def test_default(self) -> None:
        task = TaskDefinition(
            task_id="t",
            description="Create module",
            target_files=[],
        )
        tags = infer_task_tags(task)
        assert tags == ["code-generation"]


# ---------------------------------------------------------------------------
# build_playbook_context_items
# ---------------------------------------------------------------------------


class TestBuildPlaybookContextItems:
    def test_priority_and_representation(self) -> None:
        playbooks = [
            {
                "title": "Always include stubs",
                "content": "Include type stubs for Pydantic models.",
            },
        ]
        items = build_playbook_context_items(playbooks)
        assert len(items) == 1
        assert items[0].representation == Representation.PLAYBOOK
        assert items[0].priority == 5
        assert "Always include stubs" in items[0].content
        assert items[0].estimated_tokens > 0

    def test_empty_list(self) -> None:
        items = build_playbook_context_items([])
        assert items == []

    def test_multiple_playbooks(self) -> None:
        playbooks = [
            {"title": "Lesson 1", "content": "Content 1"},
            {"title": "Lesson 2", "content": "Content 2"},
        ]
        items = build_playbook_context_items(playbooks)
        assert len(items) == 2
        assert all(i.representation == Representation.PLAYBOOK for i in items)


# ---------------------------------------------------------------------------
# build_system_prompt_with_context includes playbooks
# ---------------------------------------------------------------------------


class TestBuildSystemPromptWithPlaybooks:
    def test_includes_playbooks_section(self) -> None:
        task = TaskDefinition(
            task_id="t",
            description="Create a module",
            target_files=["a.py"],
        )
        playbook_item = ContextItem(
            file_path="playbook:Test Lesson",
            content="**Test Lesson**\nAlways include stubs.",
            representation=Representation.PLAYBOOK,
            priority=5,
            importance=0.0,
            estimated_tokens=10,
        )
        packed = PackedContext(
            items=[playbook_item],
            total_estimated_tokens=10,
            budget_utilization=0.01,
            items_included=1,
        )
        prompt = build_system_prompt_with_context(task, packed)
        assert "## Relevant Playbooks" in prompt
        assert "Test Lesson" in prompt
        assert "Always include stubs" in prompt

    def test_no_playbooks_no_section(self) -> None:
        task = TaskDefinition(
            task_id="t",
            description="Create a module",
            target_files=["a.py"],
        )
        packed = PackedContext(items=[], total_estimated_tokens=0)
        prompt = build_system_prompt_with_context(task, packed)
        assert "Relevant Playbooks" not in prompt

    def test_repo_map_and_playbooks_coexist(self) -> None:
        task = TaskDefinition(
            task_id="t",
            description="Create a module",
            target_files=["a.py"],
        )
        repo_map_item = ContextItem(
            file_path="__repo_map__",
            content="src/a.py: def foo()",
            representation=Representation.REPO_MAP,
            priority=5,
            importance=0.0,
            estimated_tokens=10,
        )
        playbook_item = ContextItem(
            file_path="playbook:Lesson",
            content="**Lesson**\nDo this.",
            representation=Representation.PLAYBOOK,
            priority=5,
            importance=0.0,
            estimated_tokens=10,
        )
        packed = PackedContext(
            items=[repo_map_item, playbook_item],
            total_estimated_tokens=20,
            budget_utilization=0.02,
            items_included=2,
        )
        prompt = build_system_prompt_with_context(task, packed)
        assert "## Repository Structure" in prompt
        assert "## Relevant Playbooks" in prompt


# ---------------------------------------------------------------------------
# Playbooks respect token budget
# ---------------------------------------------------------------------------


class TestPlaybooksRespectTokenBudget:
    def test_playbooks_dropped_when_budget_exceeded(self) -> None:
        """Playbooks at priority 5 should be dropped if budget is full."""
        # Fill budget with high-priority items
        big_item = ContextItem(
            file_path="big.py",
            content="x" * 400,
            representation=Representation.FULL,
            priority=2,
            importance=1.0,
            estimated_tokens=100,
        )
        playbook_item = ContextItem(
            file_path="playbook:Lesson",
            content="**Lesson**\nDo this.",
            representation=Representation.PLAYBOOK,
            priority=5,
            importance=0.0,
            estimated_tokens=50,
        )
        # Budget only fits the big item
        packed = pack_context([big_item, playbook_item], budget_tokens=100)
        assert packed.items_included == 1
        assert packed.items[0].file_path == "big.py"


# ---------------------------------------------------------------------------
# _load_playbooks_for_task graceful on missing store
# ---------------------------------------------------------------------------


class TestLoadPlaybooksGraceful:
    def test_missing_store_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from forge.activities.context import _load_playbooks_for_task

        monkeypatch.setenv("FORGE_DB_PATH", "")
        task = TaskDefinition(
            task_id="t",
            description="Create module",
            target_files=["a.py"],
        )
        result = _load_playbooks_for_task(task)
        assert result == []

    def test_exception_returns_empty(self) -> None:
        from forge.activities.context import _load_playbooks_for_task

        task = TaskDefinition(
            task_id="t",
            description="Create module",
            target_files=["a.py"],
        )
        with patch(
            "forge.activities.context.infer_task_tags",
            side_effect=RuntimeError("boom"),
        ):
            result = _load_playbooks_for_task(task)
            assert result == []
