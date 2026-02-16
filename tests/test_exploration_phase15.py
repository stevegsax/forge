"""Tests for Phase 15 exploration prompt updates."""

from __future__ import annotations

from forge.activities.exploration import build_exploration_prompt
from forge.models import (
    ContextProviderSpec,
    ContextResult,
    ExplorationInput,
    TaskDefinition,
)


def _make_input(
    providers: list[ContextProviderSpec] | None = None,
    accumulated: list[ContextResult] | None = None,
) -> ExplorationInput:
    """Create a minimal ExplorationInput for testing."""
    return ExplorationInput(
        task=TaskDefinition(
            task_id="test",
            description="Test task",
            target_files=["test.py"],
        ),
        available_providers=providers or [],
        accumulated_context=accumulated or [],
        round_number=1,
        max_rounds=5,
    )


class TestExplorationPromptWithActions:
    def test_no_actions_no_mention(self):
        """Without MCP/Skills, prompt should not mention actions."""
        providers = [
            ContextProviderSpec(
                name="read_file",
                description="Read a file",
                parameters={"path": "File path"},
            ),
        ]
        system, user = build_exploration_prompt(_make_input(providers=providers))
        assert "actions" not in system.lower() or "actions" not in user.lower()
        assert "## Available Context Providers" in system
        assert "## Available Actions" not in system

    def test_mcp_tool_in_actions_section(self):
        """MCP tools should appear in the Actions section."""
        providers = [
            ContextProviderSpec(
                name="read_file",
                description="Read a file",
                parameters={"path": "File path"},
            ),
            ContextProviderSpec(
                name="mcp_github_search",
                description="[MCP:github] Search repos",
                parameters={"query": "Search query"},
            ),
        ]
        system, user = build_exploration_prompt(_make_input(providers=providers))
        assert "## Available Actions (MCP Tools & Skills)" in system
        assert "mcp_github_search" in system
        assert "(type: mcp_tool)" in system
        assert "actions" in user.lower()

    def test_skill_in_actions_section(self):
        """Skills should appear in the Actions section."""
        providers = [
            ContextProviderSpec(
                name="search_code",
                description="Search code",
                parameters={"query": "Search query"},
            ),
            ContextProviderSpec(
                name="skill_deploy",
                description="[Skill] Deploy the app",
                parameters={"request": "What to deploy"},
            ),
        ]
        system, user = build_exploration_prompt(_make_input(providers=providers))
        assert "skill_deploy" in system
        assert "(type: skill)" in system

    def test_readiness_signal_mentions_both(self):
        """With actions, readiness signal should mention both requests and actions."""
        providers = [
            ContextProviderSpec(
                name="mcp_test_tool",
                description="[MCP:test] A tool",
                parameters={},
            ),
        ]
        system, _ = build_exploration_prompt(_make_input(providers=providers))
        assert "EMPTY `requests` and `actions` lists" in system

    def test_readiness_signal_without_actions(self):
        """Without actions, readiness signal should only mention requests."""
        providers = [
            ContextProviderSpec(
                name="read_file",
                description="Read a file",
                parameters={},
            ),
        ]
        system, _ = build_exploration_prompt(_make_input(providers=providers))
        assert "EMPTY requests list" in system

    def test_action_instructions_present(self):
        """With actions, prompt should include action field instructions."""
        providers = [
            ContextProviderSpec(
                name="mcp_api_call",
                description="[MCP:api] Call API",
                parameters={"url": "URL to call"},
            ),
        ]
        system, _ = build_exploration_prompt(_make_input(providers=providers))
        assert "`capability`" in system
        assert "`capability_type`" in system
        assert "`params`" in system
        assert "`reasoning`" in system
