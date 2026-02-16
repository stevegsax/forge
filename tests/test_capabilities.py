"""Tests for unified capability registry (Phase 15)."""

from __future__ import annotations

from forge.capabilities import CapabilityCatalog, build_capability_catalog
from forge.models import (
    CapabilityType,
    MCPToolInfo,
    SkillDefinition,
)


class TestBuildCapabilityCatalog:
    def test_providers_only(self):
        catalog = build_capability_catalog()
        # Should contain all built-in providers
        from forge.providers import PROVIDER_SPECS

        assert len(catalog.specs) == len(PROVIDER_SPECS)
        assert len(catalog.mcp_tools) == 0
        assert len(catalog.skill_definitions) == 0
        assert len(catalog.action_names) == 0

    def test_with_mcp_tools(self):
        mcp_tools = {
            "mcp_test_search": MCPToolInfo(
                server_name="test",
                tool_name="search",
                description="Search for stuff",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
        }
        catalog = build_capability_catalog(mcp_tools=mcp_tools)

        from forge.providers import PROVIDER_SPECS

        assert len(catalog.specs) == len(PROVIDER_SPECS) + 1
        assert "mcp_test_search" in catalog.action_names
        assert catalog.is_action("mcp_test_search")
        assert not catalog.is_action("read_file")

    def test_with_skills(self):
        skill_defs = {
            "skill_deploy": SkillDefinition(
                name="deploy",
                description="Deploy the app",
                instructions="Run deploy.sh",
                skill_path="/tmp/skills/deploy",
            ),
        }
        catalog = build_capability_catalog(skill_definitions=skill_defs)

        from forge.providers import PROVIDER_SPECS

        assert len(catalog.specs) == len(PROVIDER_SPECS) + 1
        assert "skill_deploy" in catalog.action_names
        assert catalog.is_action("skill_deploy")

    def test_mixed_capabilities(self):
        mcp_tools = {
            "mcp_github_search_repos": MCPToolInfo(
                server_name="github",
                tool_name="search_repos",
                description="Search GitHub repos",
            ),
        }
        skill_defs = {
            "skill_lint": SkillDefinition(
                name="lint",
                description="Run linter",
                instructions="Run ruff check",
                skill_path="/tmp/skills/lint",
            ),
        }
        catalog = build_capability_catalog(
            mcp_tools=mcp_tools,
            skill_definitions=skill_defs,
        )

        from forge.providers import PROVIDER_SPECS

        assert len(catalog.specs) == len(PROVIDER_SPECS) + 2
        assert len(catalog.action_names) == 2

    def test_get_capability_type(self):
        mcp_tools = {
            "mcp_test_tool": MCPToolInfo(
                server_name="test",
                tool_name="tool",
            ),
        }
        skill_defs = {
            "skill_test": SkillDefinition(
                name="test",
                description="test",
                instructions="test",
                skill_path="/tmp",
            ),
        }
        catalog = build_capability_catalog(
            mcp_tools=mcp_tools,
            skill_definitions=skill_defs,
        )

        assert catalog.get_capability_type("mcp_test_tool") == CapabilityType.MCP_TOOL
        assert catalog.get_capability_type("skill_test") == CapabilityType.SKILL
        assert catalog.get_capability_type("read_file") == CapabilityType.PROVIDER
        assert catalog.get_capability_type("nonexistent") is None
