"""Unified capability registry for Forge (Phase 15).

Merges built-in context providers, MCP tools, and Agent Skills into
a single capability catalog. Provides the specs list for the exploration
LLM and dispatch metadata for fulfilling action requests.

Design follows Function Core / Imperative Shell:
- Pure functions: build_capability_catalog, get_capability_specs
- I/O shell: discover_all_capabilities (calls MCP servers + scans skills)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from forge.models import (
    CapabilityType,
    ContextProviderSpec,
    MCPConfig,
    MCPToolInfo,
    SkillDefinition,
)
from forge.providers import PROVIDER_SPECS

logger = logging.getLogger(__name__)


@dataclass
class CapabilityCatalog:
    """The full catalog of available capabilities.

    Holds the specs for the LLM plus dispatch metadata for action requests.
    """

    # Specs shown to the exploration LLM (providers + MCP tools + Skills)
    specs: list[ContextProviderSpec] = field(default_factory=list)

    # Dispatch metadata for MCP tools (keyed by capability name)
    mcp_tools: dict[str, MCPToolInfo] = field(default_factory=dict)

    # Dispatch metadata for Skills (keyed by capability name)
    skill_definitions: dict[str, SkillDefinition] = field(default_factory=dict)

    # Set of names that are action capabilities (not context providers)
    action_names: set[str] = field(default_factory=set)

    def is_action(self, name: str) -> bool:
        """Check if a capability name is an action (MCP tool or Skill)."""
        return name in self.action_names

    def get_capability_type(self, name: str) -> CapabilityType | None:
        """Get the type of a capability by name."""
        if name in self.mcp_tools:
            return CapabilityType.MCP_TOOL
        if name in self.skill_definitions:
            return CapabilityType.SKILL
        # Check if it's a built-in provider
        provider_names = {s.name for s in PROVIDER_SPECS}
        if name in provider_names:
            return CapabilityType.PROVIDER
        return None


def build_capability_catalog(
    mcp_tools: dict[str, MCPToolInfo] | None = None,
    skill_definitions: dict[str, SkillDefinition] | None = None,
) -> CapabilityCatalog:
    """Build a unified capability catalog from all sources.

    Args:
        mcp_tools: Discovered MCP tools keyed by capability name.
        skill_definitions: Discovered skills keyed by capability name.

    Returns:
        A CapabilityCatalog with all capabilities merged.
    """
    from forge.mcp.client import build_tool_specs
    from forge.skills.loader import build_skill_specs

    mcp_tools = mcp_tools or {}
    skill_definitions = skill_definitions or {}

    specs: list[ContextProviderSpec] = list(PROVIDER_SPECS)
    action_names: set[str] = set()

    # Add MCP tool specs, grouped by server
    servers_seen: dict[str, list[MCPToolInfo]] = {}
    for cap_name, tool_info in mcp_tools.items():
        servers_seen.setdefault(tool_info.server_name, []).append(tool_info)
        action_names.add(cap_name)

    for server_name, tools in servers_seen.items():
        mcp_specs = build_tool_specs(server_name, tools)
        specs.extend(mcp_specs)

    # Add Skill specs
    skill_specs = build_skill_specs(skill_definitions)
    specs.extend(skill_specs)
    action_names.update(skill_definitions.keys())

    # Check for name collisions
    all_names = [s.name for s in specs]
    seen: set[str] = set()
    for name in all_names:
        if name in seen:
            logger.warning("Capability name collision: '%s' appears multiple times", name)
        seen.add(name)

    return CapabilityCatalog(
        specs=specs,
        mcp_tools=mcp_tools,
        skill_definitions=skill_definitions,
        action_names=action_names,
    )


async def discover_all_capabilities(
    mcp_config: MCPConfig,
    skills_dirs: list[str],
) -> CapabilityCatalog:
    """Discover all external capabilities (MCP tools + Skills).

    Connects to configured MCP servers to list tools, and scans
    skills directories for SKILL.md files. Returns a unified catalog.

    This is an async function because MCP tool discovery requires
    connecting to servers.
    """
    from forge.mcp.client import _make_capability_name, discover_tools
    from forge.skills.loader import discover_skills

    # Discover MCP tools from all configured servers
    all_mcp_tools: dict[str, MCPToolInfo] = {}
    for server_name, server_config in mcp_config.mcp_servers.items():
        tools = await discover_tools(server_name, server_config)
        for tool in tools:
            cap_name = _make_capability_name(server_name, tool.tool_name)
            all_mcp_tools[cap_name] = tool

    # Merge skills_dirs from config and CLI
    combined_dirs = list(mcp_config.skills_dirs) + skills_dirs
    all_skills = discover_skills(combined_dirs)

    return build_capability_catalog(
        mcp_tools=all_mcp_tools,
        skill_definitions=all_skills,
    )
