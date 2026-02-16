"""Capability discovery activity for Forge (Phase 15).

Activity that discovers external capabilities (MCP tools and Agent Skills)
and returns the unified capability catalog as serializable data.

This runs as a Temporal activity so that the workflow remains deterministic.
"""

from __future__ import annotations

import logging
from typing import Any

from temporalio import activity

logger = logging.getLogger(__name__)


@activity.defn
async def discover_capabilities(input: dict[str, Any]) -> dict[str, Any]:
    """Activity: discover MCP tools and Agent Skills.

    Input is a plain dict (Temporal serialization) with:
    - mcp_config: serialized MCPConfig
    - skills_dirs: list of directory paths

    Returns a plain dict with:
    - specs: list of serialized ContextProviderSpec
    - mcp_tools: dict of capability_name -> serialized MCPToolInfo
    - skill_definitions: dict of capability_name -> serialized SkillDefinition
    """
    from forge.capabilities import discover_all_capabilities
    from forge.models import MCPConfig

    mcp_config = MCPConfig(**input.get("mcp_config", {}))
    skills_dirs = input.get("skills_dirs", [])

    catalog = await discover_all_capabilities(mcp_config, skills_dirs)

    return {
        "specs": [s.model_dump() for s in catalog.specs],
        "mcp_tools": {k: v.model_dump() for k, v in catalog.mcp_tools.items()},
        "skill_definitions": {k: v.model_dump() for k, v in catalog.skill_definitions.items()},
    }
