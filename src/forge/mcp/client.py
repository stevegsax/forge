"""MCP client wrapper for tool discovery and invocation (Phase 15).

Provides async functions for connecting to MCP servers, listing tools,
and calling tools. Each invocation manages its own connection lifecycle
(per-activity pattern, D87).

Design follows Function Core / Imperative Shell:
- Pure functions: format_tool_result, build_tool_specs
- I/O shell: discover_tools, call_tool
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from forge.models import ContextProviderSpec, MCPServerConfig, MCPToolInfo

logger = logging.getLogger(__name__)

MCP_TOOL_TIMEOUT_SECONDS = 60


def _make_capability_name(server_name: str, tool_name: str) -> str:
    """Build a unique capability name for an MCP tool.

    Format: mcp_<server>_<tool> to avoid collisions with providers.
    """
    return f"mcp_{server_name}_{tool_name}"


def _schema_to_param_descriptions(input_schema: dict[str, Any]) -> dict[str, str]:
    """Convert a JSON Schema to a flat param_name -> description map.

    Extracts top-level properties from the schema. Nested objects are
    described as 'JSON object' for simplicity.
    """
    params: dict[str, str] = {}
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    for name, prop in properties.items():
        desc_parts: list[str] = []
        if prop.get("description"):
            desc_parts.append(prop["description"])

        type_str = prop.get("type", "any")
        if isinstance(type_str, list):
            type_str = "/".join(type_str)
        desc_parts.append(f"(type: {type_str})")

        if name in required:
            desc_parts.append("[required]")

        params[name] = " ".join(desc_parts)

    return params


def build_tool_specs(
    server_name: str,
    tools: list[MCPToolInfo],
) -> list[ContextProviderSpec]:
    """Convert discovered MCP tools into ContextProviderSpec entries for the LLM."""
    specs: list[ContextProviderSpec] = []
    for tool in tools:
        capability_name = _make_capability_name(server_name, tool.tool_name)
        params = _schema_to_param_descriptions(tool.input_schema)
        specs.append(
            ContextProviderSpec(
                name=capability_name,
                description=f"[MCP:{server_name}] {tool.description}",
                parameters=params,
            )
        )
    return specs


@asynccontextmanager
async def _mcp_session(server_config: MCPServerConfig):
    """Context manager for an MCP client session over stdio.

    Starts the MCP server process, initializes the session, and yields it.
    Cleans up on exit.
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as e:
        raise ImportError(
            "The 'mcp' package is required for MCP integration. Install it with: pip install mcp"
        ) from e

    # Build environment: inherit current env, overlay server-specific vars
    env = {**os.environ, **server_config.env}

    server_params = StdioServerParameters(
        command=server_config.command,
        args=server_config.args,
        env=env,
    )

    async with (
        stdio_client(server_params) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        yield session


async def discover_tools(
    server_name: str,
    server_config: MCPServerConfig,
) -> list[MCPToolInfo]:
    """Connect to an MCP server and discover its available tools.

    Returns a list of MCPToolInfo with tool metadata.
    Handles connection failures gracefully by returning an empty list.
    """
    try:
        async with _mcp_session(server_config) as session:
            result = await asyncio.wait_for(
                session.list_tools(),
                timeout=MCP_TOOL_TIMEOUT_SECONDS,
            )
            tools: list[MCPToolInfo] = []
            for tool in result.tools:
                tools.append(
                    MCPToolInfo(
                        server_name=server_name,
                        tool_name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema if hasattr(tool, "inputSchema") else {},
                    )
                )
            logger.info("Discovered %d tools from MCP server '%s'", len(tools), server_name)
            return tools
    except ImportError:
        raise
    except Exception as e:
        logger.warning("Failed to discover tools from MCP server '%s': %s", server_name, e)
        return []


async def call_tool(
    server_config: MCPServerConfig,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[str, bool]:
    """Call an MCP tool and return (result_text, success).

    Opens a fresh connection, calls the tool, returns the result, and
    closes the connection. Per-activity lifecycle (D87).
    """
    try:
        async with _mcp_session(server_config) as session:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments=arguments),
                timeout=MCP_TOOL_TIMEOUT_SECONDS,
            )

            # MCP tool results contain a list of content blocks
            parts: list[str] = []
            for content_block in result.content:
                if hasattr(content_block, "text"):
                    parts.append(content_block.text)
                else:
                    parts.append(str(content_block))

            text = "\n".join(parts)
            is_error = getattr(result, "isError", False)
            return text, not is_error

    except TimeoutError:
        return f"Error: MCP tool '{tool_name}' timed out after {MCP_TOOL_TIMEOUT_SECONDS}s", False
    except ImportError:
        raise
    except Exception as e:
        return f"Error calling MCP tool '{tool_name}': {e}", False


def format_tool_result(content: str, success: bool, tool_name: str) -> str:
    """Format an MCP tool result for inclusion in the exploration context."""
    if success:
        return content
    return f"Tool '{tool_name}' failed:\n{content}"
