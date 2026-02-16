"""MCP configuration loading and validation (Phase 15).

Loads forge-mcp.json configuration files. Supports environment variable
interpolation in server configurations (${VAR} syntax).

Design follows Function Core / Imperative Shell:
- Pure functions: interpolate_env, parse_mcp_config
- I/O shell: load_mcp_config (reads file)
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from forge.models import MCPConfig, MCPServerConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILENAME = "forge-mcp.json"

# Match ${VAR_NAME} patterns for environment variable interpolation
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def interpolate_env(value: str) -> str:
    """Replace ${VAR} patterns with environment variable values.

    Missing variables are replaced with empty strings and a warning is logged.
    """

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        result = os.environ.get(var_name)
        if result is None:
            logger.warning("Environment variable %s not set, using empty string", var_name)
            return ""
        return result

    return _ENV_VAR_PATTERN.sub(_replace, value)


def parse_mcp_config(raw: dict) -> MCPConfig:
    """Parse a raw JSON dict into an MCPConfig, applying env interpolation.

    Expected format (Claude Desktop convention):

    .. code-block:: json

        {
            "mcpServers": {
                "server-name": {
                    "command": "npx",
                    "args": ["-y", "@some/mcp-server"],
                    "env": {"API_KEY": "${API_KEY}"}
                }
            },
            "skillsDirs": ["./skills", ".forge/skills"]
        }
    """
    servers: dict[str, MCPServerConfig] = {}

    for name, server_raw in raw.get("mcpServers", {}).items():
        if not isinstance(server_raw, dict):
            logger.warning(
                "Skipping MCP server %s: expected object, got %s", name, type(server_raw)
            )
            continue

        command = server_raw.get("command", "")
        if not command:
            logger.warning("Skipping MCP server %s: missing 'command'", name)
            continue

        args = [str(a) for a in server_raw.get("args", [])]
        env_raw = server_raw.get("env", {})
        env = {k: interpolate_env(str(v)) for k, v in env_raw.items()}

        servers[name] = MCPServerConfig(command=command, args=args, env=env)

    skills_dirs = [str(d) for d in raw.get("skillsDirs", [])]

    return MCPConfig(mcp_servers=servers, skills_dirs=skills_dirs)


def load_mcp_config(
    config_path: str | Path | None = None, project_root: str | Path = "."
) -> MCPConfig:
    """Load MCP configuration from a JSON file.

    If config_path is None, looks for forge-mcp.json in the project root.
    Returns an empty MCPConfig if no config file is found.
    """
    if config_path is not None:
        path = Path(config_path)
    else:
        path = Path(project_root) / DEFAULT_CONFIG_FILENAME

    if not path.is_file():
        logger.debug("No MCP config found at %s", path)
        return MCPConfig()

    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read MCP config at %s: %s", path, e)
        return MCPConfig()

    logger.info("Loaded MCP config from %s", path)
    return parse_mcp_config(raw)
