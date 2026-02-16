"""Tests for MCP configuration loading (Phase 15)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from forge.mcp.config import (
    DEFAULT_CONFIG_FILENAME,
    interpolate_env,
    load_mcp_config,
    parse_mcp_config,
)
from forge.models import MCPConfig


class TestInterpolateEnv:
    def test_no_variables(self):
        assert interpolate_env("hello world") == "hello world"

    def test_single_variable(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret123")
        assert interpolate_env("${MY_KEY}") == "secret123"

    def test_multiple_variables(self, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "8080")
        assert interpolate_env("${HOST}:${PORT}") == "localhost:8080"

    def test_missing_variable_returns_empty(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        assert interpolate_env("${NONEXISTENT_VAR}") == ""

    def test_mixed_text_and_variables(self, monkeypatch):
        monkeypatch.setenv("TOKEN", "abc")
        assert interpolate_env("Bearer ${TOKEN}") == "Bearer abc"


class TestParseMcpConfig:
    def test_empty_config(self):
        config = parse_mcp_config({})
        assert config.mcp_servers == {}
        assert config.skills_dirs == []

    def test_single_server(self):
        raw = {
            "mcpServers": {
                "test-server": {
                    "command": "npx",
                    "args": ["-y", "@test/mcp-server"],
                    "env": {},
                }
            }
        }
        config = parse_mcp_config(raw)
        assert "test-server" in config.mcp_servers
        server = config.mcp_servers["test-server"]
        assert server.command == "npx"
        assert server.args == ["-y", "@test/mcp-server"]
        assert server.env == {}

    def test_multiple_servers(self):
        raw = {
            "mcpServers": {
                "server-a": {"command": "python", "args": ["server_a.py"]},
                "server-b": {"command": "node", "args": ["server_b.js"]},
            }
        }
        config = parse_mcp_config(raw)
        assert len(config.mcp_servers) == 2

    def test_env_interpolation(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "test-key-123")
        raw = {
            "mcpServers": {
                "api-server": {
                    "command": "npx",
                    "args": [],
                    "env": {"API_KEY": "${API_KEY}"},
                }
            }
        }
        config = parse_mcp_config(raw)
        assert config.mcp_servers["api-server"].env["API_KEY"] == "test-key-123"

    def test_skills_dirs(self):
        raw = {"skillsDirs": ["./skills", ".forge/skills"]}
        config = parse_mcp_config(raw)
        assert config.skills_dirs == ["./skills", ".forge/skills"]

    def test_missing_command_skipped(self):
        raw = {
            "mcpServers": {
                "bad-server": {"args": ["test"]},
            }
        }
        config = parse_mcp_config(raw)
        assert len(config.mcp_servers) == 0


class TestLoadMcpConfig:
    def test_no_config_file(self, tmp_path):
        config = load_mcp_config(project_root=tmp_path)
        assert config == MCPConfig()

    def test_load_from_default_location(self, tmp_path):
        config_data = {
            "mcpServers": {
                "test": {"command": "echo", "args": ["hello"]},
            }
        }
        config_file = tmp_path / DEFAULT_CONFIG_FILENAME
        config_file.write_text(json.dumps(config_data))

        config = load_mcp_config(project_root=tmp_path)
        assert "test" in config.mcp_servers

    def test_load_from_explicit_path(self, tmp_path):
        config_data = {
            "mcpServers": {
                "custom": {"command": "test-cmd"},
            }
        }
        custom_path = tmp_path / "custom-mcp.json"
        custom_path.write_text(json.dumps(config_data))

        config = load_mcp_config(config_path=custom_path)
        assert "custom" in config.mcp_servers

    def test_invalid_json_returns_empty(self, tmp_path):
        config_file = tmp_path / DEFAULT_CONFIG_FILENAME
        config_file.write_text("not valid json{{{")

        config = load_mcp_config(project_root=tmp_path)
        assert config == MCPConfig()
