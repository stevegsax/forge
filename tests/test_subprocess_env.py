"""Tests for forge.subprocess_env — the allowlist for model-influenced seams."""

from __future__ import annotations

import pytest

from forge.subprocess_env import ALLOWED_SUBPROCESS_ENV_KEYS, allowlist_env


class TestAllowlistEnv:
    def test_keeps_only_allowlisted_keys(self) -> None:
        source = {
            "PATH": "/usr/bin",
            "HOME": "/home/forge",
            "VIRTUAL_ENV": "/srv/forge-app/forge/.venv",
            "LANG": "C.UTF-8",
            "TMPDIR": "/tmp/forge",
            "ANTHROPIC_API_KEY": "sk-ant-SENTINEL",
            "FORGE_DB_URL": "postgres://SENTINEL",
            "AWS_SECRET_ACCESS_KEY": "SENTINEL",
        }
        result = allowlist_env(source)
        assert set(result) == set(ALLOWED_SUBPROCESS_ENV_KEYS)

    @pytest.mark.parametrize(
        "secret", ["ANTHROPIC_API_KEY", "FORGE_DB_URL", "AWS_SECRET_ACCESS_KEY"]
    )
    def test_secrets_are_dropped(self, secret: str) -> None:
        result = allowlist_env({"PATH": "/usr/bin", secret: "SENTINEL"})
        assert secret not in result
        assert "SENTINEL" not in result.values()

    def test_absent_keys_are_omitted_not_blanked(self) -> None:
        # Only PATH is present; the other allowlisted keys must not appear as
        # empty placeholders (which would mask an unset var from the child).
        result = allowlist_env({"PATH": "/usr/bin"})
        assert result == {"PATH": "/usr/bin"}

    def test_returns_fresh_dict(self) -> None:
        source = {"PATH": "/usr/bin"}
        result = allowlist_env(source)
        result["PATH"] = "/mutated"
        assert source["PATH"] == "/usr/bin"
