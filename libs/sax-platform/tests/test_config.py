"""Tests for sax_platform.config — the frozen env-reading settings groups.

These groups are not wired into any app yet (T3.6); this file only exercises
the mechanism: env values land in the right fields, defaults apply when a var
is unset, instances are frozen, and ``DbSettings`` requires ``FORGE_DB_URL``.
"""

import pytest
from pydantic import ValidationError

from sax_platform.config import (
    BlobSettings,
    DbSettings,
    LlmSettings,
    LogSettings,
    TemporalSettings,
)

_ALL_ENV_VARS = (
    "FORGE_TEMPORAL_ADDRESS",
    "FORGE_TEMPORAL_TLS",
    "FORGE_TEMPORAL_TLS_SERVER_CA",
    "FORGE_TEMPORAL_TLS_CLIENT_CERT",
    "FORGE_TEMPORAL_TLS_CLIENT_KEY",
    "FORGE_TEMPORAL_TLS_SERVER_NAME",
    "FORGE_DB_URL",
    "FORGE_OCR_S3_BUCKET",
    "FORGE_OCR_S3_PREFIX",
    "MISTRAL_API_KEY",
    "OPENAI_API_KEY",
    "FORGE_LOG_DIR",
    "PBOOK_LOG_PATH",
    "XDG_STATE_HOME",
)


@pytest.fixture(autouse=True)
def _clear_ambient_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate every test from whatever the ambient shell env happens to hold.

    The ambient environment on this machine points at production (see
    CLAUDE.md), so tests must never inherit it — each test sets exactly the
    vars it asserts on.
    """
    for var in _ALL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestTemporalSettings:
    def test_env_values_read_into_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_TEMPORAL_ADDRESS", "temporal.example.com:7233")
        monkeypatch.setenv("FORGE_TEMPORAL_TLS", "true")
        monkeypatch.setenv("FORGE_TEMPORAL_TLS_SERVER_CA", "/etc/temporal/ca.pem")
        monkeypatch.setenv("FORGE_TEMPORAL_TLS_CLIENT_CERT", "/etc/temporal/cert.pem")
        monkeypatch.setenv("FORGE_TEMPORAL_TLS_CLIENT_KEY", "/etc/temporal/key.pem")
        monkeypatch.setenv("FORGE_TEMPORAL_TLS_SERVER_NAME", "temporal.internal")

        settings = TemporalSettings()

        assert settings.address == "temporal.example.com:7233"
        assert settings.tls is True
        assert settings.tls_server_ca == "/etc/temporal/ca.pem"
        assert settings.tls_client_cert == "/etc/temporal/cert.pem"
        assert settings.tls_client_key == "/etc/temporal/key.pem"
        assert settings.tls_server_name == "temporal.internal"

    @pytest.mark.parametrize("value", ["1", "yes", "on", "TRUE"])
    def test_tls_truthy_values_parse_true(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("FORGE_TEMPORAL_TLS", value)
        assert TemporalSettings().tls is True

    @pytest.mark.parametrize("value", ["0", "no", "off", "false"])
    def test_tls_falsey_values_parse_false(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("FORGE_TEMPORAL_TLS", value)
        assert TemporalSettings().tls is False

    def test_defaults_apply_when_unset(self) -> None:
        settings = TemporalSettings()

        assert settings.address == "localhost:7233"
        assert settings.tls is False
        assert settings.tls_server_ca is None
        assert settings.tls_client_cert is None
        assert settings.tls_client_key is None
        assert settings.tls_server_name is None

    def test_frozen_instance_rejects_mutation(self) -> None:
        settings = TemporalSettings()

        with pytest.raises(ValidationError, match="frozen"):
            settings.address = "mutated:7233"  # type: ignore[misc]


class TestDbSettings:
    def test_env_value_read_into_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_DB_URL", "postgresql://user@host/db")

        assert DbSettings().url == "postgresql://user@host/db"

    def test_missing_env_var_raises(self) -> None:
        with pytest.raises(ValidationError, match="FORGE_DB_URL"):
            DbSettings()

    def test_frozen_instance_rejects_mutation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_DB_URL", "postgresql://user@host/db")
        settings = DbSettings()

        with pytest.raises(ValidationError, match="frozen"):
            settings.url = "sqlite:///mutated.db"  # type: ignore[misc]


class TestBlobSettings:
    def test_env_values_read_into_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_OCR_S3_BUCKET", "forge-ocr-bucket")
        monkeypatch.setenv("FORGE_OCR_S3_PREFIX", "ocr/")

        settings = BlobSettings()

        assert settings.bucket == "forge-ocr-bucket"
        assert settings.prefix == "ocr/"

    def test_defaults_apply_when_unset(self) -> None:
        settings = BlobSettings()

        assert settings.bucket is None
        assert settings.prefix == ""

    def test_frozen_instance_rejects_mutation(self) -> None:
        settings = BlobSettings()

        with pytest.raises(ValidationError, match="frozen"):
            settings.prefix = "mutated/"  # type: ignore[misc]


class TestLlmSettings:
    def test_env_values_read_into_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "mistral-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

        settings = LlmSettings()

        assert settings.mistral_api_key == "mistral-secret"
        assert settings.openai_api_key == "openai-secret"

    def test_defaults_apply_when_unset(self) -> None:
        settings = LlmSettings()

        assert settings.mistral_api_key is None
        assert settings.openai_api_key is None

    def test_frozen_instance_rejects_mutation(self) -> None:
        settings = LlmSettings()

        with pytest.raises(ValidationError, match="frozen"):
            settings.mistral_api_key = "mutated"  # type: ignore[misc]


class TestLogSettings:
    def test_env_values_read_into_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_LOG_DIR", "/var/log/forge")
        monkeypatch.setenv("PBOOK_LOG_PATH", "/var/log/pbook.log")
        monkeypatch.setenv("XDG_STATE_HOME", "/home/user/.local/state")

        settings = LogSettings()

        assert settings.log_dir == "/var/log/forge"
        assert settings.pbook_log_path == "/var/log/pbook.log"
        assert settings.xdg_state_home == "/home/user/.local/state"

    def test_defaults_apply_when_unset(self) -> None:
        settings = LogSettings()

        assert settings.log_dir is None
        assert settings.pbook_log_path is None
        assert settings.xdg_state_home is None

    def test_frozen_instance_rejects_mutation(self) -> None:
        settings = LogSettings()

        with pytest.raises(ValidationError, match="frozen"):
            settings.log_dir = "/mutated"  # type: ignore[misc]
