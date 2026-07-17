"""Tests for the centralized Temporal connection / TLS helper.

Ported from ``libs/forge-contracts/tests/test_temporal.py`` (T3.4, ST2) with
imports rewritten to ``sax_platform.temporal.client``. ``Client.connect`` is
mocked throughout — no real Temporal frontend connection is ever attempted
from these tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sax_platform.contracts.constants import TEMPORAL_NAMESPACE
from sax_platform.temporal.client import (
    TemporalTLSConfigError,
    build_tls_config,
    connect_temporal,
)

if TYPE_CHECKING:
    from pathlib import Path

_TLS_VARS = (
    "FORGE_TEMPORAL_TLS",
    "FORGE_TEMPORAL_TLS_SERVER_CA",
    "FORGE_TEMPORAL_TLS_CLIENT_CERT",
    "FORGE_TEMPORAL_TLS_CLIENT_KEY",
    "FORGE_TEMPORAL_TLS_SERVER_NAME",
)


@pytest.fixture(autouse=True)
def _clear_tls_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test controls the TLS environment explicitly."""
    for var in _TLS_VARS:
        monkeypatch.delenv(var, raising=False)


class TestBuildTlsConfig:
    def test_disabled_by_default(self) -> None:
        assert build_tls_config() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_enabled_server_only_uses_system_roots(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("FORGE_TEMPORAL_TLS", value)
        assert build_tls_config() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", ""])
    def test_falsey_value_is_plaintext(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("FORGE_TEMPORAL_TLS", value)
        assert build_tls_config() is False

    def test_mtls_builds_tlsconfig_from_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from temporalio.service import TLSConfig

        ca = tmp_path / "ca.pem"
        ca.write_bytes(b"CA-PEM")
        cert = tmp_path / "client.pem"
        cert.write_bytes(b"CERT-PEM")
        key = tmp_path / "client.key"
        key.write_bytes(b"KEY-PEM")

        monkeypatch.setenv("FORGE_TEMPORAL_TLS", "1")
        monkeypatch.setenv("FORGE_TEMPORAL_TLS_SERVER_CA", str(ca))
        monkeypatch.setenv("FORGE_TEMPORAL_TLS_CLIENT_CERT", str(cert))
        monkeypatch.setenv("FORGE_TEMPORAL_TLS_CLIENT_KEY", str(key))
        monkeypatch.setenv("FORGE_TEMPORAL_TLS_SERVER_NAME", "temporal.example.com")

        cfg = build_tls_config()
        assert isinstance(cfg, TLSConfig)
        assert cfg.server_root_ca_cert == b"CA-PEM"
        assert cfg.client_cert == b"CERT-PEM"
        assert cfg.client_private_key == b"KEY-PEM"
        assert cfg.domain == "temporal.example.com"

    def test_server_ca_only_without_client_cert(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from temporalio.service import TLSConfig

        ca = tmp_path / "ca.pem"
        ca.write_bytes(b"CA-PEM")
        monkeypatch.setenv("FORGE_TEMPORAL_TLS", "1")
        monkeypatch.setenv("FORGE_TEMPORAL_TLS_SERVER_CA", str(ca))

        cfg = build_tls_config()
        assert isinstance(cfg, TLSConfig)
        assert cfg.server_root_ca_cert == b"CA-PEM"
        assert cfg.client_cert is None
        assert cfg.client_private_key is None

    def test_half_mtls_pair_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        cert = tmp_path / "client.pem"
        cert.write_bytes(b"CERT-PEM")
        monkeypatch.setenv("FORGE_TEMPORAL_TLS", "1")
        monkeypatch.setenv("FORGE_TEMPORAL_TLS_CLIENT_CERT", str(cert))
        with pytest.raises(TemporalTLSConfigError, match="both"):
            build_tls_config()

    def test_missing_pem_file_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("FORGE_TEMPORAL_TLS", "1")
        monkeypatch.setenv("FORGE_TEMPORAL_TLS_SERVER_CA", str(tmp_path / "nope.pem"))
        with pytest.raises(TemporalTLSConfigError, match="Cannot read"):
            build_tls_config()


class TestConnectTemporal:
    @pytest.mark.asyncio
    async def test_threads_tls_converter_and_namespace_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """connect_temporal must pass the data converter, resolved tls value,
        and default to the shared TEMPORAL_NAMESPACE."""
        import temporalio.client
        from temporalio.contrib.pydantic import pydantic_data_converter

        captured: dict = {}

        async def fake_connect(address: str, **kwargs: object) -> str:
            captured["address"] = address
            captured["kwargs"] = kwargs
            return "FAKE_CLIENT"

        monkeypatch.setattr(temporalio.client.Client, "connect", staticmethod(fake_connect))
        monkeypatch.setenv("FORGE_TEMPORAL_TLS", "0")

        result = await connect_temporal("temporal.example.com:7233", identity="worker-1")

        assert result == "FAKE_CLIENT"
        assert captured["address"] == "temporal.example.com:7233"
        assert captured["kwargs"]["tls"] is False
        assert captured["kwargs"]["data_converter"] is pydantic_data_converter
        assert captured["kwargs"]["identity"] == "worker-1"
        assert captured["kwargs"]["namespace"] == TEMPORAL_NAMESPACE

    @pytest.mark.asyncio
    async def test_namespace_override_is_threaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import temporalio.client

        captured: dict = {}

        async def fake_connect(address: str, **kwargs: object) -> str:
            captured["kwargs"] = kwargs
            return "FAKE_CLIENT"

        monkeypatch.setattr(temporalio.client.Client, "connect", staticmethod(fake_connect))
        monkeypatch.setenv("FORGE_TEMPORAL_TLS", "0")

        await connect_temporal("temporal.example.com:7233", namespace="ocr-namespace")

        assert captured["kwargs"]["namespace"] == "ocr-namespace"
