"""Tests for the centralized Temporal connection / TLS helper.

Ported from ``libs/forge-contracts/tests/test_temporal.py`` (T3.4, ST2) with
imports rewritten to ``sax_platform.temporal.client``. ``Client.connect`` is
mocked throughout — no real Temporal frontend connection is ever attempted
from these tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sax_platform.config import TemporalSettings
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


class TestBuildTlsConfigFromSettings:
    """``build_tls_config`` takes an explicit ``TemporalSettings`` as its single
    source. The autouse ``_clear_tls_env`` fixture guarantees the ambient env
    cannot leak into fields left unset, so a ``TemporalSettings`` built with
    explicit fields is fully in control."""

    def test_disabled_when_tls_false(self) -> None:
        settings = TemporalSettings(tls=False)
        assert build_tls_config(settings) is False

    def test_enabled_server_only_uses_system_roots(self) -> None:
        settings = TemporalSettings(tls=True)
        assert build_tls_config(settings) is True

    def test_mtls_builds_tlsconfig_from_files(self, tmp_path: Path) -> None:
        from temporalio.service import TLSConfig

        ca = tmp_path / "ca.pem"
        ca.write_bytes(b"CA-PEM")
        cert = tmp_path / "client.pem"
        cert.write_bytes(b"CERT-PEM")
        key = tmp_path / "client.key"
        key.write_bytes(b"KEY-PEM")

        settings = TemporalSettings(
            tls=True,
            tls_server_ca=str(ca),
            tls_client_cert=str(cert),
            tls_client_key=str(key),
            tls_server_name="temporal.example.com",
        )

        cfg = build_tls_config(settings)
        assert isinstance(cfg, TLSConfig)
        assert cfg.server_root_ca_cert == b"CA-PEM"
        assert cfg.client_cert == b"CERT-PEM"
        assert cfg.client_private_key == b"KEY-PEM"
        assert cfg.domain == "temporal.example.com"

    def test_server_ca_only_without_client_cert(self, tmp_path: Path) -> None:
        from temporalio.service import TLSConfig

        ca = tmp_path / "ca.pem"
        ca.write_bytes(b"CA-PEM")
        settings = TemporalSettings(tls=True, tls_server_ca=str(ca))

        cfg = build_tls_config(settings)
        assert isinstance(cfg, TLSConfig)
        assert cfg.server_root_ca_cert == b"CA-PEM"
        assert cfg.client_cert is None
        assert cfg.client_private_key is None

    def test_half_mtls_pair_raises(self, tmp_path: Path) -> None:
        cert = tmp_path / "client.pem"
        cert.write_bytes(b"CERT-PEM")
        settings = TemporalSettings(tls=True, tls_client_cert=str(cert))
        with pytest.raises(TemporalTLSConfigError, match="both"):
            build_tls_config(settings)

    def test_missing_pem_file_raises(self, tmp_path: Path) -> None:
        settings = TemporalSettings(tls=True, tls_server_ca=str(tmp_path / "nope.pem"))
        with pytest.raises(TemporalTLSConfigError, match="Cannot read"):
            build_tls_config(settings)

    def test_settings_win_over_ambient_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Env says plaintext; the explicit settings object says TLS-on and wins.
        monkeypatch.setenv("FORGE_TEMPORAL_TLS", "0")
        settings = TemporalSettings(tls=True)
        assert build_tls_config(settings) is True


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

    @pytest.mark.asyncio
    async def test_settings_thread_through_to_tls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit TemporalSettings reaches build_tls_config even when the
        ambient env would otherwise force plaintext."""
        import temporalio.client

        captured: dict = {}

        async def fake_connect(address: str, **kwargs: object) -> str:
            captured["kwargs"] = kwargs
            return "FAKE_CLIENT"

        monkeypatch.setattr(temporalio.client.Client, "connect", staticmethod(fake_connect))
        monkeypatch.setenv("FORGE_TEMPORAL_TLS", "0")

        await connect_temporal("temporal.example.com:7233", settings=TemporalSettings(tls=True))

        assert captured["kwargs"]["tls"] is True
