"""Tests for the centralized Temporal connection / TLS helper (forge re-export).

``build_tls_config`` takes an explicit ``TemporalSettings`` (its single source);
``connect_temporal`` with no settings constructs a default ``TemporalSettings``,
which reads the ``FORGE_TEMPORAL_*`` env vars via pydantic-settings.
"""

from __future__ import annotations

import pytest
from sax_platform.config import TemporalSettings

from forge.temporal_client import (
    TemporalTLSConfigError,
    build_tls_config,
    connect_temporal,
)

_TLS_VARS = (
    "FORGE_TEMPORAL_TLS",
    "FORGE_TEMPORAL_TLS_SERVER_CA",
    "FORGE_TEMPORAL_TLS_CLIENT_CERT",
    "FORGE_TEMPORAL_TLS_CLIENT_KEY",
    "FORGE_TEMPORAL_TLS_SERVER_NAME",
)


@pytest.fixture(autouse=True)
def _clear_tls_env(monkeypatch):
    """Each test controls the TLS environment explicitly."""
    for var in _TLS_VARS:
        monkeypatch.delenv(var, raising=False)


def test_tls_disabled_when_settings_tls_false():
    assert build_tls_config(TemporalSettings(tls=False)) is False


def test_tls_enabled_server_only_uses_system_roots():
    assert build_tls_config(TemporalSettings(tls=True)) is True


def test_mtls_builds_tlsconfig_from_files(tmp_path):
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


def test_half_mtls_pair_raises(tmp_path):
    cert = tmp_path / "client.pem"
    cert.write_bytes(b"CERT-PEM")
    settings = TemporalSettings(tls=True, tls_client_cert=str(cert))
    with pytest.raises(TemporalTLSConfigError, match="both"):
        build_tls_config(settings)


def test_missing_pem_file_raises(tmp_path):
    settings = TemporalSettings(tls=True, tls_server_ca=str(tmp_path / "nope.pem"))
    with pytest.raises(TemporalTLSConfigError, match="Cannot read"):
        build_tls_config(settings)


async def test_connect_temporal_threads_tls_and_converter(monkeypatch):
    """connect_temporal must pass the data converter and the resolved tls value."""
    import temporalio.client
    from temporalio.contrib.pydantic import pydantic_data_converter

    captured: dict = {}

    async def fake_connect(address, **kwargs):
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
