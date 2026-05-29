"""Single chokepoint for connecting to the Temporal frontend.

Every Forge process — each CLI command and the worker — connects to
Temporal through :func:`connect_temporal` so that the security-critical
TLS / mTLS configuration lives in exactly one place.

By default (``FORGE_TEMPORAL_TLS`` unset) the connection is plaintext.
That is correct for a worker talking to a co-located Temporal at
``127.0.0.1:7233`` behind the instance firewall. Remote CLIs that reach
the frontend over the internet set ``FORGE_TEMPORAL_TLS=1`` and provide a
client certificate/key, authenticating with mutual TLS (mTLS) — the
mechanism that gates access so only holders of a CA-signed certificate
can connect.

Environment variables
---------------------
``FORGE_TEMPORAL_TLS``
    Enable TLS when truthy (``1``/``true``/``yes``/``on``). Unset ⇒ plaintext.
``FORGE_TEMPORAL_TLS_SERVER_CA``
    PEM file holding the CA that signed the server certificate. Required
    when the server uses a private/internal CA (the usual case here).
``FORGE_TEMPORAL_TLS_CLIENT_CERT`` / ``FORGE_TEMPORAL_TLS_CLIENT_KEY``
    PEM files for this client's certificate and private key. Supplying
    both turns on mTLS; they must be provided together.
``FORGE_TEMPORAL_TLS_SERVER_NAME``
    Override the expected server name (SNI / certificate name) when the
    address dialed differs from the certificate's SAN.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from temporalio.client import Client
    from temporalio.service import TLSConfig

_TRUTHY = {"1", "true", "yes", "on"}


class TemporalTLSConfigError(RuntimeError):
    """Raised when the Temporal TLS environment configuration is invalid."""


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _TRUTHY


def _read_pem(path: str | None, *, what: str) -> bytes | None:
    """Read a PEM file as bytes, or return ``None`` when no path is set."""
    if not path:
        return None
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise TemporalTLSConfigError(f"Cannot read {what} at {path!r}: {exc}") from exc


def build_tls_config() -> TLSConfig | bool:
    """Build the ``tls=`` argument for ``Client.connect`` from the environment.

    Returns one of:

    - ``False`` when ``FORGE_TEMPORAL_TLS`` is unset/falsey (plaintext).
    - ``True`` for server-only TLS validated against the system trust store
      (TLS enabled, but no custom CA, client cert, or server-name override).
    - a :class:`temporalio.service.TLSConfig` when a private server CA and/or
      a client certificate (mTLS) is supplied.

    Raises:
        TemporalTLSConfigError: if exactly one of the client cert/key pair is
            supplied, or a referenced PEM file cannot be read.
    """
    if not _truthy(os.environ.get("FORGE_TEMPORAL_TLS")):
        return False

    server_ca = _read_pem(os.environ.get("FORGE_TEMPORAL_TLS_SERVER_CA"), what="server CA cert")
    client_cert = _read_pem(os.environ.get("FORGE_TEMPORAL_TLS_CLIENT_CERT"), what="client cert")
    client_key = _read_pem(os.environ.get("FORGE_TEMPORAL_TLS_CLIENT_KEY"), what="client key")
    server_name = os.environ.get("FORGE_TEMPORAL_TLS_SERVER_NAME") or None

    if (client_cert is None) != (client_key is None):
        raise TemporalTLSConfigError(
            "mTLS requires both FORGE_TEMPORAL_TLS_CLIENT_CERT and "
            "FORGE_TEMPORAL_TLS_CLIENT_KEY to be set (only one was provided)."
        )

    # TLS on, but nothing custom: validate against the system trust store.
    if server_ca is None and client_cert is None and server_name is None:
        return True

    from temporalio.service import TLSConfig

    return TLSConfig(
        server_root_ca_cert=server_ca,
        client_cert=client_cert,
        client_private_key=client_key,
        domain=server_name,
    )


async def connect_temporal(address: str, *, identity: str | None = None) -> Client:
    """Connect to Temporal with Forge's data converter and TLS settings.

    The one place ``Client.connect`` is called from across Forge, so TLS /
    mTLS is configured identically for every CLI command and the worker.
    """
    from temporalio.client import Client
    from temporalio.contrib.pydantic import pydantic_data_converter

    kwargs: dict[str, object] = {
        "data_converter": pydantic_data_converter,
        "tls": build_tls_config(),
    }
    if identity is not None:
        kwargs["identity"] = identity
    return await Client.connect(address, **kwargs)
