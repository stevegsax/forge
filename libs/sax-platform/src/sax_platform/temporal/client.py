"""Single chokepoint for connecting to the Temporal frontend.

Every process — the platform worker/CLI and each consumer app's worker/CLI —
connects through :func:`connect_temporal` so the security-critical TLS / mTLS
configuration and the shared data converter / namespace live in exactly one
place, shared across repos.

By default (``FORGE_TEMPORAL_TLS`` unset) the connection is plaintext — correct
for a worker talking to a co-located Temporal at ``127.0.0.1:7233`` behind the
instance firewall. Remote CLIs reaching the frontend over the internet set
``FORGE_TEMPORAL_TLS=1`` and provide a client certificate/key (mTLS).

Environment variables
---------------------
``FORGE_TEMPORAL_TLS``
    Enable TLS when truthy (``1``/``true``/``yes``/``on``). Unset ⇒ plaintext.
``FORGE_TEMPORAL_TLS_SERVER_CA``
    PEM file holding the CA that signed the server certificate.
``FORGE_TEMPORAL_TLS_CLIENT_CERT`` / ``FORGE_TEMPORAL_TLS_CLIENT_KEY``
    PEM files for this client's certificate and private key (both ⇒ mTLS).
``FORGE_TEMPORAL_TLS_SERVER_NAME``
    Override the expected server name (SNI / certificate name).

NOTE: the ``FORGE_TEMPORAL_TLS_*`` env names are retained from the pre-split
layout; generalizing them is a tracked follow-up.

Ported verbatim from ``forge_contracts.temporal`` (T3.4, ST2) — same env
reads, same behavior. Not yet routed through ``sax_platform.config``; that
wiring is a later task.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from sax_platform.contracts.constants import TEMPORAL_NAMESPACE

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
    - ``True`` for server-only TLS validated against the system trust store.
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


async def connect_temporal(
    address: str,
    *,
    identity: str | None = None,
    namespace: str = TEMPORAL_NAMESPACE,
) -> Client:
    """Connect to Temporal with the shared data converter, namespace, and TLS.

    The one place ``Client.connect`` is called from across repos, so the data
    converter, namespace, and TLS / mTLS are configured identically everywhere.
    ``namespace`` defaults to the shared :data:`TEMPORAL_NAMESPACE` (``"default"``,
    the same namespace used implicitly before the split).
    """
    from temporalio.client import Client
    from temporalio.contrib.pydantic import pydantic_data_converter

    return await Client.connect(
        address,
        namespace=namespace,
        data_converter=pydantic_data_converter,
        tls=build_tls_config(),
        identity=identity,
    )
