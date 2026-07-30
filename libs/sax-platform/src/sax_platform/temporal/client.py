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

:func:`build_tls_config` takes an explicit
:class:`~sax_platform.config.TemporalSettings` as its single source — that
settings group is the one place the ``FORGE_TEMPORAL_*`` environment variables
are read. :func:`connect_temporal` accepts the same settings; when a caller
passes none (e.g. a CLI), it constructs a default ``TemporalSettings``, which
reads those env vars via pydantic-settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from temporalio.client import Client
    from temporalio.service import TLSConfig

    from sax_platform.config import TemporalSettings


class TemporalTLSConfigError(RuntimeError):
    """Raised when the Temporal TLS environment configuration is invalid."""


def _read_pem(path: str | None, *, what: str) -> bytes | None:
    """Read a PEM file as bytes, or return ``None`` when no path is set."""
    if not path:
        return None
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise TemporalTLSConfigError(f"Cannot read {what} at {path!r}: {exc}") from exc


def _assemble_tls_config(
    *,
    tls_enabled: bool,
    server_ca_path: str | None,
    client_cert_path: str | None,
    client_key_path: str | None,
    server_name: str | None,
) -> TLSConfig | bool:
    """Assemble the ``tls=`` value from already-resolved string inputs.

    The validation semantics :func:`build_tls_config` relies on: half a
    cert/key pair raises; TLS on with no custom material validates against the
    system trust store; otherwise a :class:`~temporalio.service.TLSConfig` is
    built from the PEM files.
    """
    if not tls_enabled:
        return False

    server_ca = _read_pem(server_ca_path, what="server CA cert")
    client_cert = _read_pem(client_cert_path, what="client cert")
    client_key = _read_pem(client_key_path, what="client key")
    server_name = server_name or None

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


def build_tls_config(settings: TemporalSettings) -> TLSConfig | bool:
    """Build the ``tls=`` argument for ``Client.connect``.

    Returns one of:

    - ``False`` when TLS is disabled (plaintext).
    - ``True`` for server-only TLS validated against the system trust store.
    - a :class:`temporalio.service.TLSConfig` when a private server CA and/or
      a client certificate (mTLS) is supplied.

    ``settings`` is the single source: ``settings.tls`` is a pydantic bool
    coerced at settings construction, and the ``FORGE_TEMPORAL_*`` environment
    variables are read only by :class:`~sax_platform.config.TemporalSettings`.

    Raises:
        TemporalTLSConfigError: if exactly one of the client cert/key pair is
            supplied, or a referenced PEM file cannot be read.
    """
    return _assemble_tls_config(
        tls_enabled=settings.tls,
        server_ca_path=settings.tls_server_ca,
        client_cert_path=settings.tls_client_cert,
        client_key_path=settings.tls_client_key,
        server_name=settings.tls_server_name,
    )


async def connect_temporal(
    address: str,
    *,
    namespace: str,
    identity: str | None = None,
    settings: TemporalSettings | None = None,
) -> Client:
    """Connect to Temporal with the shared data converter, namespace, and TLS.

    The one place ``Client.connect`` is called from across repos, so the data
    converter, namespace, and TLS / mTLS are configured identically everywhere.

    ``namespace`` is required and has no default: it comes from
    :func:`~sax_platform.config.resolve_temporal_target`, derived from the
    declared environment. A default here would be a way to reach a namespace
    without declaring an environment, which is exactly what the derivation
    exists to prevent.

    ``settings`` is passed through to :func:`build_tls_config`. Workers hand in
    their already-built ``settings.temporal``; when a caller passes ``None`` (a
    CLI), a default :class:`~sax_platform.config.TemporalSettings` is constructed,
    which reads the ``FORGE_TEMPORAL_*`` env vars via pydantic-settings — the one
    sanctioned env-reading path.
    """
    from temporalio.client import Client
    from temporalio.contrib.pydantic import pydantic_data_converter

    from sax_platform.config import TemporalSettings

    resolved = settings if settings is not None else TemporalSettings()
    return await Client.connect(
        address,
        namespace=namespace,
        data_converter=pydantic_data_converter,
        tls=build_tls_config(resolved),
        identity=identity,
    )
