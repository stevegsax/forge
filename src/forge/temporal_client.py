"""Backwards-compatible re-export of the Temporal connection chokepoint.

The connect helper and TLS/mTLS configuration now live in
``sax_platform.temporal.client`` (shared with consumer apps). This module
re-exports them so existing ``from forge.temporal_client import ...`` call
sites keep working unchanged.
"""

from sax_platform.temporal.client import (
    TemporalTLSConfigError,
    build_tls_config,
    connect_temporal,
)

__all__ = ["TemporalTLSConfigError", "build_tls_config", "connect_temporal"]
