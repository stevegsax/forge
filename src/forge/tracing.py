"""OpenTelemetry tracing setup for Forge.

Provides functions to configure and manage distributed tracing across the
Forge workflow. Each activity creates child spans, and the workflow creates
the root span.

Design follows Function Core / Imperative Shell:
- Pure functions: build_resource, resolve_exporter_type, llm_call_attributes,
  validation_attributes — compute plain dicts, no OTel SDK imports.
- Imperative shell (internal): _create_tracer_provider — builds a provider
  without setting it globally, enabling isolated testing.
- Imperative shell (public): init_tracing, get_tracer, shutdown_tracing.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import Tracer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVICE_NAME = "forge"
SERVICE_VERSION = "0.1.0"

FORGE_OTEL_EXPORTER_ENV = "FORGE_OTEL_EXPORTER"
FORGE_OTEL_ENDPOINT_ENV = "FORGE_OTEL_ENDPOINT"


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class ExporterType(Enum):
    """Supported trace exporter backends."""

    CONSOLE = "console"
    OTLP_GRPC = "otlp_grpc"
    OTLP_HTTP = "otlp_http"
    NONE = "none"


# ---------------------------------------------------------------------------
# Pure functions (no OTel SDK imports)
# ---------------------------------------------------------------------------


def build_resource(
    service_name: str = SERVICE_NAME,
    service_version: str = SERVICE_VERSION,
    extra_attributes: dict[str, str] | None = None,
) -> dict[str, str]:
    """Compute resource attributes as a plain dict.

    ``service.name`` and ``service.version`` are always set from the explicit
    parameters, even if *extra_attributes* contains those keys.
    """
    attrs: dict[str, str] = {}
    if extra_attributes:
        attrs.update(extra_attributes)
    # Explicit params always win over extras.
    attrs["service.name"] = service_name
    attrs["service.version"] = service_version
    return attrs


def resolve_exporter_type(exporter: ExporterType | None = None) -> ExporterType:
    """Determine the exporter type.

    Resolution order:

    1. Explicit *exporter* parameter.
    2. ``FORGE_OTEL_EXPORTER`` environment variable.
    3. Default: ``ExporterType.CONSOLE``.

    Raises:
        ValueError: If the environment variable contains an unrecognised value.
    """
    if exporter is not None:
        return exporter

    env_value = os.environ.get(FORGE_OTEL_EXPORTER_ENV)
    if env_value is not None:
        try:
            return ExporterType(env_value)
        except ValueError:
            valid = ", ".join(e.value for e in ExporterType)
            msg = f"Invalid {FORGE_OTEL_EXPORTER_ENV} value {env_value!r}. Valid options: {valid}"
            raise ValueError(msg) from None

    return ExporterType.CONSOLE


def llm_call_attributes(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    task_id: str | None = None,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> dict[str, str | int | float]:
    """Build ``forge.llm.*`` span attributes for an LLM call."""
    attrs: dict[str, str | int | float] = {
        "forge.llm.model": model_name,
        "forge.llm.input_tokens": input_tokens,
        "forge.llm.output_tokens": output_tokens,
        "forge.llm.latency_ms": latency_ms,
    }
    if task_id is not None:
        attrs["forge.task_id"] = task_id
    if cache_creation_input_tokens:
        attrs["forge.llm.cache_creation_input_tokens"] = cache_creation_input_tokens
    if cache_read_input_tokens:
        attrs["forge.llm.cache_read_input_tokens"] = cache_read_input_tokens
    return attrs


def validation_attributes(
    checks: list[tuple[str, bool]],
    task_id: str | None = None,
) -> dict[str, str | int | bool]:
    """Build ``forge.validation.*`` span attributes from check results.

    Args:
        checks: List of ``(check_name, passed)`` tuples.
        task_id: Optional task identifier.
    """
    attrs: dict[str, str | int | bool] = {
        "forge.validation.check_count": len(checks),
        "forge.validation.pass_count": sum(1 for _, passed in checks if passed),
    }
    for name, passed in checks:
        attrs[f"forge.validation.{name}.passed"] = passed
    if task_id is not None:
        attrs["forge.task_id"] = task_id
    return attrs


# ---------------------------------------------------------------------------
# Imperative shell — internal
# ---------------------------------------------------------------------------


def _create_tracer_provider(
    resource_attrs: dict[str, str],
    exporter_type: ExporterType,
    endpoint: str | None = None,
) -> TracerProvider:
    """Build a ``TracerProvider`` without setting it globally.

    This enables tests to inspect providers in isolation.
    """
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    resource = Resource.create(resource_attrs)
    provider = TracerProvider(resource=resource)

    if exporter_type is ExporterType.CONSOLE:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    elif exporter_type is ExporterType.OTLP_GRPC:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        exporter_grpc = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter_grpc))

    elif exporter_type is ExporterType.OTLP_HTTP:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        exporter_http = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter_http))

    # ExporterType.NONE — no processors added (no-op provider).

    return provider


# ---------------------------------------------------------------------------
# Imperative shell — public
# ---------------------------------------------------------------------------


def init_tracing(
    exporter: ExporterType | None = None,
    endpoint: str | None = None,
    service_name: str = SERVICE_NAME,
    service_version: str = SERVICE_VERSION,
) -> None:
    """Create and globally register a ``TracerProvider``.

    Safe to call multiple times — each call replaces the previous provider
    (after shutting it down). Resets the OTel set-once guard so the new
    provider is accepted.
    """
    from opentelemetry import trace

    resource_attrs = build_resource(service_name, service_version)
    exporter_type = resolve_exporter_type(exporter)
    provider = _create_tracer_provider(resource_attrs, exporter_type, endpoint)

    # Shut down any previously registered SDK provider.
    current = trace.get_tracer_provider()
    if hasattr(current, "shutdown"):
        current.shutdown()

    # Reset the set-once guard so set_tracer_provider accepts the new provider.
    once = trace._TRACER_PROVIDER_SET_ONCE
    with once._lock:
        once._done = False

    trace.set_tracer_provider(provider)


def get_tracer(name: str = "forge") -> Tracer:
    """Return a tracer from the globally registered provider.

    If ``init_tracing`` has not been called, the default no-op provider is
    used — calls will succeed but produce no spans.
    """
    from opentelemetry import trace

    return trace.get_tracer(name)


def shutdown_tracing() -> None:
    """Flush pending spans and shut down the global tracer provider.

    Safe to call even if ``init_tracing`` was never called.
    """
    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()
