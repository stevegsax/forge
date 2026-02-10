"""Tests for forge.tracing — OpenTelemetry setup."""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from forge.tracing import (
    FORGE_OTEL_EXPORTER_ENV,
    SERVICE_NAME,
    SERVICE_VERSION,
    ExporterType,
    _create_tracer_provider,
    build_resource,
    get_tracer,
    init_tracing,
    llm_call_attributes,
    resolve_exporter_type,
    shutdown_tracing,
    validation_attributes,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_global_tracer_provider():
    """Reset the global tracer provider before and after each test.

    The OTel SDK uses a set-once guard that prevents subsequent calls to
    ``set_tracer_provider``. We reset the internal ``_done`` flag so each
    test can register its own provider cleanly.
    """
    _force_reset_otel_provider()
    yield
    # Shut down whatever the test registered, then restore the no-op state.
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.shutdown()
    _force_reset_otel_provider()


def _force_reset_otel_provider() -> None:
    """Reset OTel global tracer provider to the default no-op state."""
    # Reset the set-once guard so set_tracer_provider works again.
    once = trace._TRACER_PROVIDER_SET_ONCE
    with once._lock:
        once._done = False

    # Reset the global provider reference to the default proxy.
    trace._TRACER_PROVIDER = None


# ---------------------------------------------------------------------------
# build_resource
# ---------------------------------------------------------------------------


class TestBuildResource:
    def test_default_attributes(self) -> None:
        attrs = build_resource()
        assert attrs["service.name"] == SERVICE_NAME
        assert attrs["service.version"] == SERVICE_VERSION

    def test_custom_service_name(self) -> None:
        attrs = build_resource(service_name="custom-service")
        assert attrs["service.name"] == "custom-service"

    def test_extra_attributes(self) -> None:
        attrs = build_resource(extra_attributes={"environment": "test", "region": "us-east-1"})
        assert attrs["environment"] == "test"
        assert attrs["region"] == "us-east-1"
        assert attrs["service.name"] == SERVICE_NAME

    def test_service_name_not_overwritten_by_extras(self) -> None:
        attrs = build_resource(extra_attributes={"service.name": "should-be-ignored"})
        assert attrs["service.name"] == SERVICE_NAME


# ---------------------------------------------------------------------------
# resolve_exporter_type
# ---------------------------------------------------------------------------


class TestResolveExporterType:
    def test_explicit_param_wins(self) -> None:
        result = resolve_exporter_type(ExporterType.OTLP_GRPC)
        assert result is ExporterType.OTLP_GRPC

    def test_explicit_param_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(FORGE_OTEL_EXPORTER_ENV, "none")
        result = resolve_exporter_type(ExporterType.CONSOLE)
        assert result is ExporterType.CONSOLE

    def test_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(FORGE_OTEL_EXPORTER_ENV, "otlp_grpc")
        result = resolve_exporter_type()
        assert result is ExporterType.OTLP_GRPC

    def test_default_is_console(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(FORGE_OTEL_EXPORTER_ENV, raising=False)
        result = resolve_exporter_type()
        assert result is ExporterType.CONSOLE

    def test_invalid_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(FORGE_OTEL_EXPORTER_ENV, "banana")
        with pytest.raises(ValueError, match="Invalid FORGE_OTEL_EXPORTER"):
            resolve_exporter_type()


# ---------------------------------------------------------------------------
# llm_call_attributes
# ---------------------------------------------------------------------------


class TestLlmCallAttributes:
    def test_expected_keys(self) -> None:
        attrs = llm_call_attributes("claude-3", 100, 200, 1500.5)
        assert attrs == {
            "forge.llm.model": "claude-3",
            "forge.llm.input_tokens": 100,
            "forge.llm.output_tokens": 200,
            "forge.llm.latency_ms": 1500.5,
        }

    def test_with_task_id(self) -> None:
        attrs = llm_call_attributes("claude-3", 100, 200, 1500.5, task_id="task-42")
        assert attrs["forge.task_id"] == "task-42"
        assert attrs["forge.llm.model"] == "claude-3"

    def test_without_task_id(self) -> None:
        attrs = llm_call_attributes("claude-3", 100, 200, 1500.5)
        assert "forge.task_id" not in attrs

    def test_cache_tokens_included_when_nonzero(self) -> None:
        attrs = llm_call_attributes(
            "claude-3",
            100,
            200,
            1500.5,
            cache_creation_input_tokens=500,
            cache_read_input_tokens=1000,
        )
        assert attrs["forge.llm.cache_creation_input_tokens"] == 500
        assert attrs["forge.llm.cache_read_input_tokens"] == 1000

    def test_cache_tokens_omitted_when_zero(self) -> None:
        attrs = llm_call_attributes("claude-3", 100, 200, 1500.5)
        assert "forge.llm.cache_creation_input_tokens" not in attrs
        assert "forge.llm.cache_read_input_tokens" not in attrs

    def test_cache_tokens_partial(self) -> None:
        attrs = llm_call_attributes(
            "claude-3",
            100,
            200,
            1500.5,
            cache_creation_input_tokens=500,
            cache_read_input_tokens=0,
        )
        assert attrs["forge.llm.cache_creation_input_tokens"] == 500
        assert "forge.llm.cache_read_input_tokens" not in attrs


# ---------------------------------------------------------------------------
# validation_attributes
# ---------------------------------------------------------------------------


class TestValidationAttributes:
    def test_check_count(self) -> None:
        checks = [("ruff_lint", True), ("ruff_format", False)]
        attrs = validation_attributes(checks)
        assert attrs["forge.validation.check_count"] == 2

    def test_pass_count(self) -> None:
        checks = [("ruff_lint", True), ("ruff_format", False), ("tests", True)]
        attrs = validation_attributes(checks)
        assert attrs["forge.validation.pass_count"] == 2

    def test_per_check_booleans(self) -> None:
        checks = [("ruff_lint", True), ("ruff_format", False)]
        attrs = validation_attributes(checks)
        assert attrs["forge.validation.ruff_lint.passed"] is True
        assert attrs["forge.validation.ruff_format.passed"] is False

    def test_with_task_id(self) -> None:
        attrs = validation_attributes([("lint", True)], task_id="task-7")
        assert attrs["forge.task_id"] == "task-7"

    def test_without_task_id(self) -> None:
        attrs = validation_attributes([("lint", True)])
        assert "forge.task_id" not in attrs

    def test_empty_checks(self) -> None:
        attrs = validation_attributes([])
        assert attrs["forge.validation.check_count"] == 0
        assert attrs["forge.validation.pass_count"] == 0


# ---------------------------------------------------------------------------
# _create_tracer_provider
# ---------------------------------------------------------------------------


class TestCreateTracerProvider:
    def test_console_creates_valid_provider(self) -> None:
        provider = _create_tracer_provider(build_resource(), ExporterType.CONSOLE)
        assert isinstance(provider, TracerProvider)
        # Console exporter adds one processor.
        assert len(provider._active_span_processor._span_processors) == 1
        provider.shutdown()

    def test_none_creates_provider_with_no_processors(self) -> None:
        provider = _create_tracer_provider(build_resource(), ExporterType.NONE)
        assert isinstance(provider, TracerProvider)
        assert len(provider._active_span_processor._span_processors) == 0
        provider.shutdown()

    def test_resource_attributes_set(self) -> None:
        attrs = build_resource(service_name="test-svc", service_version="9.9.9")
        provider = _create_tracer_provider(attrs, ExporterType.NONE)

        resource_attrs = dict(provider.resource.attributes)
        assert resource_attrs["service.name"] == "test-svc"
        assert resource_attrs["service.version"] == "9.9.9"
        provider.shutdown()


# ---------------------------------------------------------------------------
# init_tracing / get_tracer / shutdown_tracing
# ---------------------------------------------------------------------------


class TestInitTracing:
    def test_sets_global_provider(self) -> None:
        init_tracing(exporter=ExporterType.NONE)
        provider = trace.get_tracer_provider()
        assert isinstance(provider, TracerProvider)

    def test_get_tracer_returns_working_tracer(self) -> None:
        init_tracing(exporter=ExporterType.NONE)
        tracer = get_tracer()
        assert tracer is not None
        # Verify we can start a span without error.
        with tracer.start_as_current_span("test-span"):
            pass


class TestGetTracer:
    def test_returns_noop_tracer_without_init(self) -> None:
        # Should not crash even without init_tracing.
        tracer = get_tracer()
        assert tracer is not None
        # No-op tracer should still allow span creation without error.
        with tracer.start_as_current_span("noop-span"):
            pass


class TestShutdownTracing:
    def test_no_crash_after_init(self) -> None:
        init_tracing(exporter=ExporterType.NONE)
        shutdown_tracing()  # should not raise

    def test_no_crash_without_init(self) -> None:
        shutdown_tracing()  # should not raise


# ---------------------------------------------------------------------------
# Integration: spans flow through InMemorySpanExporter
# ---------------------------------------------------------------------------


class TestSpanExportIntegration:
    def test_spans_with_attributes_exported(self) -> None:
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        provider = _create_tracer_provider(build_resource(), ExporterType.NONE)
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        # Use the provider directly to get a tracer (avoids global state).
        tracer = provider.get_tracer("forge")
        with tracer.start_as_current_span("llm-call") as span:
            attrs = llm_call_attributes("claude-3", 50, 150, 800.0, task_id="task-99")
            for key, value in attrs.items():
                span.set_attribute(key, value)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        exported_span = spans[0]
        assert exported_span.name == "llm-call"
        assert exported_span.attributes["forge.llm.model"] == "claude-3"
        assert exported_span.attributes["forge.llm.input_tokens"] == 50
        assert exported_span.attributes["forge.llm.output_tokens"] == 150
        assert exported_span.attributes["forge.llm.latency_ms"] == 800.0
        assert exported_span.attributes["forge.task_id"] == "task-99"
        provider.shutdown()

    def test_validation_span_attributes_exported(self) -> None:
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        provider = _create_tracer_provider(build_resource(), ExporterType.NONE)
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        tracer = provider.get_tracer("forge")
        checks = [("ruff_lint", True), ("ruff_format", False)]
        with tracer.start_as_current_span("validation") as span:
            attrs = validation_attributes(checks, task_id="task-100")
            for key, value in attrs.items():
                span.set_attribute(key, value)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        exported_span = spans[0]
        assert exported_span.name == "validation"
        assert exported_span.attributes["forge.validation.check_count"] == 2
        assert exported_span.attributes["forge.validation.pass_count"] == 1
        assert exported_span.attributes["forge.validation.ruff_lint.passed"] is True
        assert exported_span.attributes["forge.validation.ruff_format.passed"] is False
        assert exported_span.attributes["forge.task_id"] == "task-100"
        provider.shutdown()
