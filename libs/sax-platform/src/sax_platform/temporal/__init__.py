"""Temporal plumbing shared by the platform and its consumer apps (T3.4, ST2).

``retries`` and ``heartbeat`` import only ``temporalio.common`` /
``temporalio.activity`` plus pydantic — cheap and workflow/activity-sandbox
safe — so they may be imported eagerly. ``client`` and ``worker`` import
``temporalio.client`` / ``temporalio.worker`` (real connection and worker
machinery), so — mirroring ``sax_platform.llm`` and
``sax_platform.contracts`` — they are exported lazily via PEP 562:
`import sax_platform.temporal` or `from sax_platform.temporal import retries`
inside a Temporal workflow sandbox must never drag in the worker stack.
"""

from typing import TYPE_CHECKING, Any

from sax_platform.temporal.heartbeat import heartbeat_during
from sax_platform.temporal.retries import (
    DB_RETRY,
    IO_RETRY,
    LLM_RETRY,
    PERSIST_RETRY,
    classify_llm_error,
)

if TYPE_CHECKING:
    from sax_platform.temporal.client import (
        TemporalTLSConfigError,
        build_tls_config,
        connect_temporal,
    )
    from sax_platform.temporal.worker import (
        DEFAULT_GRACEFUL_SHUTDOWN,
        DEFAULT_MAX_CONCURRENT_ACTIVITIES,
        build_sandbox_runner,
        run_worker,
        worker_kwargs,
    )

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "TemporalTLSConfigError": ("sax_platform.temporal.client", "TemporalTLSConfigError"),
    "build_tls_config": ("sax_platform.temporal.client", "build_tls_config"),
    "connect_temporal": ("sax_platform.temporal.client", "connect_temporal"),
    "DEFAULT_GRACEFUL_SHUTDOWN": ("sax_platform.temporal.worker", "DEFAULT_GRACEFUL_SHUTDOWN"),
    "DEFAULT_MAX_CONCURRENT_ACTIVITIES": (
        "sax_platform.temporal.worker",
        "DEFAULT_MAX_CONCURRENT_ACTIVITIES",
    ),
    "build_sandbox_runner": ("sax_platform.temporal.worker", "build_sandbox_runner"),
    "run_worker": ("sax_platform.temporal.worker", "run_worker"),
    "worker_kwargs": ("sax_platform.temporal.worker", "worker_kwargs"),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy export of the client/worker-importing surfaces (see module docstring)."""
    try:
        module_name, attr = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module_name), attr)


__all__ = [
    "DB_RETRY",
    "DEFAULT_GRACEFUL_SHUTDOWN",
    "DEFAULT_MAX_CONCURRENT_ACTIVITIES",
    "IO_RETRY",
    "LLM_RETRY",
    "PERSIST_RETRY",
    "TemporalTLSConfigError",
    "build_sandbox_runner",
    "build_tls_config",
    "classify_llm_error",
    "connect_temporal",
    "heartbeat_during",
    "run_worker",
    "worker_kwargs",
]
