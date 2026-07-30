"""Sandbox-light copy of the forge-contracts wire surface (T3.4, ST1).

``constants``, ``models``, and ``persist`` import only pydantic/temporalio —
cheap and workflow-sandbox-safe — so they are imported eagerly. ``types`` and
``batch_jobs`` import sqlalchemy, and ``s3_blobs`` is where boto3 eventually
gets pulled in, so those three are exported lazily via PEP 562, mirroring
``sax_platform.llm``'s pattern: `import sax_platform.contracts` or
`from sax_platform.contracts import persist` inside a Temporal workflow
sandbox must never drag in sqlalchemy or boto3.

Note: the ``batch_jobs`` module's own table constant is also named
``batch_jobs``. Re-exporting it under the same name here would collide with
the submodule name itself once anything does `import
sax_platform.contracts.batch_jobs` (Python's import machinery overwrites the
package attribute with the module object as a side effect), so the table and
its metadata are reached via the qualified path
(``sax_platform.contracts.batch_jobs.batch_jobs`` /
``sax_platform.contracts.batch_jobs.metadata``) rather than a package-level
re-export.
"""

from typing import TYPE_CHECKING, Any

from sax_platform.contracts.constants import (
    FORGE_TASK_QUEUE,
    OCR_TASK_QUEUE,
    PRODUCT_SLUG,
)
from sax_platform.contracts.models import (
    BatchJobStatus,
    dump_batch_result_payload,
    parse_batch_result_payload,
)
from sax_platform.contracts.persist import (
    PersistBatchFailure,
    PersistBatchOutcome,
    PersistBatchSubmission,
    PersistResult,
    persist_block,
)

if TYPE_CHECKING:
    from sax_platform.contracts.s3_blobs import S3ConfigError
    from sax_platform.contracts.types import UTCDateTime

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "UTCDateTime": ("sax_platform.contracts.types", "UTCDateTime"),
    "S3ConfigError": ("sax_platform.contracts.s3_blobs", "S3ConfigError"),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy export of the sqlalchemy/boto3-importing surfaces (see module docstring)."""
    try:
        module_name, attr = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module_name), attr)


__all__ = [
    "FORGE_TASK_QUEUE",
    "OCR_TASK_QUEUE",
    "PRODUCT_SLUG",
    "BatchJobStatus",
    "PersistBatchFailure",
    "PersistBatchOutcome",
    "PersistBatchSubmission",
    "PersistResult",
    "S3ConfigError",
    "UTCDateTime",
    "dump_batch_result_payload",
    "parse_batch_result_payload",
    "persist_block",
]
