"""LLM client, both lanes: sync (structured outputs) and batch (output_config.format).

The sync lane raises typed failures; the batch lane returns them as values —
both share one classification core (`classify_message`), so refusal and
truncation can never masquerade as schema-validation errors on either lane.

The client and batch surfaces (which import the ``anthropic`` SDK) are
exported lazily via PEP 562 so that importing the pure layers — tiers,
models, cache, schema — stays cheap and safe inside the Temporal workflow
sandbox. `from sax_platform.llm import CapabilityTier` must never drag in
an HTTP stack.
"""

from typing import TYPE_CHECKING, Any

from sax_platform.llm.cache import (
    CacheSpec,
    apply_cache_control,
    estimate_tokens,
    min_cacheable_tokens,
)
from sax_platform.llm.models import (
    ClassifiedMessage,
    Completion,
    LLMOutcomeError,
    LLMRefused,
    LLMSchemaMismatch,
    LLMTruncated,
    MismatchOutcome,
    RefusedOutcome,
    Telemetry,
    TruncatedOutcome,
    classify_message,
    outcome_from_error,
    telemetry_from_message,
)
from sax_platform.llm.schema import to_json_schema, to_output_format
from sax_platform.llm.tiers import (
    CapabilityTier,
    Effort,
    ModelConfig,
    ThinkingPolicy,
    resolve_model,
    split_provider,
)

if TYPE_CHECKING:
    from sax_platform.llm.batch import (
        BatchHandle,
        BatchItemResult,
        BatchRequestFailed,
        BatchStatus,
        build_batch_request,
        fetch_batch_results,
        get_batch_status,
        submit_batch,
    )
    from sax_platform.llm.client import AnthropicLLM, make_client

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BatchHandle": ("sax_platform.llm.batch", "BatchHandle"),
    "BatchItemResult": ("sax_platform.llm.batch", "BatchItemResult"),
    "BatchRequestFailed": ("sax_platform.llm.batch", "BatchRequestFailed"),
    "BatchStatus": ("sax_platform.llm.batch", "BatchStatus"),
    "build_batch_request": ("sax_platform.llm.batch", "build_batch_request"),
    "fetch_batch_results": ("sax_platform.llm.batch", "fetch_batch_results"),
    "get_batch_status": ("sax_platform.llm.batch", "get_batch_status"),
    "submit_batch": ("sax_platform.llm.batch", "submit_batch"),
    "AnthropicLLM": ("sax_platform.llm.client", "AnthropicLLM"),
    "make_client": ("sax_platform.llm.client", "make_client"),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy export of the SDK-importing surfaces (see module docstring)."""
    try:
        module_name, attr = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module_name), attr)


__all__ = [
    "AnthropicLLM",
    "BatchHandle",
    "BatchItemResult",
    "BatchRequestFailed",
    "BatchStatus",
    "CacheSpec",
    "CapabilityTier",
    "ClassifiedMessage",
    "Completion",
    "Effort",
    "LLMOutcomeError",
    "LLMRefused",
    "LLMSchemaMismatch",
    "LLMTruncated",
    "MismatchOutcome",
    "ModelConfig",
    "RefusedOutcome",
    "Telemetry",
    "ThinkingPolicy",
    "TruncatedOutcome",
    "apply_cache_control",
    "build_batch_request",
    "classify_message",
    "estimate_tokens",
    "fetch_batch_results",
    "get_batch_status",
    "make_client",
    "min_cacheable_tokens",
    "outcome_from_error",
    "resolve_model",
    "split_provider",
    "submit_batch",
    "telemetry_from_message",
    "to_json_schema",
    "to_output_format",
]
