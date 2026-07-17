"""LLM client, both lanes: sync (structured outputs) and batch (output_config.format).

The sync lane raises typed failures; the batch lane returns them as values —
both share one classification core (`classify_message`), so refusal and
truncation can never masquerade as schema-validation errors on either lane.
"""

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
from sax_platform.llm.cache import (
    CacheSpec,
    apply_cache_control,
    estimate_tokens,
    min_cacheable_tokens,
)
from sax_platform.llm.client import AnthropicLLM, make_client
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

__all__ = [
    "AnthropicLLM",
    "BatchHandle",
    "BatchItemResult",
    "BatchRequestFailed",
    "BatchStatus",
    "CacheSpec",
    "ClassifiedMessage",
    "Completion",
    "LLMOutcomeError",
    "LLMRefused",
    "LLMSchemaMismatch",
    "LLMTruncated",
    "MismatchOutcome",
    "RefusedOutcome",
    "Telemetry",
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
    "submit_batch",
    "telemetry_from_message",
    "to_json_schema",
    "to_output_format",
]
