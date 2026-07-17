"""Batch submit activity for Forge.

Submits an assembled context to the LLM provider's batch API.

Design follows Function Core / Imperative Shell:
- Testable function: execute_batch_submit (takes provider as argument)
- Imperative shell: submit_batch_request
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING

from sax_llm import get_output_type_registry
from sax_llm.models import text_messages
from temporalio import activity

from forge.message_log import write_message_log

# BatchSubmitResult is instantiated below (runtime); BatchSubmitInput/BatchSubmitSpiInput
# are activity-parameter types Temporal resolves at registration, so all three must be
# real runtime imports (not TYPE_CHECKING-guarded).
from forge.models import (
    BatchSubmitInput,
    BatchSubmitResult,
    BatchSubmitSpiInput,
    CapabilityTier,
    ModelConfig,
    resolve_model,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Protocol

    from sax_llm.protocol import LLMProvider

    class _BlobSubmitProvider(Protocol):
        """Structural contract for the opaque-blob submit SPI.

        Narrower than sax_llm.protocol.LLMProvider (the sync-mode / chat
        members are omitted) so sax_platform.ocr.MistralOcr — which
        implements OCR-only methods, not the full LLMProvider protocol —
        satisfies it too (T3.3: mistral moved out of sax_llm's registry).
        """

        async def submit_batch(
            self, requests: list[dict[str, Any]], model: str, *, endpoint: str = ""
        ) -> str: ...


# Shadow fallback for a missing context.model_name (see forge.activities.llm).
DEFAULT_MODEL = resolve_model(CapabilityTier.GENERATION, ModelConfig())
DEFAULT_MAX_TOKENS = 4096

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


async def execute_batch_submit(
    input: BatchSubmitInput,
    provider: LLMProvider,
) -> BatchSubmitResult:
    """Build and submit a batch request to the LLM provider.

    Separated from the imperative shell so tests can inject a mock provider.
    """
    from sax_llm import parse_model_id

    output_type = None
    if input.output_type_name:
        registry = get_output_type_registry()
        output_type = registry[input.output_type_name]
    full_model = input.context.model_name or DEFAULT_MODEL
    _, model = parse_model_id(full_model)

    params = provider.build_request_params(
        messages=text_messages(input.context.system_prompt, input.context.user_prompt),
        output_type=output_type,
        model=model,
        max_tokens=input.max_tokens,
        thinking_enabled=input.thinking.enabled,
        effort=input.thinking.effort,
    )

    if input.context.log_messages and input.context.worktree_path:
        request_json = json.dumps(params, indent=2, default=str)
        write_message_log(input.context.worktree_path, "request", request_json)

    request_id = str(uuid.uuid4())
    batch_request = provider.build_batch_request(request_id, params)

    batch_id = await provider.submit_batch([batch_request], model)

    return BatchSubmitResult(request_id=request_id, batch_id=batch_id)


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


@activity.defn
async def submit_batch_request(input: BatchSubmitInput) -> BatchSubmitResult:
    """Activity wrapper — creates provider and delegates to execute_batch_submit."""
    from sax_llm import get_provider

    from forge.tracing import get_tracer

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.submit_batch_request") as span:
        logger.info(
            "Batch submitted: task_id=%s output_type=%s",
            input.context.task_id,
            input.output_type_name,
        )
        provider = get_provider(input.context.model_name or DEFAULT_MODEL)
        result = await execute_batch_submit(input, provider)

        span.set_attributes(
            {
                "forge.batch.request_id": result.request_id,
                "forge.batch.batch_id": result.batch_id,
                "forge.batch.output_type": input.output_type_name,
                "forge.batch.workflow_id": input.workflow_id,
            }
        )

        from sax_llm import parse_model_id

        provider_name, _ = parse_model_id(input.context.model_name or DEFAULT_MODEL)
        # Thread the provider back to the workflow, which persists the submission
        # survivably (Phase C) — the activity no longer writes to the store.
        return result.model_copy(update={"provider": provider_name})


# ---------------------------------------------------------------------------
# Opaque-blob submit SPI (Option 1) — the cross-queue batch service
# ---------------------------------------------------------------------------


async def execute_submit_batch_blob(
    input: BatchSubmitSpiInput,
    provider: _BlobSubmitProvider,
    fetch_blob: Callable[[str], bytes],
) -> BatchSubmitResult:
    """Fetch the pre-built request blob and submit it verbatim.

    The platform never parses the body — it treats the blob as opaque, so the
    submit path is domain-agnostic. Writes nothing: the caller records the
    submission separately, so a provider submit and a ``batch_jobs`` write never
    share a re-runnable activity (double-submit safety). Separated from the shell
    so tests can inject a mock provider and a fake blob fetcher.
    """
    import json

    raw = fetch_blob(input.s3_key)
    requests = json.loads(raw.decode("utf-8"))
    batch_id = await provider.submit_batch(requests, input.model, endpoint=input.endpoint)
    return BatchSubmitResult(
        request_id=input.custom_id,
        batch_id=batch_id,
        provider=input.provider,
    )


def _resolve_blob_submit_provider(provider_name: str) -> _BlobSubmitProvider:
    """Resolve the batch-submit provider for the opaque-blob SPI.

    Mistral routes through ``sax_platform.ocr.MistralOcr`` — sax_llm carries no
    Mistral provider (T3.3 moved OCR's Mistral capability to the platform
    library and deleted ``sax_llm.mistral`` entirely). Every other provider
    name still resolves through sax_llm's registry, unchanged.
    """
    if provider_name == "mistral":
        from sax_platform.ocr import MistralOcr, make_mistral_client

        return MistralOcr(make_mistral_client())

    from sax_llm import get_provider_by_name

    return get_provider_by_name(provider_name)


@activity.defn
async def submit_batch_blob(input: BatchSubmitSpiInput) -> BatchSubmitResult:
    """Activity: submit an opaque pre-built request blob to the provider.

    Wires the real provider (by ``input.provider``) and S3 blob fetch. Used by
    consumer apps (e.g. OCR) cross-queue; the generic in-platform path keeps its
    own ``submit_batch_request`` builder.
    """
    from forge_contracts import s3_blobs

    from forge.tracing import get_tracer

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.submit_batch_blob") as span:
        provider = _resolve_blob_submit_provider(input.provider)
        result = await execute_submit_batch_blob(input, provider, s3_blobs.get)
        span.set_attributes(
            {
                "forge.batch.request_id": result.request_id,
                "forge.batch.batch_id": result.batch_id,
                "forge.batch.provider": input.provider,
                "forge.batch.endpoint": input.endpoint or "",
            }
        )
        return result
