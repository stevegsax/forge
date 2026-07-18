"""Batch submit activity for Forge.

Submits an assembled context to the Anthropic Message Batches API via the
platform batch lane (`sax_platform.llm.batch`).

Design follows Function Core / Imperative Shell:
- Testable function: execute_batch_submit (takes the AsyncAnthropic client as an argument)
- Imperative shell: submit_batch_request
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING

from sax_platform.llm.tiers import split_provider
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
from forge.output_types import resolve_output_type

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Protocol

    from anthropic import AsyncAnthropic

    class _BlobSubmitProvider(Protocol):
        """Structural contract for the opaque-blob submit SPI.

        Both the platform's anthropic batch submit (adapted by
        ``_AnthropicBlobSubmit``) and ``sax_platform.ocr.MistralOcr`` — which
        implements OCR-only methods — satisfy this narrow shape.
        """

        async def submit_batch(
            self, requests: list[dict[str, Any]], model: str, *, endpoint: str = ""
        ) -> str: ...


# Shadow fallback for a missing context.model_name (see forge.activities.llm).
DEFAULT_MODEL = resolve_model(CapabilityTier.GENERATION, ModelConfig())
DEFAULT_MAX_TOKENS = 4096

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-global AsyncAnthropic client for batch submit calls
# ---------------------------------------------------------------------------
#
# The batch lane needs the raw ``AsyncAnthropic`` SDK client (submit_batch takes
# the client as its first argument), not the ``AnthropicLLM`` wrapper that
# ``forge.llm_client.get_llm()`` hands out for the sync lane — hence a small
# local cache here, mirroring that seam's shape, rather than reusing it.

_batch_client: AsyncAnthropic | None = None


def _get_batch_client() -> AsyncAnthropic:
    """Return the process-wide AsyncAnthropic client, building it on first use."""
    global _batch_client
    if _batch_client is None:
        from sax_platform.llm import make_client

        _batch_client = make_client()
    return _batch_client


# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


async def execute_batch_submit(
    input: BatchSubmitInput,
    client: AsyncAnthropic,
) -> BatchSubmitResult:
    """Build and submit a batch request via the platform batch lane.

    Separated from the imperative shell so tests can inject the client and a
    mocked ``submit_batch``.
    """
    output_type = resolve_output_type(input.output_type_name) if input.output_type_name else None
    # Imported here, not at module level: sax_platform.llm.batch loads the
    # anthropic SDK, and forge.activities is chain-imported inside the Temporal
    # workflow sandbox (via workflow-bearing modules importing activity fns).
    from sax_platform.llm.batch import build_batch_request, submit_batch

    full_model = input.context.model_name or DEFAULT_MODEL
    _, model = split_provider(full_model)

    # The retired provider silently dropped `thinking` for haiku-family models
    # (build_thinking_param returned None whenever the model name contained
    # "haiku"). The current API rejects every thinking shape on haiku, so keep
    # that drop here — passing None omits the param entirely, rather than letting
    # the platform builder emit an adaptive/disabled shape haiku would 400 on.
    thinking = None if "haiku" in model else input.thinking

    request_id = str(uuid.uuid4())
    request = build_batch_request(
        request_id,
        model=model,
        max_tokens=input.max_tokens,
        messages=[{"role": "user", "content": input.context.user_prompt}],
        system=input.context.system_prompt,
        output_type=output_type,
        thinking=thinking,
    )

    if input.context.log_messages and input.context.worktree_path:
        request_json = json.dumps(request["params"], indent=2, default=str)
        write_message_log(input.context.worktree_path, "request", request_json)

    handle = await submit_batch(client, [request])

    return BatchSubmitResult(request_id=request_id, batch_id=handle.batch_id)


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


@activity.defn
async def submit_batch_request(input: BatchSubmitInput) -> BatchSubmitResult:
    """Activity wrapper — wires the client and delegates to execute_batch_submit."""
    from forge.tracing import get_tracer

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.submit_batch_request") as span:
        logger.info(
            "Batch submitted: task_id=%s output_type=%s",
            input.context.task_id,
            input.output_type_name,
        )
        client = _get_batch_client()
        result = await execute_batch_submit(input, client)

        span.set_attributes(
            {
                "forge.batch.request_id": result.request_id,
                "forge.batch.batch_id": result.batch_id,
                "forge.batch.output_type": input.output_type_name,
                "forge.batch.workflow_id": input.workflow_id,
            }
        )

        provider_name, _ = split_provider(input.context.model_name or DEFAULT_MODEL)
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
    raw = fetch_blob(input.s3_key)
    requests = json.loads(raw.decode("utf-8"))
    batch_id = await provider.submit_batch(requests, input.model, endpoint=input.endpoint)
    return BatchSubmitResult(
        request_id=input.custom_id,
        batch_id=batch_id,
        provider=input.provider,
    )


class _AnthropicBlobSubmit:
    """Adapts the platform's anthropic batch submit onto the opaque-blob SPI shape.

    The SPI's provider contract is ``submit_batch(requests, model, *, endpoint="")
    -> str`` (shared with ``MistralOcr``). The platform's anthropic submit takes
    only ``(client, requests)`` — the model rides inside each request's ``params``
    and there is no per-submit endpoint — so ``model``/``endpoint`` are accepted
    and ignored here to satisfy that shared shape, returning the new batch id.
    """

    def __init__(self, client: AsyncAnthropic) -> None:
        self._client = client

    async def submit_batch(
        self, requests: list[dict[str, Any]], model: str, *, endpoint: str = ""
    ) -> str:
        # Local import for sandbox safety (see execute_batch_submit); also
        # shadows this method's own name with the platform function.
        from sax_platform.llm.batch import submit_batch

        handle = await submit_batch(self._client, requests)
        return handle.batch_id


def _resolve_blob_submit_provider(provider_name: str) -> _BlobSubmitProvider:
    """Resolve the batch-submit provider for the opaque-blob SPI.

    Mistral routes through the shared lazily-cached ``MistralOcr`` resolver
    (``forge.activities._mistral``). Every other provider name is Anthropic's
    Message Batches API, submitted through the platform batch lane via a thin
    ``_AnthropicBlobSubmit`` adapter over the shared client.
    """
    if provider_name == "mistral":
        from forge.activities._mistral import get_mistral_ocr

        return get_mistral_ocr()

    return _AnthropicBlobSubmit(_get_batch_client())


@activity.defn
async def submit_batch_blob(input: BatchSubmitSpiInput) -> BatchSubmitResult:
    """Activity: submit an opaque pre-built request blob to the provider.

    Wires the real provider (by ``input.provider``) and S3 blob fetch. Used by
    consumer apps (e.g. OCR) cross-queue; the generic in-platform path keeps its
    own ``submit_batch_request`` builder.
    """
    from sax_platform.contracts import s3_blobs

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
