"""Batch submit activity for Forge.

Submits an assembled context to the Anthropic Message Batches API via the
platform batch lane (`sax_platform.llm.batch`).

Design follows Function Core / Imperative Shell:
- Testable functions: execute_batch_submit / execute_submit_batch_blob (take the
  AsyncAnthropic client / provider as arguments)
- Imperative shell: the ``submit_batch_request`` / ``submit_batch_blob`` bound
  methods on ``BatchActivities`` (forge.activities.roots), which pass the
  composition-root client, output-type registry, blob store, and mistral client.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from sax_platform.llm.tiers import split_provider

from forge.message_log import write_message_log

# BatchSubmitResult is instantiated below (runtime); CapabilityTier/ModelConfig/
# resolve_model compute DEFAULT_MODEL at import — all runtime. The input types are
# annotation-only here (the activity params live on the BatchActivities methods).
from forge.models import (
    BatchSubmitResult,
    CapabilityTier,
    ModelConfig,
    resolve_model,
)
from forge.output_types import OUTPUT_TYPES, resolve_output_type

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any, Protocol

    from anthropic import AsyncAnthropic
    from pydantic import BaseModel
    from sax_platform.ocr import MistralOcr

    from forge.models import BatchSubmitInput, BatchSubmitSpiInput

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
# Testable function
# ---------------------------------------------------------------------------


async def execute_batch_submit(
    input: BatchSubmitInput,
    client: AsyncAnthropic,
    output_types: Mapping[str, type[BaseModel]] = OUTPUT_TYPES,
) -> BatchSubmitResult:
    """Build and submit a batch request via the platform batch lane.

    The AsyncAnthropic *client* and the *output_types* registry are injected by
    the ``BatchActivities`` composition root (``output_types`` defaults to the
    frozen ``OUTPUT_TYPES`` for direct callers); separated from the imperative
    shell so tests can inject the client and a mocked ``submit_batch``.

    The provider custom_id is ``input.request_id``, minted in the workflow via
    ``workflow.uuid4()`` (D88): a retried submit reuses the same custom_id and the
    provider dedupes to one paid batch, closing the submit-retry orphan window.
    """
    output_type = (
        resolve_output_type(input.output_type_name, output_types)
        if input.output_type_name
        else None
    )
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

    request_id = input.request_id
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


def _resolve_blob_submit_provider(
    provider_name: str,
    *,
    client: AsyncAnthropic,
    mistral_ocr: MistralOcr | None,
) -> _BlobSubmitProvider:
    """Resolve the batch-submit provider for the opaque-blob SPI.

    The composition root supplies both the AsyncAnthropic *client* and the
    optional *mistral_ocr* (``BatchActivities`` state). Mistral routes through
    the injected ``MistralOcr`` — a ``None`` here means ``MISTRAL_API_KEY`` was
    unset at worker startup, so it raises a clear error. Every other provider
    name is Anthropic's Message Batches API, submitted through the platform batch
    lane via a thin ``_AnthropicBlobSubmit`` adapter over the shared client.
    """
    if provider_name == "mistral":
        if mistral_ocr is None:
            msg = (
                "mistral batch submit requires MISTRAL_API_KEY to be set at worker "
                "startup (no MistralOcr was constructed)."
            )
            raise RuntimeError(msg)
        return mistral_ocr

    return _AnthropicBlobSubmit(client)
