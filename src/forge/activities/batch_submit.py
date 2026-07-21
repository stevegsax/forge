"""Batch submit activity for Forge.

Submits an assembled context to the Anthropic Message Batches API via the
platform batch lane (`sax_platform.llm.batch`). Forge submits anthropic only
(T4.2 ST3): the cross-queue opaque-blob submit SPI is gone — a non-anthropic
batch is its owning app's concern.

Design follows Function Core / Imperative Shell:
- Testable function: execute_batch_submit (takes the AsyncAnthropic client as an
  argument)
- Imperative shell: the ``submit_batch_request`` bound method on
  ``BatchActivities`` (forge.activities.roots), which passes the composition-root
  client and output-type registry.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from sax_platform.llm.tiers import split_provider

from forge.message_log import write_message_log

# BatchSubmitResult is instantiated below (runtime); CapabilityTier/ModelConfig/
# resolve_model compute DEFAULT_MODEL at import — all runtime. The input type is
# annotation-only here (the activity param lives on the BatchActivities method).
from forge.models import (
    BatchSubmitResult,
    CapabilityTier,
    ModelConfig,
    resolve_model,
)
from forge.output_types import OUTPUT_TYPES, resolve_output_type

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anthropic import AsyncAnthropic
    from pydantic import BaseModel

    from forge.models import BatchSubmitInput


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
