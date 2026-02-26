"""Batch submit activity for Forge.

Submits an assembled context to the Anthropic Message Batches API.

Design follows Function Core / Imperative Shell:
- Testable function: execute_batch_submit (takes client as argument)
- Imperative shell: submit_batch_request, _record_submission
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING

from temporalio import activity

from forge.llm_client import build_batch_request, build_messages_params, get_output_type_registry
from forge.message_log import write_message_log
from forge.models import BatchSubmitInput, BatchSubmitResult

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_MAX_TOKENS = 4096

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


async def execute_batch_submit(
    input: BatchSubmitInput,
    client: AsyncAnthropic,
) -> BatchSubmitResult:
    """Build and submit a batch request to the Anthropic API.

    Separated from the imperative shell so tests can inject a mock client.
    """
    registry = get_output_type_registry()
    output_type = registry[input.output_type_name]
    model = input.context.model_name or DEFAULT_MODEL

    params = build_messages_params(
        system_prompt=input.context.system_prompt,
        user_prompt=input.context.user_prompt,
        output_type=output_type,
        model=model,
        max_tokens=input.max_tokens,
        thinking_budget_tokens=input.thinking_budget_tokens,
        thinking_effort=input.thinking_effort,
    )

    if input.context.log_messages and input.context.worktree_path:
        request_json = json.dumps(params, indent=2, default=str)
        write_message_log(input.context.worktree_path, "request", request_json)

    request_id = str(uuid.uuid4())
    batch_request = build_batch_request(request_id, params)

    batch = await client.messages.batches.create(requests=[batch_request])

    return BatchSubmitResult(request_id=request_id, batch_id=batch.id)


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def _record_submission(result: BatchSubmitResult, workflow_id: str) -> None:
    """Best-effort store write. Never raises (D42)."""
    try:
        from forge.store import get_db_path, get_engine, record_batch_submission

        db_path = get_db_path()
        if db_path is None:
            return

        engine = get_engine(db_path)
        record_batch_submission(
            engine,
            request_id=result.request_id,
            batch_id=result.batch_id,
            workflow_id=workflow_id,
        )
    except Exception:
        logger.warning("Failed to record batch submission to store", exc_info=True)


@activity.defn
async def submit_batch_request(input: BatchSubmitInput) -> BatchSubmitResult:
    """Activity wrapper — creates client and delegates to execute_batch_submit."""
    from forge.llm_client import get_anthropic_client
    from forge.tracing import get_tracer

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.submit_batch_request") as span:
        logger.info(
            "Batch submitted: task_id=%s output_type=%s",
            input.context.task_id,
            input.output_type_name,
        )
        client = get_anthropic_client()
        result = await execute_batch_submit(input, client)

        span.set_attributes(
            {
                "forge.batch.request_id": result.request_id,
                "forge.batch.batch_id": result.batch_id,
                "forge.batch.output_type": input.output_type_name,
                "forge.batch.workflow_id": input.workflow_id,
            }
        )

        _record_submission(result, input.workflow_id)
        return result
