"""Batch submit activity for Forge.

Submits an assembled context to the LLM provider's batch API.

Design follows Function Core / Imperative Shell:
- Testable function: execute_batch_submit (takes provider as argument)
- Imperative shell: submit_batch_request, _record_submission
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING

from temporalio import activity

from forge.llm_client import get_output_type_registry
from forge.message_log import write_message_log
from forge.models import BatchSubmitInput, BatchSubmitResult

if TYPE_CHECKING:
    from forge.llm_providers.protocol import LLMProvider

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
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
    from forge.llm_providers import parse_model_id

    registry = get_output_type_registry()
    output_type = registry[input.output_type_name]
    full_model = input.context.model_name or DEFAULT_MODEL
    _, model = parse_model_id(full_model)

    params = provider.build_request_params(
        system_prompt=input.context.system_prompt,
        user_prompt=input.context.user_prompt,
        output_type=output_type,
        model=model,
        max_tokens=input.max_tokens,
        thinking_budget_tokens=input.thinking.budget_tokens,
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


def _record_submission(
    result: BatchSubmitResult, workflow_id: str, provider: str = "anthropic"
) -> None:
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
            provider=provider,
        )
    except Exception:
        logger.warning("Failed to record batch submission to store", exc_info=True)


@activity.defn
async def submit_batch_request(input: BatchSubmitInput) -> BatchSubmitResult:
    """Activity wrapper — creates provider and delegates to execute_batch_submit."""
    from forge.llm_providers import get_provider
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

        from forge.llm_providers import parse_model_id

        provider_name, _ = parse_model_id(input.context.model_name or DEFAULT_MODEL)
        _record_submission(result, input.workflow_id, provider=provider_name)
        return result
