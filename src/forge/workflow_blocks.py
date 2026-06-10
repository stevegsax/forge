"""Shared workflow building blocks for Forge.

Reusable async functions that execute within a Temporal workflow context.
These functions call workflow.execute_activity and workflow.wait_condition,
so they must be called from within a Temporal workflow.

Extracted from workflows.py to enable composition into purpose-built workflows
(e.g., OCR, research) without duplicating batch dispatch logic.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from forge_contracts.persist import (
        PersistBatchSubmission,
        persist_block,
    )

    from forge.models import (
        AssembledContext,
        BatchResult,
        BatchSubmitInput,
        BatchSubmitResult,
        ConflictResolutionCallInput,
        ConflictResolutionCallResult,
        ConflictResolutionResponse,
        LLMCallResult,
        LLMResponse,
        ParsedLLMResponse,
        ParseResponseInput,
        RemoveWorktreeInput,
        ThinkingConfig,
    )


__all__ = [
    "batch_submit_and_wait",
    "conflict_resolution_dispatch",
    "generation_dispatch",
    "persist_block",
    "remove_worktree",
]


# ---------------------------------------------------------------------------
# Timeout and retry presets (shared with workflows.py)
# ---------------------------------------------------------------------------

_GIT_TIMEOUT = timedelta(seconds=30)
_LLM_TIMEOUT = timedelta(minutes=5)
_SUBMIT_TIMEOUT = timedelta(seconds=60)
_PARSE_TIMEOUT = timedelta(seconds=30)
_BATCH_WAIT_TIMEOUT = timedelta(hours=25)
_CONFLICT_RESOLUTION_TIMEOUT = timedelta(minutes=5)

_LLM_HEARTBEAT = timedelta(seconds=60)

_LLM_RETRY = RetryPolicy(
    maximum_attempts=3,
    non_retryable_error_types=[
        "BadRequestError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
    ],
)
_LOCAL_RETRY = RetryPolicy(maximum_attempts=2)


# ---------------------------------------------------------------------------
# Batch dispatch
# ---------------------------------------------------------------------------


async def batch_submit_and_wait(
    batch_results: list[BatchResult],
    context: AssembledContext,
    output_type_name: str | None,
    *,
    thinking: ThinkingConfig | None = None,
    max_tokens: int = 4096,
    submit_timeout: timedelta = _SUBMIT_TIMEOUT,
    wait_timeout: timedelta = _BATCH_WAIT_TIMEOUT,
    parse_timeout: timedelta = _PARSE_TIMEOUT,
) -> ParsedLLMResponse:
    """Submit batch request, wait for signal, parse response.

    Generalized from _call_llm_batch_dispatch in workflows.py.
    """
    submit_result: BatchSubmitResult = await workflow.execute_activity(
        "submit_batch_request",
        BatchSubmitInput(
            context=context,
            output_type_name=output_type_name or "",
            workflow_id=workflow.info().workflow_id,
            thinking=thinking or ThinkingConfig(),
            max_tokens=max_tokens,
        ),
        start_to_close_timeout=submit_timeout,
        retry_policy=_LLM_RETRY,
        result_type=BatchSubmitResult,
    )
    # Survivably record the submission before waiting, so the poller can find the
    # job (and a DB blip retries only this cheap write, not the submit call).
    await persist_block(
        PersistBatchSubmission(
            request_id=submit_result.request_id,
            batch_id=submit_result.batch_id,
            workflow_id=workflow.info().workflow_id,
            provider=submit_result.provider,
        )
    )
    await workflow.wait_condition(
        lambda: len(batch_results) > 0,
        timeout=wait_timeout,
    )
    result = batch_results.pop(0)
    if result.error:
        raise ApplicationError(f"Batch error: {result.error}")
    if result.raw_response_json is None and result.s3_key is None:
        raise ApplicationError("Batch result has neither inline body nor s3_key")
    # The body travels inline or by S3 pointer (size-chosen by the poller); the
    # parse activity fetches the blob when only s3_key is set.
    parsed: ParsedLLMResponse = await workflow.execute_activity(
        "parse_llm_response",
        ParseResponseInput(
            raw_response_json=result.raw_response_json,
            s3_key=result.s3_key,
            output_type_name=output_type_name,
            task_id=context.task_id,
            log_messages=context.log_messages,
            worktree_path=context.worktree_path,
        ),
        start_to_close_timeout=parse_timeout,
        retry_policy=_LOCAL_RETRY,
        result_type=ParsedLLMResponse,
    )
    return parsed


# ---------------------------------------------------------------------------
# Generation dispatch
# ---------------------------------------------------------------------------


async def generation_dispatch(
    batch_results: list[BatchResult],
    sync_mode: bool,
    context: AssembledContext,
) -> LLMCallResult:
    """Route LLM generation call through sync or batch path."""
    if sync_mode:
        sync_result: LLMCallResult = await workflow.execute_activity(
            "call_llm",
            context,
            start_to_close_timeout=_LLM_TIMEOUT,
            heartbeat_timeout=_LLM_HEARTBEAT,
            retry_policy=_LLM_RETRY,
            result_type=LLMCallResult,
        )
        return sync_result
    parsed = await batch_submit_and_wait(batch_results, context, "LLMResponse")
    return LLMCallResult(
        task_id=context.task_id,
        response=LLMResponse.model_validate_json(parsed.parsed_json),
        model_name=parsed.model_name,
        input_tokens=parsed.input_tokens,
        output_tokens=parsed.output_tokens,
        latency_ms=parsed.latency_ms,
        cache_creation_input_tokens=parsed.cache_creation_input_tokens,
        cache_read_input_tokens=parsed.cache_read_input_tokens,
    )


# ---------------------------------------------------------------------------
# Conflict resolution dispatch
# ---------------------------------------------------------------------------


async def conflict_resolution_dispatch(
    batch_results: list[BatchResult],
    sync_mode: bool,
    call_input: ConflictResolutionCallInput,
) -> ConflictResolutionCallResult:
    """Dispatch conflict resolution LLM call via sync or batch path."""
    if sync_mode:
        sync_result: ConflictResolutionCallResult = await workflow.execute_activity(
            "call_conflict_resolution",
            call_input,
            start_to_close_timeout=_CONFLICT_RESOLUTION_TIMEOUT,
            heartbeat_timeout=_LLM_HEARTBEAT,
            retry_policy=_LLM_RETRY,
            result_type=ConflictResolutionCallResult,
        )
        return sync_result
    context = AssembledContext(
        task_id=call_input.task_id,
        system_prompt=call_input.system_prompt,
        user_prompt=call_input.user_prompt,
        model_name=call_input.model_name,
        log_messages=call_input.log_messages,
        worktree_path=call_input.worktree_path,
    )
    parsed = await batch_submit_and_wait(
        batch_results,
        context,
        "ConflictResolutionResponse",
        thinking=call_input.thinking,
    )
    response = ConflictResolutionResponse.model_validate_json(parsed.parsed_json)
    return ConflictResolutionCallResult(
        task_id=call_input.task_id,
        resolved_files={f.file_path: f.content for f in response.resolved_files},
        explanation=response.explanation,
        model_name=parsed.model_name,
        input_tokens=parsed.input_tokens,
        output_tokens=parsed.output_tokens,
        latency_ms=parsed.latency_ms,
        cache_creation_input_tokens=parsed.cache_creation_input_tokens,
        cache_read_input_tokens=parsed.cache_read_input_tokens,
    )


# ---------------------------------------------------------------------------
# Worktree cleanup
# ---------------------------------------------------------------------------


async def remove_worktree(repo_root: str, task_id: str) -> None:
    """Remove a worktree via activity. Shared by multiple workflow classes."""
    await workflow.execute_activity(
        "remove_worktree_activity",
        RemoveWorktreeInput(
            repo_root=repo_root,
            task_id=task_id,
            force=True,
        ),
        start_to_close_timeout=_GIT_TIMEOUT,
        retry_policy=_LOCAL_RETRY,
        result_type=type(None),
    )
