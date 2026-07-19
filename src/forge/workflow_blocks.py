"""Shared workflow building blocks for Forge.

Reusable async functions that execute within a Temporal workflow context.
These functions call workflow.execute_activity and workflow.sleep,
so they must be called from within a Temporal workflow.

Extracted from workflows.py to enable composition into purpose-built workflows
(e.g., OCR, research) without duplicating batch dispatch logic.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.exceptions import ActivityError, ApplicationError

with workflow.unsafe.imports_passed_through():
    from sax_platform.contracts.models import BatchJobStatus
    from sax_platform.contracts.persist import (
        PersistBatchSubmission,
        persist_block,
    )
    from sax_platform.temporal.retries import IO_RETRY, LLM_RETRY

    from forge.models import (
        BATCH_WAIT_CEILING,
        AssembledContext,
        BatchFetchResult,
        BatchStatusInput,
        BatchStatusResult,
        BatchSubmitInput,
        BatchSubmitResult,
        ConflictResolutionCallInput,
        ConflictResolutionCallResult,
        ConflictResolutionResponse,
        FetchBatchResultInput,
        LLMCallResult,
        LLMResponse,
        ParsedLLMResponse,
        ParseResponseInput,
        RemoveWorktreeInput,
        ThinkingPolicy,
    )
    from forge.persist_models import PersistBatchOutcome


__all__ = [
    "BATCH_WAIT_FAILURES",
    "THINKING_MAX_TOKENS",
    "batch_submit_and_wait",
    "cleanup_worktree_after_failure",
    "conflict_resolution_dispatch",
    "generation_dispatch",
    "persist_block",
    "remove_worktree",
]

# Explicit cap for the three thinking-enabled batch call paths (planner,
# sanity-check, conflict-resolution): adaptive thinking now competes for
# tokens inside max_tokens instead of riding on top of it, so the old 4096
# default batch-lane cap left too little room for both the thinking budget
# and the structured output it must still emit. Sized for adaptive thinking +
# structured output on the batch lane; tokens-vs-cap telemetry decides future
# tuning (owner-adjudicated, 2026-07 Phase 3 code review). The generation
# path stays thinking-disabled and keeps its own (lower) cap untouched.
THINKING_MAX_TOKENS = 16384

# A batch wait fails in exactly three shapes under the timer-loop transport
# (T4.1, D88), and all must leave a terminal run row and no orphaned worktree
# instead of crashing out of the workflow (T1.6b):
#   * the 25h ``wait_timeout`` ceiling expiring inside the poll loop — the waiter
#     gave up (persisted MISSING);
#   * a provider-terminal status (``failed``/``expired``/``canceled``) reported by
#     ``batch_status`` (persisted FAILED/EXPIRED);
#   * an error-bearing ``fetch_batch_result`` — a failed result line or an absent
#     custom_id (persisted FAILED).
# All three are raised by ``batch_submit_and_wait`` as a non-retryable
# ``ApplicationError`` in workflow code (not wrapped in ``ActivityError``), so a
# workflow ``run()`` catches them directly; ordinary activity failures are
# untouched. (Before T4.1 a fourth shape existed — the ``wait_condition`` builtin
# ``TimeoutError`` from the signal wait — but the signal path is gone.)
BATCH_WAIT_FAILURES: tuple[type[BaseException], ...] = (ApplicationError,)


# ---------------------------------------------------------------------------
# Timeout and retry presets (shared with workflows.py)
# ---------------------------------------------------------------------------

_GIT_TIMEOUT = timedelta(seconds=30)
_LLM_TIMEOUT = timedelta(minutes=5)
_SUBMIT_TIMEOUT = timedelta(seconds=60)
_PARSE_TIMEOUT = timedelta(seconds=30)
# One source of truth for the 25h ceiling: shared with _child_timeout /
# derive_execution_timeout via forge.models.BATCH_WAIT_CEILING (T4.1 ST3c).
_BATCH_WAIT_TIMEOUT = BATCH_WAIT_CEILING
_BATCH_STATUS_TIMEOUT = timedelta(seconds=60)
_BATCH_FETCH_TIMEOUT = timedelta(minutes=5)
_CONFLICT_RESOLUTION_TIMEOUT = timedelta(minutes=5)

# Timer-loop poll cadence (T4.1, D88). Default 600s; floored at 300s in the loop
# as defense in depth (the ``batch_poll_interval_seconds`` input fields also
# validate ge=300). A batch is never done instantly, so the loop sleeps before
# its first status poll.
_BATCH_POLL_INTERVAL = timedelta(seconds=600)
_BATCH_POLL_FLOOR = timedelta(seconds=300)

_LLM_HEARTBEAT = timedelta(seconds=60)

# LLM_RETRY and IO_RETRY are the shared presets from sax_platform.temporal.retries
# (T3.4, ST7) — forge's former per-module _LLM_RETRY/_LOCAL_RETRY copies are
# retired in favor of the single platform source.


# ---------------------------------------------------------------------------
# Batch dispatch
# ---------------------------------------------------------------------------


async def batch_submit_and_wait(
    context: AssembledContext,
    output_type_name: str | None,
    *,
    thinking: ThinkingPolicy | None = None,
    max_tokens: int = 4096,
    poll_interval: timedelta = _BATCH_POLL_INTERVAL,
    submit_timeout: timedelta = _SUBMIT_TIMEOUT,
    wait_timeout: timedelta = _BATCH_WAIT_TIMEOUT,
    parse_timeout: timedelta = _PARSE_TIMEOUT,
) -> ParsedLLMResponse:
    """Submit a batch request, poll it to completion on a timer, fetch and parse.

    The single choke point for every forge batch wait under the D88 timer-loop
    transport: the requesting workflow mints ``request_id`` (the custom_id, so a
    retried submit reuses it), submits, then polls ``batch_status`` every
    ``poll_interval`` until the batch ends or the ``wait_timeout`` ceiling passes,
    then fetches this waiter's own result line and classifies it via
    ``parse_llm_response``. No signal and no shared poller — the requester is the
    recipient of its own result.
    """
    # Defense in depth: floor the poll interval (the input models also validate
    # ge=300). Never poll a provider batch API faster than every five minutes.
    poll_interval = max(poll_interval, _BATCH_POLL_FLOOR)
    request_id = str(workflow.uuid4())
    submit_result: BatchSubmitResult = await workflow.execute_activity(
        "submit_batch_request",
        BatchSubmitInput(
            request_id=request_id,
            context=context,
            output_type_name=output_type_name or "",
            workflow_id=workflow.info().workflow_id,
            # Invariant: batch calls think only when a caller explicitly opts in.
            # ThinkingPolicy()'s bare default is enabled=True (D94), unlike the
            # old ThinkingConfig() default, so a missing ``thinking`` argument
            # must resolve to disabled here, not to the type's own default.
            thinking=thinking if thinking is not None else ThinkingPolicy(enabled=False),
            max_tokens=max_tokens,
        ),
        start_to_close_timeout=submit_timeout,
        retry_policy=LLM_RETRY,
        result_type=BatchSubmitResult,
    )
    # Survivably record the submission before waiting, so batch_jobs carries the
    # in-flight row (and a DB blip retries only this cheap write, not the submit).
    await persist_block(
        PersistBatchSubmission(
            request_id=request_id,
            batch_id=submit_result.batch_id,
            workflow_id=workflow.info().workflow_id,
            provider=submit_result.provider,
        )
    )
    # Timer loop: sleep, then poll the provider's normalized status. Sleep first
    # (a batch is never done instantly). Break on ``ended``; a provider-terminal
    # status or the 25h ceiling records a terminal ledger outcome and raises a
    # non-retryable ApplicationError, preserving the T1.6b failure symmetry.
    deadline = workflow.now() + wait_timeout
    while True:
        remaining = deadline - workflow.now()
        if remaining <= timedelta(0):
            await persist_block(
                PersistBatchOutcome(
                    request_id=request_id,
                    status=BatchJobStatus.MISSING.value,
                    error_message=f"batch wait exceeded {wait_timeout} ceiling",
                )
            )
            raise ApplicationError(
                f"Batch wait exceeded {wait_timeout} ceiling for request {request_id}",
                non_retryable=True,
            )
        await workflow.sleep(min(poll_interval, remaining))
        status: BatchStatusResult = await workflow.execute_activity(
            "batch_status",
            BatchStatusInput(batch_id=submit_result.batch_id, provider=submit_result.provider),
            start_to_close_timeout=_BATCH_STATUS_TIMEOUT,
            retry_policy=IO_RETRY,
            result_type=BatchStatusResult,
        )
        if status.state == "ended":
            break
        if status.state in ("failed", "expired", "canceled"):
            terminal = (
                BatchJobStatus.EXPIRED if status.state == "expired" else BatchJobStatus.FAILED
            )
            await persist_block(
                PersistBatchOutcome(
                    request_id=request_id,
                    status=terminal.value,
                    error_message=f"provider batch {status.state}",
                )
            )
            raise ApplicationError(
                f"Batch {submit_result.batch_id} {status.state} for request {request_id}",
                non_retryable=True,
            )
    fetch: BatchFetchResult = await workflow.execute_activity(
        "fetch_batch_result",
        FetchBatchResultInput(
            batch_id=submit_result.batch_id,
            request_id=request_id,
            provider=submit_result.provider,
        ),
        start_to_close_timeout=_BATCH_FETCH_TIMEOUT,
        retry_policy=IO_RETRY,
        result_type=BatchFetchResult,
    )
    if fetch.error:
        await persist_block(
            PersistBatchOutcome(
                request_id=request_id,
                status=BatchJobStatus.FAILED.value,
                error_message=fetch.error,
            )
        )
        raise ApplicationError(
            f"Batch fetch failed for request {request_id}: {fetch.error}",
            non_retryable=True,
        )
    # Successful fetch: record the terminal ENDED outcome, then classify the body
    # (inline or via S3 pointer — the parse activity fetches the blob when only
    # s3_key is set) with the call site's output class.
    await persist_block(
        PersistBatchOutcome(
            request_id=request_id,
            status=BatchJobStatus.ENDED.value,
        )
    )
    parsed: ParsedLLMResponse = await workflow.execute_activity(
        "parse_llm_response",
        ParseResponseInput(
            raw_response_json=fetch.raw_response_json,
            s3_key=fetch.s3_key,
            output_type_name=output_type_name,
            task_id=context.task_id,
            log_messages=context.log_messages,
            worktree_path=context.worktree_path,
            max_tokens=max_tokens,
        ),
        start_to_close_timeout=parse_timeout,
        retry_policy=IO_RETRY,
        result_type=ParsedLLMResponse,
    )
    return parsed


# ---------------------------------------------------------------------------
# Generation dispatch
# ---------------------------------------------------------------------------


async def generation_dispatch(
    sync_mode: bool,
    context: AssembledContext,
    *,
    poll_interval: timedelta = _BATCH_POLL_INTERVAL,
) -> LLMCallResult:
    """Route LLM generation call through sync or batch path."""
    if sync_mode:
        sync_result: LLMCallResult = await workflow.execute_activity(
            "call_llm",
            context,
            start_to_close_timeout=_LLM_TIMEOUT,
            heartbeat_timeout=_LLM_HEARTBEAT,
            retry_policy=LLM_RETRY,
            result_type=LLMCallResult,
        )
        return sync_result
    # Generation stays thinking-disabled, as today. Omitting `thinking` relies
    # on batch_submit_and_wait's shared fallback (disabled by default).
    parsed = await batch_submit_and_wait(context, "LLMResponse", poll_interval=poll_interval)
    return LLMCallResult(
        task_id=context.task_id,
        response=LLMResponse.model_validate_json(parsed.parsed_json),
        model_name=parsed.model_name,
        input_tokens=parsed.input_tokens,
        output_tokens=parsed.output_tokens,
        latency_ms=parsed.latency_ms,
        cache_creation_input_tokens=parsed.cache_creation_input_tokens,
        cache_read_input_tokens=parsed.cache_read_input_tokens,
        stop_reason=parsed.stop_reason,
    )


# ---------------------------------------------------------------------------
# Conflict resolution dispatch
# ---------------------------------------------------------------------------


async def conflict_resolution_dispatch(
    sync_mode: bool,
    call_input: ConflictResolutionCallInput,
    *,
    poll_interval: timedelta = _BATCH_POLL_INTERVAL,
) -> ConflictResolutionCallResult:
    """Dispatch conflict resolution LLM call via sync or batch path."""
    if sync_mode:
        sync_result: ConflictResolutionCallResult = await workflow.execute_activity(
            "call_conflict_resolution",
            call_input,
            start_to_close_timeout=_CONFLICT_RESOLUTION_TIMEOUT,
            heartbeat_timeout=_LLM_HEARTBEAT,
            retry_policy=LLM_RETRY,
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
        context,
        "ConflictResolutionResponse",
        thinking=call_input.thinking,
        max_tokens=THINKING_MAX_TOKENS,
        poll_interval=poll_interval,
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
        stop_reason=parsed.stop_reason,
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
        retry_policy=IO_RETRY,
        result_type=type(None),
    )


async def cleanup_worktree_after_failure(repo_root: str, task_id: str, exc: BaseException) -> None:
    """Clean up a worktree after a batch wait fails; never raises.

    The shared failure-symmetry handler for ``ForgeTaskWorkflow`` and
    ``ForgeSubTaskWorkflow`` (T1.6b): a batch wait that gives up at the 25h ceiling
    or hits a provider-terminal status / error-bearing fetch (T4.1) must leave no
    orphaned worktree, yet it must not let a cleanup blip mask the terminal run
    record the caller still has to write.
    Worktree removal is therefore best-effort — only ``ActivityError`` is swallowed
    (``CancelledError`` must still propagate), so the caller can persist its
    FAILURE_TERMINAL row unconditionally.
    """
    workflow.logger.warning("Batch wait failed; cleaning worktree task_id=%s: %r", task_id, exc)
    try:
        await remove_worktree(repo_root, task_id)
    except ActivityError as cleanup_exc:
        workflow.logger.warning(
            "Worktree cleanup after batch failure did not complete: task_id=%s: %r",
            task_id,
            cleanup_exc,
        )
