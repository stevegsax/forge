"""The batch transport block (T4.1/D88) — one submit/poll/fetch/parse implementation.

Every forge batch wait goes through :func:`batch_submit_and_wait`: the requesting
workflow mints its own ``request_id``, submits, polls on a timer until the batch
ends, fetches its own result line, and classifies it. There is no shared poller
and no signal — the requester is the recipient of its own result (D88, reversal
R1).

This module and :mod:`forge.blocks.worktree` are what remained of the former
``forge/workflow_blocks.py`` once the typed dispatch layer moved into
:mod:`forge.blocks.dispatch` (T5.3); T5.4 dissolved that module so every shape a
workflow composes lives under ``blocks/``. Nothing about the transport changed in
the move — it is imported by :mod:`forge.blocks.dispatch` (all five LLM arms) and
by :mod:`forge.ingestion_workflow`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from temporalio import workflow
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from sax_platform.contracts.models import BatchJobStatus
    from sax_platform.contracts.persist import PersistBatchSubmission, persist_block
    from sax_platform.temporal.polling import FixedInterval, wait_batch_ended
    from sax_platform.temporal.retries import IO_RETRY, LLM_RETRY

    from forge.models import (
        AssembledContext,
        BatchFetchResult,
        BatchStatusInput,
        BatchStatusResult,
        BatchSubmitInput,
        BatchSubmitResult,
        FetchBatchResultInput,
        ParsedLLMResponse,
        ParseResponseInput,
        ThinkingPolicy,
    )
    from forge.persist_models import PersistBatchOutcome
    from forge.presets import (
        BATCH_FETCH_TIMEOUT,
        BATCH_POLL_FLOOR,
        BATCH_POLL_INTERVAL,
        BATCH_STATUS_TIMEOUT,
        BATCH_WAIT_TIMEOUT,
        DEFAULT_MAX_TOKENS,
        PARSE_TIMEOUT,
        SUBMIT_TIMEOUT,
    )

if TYPE_CHECKING:
    from datetime import timedelta

__all__ = [
    "BATCH_WAIT_FAILURES",
    "batch_submit_and_wait",
]

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


async def batch_submit_and_wait(
    context: AssembledContext,
    output_type_name: str | None,
    *,
    thinking: ThinkingPolicy | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    poll_interval: timedelta = BATCH_POLL_INTERVAL,
    submit_timeout: timedelta = SUBMIT_TIMEOUT,
    wait_timeout: timedelta = BATCH_WAIT_TIMEOUT,
    parse_timeout: timedelta = PARSE_TIMEOUT,
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
    poll_interval = max(poll_interval, BATCH_POLL_FLOOR)
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

    # Timer loop (shared skeleton in sax_platform.temporal.polling): sleep, then
    # poll the provider's normalized status. FixedInterval reproduces forge's exact
    # pre-extraction sleep sequence (min(poll_interval, remaining) each iteration),
    # so the committed replay histories still replay unregenerated. The loop returns
    # a plain outcome string; the persists + non-retryable ApplicationErrors that
    # keep the T1.6b failure symmetry stay here, where the outcomes now surface.
    async def _poll_status() -> str:
        status: BatchStatusResult = await workflow.execute_activity(
            "batch_status",
            BatchStatusInput(batch_id=submit_result.batch_id, provider=submit_result.provider),
            start_to_close_timeout=BATCH_STATUS_TIMEOUT,
            retry_policy=IO_RETRY,
            result_type=BatchStatusResult,
        )
        return status.state

    outcome = await wait_batch_ended(
        _poll_status,
        schedule=FixedInterval(poll_interval),
        ceiling=wait_timeout,
    )
    if outcome == "gave_up":
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
    if outcome in ("failed", "expired", "canceled"):
        terminal = BatchJobStatus.EXPIRED if outcome == "expired" else BatchJobStatus.FAILED
        await persist_block(
            PersistBatchOutcome(
                request_id=request_id,
                status=terminal.value,
                error_message=f"provider batch {outcome}",
            )
        )
        raise ApplicationError(
            f"Batch {submit_result.batch_id} {outcome} for request {request_id}",
            non_retryable=True,
        )
    fetch: BatchFetchResult = await workflow.execute_activity(
        "fetch_batch_result",
        FetchBatchResultInput(
            batch_id=submit_result.batch_id,
            request_id=request_id,
            provider=submit_result.provider,
        ),
        start_to_close_timeout=BATCH_FETCH_TIMEOUT,
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
