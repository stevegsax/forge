"""Shared workflow building blocks for Forge.

Reusable async functions that execute within a Temporal workflow context.
These functions call workflow.execute_activity and workflow.sleep,
so they must be called from within a Temporal workflow.

Extracted from workflows.py to enable composition into purpose-built workflows
(e.g., OCR, research) without duplicating batch dispatch logic.

What remains here is the transport and the worktree cleanup shared by both
workflow classes: ``batch_submit_and_wait`` is the one submit/poll/fetch/parse
implementation (T4.1, D88), imported by ``forge.blocks.dispatch`` and by
``forge.ingestion_workflow``. The *typed* dispatch that used to sit on top of it
in two hand-rolled copies (``generation_dispatch`` /
``conflict_resolution_dispatch``) moved into ``forge.blocks.dispatch``, where all
five arms share one implementation (T5.3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from temporalio import workflow
from temporalio.exceptions import ActivityError, ApplicationError

with workflow.unsafe.imports_passed_through():
    from sax_platform.contracts.models import BatchJobStatus
    from sax_platform.contracts.persist import (
        PersistBatchSubmission,
        persist_block,
    )
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
        RemoveWorktreeInput,
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
        GIT_TIMEOUT,
        PARSE_TIMEOUT,
        SUBMIT_TIMEOUT,
    )


if TYPE_CHECKING:
    from datetime import timedelta

__all__ = [
    "BATCH_WAIT_FAILURES",
    "batch_submit_and_wait",
    "cleanup_worktree_after_exception",
    "persist_block",
    "remove_worktree",
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


# ---------------------------------------------------------------------------
# Batch dispatch
# ---------------------------------------------------------------------------


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
        start_to_close_timeout=GIT_TIMEOUT,
        retry_policy=IO_RETRY,
        result_type=type(None),
    )


async def cleanup_worktree_after_exception(
    repo_root: str, task_id: str, exc: BaseException
) -> None:
    """Remove a worktree and its branch after a failure; never masks *exc*.

    The one cleanup helper for every owner of a worktree (T5.3 ST8 folded the
    two same-bodied copies together). Two kinds of caller use it:

    - the blocks that create a worktree (``blocks/step.py``'s fresh modes,
      ``blocks/gather.py``'s owned mode) and ``_run_planned``, which own the
      worktree an exception escaped from and re-raise afterwards;
    - both workflow ``run()`` batch-wait handlers (T1.6b), where a wait that
      gave up at the 25h ceiling or hit a provider-terminal status / error-bearing
      fetch (T4.1) must leave no orphaned worktree, yet must still let the caller
      record its terminal run row.

    Removal is forced, so the activity deletes the ``forge/<task_id>`` branch too
    — the leak that used to make the next run of the same task id fail on
    ``worktree add``. It is also best-effort: only ``ActivityError`` is swallowed
    (``CancelledError`` must still propagate), so a cleanup blip can neither
    replace the real failure nor block the FAILURE_TERMINAL persist.
    """
    workflow.logger.warning("Cleaning worktree after failure: task_id=%s: %r", task_id, exc)
    try:
        await remove_worktree(repo_root, task_id)
    except ActivityError as cleanup_exc:
        workflow.logger.warning(
            "Worktree cleanup did not complete: task_id=%s: %r",
            task_id,
            cleanup_exc,
        )
