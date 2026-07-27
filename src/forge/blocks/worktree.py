"""Worktree removal and post-failure cleanup, shared by every worktree owner.

Three kinds of caller create a worktree and must give it back: the step block's
fresh modes, the gather block's owned mode, and ``ForgeTaskWorkflow._run_planned``
(which creates the plan's worktree and lends it to every step). The two workflow
``run()`` methods use the same cleanup on their batch-wait failure path (T1.6b).
One implementation serves all of them.

Split out of the former ``forge/workflow_blocks.py`` by T5.4 alongside
:mod:`forge.blocks.transport`; the helpers themselves are unchanged (T5.3 ST8
already folded the two same-bodied cleanup copies into one).
"""

from __future__ import annotations

from temporalio import workflow
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    from sax_platform.temporal.retries import IO_RETRY

    from forge.models import RemoveWorktreeInput
    from forge.presets import GIT_TIMEOUT

__all__ = [
    "cleanup_worktree_after_exception",
    "remove_worktree",
]


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
