"""The LLM-guided context exploration loop (Phase 7), as free functions.

Exploration is a *conversation about what to read*: the exploration LLM asks
providers for context, the workflow fulfills those requests, and the loop repeats
until the LLM signals readiness (an empty request list) or the round budget runs
out. What it accumulates is rendered into a prompt section that the caller
appends to whatever prompt it is about to send.

Both callers live in ``forge.workflows.task`` — one exploration pass before
planning, and one per generation attempt inside the step block's hook, so each
attempt explores against its *own* worktree. A sub-task never explores. Nothing
here holds workflow state: the loop takes the dispatch host (for the exploration
arm, which persists its own interaction record) and plain values, so T5.4 moved
it out of the workflow class it used to be a method on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from sax_platform.temporal.retries import IO_RETRY

    from forge.blocks.dispatch import dispatch_exploration
    from forge.models import ContextResult, ExplorationInput, FulfillContextInput
    from forge.presets import EXPLORATION_FULFILL_TIMEOUT
    from forge.providers import PROVIDER_SPECS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from forge.blocks.dispatch import DispatchHost
    from forge.models import TaskDefinition

__all__ = [
    "format_exploration_context",
    "run_exploration_loop",
]

# Per-provider cap on what one exploration result may contribute to a prompt.
_MAX_RESULT_CHARS = 8000


async def run_exploration_loop(
    host: DispatchHost,
    *,
    task: TaskDefinition,
    repo_root: str,
    worktree_path: str,
    max_rounds: int,
    model_name: str = "",
    log_messages: bool = False,
) -> list[ContextResult]:
    """Run the exploration rounds and return everything the providers returned.

    The LLM requests context from providers until it signals readiness (an empty
    ``requests`` list) or ``max_rounds`` is reached. Each round's results are
    accumulated and shown to the next round, so the LLM can ask follow-ups.
    """
    accumulated: list[ContextResult] = []
    round_num = 0

    for round_num in range(1, max_rounds + 1):
        workflow.logger.debug(
            "Exploration round %d/%d: task_id=%s", round_num, max_rounds, task.task_id
        )
        exploration_input = ExplorationInput(
            task_id=task.task_id,
            task_description=task.description,
            target_files=task.target_files,
            context_files=task.context_files,
            context_config=task.context,
            available_providers=PROVIDER_SPECS,
            accumulated_context=accumulated,
            round_number=round_num,
            max_rounds=max_rounds,
            repo_root=repo_root,
            model_name=model_name,
            log_messages=log_messages,
            worktree_path=worktree_path,
        )
        exploration_call = await dispatch_exploration(host, exploration_input)
        requests = exploration_call.response.requests
        workflow.logger.debug(
            "Exploration round %d: %d provider requests",
            round_num,
            len(requests),
        )

        if not requests:
            break  # LLM is ready to generate

        context_results = await workflow.execute_activity(
            "fulfill_context_requests",
            FulfillContextInput(
                requests=requests,
                repo_root=repo_root,
                worktree_path=worktree_path,
            ),
            start_to_close_timeout=EXPLORATION_FULFILL_TIMEOUT,
            retry_policy=IO_RETRY,
            result_type=list[ContextResult],
        )
        accumulated.extend(context_results)

    workflow.logger.info(
        "Exploration complete: task_id=%s rounds_used=%d results=%d",
        task.task_id,
        min(round_num, max_rounds),
        len(accumulated),
    )
    return accumulated


def format_exploration_context(results: Sequence[ContextResult]) -> str:
    """Render exploration results as a prompt section (pure).

    Returns ``""`` for no results, so a caller can append unconditionally and
    leave its prompt untouched when exploration found nothing.
    """
    if not results:
        return ""

    parts = ["", "## Exploration Results"]
    for ctx in results:
        parts.append("")
        parts.append(f"### From: {ctx.provider}")
        content = ctx.content
        if len(content) > _MAX_RESULT_CHARS:
            content = content[:_MAX_RESULT_CHARS] + "\n... (truncated)"
        parts.append(content)

    return "\n".join(parts)
