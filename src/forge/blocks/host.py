"""The shared persisting dispatch host (T5.3).

``ForgeTaskWorkflow`` and ``ForgeSubTaskWorkflow`` each carried a verbatim copy
of the same members: the per-run lane/cadence state, the per-role occurrence
counters, the interaction persist, and the LLM dispatch wrappers. Hand-synchronized
copies are how the T1.5 propagation bug was bred — the nested copy silently
dropped fields the parent honored — so these exist once here and both workflow
classes inherit them.

The base class is deliberately **not** decorated with ``@workflow.defn``: it
defines no ``@workflow.run``, so it is a plain Python base that contributes
methods and instance state to whichever workflow class inherits it. The pattern
was verified against the sandbox and the ``Replayer`` before adoption — an
inherited method executing activities, and inherited instance state mutated
during ``run()``, replay exactly as if they were written in the subclass.

Everything the dispatch arms need beyond this state lives in
``forge.blocks.dispatch``; this module is only the seam that gives the block a
place to persist through.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Protocol

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from sax_platform.contracts.persist import persist_block

    from forge.blocks.dispatch import PersistFields, dispatch_generation
    from forge.persist_models import (
        PersistableLLMResult,
        build_interaction_idempotency_key,
        build_persist_interaction,
    )
    from forge.presets import BATCH_POLL_INTERVAL

if TYPE_CHECKING:
    from forge.models import AssembledContext, LLMCallResult

__all__ = ["DispatchHostBase", "RunSettings"]


class RunInput(Protocol):
    """The three per-run dispatch fields both workflow inputs carry."""

    sync_mode: bool
    log_messages: bool
    batch_poll_interval_seconds: int


@dataclass(frozen=True, slots=True, kw_only=True)
class RunSettings:
    """How this run dispatches: which lane, whether to log messages, how often to poll.

    Frozen and set once, so there is no half-configured state and no code path
    that can flip a run's lane mid-flight. The defaults are the safe ones a
    freshly constructed workflow instance holds until ``run`` configures it.
    """

    sync_mode: bool = True
    log_messages: bool = False
    poll_interval: timedelta = BATCH_POLL_INTERVAL

    @classmethod
    def from_input(cls, input: RunInput) -> RunSettings:
        """Read the settings off a workflow input (``ForgeTaskInput``/``SubTaskInput``)."""
        return cls(
            sync_mode=input.sync_mode,
            log_messages=input.log_messages,
            poll_interval=timedelta(seconds=input.batch_poll_interval_seconds),
        )


class DispatchHostBase:
    """Per-run LLM dispatch state, shared by both forge workflow classes.

    Satisfies ``dispatch.DispatchHost`` (lane, poll cadence, persist) and
    ``blocks.step.StepHost`` (``call_generation``). Subclasses set the state once
    from their workflow input via :meth:`configure`.
    """

    def __init__(self) -> None:
        # Replaced wholesale by run()'s configure(); held in workflow state so it
        # replays identically.
        self._settings = RunSettings()
        # Per-role occurrence counters for deterministic, replay-stable
        # interaction idempotency keys (Phase C survivable writes). Held in
        # workflow state so they replay identically; per-role so a repeated
        # same-role call never collides (T1.6a). Genuinely mutable run state —
        # this is why the host stays an object rather than becoming a value
        # threaded through every block seam.
        self._persist_occurrences: dict[str, int] = {}

    def configure(self, settings: RunSettings) -> None:
        """Adopt the run's dispatch settings (called once, at the top of ``run``)."""
        self._settings = settings

    @property
    def sync_mode(self) -> bool:
        """Whether this run calls the LLM synchronously instead of via batch."""
        return self._settings.sync_mode

    @property
    def poll_interval(self) -> timedelta:
        """The timer-loop batch poll cadence for this run (D88)."""
        return self._settings.poll_interval

    @property
    def log_messages(self) -> bool:
        """Whether activities write their request/response logs into the worktree."""
        return self._settings.log_messages

    async def persist_interaction(
        self, *, role: str, result: PersistableLLMResult, fields: PersistFields
    ) -> None:
        """Survivably persist one LLM interaction (idempotent on a per-run key)."""
        occurrence = self._persist_occurrences.get(role, 0)
        self._persist_occurrences[role] = occurrence + 1
        req = build_persist_interaction(
            idempotency_key=build_interaction_idempotency_key(
                workflow_id=workflow.info().workflow_id,
                run_id=workflow.info().run_id,
                role=role,
                occurrence=occurrence,
            ),
            role=role,
            task_id=fields.task_id,
            system_prompt=fields.system_prompt,
            user_prompt=fields.user_prompt,
            result=result,
            step_id=fields.step_id,
            sub_task_id=fields.sub_task_id,
            context_stats=fields.context_stats,
        )
        await persist_block(req)

    async def call_generation(self, context: AssembledContext) -> LLMCallResult:
        """Persisting LLM generation dispatch — the ``StepHost`` seam for blocks/step.py."""
        return await dispatch_generation(self, context)
