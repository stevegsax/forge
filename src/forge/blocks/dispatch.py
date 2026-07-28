"""The single dispatch block (T5.3) — one copy of the typed LLM-call shape.

Five forge LLM calls share one shape: pick a lane (sync activity, or the
Phase-4 batch transport), turn what comes back into the arm's typed result, and
write one interaction record. Before T5.3 that shape existed five times —
planner and sanity check inline in ``workflows.py``, generation and conflict
resolution as hand-rolled twins in the old ``workflow_blocks.py``, and exploration
inline, divergent, and persisting nothing at all.

Only four things actually differ between the arms, and they live in one pure
table (:data:`ARMS`): which activity the sync lane calls, how long that call may
take, which output type the batch lane asks for, and the batch ``max_tokens``
cap. Everything else — the lane fork, the eight stats fields mapped off
``ParsedLLMResponse``, the persist — is written once here.

The transport itself is *not* duplicated: ``blocks.transport.batch_submit_and_wait``
(T4.1, D88) stays the one submit/poll/fetch/parse implementation and this module
imports it, the same import direction ``blocks/step.py`` chose. Splitting the
monolith is T5.4's business.

One arm carries a gate as well as a call: :func:`dispatch_planner` is where the
deterministic plan preflight runs (T5.6). Because every planner dispatch — sync
and batch — passes through it, the checks and their retry arm exist in exactly
one place.

Command identity is load-bearing. Each lane emits exactly the activity sequence
it emitted before the consolidation, in the same order with the same timeouts
and retry policies, so the committed replay histories under ``tests/replay/``
replay unregenerated. The one addition is the exploration arm's interaction
persist, which appears in no committed history (every scenario runs
``max_exploration_rounds=0``).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from sax_platform.temporal.retries import IO_RETRY, LLM_RETRY

    from forge.blocks.transport import batch_submit_and_wait
    from forge.models import (
        MAX_PLANNER_ATTEMPTS,
        AssembledContext,
        ConflictResolutionCallInput,
        ConflictResolutionCallResult,
        ConflictResolutionResponse,
        ContextStats,
        ExplorationCallResult,
        ExplorationInput,
        ExplorationResponse,
        LLMCallResult,
        LLMResponse,
        ParsedLLMResponse,
        Plan,
        PlanCallResult,
        PlannerInput,
        SanityCheckCallResult,
        SanityCheckInput,
        SanityCheckResponse,
        ThinkingPolicy,
    )
    from forge.persist_models import PersistableLLMResult
    from forge.plan_checks import (
        PlanViolation,
        escalate_thinking,
        preflight_plan,
        retry_prompt_section,
        violation_summary,
    )
    from forge.presets import (
        CONFLICT_RESOLUTION_TIMEOUT,
        CONTEXT_TIMEOUT,
        DEFAULT_MAX_TOKENS,
        EXPLORATION_LLM_TIMEOUT,
        LLM_HEARTBEAT,
        LLM_TIMEOUT,
        SANITY_CHECK_TIMEOUT,
        THINKING_MAX_TOKENS,
    )

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from datetime import timedelta

    from pydantic import BaseModel

__all__ = [
    "ARMS",
    "ArmName",
    "DispatchArm",
    "DispatchHost",
    "PersistFields",
    "PlanAttempt",
    "PlanPreflightHalt",
    "call_stats",
    "conflict_result",
    "dispatch_conflict_resolution",
    "dispatch_exploration",
    "dispatch_generation",
    "dispatch_planner",
    "dispatch_sanity_check",
    "exploration_result",
    "generation_result",
    "plan_result",
    "sanity_result",
]


# ---------------------------------------------------------------------------
# The arm table (pure)
# ---------------------------------------------------------------------------

type ArmName = Literal[
    "generation",
    "planner",
    "sanity_check",
    "conflict_resolution",
    "exploration",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class DispatchArm:
    """The four things that actually differ between the dispatch arms.

    ``role`` is the interactions-table role the arm persists under — the only
    field that is not simply a call parameter, and the reason ``generation``
    (role ``llm``) cannot be derived from the arm name.
    """

    role: str
    sync_activity: str
    sync_timeout: timedelta
    output_type_name: str
    max_tokens: int = DEFAULT_MAX_TOKENS


ARMS: Mapping[ArmName, DispatchArm] = MappingProxyType(
    {
        "generation": DispatchArm(
            role="llm",
            sync_activity="call_llm",
            sync_timeout=LLM_TIMEOUT,
            output_type_name="LLMResponse",
        ),
        "planner": DispatchArm(
            role="planner",
            sync_activity="call_planner",
            sync_timeout=LLM_TIMEOUT,
            output_type_name="Plan",
            max_tokens=THINKING_MAX_TOKENS,
        ),
        "sanity_check": DispatchArm(
            role="sanity_check",
            sync_activity="call_sanity_check",
            sync_timeout=SANITY_CHECK_TIMEOUT,
            output_type_name="SanityCheckResponse",
            max_tokens=THINKING_MAX_TOKENS,
        ),
        "conflict_resolution": DispatchArm(
            role="conflict_resolution",
            sync_activity="call_conflict_resolution",
            sync_timeout=CONFLICT_RESOLUTION_TIMEOUT,
            output_type_name="ConflictResolutionResponse",
            max_tokens=THINKING_MAX_TOKENS,
        ),
        "exploration": DispatchArm(
            role="exploration",
            sync_activity="call_exploration_llm",
            sync_timeout=EXPLORATION_LLM_TIMEOUT,
            output_type_name="ExplorationResponse",
        ),
    }
)


# ---------------------------------------------------------------------------
# Stats mapping (pure) — the eight fields every arm maps identically
# ---------------------------------------------------------------------------


class CallStats(TypedDict):
    """The ``LLMStats`` fields, ready to splat into any arm's result model."""

    model_name: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    stop_reason: str | None


def call_stats(parsed: ParsedLLMResponse) -> CallStats:
    """Map a parsed batch response onto the stats every arm result carries.

    One copy, not five: a batch arm whose result forgot ``stop_reason`` or the
    cache-token fields would silently under-report spend in the interactions
    store, which is what T7.4's budget enforcement will read.
    """
    return CallStats(
        model_name=parsed.model_name,
        input_tokens=parsed.input_tokens,
        output_tokens=parsed.output_tokens,
        latency_ms=parsed.latency_ms,
        cache_creation_input_tokens=parsed.cache_creation_input_tokens,
        cache_read_input_tokens=parsed.cache_read_input_tokens,
        stop_reason=parsed.stop_reason,
    )


# ---------------------------------------------------------------------------
# Typed result builders (pure) — batch lane only; the sync lane's activity
# returns the arm's result type directly.
# ---------------------------------------------------------------------------


def generation_result(context: AssembledContext, parsed: ParsedLLMResponse) -> LLMCallResult:
    """Build a generation result from a parsed batch body."""
    return LLMCallResult(
        task_id=context.task_id,
        response=LLMResponse.model_validate_json(parsed.parsed_json),
        **call_stats(parsed),
    )


def plan_result(task_id: str, parsed: ParsedLLMResponse) -> PlanCallResult:
    """Build a planner result from a parsed batch body."""
    return PlanCallResult(
        task_id=task_id,
        plan=Plan.model_validate_json(parsed.parsed_json),
        **call_stats(parsed),
    )


def sanity_result(task_id: str, parsed: ParsedLLMResponse) -> SanityCheckCallResult:
    """Build a sanity-check result from a parsed batch body."""
    return SanityCheckCallResult(
        task_id=task_id,
        response=SanityCheckResponse.model_validate_json(parsed.parsed_json),
        **call_stats(parsed),
    )


def conflict_result(task_id: str, parsed: ParsedLLMResponse) -> ConflictResolutionCallResult:
    """Build a conflict-resolution result from a parsed batch body."""
    response = ConflictResolutionResponse.model_validate_json(parsed.parsed_json)
    return ConflictResolutionCallResult(
        task_id=task_id,
        resolved_files={f.file_path: f.content for f in response.resolved_files},
        explanation=response.explanation,
        **call_stats(parsed),
    )


def exploration_result(
    context: AssembledContext, parsed: ParsedLLMResponse
) -> ExplorationCallResult:
    """Build an exploration result from a parsed batch body.

    The prompts come from the assembled context, so the batch lane's envelope is
    the same shape the sync activity returns.
    """
    return ExplorationCallResult(
        task_id=context.task_id,
        response=ExplorationResponse.model_validate_json(parsed.parsed_json),
        system_prompt=context.system_prompt,
        user_prompt=context.user_prompt,
        **call_stats(parsed),
    )


# ---------------------------------------------------------------------------
# Host seam and persist fields
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PersistFields:
    """The per-call half of an interaction row; the arm supplies its ``role``."""

    task_id: str
    system_prompt: str
    user_prompt: str
    step_id: str | None = None
    sub_task_id: str | None = None
    context_stats: ContextStats | None = None


class DispatchHost(Protocol):
    """The workflow instance's per-run dispatch state and its persisting seam.

    Both workflow classes satisfy it through the one shared implementation in
    ``forge.blocks.host``; the block stays free of the workflow state (lane,
    poll cadence, per-role occurrence counters) it hides.
    """

    @property
    def sync_mode(self) -> bool: ...

    @property
    def poll_interval(self) -> timedelta: ...

    async def persist_interaction(
        self, *, role: str, result: PersistableLLMResult, fields: PersistFields
    ) -> None: ...


# ---------------------------------------------------------------------------
# Batch-lane context: prebuilt, or assembled by an activity first
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class AssembleVia:
    """A batch context that an activity must build first (the exploration arm).

    The four other arms hand ``typed_dispatch`` a context built purely from
    their own input, which emits no command — so an arm's assemble activity runs
    on the batch lane and *only* on the batch lane.
    """

    activity: str
    input: BaseModel
    timeout: timedelta = CONTEXT_TIMEOUT


type BatchContext = AssembledContext | AssembleVia

type BuildResult[R] = Callable[[AssembledContext, ParsedLLMResponse], R]
"""Batch-lane result builder: the assembled context plus the parsed body."""

type PersistFieldsOf[R] = Callable[[R], PersistFields]
"""Per-call interaction fields, derived from the arm's own result."""


async def _resolve_batch_context(source: BatchContext) -> AssembledContext:
    """Return the batch lane's assembled context, running its activity if needed."""
    if isinstance(source, AssembledContext):
        return source
    context: AssembledContext = await workflow.execute_activity(
        source.activity,
        source.input,
        start_to_close_timeout=source.timeout,
        retry_policy=IO_RETRY,
        result_type=AssembledContext,
    )
    return context


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------


async def typed_dispatch[R: PersistableLLMResult](
    host: DispatchHost,
    arm_name: ArmName,
    *,
    sync_input: BaseModel,
    result_type: type[R],
    batch_context: BatchContext,
    build: BuildResult[R],
    persist: PersistFieldsOf[R],
    thinking: ThinkingPolicy | None = None,
) -> R:
    """Run one LLM call on the host's lane and record its interaction.

    Sync lane: one activity call, whose result *is* the arm's typed result.
    Batch lane: the shared T4.1 transport, then ``build`` turns the parsed body
    into that same typed result. Either way exactly one interaction row is
    written, with the arm's role and the call's token counts.
    """
    arm = ARMS[arm_name]
    if host.sync_mode:
        result: R = await workflow.execute_activity(
            arm.sync_activity,
            sync_input,
            start_to_close_timeout=arm.sync_timeout,
            heartbeat_timeout=LLM_HEARTBEAT,
            retry_policy=LLM_RETRY,
            result_type=result_type,
        )
    else:
        context = await _resolve_batch_context(batch_context)
        parsed = await batch_submit_and_wait(
            context,
            arm.output_type_name,
            thinking=thinking,
            max_tokens=arm.max_tokens,
            poll_interval=host.poll_interval,
        )
        result = build(context, parsed)
    await host.persist_interaction(role=arm.role, result=result, fields=persist(result))
    return result


# ---------------------------------------------------------------------------
# The five arms
# ---------------------------------------------------------------------------


async def dispatch_generation(host: DispatchHost, context: AssembledContext) -> LLMCallResult:
    """Generation (role ``llm``) — the ``StepHost.call_generation`` payload."""

    def _persist(_result: LLMCallResult) -> PersistFields:
        return PersistFields(
            task_id=context.task_id,
            system_prompt=context.system_prompt,
            user_prompt=context.user_prompt,
            step_id=context.step_id,
            sub_task_id=context.sub_task_id,
            context_stats=context.context_stats,
        )

    # Generation stays thinking-disabled: omitting ``thinking`` relies on
    # batch_submit_and_wait's shared fallback (disabled by default, D94).
    return await typed_dispatch(
        host,
        "generation",
        sync_input=context,
        result_type=LLMCallResult,
        batch_context=context,
        build=generation_result,
        persist=_persist,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanAttempt:
    """One planner attempt the gate rejected: what it returned, and why.

    Pairing the two means a halt cannot carry stats and violations that have
    drifted out of alignment — there is no index invariant to get wrong.
    """

    result: PlanCallResult
    violations: tuple[PlanViolation, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanPreflightHalt:
    """No structurally valid plan after :data:`MAX_PLANNER_ATTEMPTS` attempts (T5.6).

    Returned rather than raised: ``ForgeTaskWorkflow.run`` treats a bare
    ``ApplicationError`` as a batch-wait failure (T1.6b), so raising would record
    this halt under the wrong ``failure_kind``. The caller turns it into a clean
    terminal result (Principle 5 — halt when confused).

    **Every** rejected attempt is carried, not just the last. A halt that paid
    for three planner calls must report three planner calls: the caller sums
    :attr:`attempts` into the terminal result's ``planner_stats`` and its
    ``llm_totals``, and the wording below names each attempt's own violations, so
    "why did this halt cost that much?" is answerable from the result alone.
    """

    attempts: tuple[PlanAttempt, ...]

    @property
    def attempt_count(self) -> int:
        """How many planner calls the halt paid for."""
        return len(self.attempts)

    @property
    def error(self) -> str:
        """The terminal-result wording: every attempt, in order, with its violations."""
        detail = "; ".join(
            f"attempt {number}: {violation_summary(attempt.violations)}"
            for number, attempt in enumerate(self.attempts, start=1)
        )
        return f"Plan rejected by preflight after {self.attempt_count} planner attempts: {detail}"


async def _plan_attempt(host: DispatchHost, planner_input: PlannerInput) -> PlanCallResult:
    """One planner call on the host's lane, with its interaction record."""
    return await typed_dispatch(
        host,
        "planner",
        sync_input=planner_input,
        result_type=PlanCallResult,
        batch_context=_context_from(planner_input),
        thinking=planner_input.thinking,
        build=lambda _ctx, parsed: plan_result(planner_input.task_id, parsed),
        persist=lambda _result: _prompt_fields(planner_input),
    )


def _retry_input(
    planner_input: PlannerInput, violations: Sequence[PlanViolation], *, attempt: int
) -> PlannerInput:
    """The next attempt's input: the violations appended, thinking escalated last.

    Pure — the escalating context is a longer prompt, not another activity, so a
    preflight retry costs exactly one more planner call.
    """
    update: dict[str, object] = {
        "user_prompt": planner_input.user_prompt
        + retry_prompt_section(violations, attempt=attempt, max_attempts=MAX_PLANNER_ATTEMPTS)
    }
    if attempt == MAX_PLANNER_ATTEMPTS:
        update["thinking"] = escalate_thinking(planner_input.thinking)
    return planner_input.model_copy(update=update)


async def dispatch_planner(
    host: DispatchHost, planner_input: PlannerInput
) -> PlanCallResult | PlanPreflightHalt:
    """Planning — the most expensive arm; thinks with the caller's policy.

    This is also the plan **preflight gate** (T5.6). Both lanes dispatch through
    here, so the deterministic structural checks
    (:func:`forge.plan_checks.preflight_plan`) run exactly once, in one place, on
    whatever the planner produced. A plan carrying duplicate ids, overlapping
    fan-out targets, unsafe paths, or an output-less leaf is rejected *before*
    any step runs, and the planner is asked again with the specific violations
    appended to its context — up to :data:`MAX_PLANNER_ATTEMPTS` attempts total.

    There is deliberately no backoff between attempts: a preflight failure is
    semantic, not transient (the transport already owns transient retry), so a
    timer would add nothing on the sync lane and hours on the batch lane.
    """
    attempt_input = planner_input
    rejected: list[PlanAttempt] = []
    for attempt in range(1, MAX_PLANNER_ATTEMPTS + 1):
        result = await _plan_attempt(host, attempt_input)
        violations = preflight_plan(result.plan)
        if not violations:
            return result
        rejected.append(PlanAttempt(result=result, violations=violations))
        workflow.logger.warning(
            "Plan preflight rejected the plan: task_id=%s attempt=%d/%d violations=%s",
            planner_input.task_id,
            attempt,
            MAX_PLANNER_ATTEMPTS,
            violation_summary(violations),
        )
        if attempt == MAX_PLANNER_ATTEMPTS:
            return PlanPreflightHalt(attempts=tuple(rejected))
        attempt_input = _retry_input(attempt_input, violations, attempt=attempt + 1)

    # Unreachable: MAX_PLANNER_ATTEMPTS >= 1, so the loop always returns.
    msg = f"Planner loop for {planner_input.task_id} ended without a verdict"
    raise RuntimeError(msg)


async def dispatch_sanity_check(
    host: DispatchHost, sanity_input: SanityCheckInput
) -> SanityCheckCallResult:
    """Mid-plan sanity check (continue / revise / abort)."""
    return await typed_dispatch(
        host,
        "sanity_check",
        sync_input=sanity_input,
        result_type=SanityCheckCallResult,
        batch_context=_context_from(sanity_input),
        thinking=sanity_input.thinking,
        build=lambda _ctx, parsed: sanity_result(sanity_input.task_id, parsed),
        persist=lambda _result: _prompt_fields(sanity_input),
    )


async def dispatch_conflict_resolution(
    host: DispatchHost, call_input: ConflictResolutionCallInput
) -> ConflictResolutionCallResult:
    """Merge of file versions two fan-out children both produced."""
    return await typed_dispatch(
        host,
        "conflict_resolution",
        sync_input=call_input,
        result_type=ConflictResolutionCallResult,
        batch_context=_context_from(call_input),
        thinking=call_input.thinking,
        build=lambda _ctx, parsed: conflict_result(call_input.task_id, parsed),
        persist=lambda _result: _prompt_fields(call_input, step_id=call_input.step_id),
    )


async def dispatch_exploration(
    host: DispatchHost, exploration_input: ExplorationInput
) -> ExplorationCallResult:
    """LLM-guided context exploration (Phase 7).

    The one arm whose prompts the workflow does not already hold: the sync
    activity builds them internally and returns them in the result envelope, and
    the batch lane gets them from ``assemble_exploration_context``. Either way
    the persist below has real prompts to write.
    """

    def _persist(result: ExplorationCallResult) -> PersistFields:
        return PersistFields(
            task_id=exploration_input.task_id,
            system_prompt=result.system_prompt,
            user_prompt=result.user_prompt,
        )

    # Exploration stays thinking-disabled (see dispatch_generation).
    return await typed_dispatch(
        host,
        "exploration",
        sync_input=exploration_input,
        result_type=ExplorationCallResult,
        batch_context=AssembleVia(
            activity="assemble_exploration_context",
            input=exploration_input,
        ),
        build=exploration_result,
        persist=_persist,
    )


# ---------------------------------------------------------------------------
# Shared adapter helpers (pure)
# ---------------------------------------------------------------------------

type PromptCarrier = PlannerInput | SanityCheckInput | ConflictResolutionCallInput
"""The three arm inputs that already carry their own assembled prompts."""


def _context_from(input: PromptCarrier) -> AssembledContext:
    """Rebuild the batch lane's context from an input that already has prompts."""
    return AssembledContext(
        task_id=input.task_id,
        system_prompt=input.system_prompt,
        user_prompt=input.user_prompt,
        model_name=input.model_name,
        log_messages=input.log_messages,
        worktree_path=input.worktree_path,
    )


def _prompt_fields(input: PromptCarrier, *, step_id: str | None = None) -> PersistFields:
    """Interaction fields for an arm whose prompts came in with its input."""
    return PersistFields(
        task_id=input.task_id,
        system_prompt=input.system_prompt,
        user_prompt=input.user_prompt,
        step_id=step_id,
    )
