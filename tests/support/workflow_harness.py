"""The per-test workflow harness (T5.5) — one canonical set of by-name mocks.

Every forge workflow test drives real workflow code against mocked activities.
Before T5.5 each section of ``tests/test_workflows.py`` re-stubbed the world
against its own module-level scenario globals: 184 ``@activity.defn``
decorators over 26 distinct names, 55 mutable module bindings, and 12
``_reset_*`` functions a test had to remember to call. Two tests could never
run at once (so ``pytest-xdist`` was impossible), and scripting by consumption
order (``list.pop(0)``, ``count % 3``) handed the next scripted outcome to
whichever *call arrived first* — a coin flip once parallel children are in
play, and the mechanism of the ~1/8
``TestRecursiveFanOutNestedFailure::test_failure_propagates`` flake.

This module replaces all of it with two pieces:

- :class:`ScenarioState` — one instance per test, constructed *in* the test, so
  it is unreachable from any other test by construction.
- :func:`build_activities` — the one canonical mock set, every activity a
  closure over that instance. Mock activities run worker-side (outside the
  workflow sandbox), so closing over test state is legal.

**Scripting is keyed by identity, never by arrival order.** There are two key
spaces, because the two activities that need scripting see different identity:

===================== ========================================================
Field                 Key
===================== ========================================================
``transitions``       the *validate identity* — ``ValidateOutputInput.task_id``:
``validations``       the task id for a root or planned step, the compound
                      ``<parent>.sub.<child>`` id for a sub-task or nested node.
``llm_responses``     the *call identity* — :func:`call_key` of the assembled
                      context: the compound sub-task id, else the step id, else
                      the task id.
``conflict_responses`` the resolving node's ``task_id``.
``sanity_responses``  the task's ``task_id``.
``parsed_responses``  the batch ``output_type_name``.
===================== ========================================================

Scripts handed in by a test are **read-only**: nothing here pops, clears, or
reassigns them. Sequential consumption (a plan's successive steps, a step's
successive attempts) uses an internal per-key cursor instead. Within one key,
every call comes from a single workflow's own sequential activity stream, so
cursor order is deterministic — it is never the arrival race D3 forbids.

The batch lane resolves identity through the ``request_id``, exactly as the
real transport does: the submit mock records ``request_id -> BatchSubmitInput``,
the fetch mock returns a body naming that ``request_id``, and the parse mock
recovers the originating context. ``ParseResponseInput`` alone carries only a
``task_id``, which cannot tell two fan-out children apart.

Call-log vocabulary: every entry is ``name`` or ``name:detail`` with one format
per activity. Use :meth:`ScenarioState.called` / :meth:`~ScenarioState.count` /
:meth:`~ScenarioState.entries` rather than matching raw strings when only the
activity name matters.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sax_platform.contracts.constants import FORGE_TASK_QUEUE
from temporalio import activity
from temporalio.worker import Worker

from forge.activities.conflict_resolution import classify_file_conflicts
from forge.models import (
    AssembleContextInput,
    AssembledContext,
    AssembleSanityCheckContextInput,
    AssembleStepContextInput,
    AssembleSubTaskContextInput,
    BatchFetchResult,
    BatchStatusInput,
    BatchStatusResult,
    BatchSubmitInput,
    BatchSubmitResult,
    CommitChangesInput,
    CommitChangesOutput,
    ConflictResolutionCallInput,
    ConflictResolutionCallResult,
    ConflictResolutionInput,
    ConflictResolutionResponse,
    ContextRequest,
    ContextResult,
    CreateWorktreeInput,
    CreateWorktreeOutput,
    DetectFileConflictsInput,
    DetectFileConflictsOutput,
    ExplorationCallResult,
    ExplorationInput,
    ExplorationResponse,
    FetchBatchResultInput,
    FileOutput,
    ForgeTaskInput,
    FulfillContextInput,
    LLMCallResult,
    LLMResponse,
    ParsedLLMResponse,
    ParseResponseInput,
    Plan,
    PlanCallResult,
    PlannerInput,
    RemoveWorktreeInput,
    ResetWorktreeInput,
    SanityCheckCallResult,
    SanityCheckInput,
    SanityCheckResponse,
    SanityCheckVerdict,
    SubTaskInput,
    SubTaskResult,
    TaskResult,
    TransitionSignal,
    ValidateOutputInput,
    ValidationResult,
    WriteFilesInput,
    WriteOutputInput,
    WriteResult,
)
from forge.persist_models import PersistRequest, PersistResult
from forge.workflows import ForgeSubTaskWorkflow, ForgeTaskWorkflow

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pydantic import BaseModel
    from temporalio.testing import WorkflowEnvironment

__all__ = [
    "DEFAULT_LLM_RESPONSE",
    "FAIL_VALIDATION",
    "PASS_VALIDATION",
    "ScenarioState",
    "build_activities",
    "call_key",
    "compound",
    "make_parsed",
    "run_sub_task",
    "run_task",
]


# ---------------------------------------------------------------------------
# Frozen test data (read-only; D8 — constants stay module level)
# ---------------------------------------------------------------------------

PASS_VALIDATION = ValidationResult(check_name="ruff_lint", passed=True, summary="ruff_lint passed")
FAIL_VALIDATION = ValidationResult(check_name="ruff_lint", passed=False, summary="ruff_lint failed")

DEFAULT_LLM_RESPONSE = LLMResponse(
    files=[FileOutput(file_path="hello.py", content="print('hello')\n")],
    explanation="Created hello module.",
)

DEFAULT_CONFLICT_RESOLUTION = ConflictResolutionCallResult(
    task_id="",
    resolved_files={},
    explanation="No conflicts resolved (default mock).",
    model_name="mock-reasoning",
    input_tokens=200,
    output_tokens=100,
    latency_ms=300.0,
)

DEFAULT_SANITY_CHECK = SanityCheckCallResult(
    task_id="",
    response=SanityCheckResponse(
        verdict=SanityCheckVerdict.CONTINUE,
        explanation="Plan looks good.",
    ),
    model_name="mock-reasoning",
    input_tokens=200,
    output_tokens=100,
    latency_ms=300.0,
)

DEFAULT_EXPLORATION_RESPONSE = ExplorationResponse(
    requests=[ContextRequest(provider="file_content", reasoning="need a peek")]
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def compound(parent_task_id: str, sub_task_id: str) -> str:
    """The compound sub-task id (``forge.step_logic.compound_sub_task_id``'s shape)."""
    return f"{parent_task_id}.sub.{sub_task_id}"


def call_key(context: AssembledContext) -> str:
    """The *call identity* an assembled context stands for.

    A sub-task's compound id, else the plan step's id, else the task id — the
    key ``ScenarioState.llm_responses`` is scripted by. The canonical assemble
    mocks stamp ``sub_task_id``/``step_id`` the way the real activities do, so
    this reads a real field rather than sniffing prompt text.
    """
    if context.sub_task_id:
        return compound(context.task_id, context.sub_task_id)
    return context.step_id or context.task_id


def make_parsed(
    model: "BaseModel",
    *,
    model_name: str = "mock-model",
    input_tokens: int = 100,
    output_tokens: int = 50,
    latency_ms: float = 200.0,
) -> ParsedLLMResponse:
    """Build a ``ParsedLLMResponse`` carrying any pydantic model's JSON."""
    return ParsedLLMResponse(
        parsed_json=model.model_dump_json(),
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )


def _parsed_like(
    model: "BaseModel", result: ConflictResolutionCallResult | SanityCheckCallResult
) -> ParsedLLMResponse:
    """Parse a batch body carrying the same spend its sync-lane twin reports."""
    return make_parsed(
        model,
        model_name=result.model_name,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
    )


# ---------------------------------------------------------------------------
# Per-test state
# ---------------------------------------------------------------------------


@dataclass
class ScenarioState:
    """One test's scripting and capture. Constructed in the test, never shared.

    Every ``Mapping``/``Sequence`` field is scripting the test hands in and the
    harness only reads; every ``list``/``dict`` with a ``default_factory`` below
    the capture banner is something the mocks record for the test to assert on.
    """

    # --- scripting: keyed by identity (see the module docstring's table) ---
    plan: Plan | None = None
    """The plan every planner call returns (sync activity and batch parse)."""

    transitions: "Mapping[str, Sequence[str]]" = field(default_factory=dict)
    """Validate identity -> transition tokens. A ``failure_terminal`` token is
    sticky: it stays current so every remaining attempt for that key fails."""

    validations: "Mapping[str, Sequence[Sequence[ValidationResult]]]" = field(default_factory=dict)
    """Validate identity -> explicit validation results, consumed before
    ``transitions`` (for tests that assert on the errors threaded into a retry)."""

    llm_responses: "Mapping[str, LLMResponse]" = field(default_factory=dict)
    """Call identity -> generation response; ``""`` is the fallback for
    unlisted identities, and :data:`DEFAULT_LLM_RESPONSE` backs that."""

    conflict_responses: "Mapping[str, ConflictResolutionCallResult]" = field(default_factory=dict)
    """Resolving node's task id -> resolution result (default: resolves nothing)."""

    sanity_responses: "Mapping[str, Sequence[SanityCheckCallResult]]" = field(default_factory=dict)
    """Task id -> successive sanity verdicts (default: CONTINUE)."""

    exploration_response: ExplorationResponse = field(
        default_factory=lambda: DEFAULT_EXPLORATION_RESPONSE
    )
    """What the exploration arm asks for each round."""

    parsed_responses: "Mapping[str, ParsedLLMResponse]" = field(default_factory=dict)
    """Batch ``output_type_name`` -> a verbatim parsed body. When non-empty the
    parse mock is strict: an unlisted output type is an error, not a default."""

    # --- scripting: batch transport knobs ---
    in_progress_polls: "Mapping[str, int]" = field(default_factory=dict)
    """Batch ``output_type_name`` -> ``in_progress`` polls before ``ended``
    (``""`` applies to every type). A value beyond 25h / poll interval stalls a
    waiter to its ceiling."""

    batch_state: str = ""
    """Force every ``batch_status`` poll to this provider-terminal state
    (``failed``/``expired``/``canceled``); ``""`` uses the normal progression."""

    fetch_error: str = ""
    """Make every ``fetch_batch_result`` return this error instead of a body."""

    # --- capture: assertion surfaces ---
    call_log: list[str] = field(default_factory=list)
    persisted: list[PersistRequest] = field(default_factory=list)
    submits: list[BatchSubmitInput] = field(default_factory=list)
    submits_by_type: dict[str, BatchSubmitInput] = field(default_factory=dict)
    status_polls: dict[str, int] = field(default_factory=dict)
    context_inputs: list[AssembleContextInput] = field(default_factory=list)
    step_context_inputs: list[AssembleStepContextInput] = field(default_factory=list)
    sub_task_context_inputs: list[AssembleSubTaskContextInput] = field(default_factory=list)
    conflict_inputs: list[ConflictResolutionInput] = field(default_factory=list)

    # --- internal: consumption cursors and request bookkeeping ---
    _cursors: dict[tuple[str, str], int] = field(default_factory=dict, repr=False)
    _submits_by_request: dict[str, BatchSubmitInput] = field(default_factory=dict, repr=False)

    # -- call-log helpers ------------------------------------------------

    def entries(self, name: str) -> list[str]:
        """Every call-log entry for one activity name (with or without detail)."""
        return [e for e in self.call_log if e == name or e.startswith(f"{name}:")]

    def called(self, name: str) -> bool:
        """Whether an activity was called at all."""
        return bool(self.entries(name))

    def count(self, name: str) -> int:
        """How many times an activity was called."""
        return len(self.entries(name))

    # -- scripting lookups ----------------------------------------------

    def _next(self, kind: str, key: str, length: int) -> int:
        """Return the current cursor for ``(kind, key)`` and advance it."""
        index = self._cursors.get((kind, key), 0)
        if index < length:
            self._cursors[(kind, key)] = index + 1
        return index

    def _peek(self, kind: str, key: str) -> int:
        return self._cursors.get((kind, key), 0)

    def next_validation(self, task_id: str) -> list[ValidationResult]:
        """The validation result driving this key's next scripted transition.

        Keyed successor of the old ``_validation_for_transition``: same sticky
        terminal semantics, but per key, so one child's terminal token can never
        be drawn by a sibling that happened to run first.
        """
        explicit = self.validations.get(task_id)
        if explicit:
            index = self._next("validations", task_id, len(explicit))
            if index < len(explicit):
                return list(explicit[index])

        sequence = self.transitions.get(task_id, ())
        index = self._peek("transitions", task_id)
        if index >= len(sequence):
            return [PASS_VALIDATION]
        token = sequence[index]
        if token == TransitionSignal.FAILURE_TERMINAL.value:
            return [FAIL_VALIDATION]  # sticky: the cursor stays put, so it keeps failing
        self._cursors[("transitions", task_id)] = index + 1
        if token == TransitionSignal.SUCCESS.value:
            return [PASS_VALIDATION]
        return [FAIL_VALIDATION]  # failure_retryable

    def llm_response_for(self, key: str) -> LLMResponse:
        """The generation response scripted for one call identity."""
        if key in self.llm_responses:
            return self.llm_responses[key]
        return self.llm_responses.get("", DEFAULT_LLM_RESPONSE)

    def conflict_response_for(self, task_id: str) -> ConflictResolutionCallResult:
        """The conflict resolution scripted for one resolving node."""
        canned = self.conflict_responses.get(task_id, self.conflict_responses.get(""))
        if canned is None:
            return DEFAULT_CONFLICT_RESOLUTION.model_copy(update={"task_id": task_id})
        return canned

    def sanity_response_for(self, task_id: str) -> SanityCheckCallResult:
        """The next sanity verdict scripted for one task (default: CONTINUE)."""
        scripted = self.sanity_responses.get(task_id, ())
        index = self._next("sanity", task_id, len(scripted))
        if index < len(scripted):
            return scripted[index]
        return DEFAULT_SANITY_CHECK.model_copy(update={"task_id": task_id})

    def the_plan(self) -> Plan:
        """The scripted plan, or a loud failure — a planner call needs one."""
        if self.plan is None:
            msg = "ScenarioState.plan is not set, but the workflow called the planner"
            raise RuntimeError(msg)
        return self.plan

    # -- batch bookkeeping ----------------------------------------------

    def record_submit(self, input: BatchSubmitInput) -> None:
        """Record a submit so the fetch/parse pair can recover its identity."""
        self.submits.append(input)
        self.submits_by_type[input.output_type_name] = input
        self._submits_by_request[input.request_id] = input

    def submit_for(self, request_id: str) -> BatchSubmitInput | None:
        """The submit a ``request_id`` came from, if this scenario saw it."""
        return self._submits_by_request.get(request_id)

    def next_batch_state(self, batch_id: str, output_type: str) -> str:
        """The provider status for one poll of one batch."""
        seen = self.status_polls.get(batch_id, 0)
        self.status_polls[batch_id] = seen + 1
        if self.batch_state:
            return self.batch_state
        stall = self.in_progress_polls.get(output_type, self.in_progress_polls.get("", 0))
        return "in_progress" if seen < stall else "ended"

    def parsed_for(self, output_type: str, context: AssembledContext | None) -> ParsedLLMResponse:
        """The parsed body for one batch result line.

        ``context`` is the submit's own assembled context, recovered from the
        ``request_id`` — that is what makes a per-child canned response possible
        on the batch lane.
        """
        if self.parsed_responses:
            canned = self.parsed_responses.get(output_type)
            if canned is None:
                msg = f"No parse response queued for output type {output_type!r}"
                raise RuntimeError(msg)
            return canned

        task_id = context.task_id if context is not None else ""
        if output_type == "Plan":
            return make_parsed(
                self.the_plan(), model_name="mock-planner", input_tokens=300, output_tokens=150
            )
        if output_type == "SanityCheckResponse":
            sanity = self.sanity_response_for(task_id)
            return _parsed_like(sanity.response, sanity)
        if output_type == "ConflictResolutionResponse":
            resolution = self.conflict_response_for(task_id)
            response = ConflictResolutionResponse(
                resolved_files=[
                    FileOutput(file_path=path, content=content)
                    for path, content in resolution.resolved_files.items()
                ],
                explanation=resolution.explanation,
            )
            return _parsed_like(response, resolution)
        if output_type == "ExplorationResponse":
            return make_parsed(self.exploration_response, model_name="mock-explorer")
        key = call_key(context) if context is not None else task_id
        return make_parsed(self.llm_response_for(key))


# ---------------------------------------------------------------------------
# The one canonical activity set
# ---------------------------------------------------------------------------


def build_activities(
    state: ScenarioState,
    *,
    replace: "Mapping[str, Callable[..., object]] | None" = None,
) -> "list[Callable[..., object]]":
    """Every activity name the forge workflows call, bound to one scenario.

    Registering the full set is deliberate: which activities a run actually
    calls is decided by the workflow input, not by what the worker knows about,
    and an activity the workflow calls but the worker lacks does not fail fast —
    the test hangs to its timeout (guide §4.1).

    ``replace`` swaps one canonical mock for a test-local one, keyed by activity
    name (the replacement must carry the same ``@activity.defn(name=...)``) —
    for the rare scenario whose behavior is not expressible as data on
    :class:`ScenarioState`, such as an activity that must block until a worker
    is shut down out from under it.
    """

    @activity.defn(name="persist_to_store")
    async def persist_to_store(req: PersistRequest) -> PersistResult:
        state.persisted.append(req)
        return PersistResult(kind=req.kind, applied=True)

    # -- git -------------------------------------------------------------

    @activity.defn(name="create_worktree_activity")
    async def create_worktree(input: CreateWorktreeInput) -> CreateWorktreeOutput:
        state.call_log.append(f"create_worktree:{input.task_id}")
        return CreateWorktreeOutput(
            worktree_path=f"/tmp/repo/.forge-worktrees/{input.task_id}",
            branch_name=f"forge/{input.task_id}",
        )

    @activity.defn(name="remove_worktree_activity")
    async def remove_worktree(input: RemoveWorktreeInput) -> None:
        state.call_log.append(f"remove_worktree:{input.task_id}")

    @activity.defn(name="reset_worktree_activity")
    async def reset_worktree(input: ResetWorktreeInput) -> None:
        state.call_log.append(f"reset_worktree:{input.task_id}")

    @activity.defn(name="commit_changes_activity")
    async def commit_changes(input: CommitChangesInput) -> CommitChangesOutput:
        state.call_log.append(f"commit:{input.message or input.status}")
        return CommitChangesOutput(commit_sha="a" * 40)

    # -- context assembly ------------------------------------------------

    @activity.defn(name="assemble_context")
    async def assemble_context(input: AssembleContextInput) -> AssembledContext:
        state.call_log.append(f"assemble_context:{input.task_id}")
        state.context_inputs.append(input)
        return AssembledContext(
            task_id=input.task_id,
            system_prompt="system prompt",
            user_prompt="user prompt",
            worktree_path=input.worktree_path,
        )

    @activity.defn(name="assemble_step_context")
    async def assemble_step_context(input: AssembleStepContextInput) -> AssembledContext:
        state.call_log.append(f"assemble_step_context:{input.step.step_id}")
        state.step_context_inputs.append(input)
        return AssembledContext(
            task_id=input.task_id,
            system_prompt=f"step system prompt for {input.step.step_id}",
            user_prompt=f"step user prompt for {input.step.step_id}",
            step_id=input.step.step_id,
            worktree_path=input.worktree_path,
        )

    @activity.defn(name="assemble_sub_task_context")
    async def assemble_sub_task_context(input: AssembleSubTaskContextInput) -> AssembledContext:
        sub_task_id = input.sub_task.sub_task_id
        state.call_log.append(f"assemble_sub_task_context:{sub_task_id}")
        state.sub_task_context_inputs.append(input)
        return AssembledContext(
            task_id=input.parent_task_id,
            system_prompt=f"sub-task prompt for {sub_task_id}",
            user_prompt=f"execute {sub_task_id}",
            sub_task_id=sub_task_id,
            worktree_path=input.worktree_path,
        )

    @activity.defn(name="assemble_planner_context")
    async def assemble_planner_context(input: AssembleContextInput) -> PlannerInput:
        state.call_log.append("assemble_planner_context")
        return PlannerInput(
            task_id=input.task_id,
            system_prompt="planner system prompt",
            user_prompt="planner user prompt",
        )

    @activity.defn(name="assemble_sanity_check_context")
    async def assemble_sanity_check_context(
        input: AssembleSanityCheckContextInput,
    ) -> SanityCheckInput:
        state.call_log.append("assemble_sanity_check_context")
        return SanityCheckInput(
            task_id=input.task_id,
            system_prompt="sanity check system prompt",
            user_prompt="sanity check user prompt",
        )

    @activity.defn(name="assemble_conflict_resolution_context")
    async def assemble_conflict_resolution_context(
        input: ConflictResolutionInput,
    ) -> ConflictResolutionCallInput:
        state.call_log.append("assemble_conflict_resolution_context")
        state.conflict_inputs.append(input)
        # Mirrors the real activity, which threads model_name and thinking
        # through: a mock that dropped them would mask propagation bugs.
        return ConflictResolutionCallInput(
            task_id=input.task_id,
            step_id=input.step_id,
            system_prompt="conflict resolution system prompt",
            user_prompt="conflict resolution user prompt",
            model_name=input.model_name,
            thinking=input.thinking,
        )

    @activity.defn(name="assemble_exploration_context")
    async def assemble_exploration_context(input: ExplorationInput) -> AssembledContext:
        state.call_log.append("assemble_exploration_context")
        return AssembledContext(
            task_id=input.task_id,
            system_prompt="exploration system",
            user_prompt="exploration user",
            worktree_path=input.worktree_path,
        )

    # -- sync LLM lane ---------------------------------------------------

    @activity.defn(name="call_llm")
    async def call_llm(context: AssembledContext) -> LLMCallResult:
        state.call_log.append("call_llm")
        return LLMCallResult(
            task_id=context.task_id,
            response=state.llm_response_for(call_key(context)),
            model_name="mock-model",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200.0,
        )

    @activity.defn(name="call_planner")
    async def call_planner(input: PlannerInput) -> PlanCallResult:
        state.call_log.append("call_planner")
        return PlanCallResult(
            task_id=input.task_id,
            plan=state.the_plan(),
            model_name="mock-planner",
            input_tokens=300,
            output_tokens=150,
            latency_ms=500.0,
        )

    @activity.defn(name="call_sanity_check")
    async def call_sanity_check(input: SanityCheckInput) -> SanityCheckCallResult:
        state.call_log.append("call_sanity_check")
        return state.sanity_response_for(input.task_id)

    @activity.defn(name="call_conflict_resolution")
    async def call_conflict_resolution(
        input: ConflictResolutionCallInput,
    ) -> ConflictResolutionCallResult:
        state.call_log.append("call_conflict_resolution")
        return state.conflict_response_for(input.task_id)

    @activity.defn(name="call_exploration_llm")
    async def call_exploration_llm(input: ExplorationInput) -> ExplorationCallResult:
        state.call_log.append("call_exploration_llm")
        return ExplorationCallResult(
            task_id=input.task_id,
            response=state.exploration_response,
            system_prompt="exploration system",
            user_prompt="exploration user",
            model_name="mock-explorer",
            input_tokens=41,
            output_tokens=17,
            latency_ms=90.0,
        )

    @activity.defn(name="fulfill_context_requests")
    async def fulfill_context_requests(input: FulfillContextInput) -> list[ContextResult]:
        state.call_log.append("fulfill_context_requests")
        return [
            ContextResult(provider="file_content", content="explored content", estimated_tokens=10)
        ]

    # -- batch lane ------------------------------------------------------

    @activity.defn(name="submit_batch_request")
    async def submit_batch_request(input: BatchSubmitInput) -> BatchSubmitResult:
        state.call_log.append(f"submit_batch_request:{input.output_type_name}")
        state.record_submit(input)
        # Echo the workflow-minted request_id (T4.1) and give every waiter its
        # own batch_id, so poll counters never interleave between waiters.
        return BatchSubmitResult(
            request_id=input.request_id,
            batch_id=f"batch-{input.request_id}",
            provider="anthropic",
        )

    @activity.defn(name="batch_status")
    async def batch_status(input: BatchStatusInput) -> BatchStatusResult:
        state.call_log.append(f"batch_status:{input.batch_id}")
        submit = state.submit_for(input.batch_id.removeprefix("batch-"))
        output_type = submit.output_type_name if submit else ""
        return BatchStatusResult(
            batch_id=input.batch_id,
            state=state.next_batch_state(input.batch_id, output_type),
        )

    @activity.defn(name="fetch_batch_result")
    async def fetch_batch_result(input: FetchBatchResultInput) -> BatchFetchResult:
        state.call_log.append(f"fetch_batch_result:{input.request_id}")
        if state.fetch_error:
            return BatchFetchResult(error=state.fetch_error)
        # The body names its own request_id, exactly as a provider result line
        # carries its custom_id — that is how the parse mock knows whose line
        # this is when two children are in flight.
        return BatchFetchResult(raw_response_json=f'{{"request_id": "{input.request_id}"}}')

    @activity.defn(name="parse_llm_response")
    async def parse_llm_response(input: ParseResponseInput) -> ParsedLLMResponse:
        output_type = input.output_type_name or ""
        state.call_log.append(f"parse_llm_response:{output_type}")
        submit = state.submit_for(_request_id_of(input.raw_response_json))
        return state.parsed_for(output_type, submit.context if submit else None)

    # -- write / validate / merge ----------------------------------------

    @activity.defn(name="write_output")
    async def write_output(input: WriteOutputInput) -> WriteResult:
        state.call_log.append("write_output")
        files = input.llm_result.response.files
        return WriteResult(
            task_id=input.llm_result.task_id,
            files_written=[f.file_path for f in files],
            output_files={f.file_path: f.content for f in files},
        )

    @activity.defn(name="write_files")
    async def write_files(input: WriteFilesInput) -> WriteResult:
        state.call_log.append(f"write_files:{len(input.files)}")
        return WriteResult(
            task_id=input.task_id,
            files_written=list(input.files.keys()),
            output_files=dict(input.files),
        )

    @activity.defn(name="validate_output")
    async def validate_output(input: ValidateOutputInput) -> list[ValidationResult]:
        state.call_log.append(f"validate_output:{input.task_id}")
        return state.next_validation(input.task_id)

    @activity.defn(name="detect_file_conflicts_activity")
    async def detect_file_conflicts(input: DetectFileConflictsInput) -> DetectFileConflictsOutput:
        state.call_log.append("detect_file_conflicts")
        non_conflicting, conflicts = classify_file_conflicts(input.sub_task_results)
        return DetectFileConflictsOutput(
            non_conflicting_files=non_conflicting,
            conflicts=conflicts,
        )

    canonical: dict[str, Callable[..., object]] = {
        "persist_to_store": persist_to_store,
        "create_worktree_activity": create_worktree,
        "remove_worktree_activity": remove_worktree,
        "reset_worktree_activity": reset_worktree,
        "commit_changes_activity": commit_changes,
        "assemble_context": assemble_context,
        "assemble_step_context": assemble_step_context,
        "assemble_sub_task_context": assemble_sub_task_context,
        "assemble_planner_context": assemble_planner_context,
        "assemble_sanity_check_context": assemble_sanity_check_context,
        "assemble_conflict_resolution_context": assemble_conflict_resolution_context,
        "assemble_exploration_context": assemble_exploration_context,
        "call_llm": call_llm,
        "call_planner": call_planner,
        "call_sanity_check": call_sanity_check,
        "call_conflict_resolution": call_conflict_resolution,
        "call_exploration_llm": call_exploration_llm,
        "fulfill_context_requests": fulfill_context_requests,
        "submit_batch_request": submit_batch_request,
        "batch_status": batch_status,
        "fetch_batch_result": fetch_batch_result,
        "parse_llm_response": parse_llm_response,
        "write_output": write_output,
        "write_files": write_files,
        "validate_output": validate_output,
        "detect_file_conflicts_activity": detect_file_conflicts,
    }
    for name, replacement in (replace or {}).items():
        if name not in canonical:
            msg = f"{name!r} is not a canonical activity name"
            raise KeyError(msg)
        canonical[name] = replacement
    return list(canonical.values())


def _request_id_of(raw_response_json: str | None) -> str:
    """Recover the ``request_id`` the fetch mock stamped into a result body."""
    if not raw_response_json:
        return ""
    marker = '"request_id": "'
    start = raw_response_json.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = raw_response_json.find('"', start)
    return raw_response_json[start:end]


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

WORKFLOWS = (ForgeTaskWorkflow, ForgeSubTaskWorkflow)


async def run_task(
    env: "WorkflowEnvironment",
    input: ForgeTaskInput,
    state: ScenarioState,
    *,
    workflow_id: str = "",
    activities: "Sequence[Callable[..., object]] | None" = None,
) -> TaskResult:
    """Run a ``ForgeTaskWorkflow`` to completion against one scenario."""
    async with Worker(
        env.client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=WORKFLOWS,
        activities=list(activities) if activities is not None else build_activities(state),
    ):
        return await env.client.execute_workflow(
            ForgeTaskWorkflow.run,
            input,
            id=workflow_id or f"test-{input.task.task_id}",
            task_queue=FORGE_TASK_QUEUE,
        )


async def run_sub_task(
    env: "WorkflowEnvironment",
    input: SubTaskInput,
    state: ScenarioState,
    *,
    workflow_id: str = "",
    activities: "Sequence[Callable[..., object]] | None" = None,
) -> SubTaskResult:
    """Run a ``ForgeSubTaskWorkflow`` to completion against one scenario."""
    async with Worker(
        env.client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=WORKFLOWS,
        activities=list(activities) if activities is not None else build_activities(state),
    ):
        return await env.client.execute_workflow(
            ForgeSubTaskWorkflow.run,
            input,
            id=workflow_id or f"test-subtask-{input.sub_task.sub_task_id}",
            task_queue=FORGE_TASK_QUEUE,
        )
