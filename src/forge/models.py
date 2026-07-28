"""Core data models for Forge."""

from __future__ import annotations

from datetime import timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from collections.abc import Iterable

# BatchJobStatus is the cross-queue wire contract; it now lives in sax-platform
# (T3.4, ST7) and is re-exported here so existing `from forge.models import ...`
# call sites keep working. BatchSubmitResult used to live there too, but as of
# T4.2 ST3 it no longer crosses a queue (ocr owns its own submit and typed
# output) — it is defined locally below with the rest of the batch models.
from sax_platform.contracts.models import (
    BatchJobStatus as BatchJobStatus,
)

# The model-tier registry and thinking policy are single-sourced on the platform
# (D94, T3.2): forge's former CapabilityTier/ModelConfig/resolve_model/
# ThinkingConfig are retired in favor of these re-exports, so existing
# `from forge.models import ...` call sites keep working unchanged. Explicit
# `as` aliases satisfy mypy strict's no_implicit_reexport.
from sax_platform.llm.tiers import (
    CapabilityTier as CapabilityTier,
)
from sax_platform.llm.tiers import (
    Effort as Effort,
)
from sax_platform.llm.tiers import (
    ModelConfig as ModelConfig,
)
from sax_platform.llm.tiers import (
    ThinkingPolicy as ThinkingPolicy,
)
from sax_platform.llm.tiers import (
    resolve_model as resolve_model,
)

# The per-wait batch ceiling now lives in ``sax_platform.temporal.polling`` (T4.2
# ST1), the module that owns the shared batch poll loop. It is re-exported here so
# forge's execution-timeout math (``_batch_execution_timeout`` /
# ``derive_execution_timeout``) and ``forge.step_logic.child_timeout`` keep
# importing the 25h ceiling from ``forge.models`` unchanged.
from sax_platform.temporal.polling import (
    BATCH_WAIT_CEILING as BATCH_WAIT_CEILING,
)

# ---------------------------------------------------------------------------
# Execution-timeout derivation constants (T4.1 ST3c)
# ---------------------------------------------------------------------------

# Hard cap on planner-produced plan length. An oversized plan becomes a validation
# failure at the parse seam (retryable there) instead of an unbounded execution
# timeout; it is what lets ``derive_execution_timeout`` close the timeout tree for
# planned mode. Enforced as ``max_length`` on ``Plan.steps``.
MAX_PLAN_STEPS: int = 25

# Total planner calls one run may make, including the first (T5.6). A plan that
# fails the deterministic preflight gate is re-planned with the specific
# violations appended to the planner's context; after this many attempts the run
# halts cleanly rather than executing a plan known to be malformed (Principle 5).
# Preflight failures are semantic, not transient — the transport owns transient
# retry — so the attempts carry escalating context and no backoff.
MAX_PLANNER_ATTEMPTS: int = 3

# How many times a sanity check may replace the plan's remaining steps (T5.6).
# The REVISE splice is the one path that rewrites a running plan; execution is
# already bounded by MAX_PLAN_STEPS (the step index only advances), so this cap
# is about thrash rather than termination: more than a handful of re-plans means
# the sanity checker and the planner disagree persistently, which is a halt.
MAX_PLAN_REVISIONS: int = 5

# Flat 48h execution timeout for sync mode (no batch waits) — the pre-T4.1 default.
_SYNC_EXECUTION_TIMEOUT: timedelta = timedelta(hours=48)

# Orchestration headroom added on top of the pure batch-wait budget in batch mode:
# non-batch git/context/write/validate/transition activity time surrounds each wait
# (per-wait), plus one-time worktree/planner setup and scheduling slack (base).
_PER_WAIT_ORCHESTRATION: timedelta = timedelta(minutes=10)
_EXECUTION_TIMEOUT_BASE: timedelta = timedelta(hours=1)


class MatchLevel(StrEnum):
    """Which matching strategy succeeded for a search/replace edit."""

    EXACT = "exact"
    WHITESPACE = "whitespace"
    INDENTATION = "indentation"
    FUZZY = "fuzzy"


class TransitionSignal(StrEnum):
    """Outcome signals that the orchestrator acts on."""

    SUCCESS = "success"
    FAILURE_RETRYABLE = "failure_retryable"
    FAILURE_TERMINAL = "failure_terminal"

    # Future phases:
    # NEW_TASKS_DISCOVERED = "new_tasks_discovered"
    # BLOCKED_ON_HUMAN = "blocked_on_human"
    # BLOCKED_ON_SIBLING = "blocked_on_sibling"


class TaskDomain(StrEnum):
    """The kind of task being performed."""

    CODE_GENERATION = "code_generation"
    RESEARCH = "research"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    GENERIC = "generic"


class SanityCheckVerdict(StrEnum):
    """Verdict from a plan-level sanity check."""

    CONTINUE = "continue"
    REVISE = "revise"
    ABORT = "abort"


class ValidationConfig(BaseModel):
    """Configuration for deterministic validation checks."""

    auto_fix: bool = True
    run_ruff_lint: bool = True
    run_ruff_format: bool = True
    run_tests: bool = False
    test_command: str | None = None
    test_timeout_seconds: int | None = Field(
        default=None,
        description=(
            "Cap in seconds for the test command. Falls back to "
            "TEST_TIMEOUT_SECONDS (aligned to the validate activity timeout) when "
            "unset. On timeout the check fails with a ValidationResult rather than "
            "crashing the activity."
        ),
    )


class ContextConfig(BaseModel):
    """Configuration for automatic context discovery."""

    auto_discover: bool = True
    include_dependencies: bool = Field(
        default=False,
        description=(
            "Include direct import contents and transitive symbol signatures. "
            "When False (default), only target files and repo map are assembled "
            "upfront; the LLM can pull dependencies on demand via exploration."
        ),
    )
    token_budget: int = Field(default=100_000, description="Token budget for context.")
    output_reserve: int = Field(default=16_000, description="Tokens reserved for LLM output.")
    max_import_depth: int = Field(default=2, description="How deep to trace imports.")
    include_repo_map: bool = True
    repo_map_tokens: int = Field(default=2048, description="Token budget for the repo map.")
    package_name: str | None = Field(
        default=None,
        description="Python package name for import graph. Auto-detected if None.",
    )


class ContextStats(BaseModel):
    """Observability stats from context assembly."""

    files_discovered: int = 0
    files_included_full: int = 0
    files_included_signatures: int = 0
    files_truncated: int = 0
    total_estimated_tokens: int = 0
    budget_utilization: float = Field(default=0.0, description="0.0 to 1.0.")
    repo_map_tokens: int = 0


class TaskDefinition(BaseModel):
    """A single unit of work to be executed by the workflow."""

    task_id: str
    description: str = Field(description="What the task should produce.")
    domain: TaskDomain = Field(
        default=TaskDomain.CODE_GENERATION,
        description="The kind of task: code generation, research, review, documentation.",
    )
    target_files: list[str] = Field(
        default_factory=list,
        description="Files to create or modify. Optional when planning.",
    )
    context_files: list[str] = Field(
        default_factory=list,
        description="Files to include as context for the LLM.",
    )
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    base_branch: str = Field(
        default="main",
        description="Branch to create the worktree from.",
    )
    context: ContextConfig = Field(default_factory=ContextConfig)


class ValidationResult(BaseModel):
    """Output from a single validation check."""

    check_name: str
    passed: bool
    summary: str = Field(description="Concise summary of the result.")
    details: str | None = Field(
        default=None,
        description="Extended details, available on request. Not sent to LLM by default.",
    )


# ---------------------------------------------------------------------------
# Run outcome classification (T5.1, D95)
# ---------------------------------------------------------------------------

# One-field terminal-failure classifier set on Task/Step/SubTaskResult (None on
# success). The values map 1:1 to the terminal construction sites that the pure
# step_logic result builders route through, so a failure's category is a typed
# field rather than something a reader must recover from the free-text ``error``.
FailureKind = Literal[
    "validation",  # terminal validation failure (incl. leaf sub-tasks)
    "batch_wait",  # the batch wait gave up / provider reported terminal / fetch error
    "step_failed",  # TaskResult: a planned step failed
    "sub_task_failed",  # fan-out gather saw failed children
    "child_crashed",  # a child workflow raised instead of returning a result (T5.3)
    "sanity_abort",  # sanity check returned ABORT
    "plan_preflight",  # T5.6: no structurally valid plan in MAX_PLANNER_ATTEMPTS tries
    "plan_revision",  # T5.6: a REVISE splice is over a cap or structurally invalid
    "duplicate_sub_task_ids",  # fan-out sub-task ids not unique
    "conflict_unresolved",  # D27: conflicts with resolve_conflicts=False
    "conflict_incomplete",  # conflict resolution left paths unresolved
    "merged_validation",  # merged fan-out output failed validation
]


class LLMRunTotals(BaseModel):
    """Run-level aggregate of every LLM call in a finished result tree (D97).

    Summed across models, so there is deliberately no ``model_name`` (it would
    lie) and the latency field is named ``llm_time_ms`` — the sum of per-call
    latencies is total LLM time, not wall-clock time, which for parallel fan-out
    children is much less than the sum. This is result-level spend *visibility*;
    the interactions table remains the authoritative accounting record, and stats
    of earlier retried attempts are not in the tree (only surviving calls count).
    """

    call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    llm_time_ms: float = 0.0

    @classmethod
    def from_stats(cls, stats: Iterable[LLMStats]) -> LLMRunTotals:
        """Sum per-call stats into run totals (the field mapping lives here)."""
        stats_list = list(stats)
        return cls(
            call_count=len(stats_list),
            input_tokens=sum(s.input_tokens for s in stats_list),
            output_tokens=sum(s.output_tokens for s in stats_list),
            cache_creation_input_tokens=sum(s.cache_creation_input_tokens for s in stats_list),
            cache_read_input_tokens=sum(s.cache_read_input_tokens for s in stats_list),
            llm_time_ms=sum(s.latency_ms for s in stats_list),
        )


# ---------------------------------------------------------------------------
# Planning models
# ---------------------------------------------------------------------------


class SubTask(BaseModel):
    """A single sub-task within a fan-out step."""

    sub_task_id: str = Field(description="Unique identifier within the parent step.")
    description: str = Field(description="What this sub-task should produce.")
    target_files: list[str] = Field(description="Files to create or modify.")
    context_files: list[str] = Field(
        default_factory=list,
        description="Files to include as context (read from parent worktree).",
    )
    sub_tasks: list[SubTask] | None = Field(
        default=None,
        description="Optional nested sub-tasks for recursive fan-out.",
    )


class PlanStep(BaseModel):
    """A single step within a plan."""

    step_id: str = Field(description="Unique step identifier within the plan.")
    description: str = Field(description="What this step should accomplish.")
    target_files: list[str] = Field(description="Files to create or modify in this step.")
    context_files: list[str] = Field(
        default_factory=list,
        description="Files to include as context for this step.",
    )
    sub_tasks: list[SubTask] | None = Field(
        default=None,
        description="Optional sub-tasks for fan-out parallel execution.",
    )
    capability_tier: CapabilityTier | None = Field(
        default=None,
        description="Optional capability tier override for this step's LLM call.",
    )


class Plan(BaseModel):
    """A decomposed plan for a task."""

    task_id: str
    # ``max_length`` caps the plan at MAX_PLAN_STEPS: an oversized LLM-produced plan
    # is rejected at the parse seam (retryable there) rather than yielding an
    # unbounded workflow execution timeout (T4.1 ST3c).
    steps: list[PlanStep] = Field(min_length=1, max_length=MAX_PLAN_STEPS)
    explanation: str = Field(description="Brief explanation of the decomposition strategy.")


class SubTaskResult(BaseModel):
    """The outcome of a single sub-task execution."""

    sub_task_id: str
    status: TransitionSignal
    output_files: dict[str, str] = Field(default_factory=dict)
    output_digests: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "path -> sha256 hex of this node's own produced output (T5.1). "
            "Mutually exclusive with output_files: the parent's conflict "
            "detection consumes output_files during gather, then the slim "
            "builders empty it into digests, so contents travel at most once "
            "— in the top-level TaskResult.output_files for folded successful "
            "steps; a failed step's contents are dropped from the result and "
            "survive only in its worktree."
        ),
    )
    validation_results: list[ValidationResult] = Field(default_factory=list)
    digest: str = Field(default="", description="From LLMResponse.explanation (D18).")
    error: str | None = None
    failure_kind: FailureKind | None = None
    llm_stats: LLMStats | None = None
    sub_task_results: list[SubTaskResult] = Field(default_factory=list)
    conflict_resolution: ConflictResolutionCallResult | None = None

    @model_validator(mode="after")
    def _contents_travel_once(self) -> Self:
        if self.output_files and self.output_digests:
            msg = "output_files and output_digests are mutually exclusive (contents travel once)"
            raise ValueError(msg)
        return self

    @property
    def file_count(self) -> int:
        """Number of output files this node produced, in either lifecycle state."""
        return len(self.output_files) + len(self.output_digests)


class StepResult(BaseModel):
    """The outcome of executing a single plan step."""

    step_id: str
    status: TransitionSignal
    output_files: dict[str, str] = Field(default_factory=dict)
    output_digests: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "path -> sha256 hex of this step's own produced output (T5.1). "
            "Mutually exclusive with output_files. See SubTaskResult.output_digests."
        ),
    )
    validation_results: list[ValidationResult] = Field(default_factory=list)
    commit_sha: str | None = None
    error: str | None = None
    failure_kind: FailureKind | None = None
    sub_task_results: list[SubTaskResult] = Field(default_factory=list)
    llm_stats: LLMStats | None = None
    digest: str = Field(
        default="",
        description="Compact summary of step outcome for sanity check consumption.",
    )
    conflict_resolution: ConflictResolutionCallResult | None = None

    @model_validator(mode="after")
    def _contents_travel_once(self) -> Self:
        if self.output_files and self.output_digests:
            msg = "output_files and output_digests are mutually exclusive (contents travel once)"
            raise ValueError(msg)
        return self

    @property
    def file_count(self) -> int:
        """Number of output files this step produced, in either lifecycle state."""
        return len(self.output_files) + len(self.output_digests)


class TaskResult(BaseModel):
    """The outcome of a workflow execution."""

    task_id: str
    status: TransitionSignal
    output_files: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of file path to content.",
    )
    validation_results: list[ValidationResult] = Field(default_factory=list)
    error: str | None = Field(
        default=None,
        description="If the task failed, a concise explanation of why.",
    )
    failure_kind: FailureKind | None = None
    worktree_path: str | None = None
    worktree_branch: str | None = None
    step_results: list[StepResult] = Field(default_factory=list)
    plan: Plan | None = None
    llm_stats: LLMStats | None = None
    planner_stats: LLMStats | None = None
    context_stats: ContextStats | None = None
    llm_totals: LLMRunTotals | None = Field(
        default=None,
        description=(
            "Run-level LLM spend aggregated across the finished result tree "
            "(D97), computed once before the run is persisted."
        ),
    )
    sanity_check_count: int = 0


# ---------------------------------------------------------------------------
# LLM statistics (Phase 5)
# ---------------------------------------------------------------------------


class LLMStats(BaseModel):
    """Lightweight LLM call statistics for Temporal payloads."""

    model_name: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    stop_reason: str | None = Field(
        default=None,
        description=(
            "Provider stop_reason for the response (e.g. end_turn, max_tokens, "
            "tool_use). None if unavailable or unparseable."
        ),
    )


def build_llm_stats(result: LLMStats) -> LLMStats:
    """Extract an LLMStats from any LLMStats subclass (strips domain-specific fields)."""
    return LLMStats(
        model_name=result.model_name,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
        cache_creation_input_tokens=result.cache_creation_input_tokens,
        cache_read_input_tokens=result.cache_read_input_tokens,
        stop_reason=result.stop_reason,
    )


# ---------------------------------------------------------------------------
# LLM structured output
# ---------------------------------------------------------------------------


class FileOutput(BaseModel):
    """A single file produced by the LLM."""

    file_path: str = Field(description="Relative path within the worktree.")
    content: str = Field(description="Complete file content.")


class SearchReplaceEdit(BaseModel):
    """A single search/replace operation within a file."""

    search: str = Field(description="Exact text to find in the file. Must match exactly once.")
    replace: str = Field(description="Text to replace the match with.")


class FileEdit(BaseModel):
    """Edits to an existing file via search/replace."""

    file_path: str = Field(description="Relative path within the worktree.")
    edits: list[SearchReplaceEdit] = Field(description="Ordered search/replace edits to apply.")


class LLMResponse(BaseModel):
    """Structured output from the LLM call.

    Invariant (T5.6): a response must carry at least one file or one edit. A
    do-nothing response used to parse cleanly and then sail through the pipeline
    — nothing written, validation run over zero files, the step reported SUCCESS
    having produced nothing. Rejecting it here turns that silent no-op into a
    schema mismatch at the parse seam, where ``LLMSchemaMismatch`` is
    deliberately *retryable* (a differently-sampled call can produce real
    output). This is a contract violation, distinct from T3.5's refusal and
    truncation outcomes: there the model declined or was cut off, here it
    answered with nothing.
    """

    files: list[FileOutput] = Field(
        default_factory=list,
        description="New files to create with complete content.",
    )
    edits: list[FileEdit] = Field(
        default_factory=list,
        description="Search/replace edits for existing files.",
    )
    explanation: str = Field(description="Brief explanation of what was produced.")

    @model_validator(mode="after")
    def _require_output(self) -> LLMResponse:
        """Reject a response that produces neither a file nor an edit."""
        if not self.files and not self.edits:
            msg = "LLMResponse produced no output: both 'files' and 'edits' are empty"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Exploration models (Phase 7)
# ---------------------------------------------------------------------------


class ContextProviderSpec(BaseModel):
    """Description of an available context provider shown to the LLM."""

    name: str
    description: str
    parameters: dict[str, str] = Field(description="param_name -> description")


class ContextRequest(BaseModel):
    """A request for specific context from a provider."""

    provider: str
    params: dict[str, str] = Field(default_factory=dict)
    reasoning: str = Field(description="Why this context is needed.")


class ExplorationResponse(BaseModel):
    """Output from the exploration LLM call."""

    requests: list[ContextRequest] = Field(
        description="Context requests. Empty list signals readiness to generate.",
    )


class ContextResult(BaseModel):
    """Result of fulfilling a context request."""

    provider: str
    content: str
    estimated_tokens: int


class FulfillContextInput(BaseModel):
    """Input to the fulfill_context_requests activity."""

    requests: list[ContextRequest]
    repo_root: str
    worktree_path: str


class ExplorationInput(BaseModel):
    """Input to the exploration LLM call."""

    task_id: str
    task_description: str
    target_files: list[str]
    context_files: list[str]
    context_config: ContextConfig
    available_providers: list[ContextProviderSpec]
    domain: TaskDomain = Field(default=TaskDomain.CODE_GENERATION)
    accumulated_context: list[ContextResult] = Field(default_factory=list)
    round_number: int
    max_rounds: int
    repo_root: str = Field(default="", description="Repo root for reading project instructions.")
    model_name: str = ""
    log_messages: bool = False
    worktree_path: str = ""


class ExplorationCallResult(LLMStats):
    """Output of the exploration dispatch arm — response, spend, and prompts.

    The other four arms already return an ``LLMStats`` subclass; exploration
    returned a bare ``ExplorationResponse``, so its token counts died in a trace
    span and there was nothing to write to the interactions store (T5.3). The
    prompts travel with the result because the sync lane assembles them *inside*
    the activity — the workflow never sees them otherwise, and an interaction row
    without its prompts is not an interaction record.
    """

    task_id: str
    response: ExplorationResponse
    system_prompt: str
    user_prompt: str


# ---------------------------------------------------------------------------
# Knowledge extraction models (Phase 6)
# ---------------------------------------------------------------------------


class PlaybookEntry(BaseModel):
    """A structured lesson extracted from completed work."""

    title: str = Field(description="Short descriptive title of the lesson.")
    content: str = Field(description="The actionable lesson or pattern.")
    tags: list[str] = Field(
        default_factory=list,
        description="Index tags: task type, domain, error pattern, etc.",
    )
    source_task_id: str = Field(description="Task ID this was extracted from.")
    source_workflow_id: str = Field(
        default="",
        description="Workflow ID this was extracted from.",
    )


class PlaybookReviewResult(BaseModel):
    """Structured output from playbook entry review."""

    approved: bool = Field(description="Whether the entry is acceptable for storage.")
    rejection_reason: str = Field(
        default="",
        description="If rejected: why (confusing, incomplete, duplicate, etc.).",
    )
    suggested_tags: list[str] = Field(
        default_factory=list,
        description="Tags the reviewer recommends adding or correcting.",
    )
    suggested_title: str = Field(
        default="",
        description="Improved title, or empty if the original is fine.",
    )
    suggested_content: str = Field(
        default="",
        description="Improved content, or empty if the original is fine.",
    )


class ExtractionResult(BaseModel):
    """Structured output from the knowledge extraction LLM call."""

    entries: list[PlaybookEntry] = Field(
        description="Extracted playbook entries from the completed work.",
    )
    summary: str = Field(
        description="Brief summary of what was extracted and why.",
    )


# ---------------------------------------------------------------------------
# Manual playbook workflow models
# ---------------------------------------------------------------------------


class ExportPlaybookInput(BaseModel):
    """Input to ExportPlaybookWorkflow."""

    tags: list[str] = Field(default_factory=list)
    source_task_id: str = ""
    limit: int = 0  # 0 = no limit


class ExportPlaybookResult(BaseModel):
    """Result from ExportPlaybookWorkflow."""

    entries: list[PlaybookEntry]
    count: int


class FetchPlaybookIdsInput(BaseModel):
    """Input for fetch_playbook_ids activity."""

    tags: list[str] = Field(default_factory=list)
    source_task_id: str = ""
    limit: int = 0


class ExportSinglePlaybookInput(BaseModel):
    """Input for export_single_playbook activity."""

    playbook_id: int


class ValidatePlaybookInput(BaseModel):
    """Input for playbook validation activity."""

    raw_json: str = Field(description="Raw JSON string to validate.")


class ValidatePlaybookResult(BaseModel):
    """Result from playbook validation activity."""

    valid: bool
    error: str = ""
    entry: PlaybookEntry | None = None


class FetchExistingPlaybooksInput(BaseModel):
    """Input for fetching existing playbooks for duplication context."""

    limit: int = 50


class ReviewManualPlaybookInput(BaseModel):
    """Input for the review activity."""

    entry: PlaybookEntry
    existing_playbooks: list[dict[str, Any]] = Field(default_factory=list)
    model_name: str = ""


class ReviewManualPlaybookResult(BaseModel):
    """Output from the review activity."""

    approved: bool
    rejection_reason: str = ""
    final_entry: PlaybookEntry


class ManualPlaybookInput(BaseModel):
    """Input for the manual playbook add workflow."""

    raw_json: str = Field(description="Raw JSON string for the playbook entry.")
    model_routing: ModelConfig = Field(default_factory=ModelConfig)


class ManualPlaybookResult(BaseModel):
    """Result from the manual playbook add workflow."""

    approved: bool
    rejection_reason: str = ""
    validation_error: str = ""
    entry: PlaybookEntry | None = None


# ---------------------------------------------------------------------------
# Conflict resolution models
# ---------------------------------------------------------------------------


class FileConflictVersion(BaseModel):
    """One competing version of a conflicting file."""

    source_id: str = Field(description="Sub-task ID that produced this version.")
    content: str = Field(description="Full file content.")


class FileConflict(BaseModel):
    """A file produced by multiple sub-tasks."""

    file_path: str = Field(description="Relative path in worktree.")
    versions: list[FileConflictVersion] = Field(description="2+ competing versions.")
    original_content: str | None = Field(
        default=None,
        description="Pre-existing content (None = new file).",
    )


class DetectFileConflictsInput(BaseModel):
    """Input to detect_file_conflicts_activity."""

    sub_task_results: list[SubTaskResult]
    worktree_path: str | None = None


class DetectFileConflictsOutput(BaseModel):
    """Output from detect_file_conflicts_activity."""

    non_conflicting_files: dict[str, str]
    conflicts: list[FileConflict]


class ConflictResolutionInput(BaseModel):
    """Input to assemble_conflict_resolution_context activity."""

    task_id: str
    step_id: str
    conflicts: list[FileConflict]
    non_conflicting_files: dict[str, str] = Field(
        description="Already-merged files for context.",
    )
    task_description: str
    step_description: str
    repo_root: str
    worktree_path: str
    domain: TaskDomain
    model_name: str = ""
    thinking: ThinkingPolicy = Field(default_factory=ThinkingPolicy)


class ConflictResolutionResponse(BaseModel):
    """Structured LLM output -- resolved files."""

    resolved_files: list[FileOutput] = Field(
        description="Resolved file contents for each conflicting path.",
    )
    explanation: str = Field(description="How the conflicts were resolved.")


class ConflictResolutionCallInput(BaseModel):
    """Input to call_conflict_resolution activity (assembled prompts)."""

    task_id: str
    step_id: str
    system_prompt: str
    user_prompt: str
    model_name: str = ""
    thinking: ThinkingPolicy = Field(default_factory=ThinkingPolicy)
    log_messages: bool = False
    worktree_path: str = ""


class ConflictResolutionCallResult(LLMStats):
    """Output of call_conflict_resolution activity."""

    task_id: str
    resolved_files: dict[str, str] = Field(
        description="file_path -> merged content.",
    )
    explanation: str


# ---------------------------------------------------------------------------
# Inter-activity transport
# ---------------------------------------------------------------------------


class AssembledContext(BaseModel):
    """Output of assemble_context, input to call_llm."""

    task_id: str
    system_prompt: str
    user_prompt: str
    context_stats: ContextStats | None = None
    step_id: str | None = None
    sub_task_id: str | None = None
    model_name: str = ""
    log_messages: bool = False
    worktree_path: str = ""


class LLMCallResult(LLMStats):
    """Output of call_llm, input to write_output."""

    task_id: str
    response: LLMResponse


class WriteResult(BaseModel):
    """Output of write_output."""

    task_id: str
    files_written: list[str]
    output_files: dict[str, str] = Field(
        default_factory=dict,
        description="Final file contents (path -> content) for all written files.",
    )


# ---------------------------------------------------------------------------
# Activity input models (Temporal single-arg convention)
# ---------------------------------------------------------------------------


class AssembleContextInput(BaseModel):
    """Input to the assemble_context activity."""

    task_id: str
    description: str
    target_files: list[str]
    context_files: list[str]
    context_config: ContextConfig
    repo_root: str
    worktree_path: str
    prior_errors: list[ValidationResult] = Field(default_factory=list)
    attempt: int = Field(default=1)
    max_attempts: int = Field(default=2)


class WriteOutputInput(BaseModel):
    """Input to the write_output activity."""

    llm_result: LLMCallResult
    worktree_path: str


class ValidateOutputInput(BaseModel):
    """Input to the validate_output activity."""

    task_id: str
    worktree_path: str
    files: list[str]
    validation: ValidationConfig


# ---------------------------------------------------------------------------
# Workflow input model
# ---------------------------------------------------------------------------


class ForgeTaskInput(BaseModel):
    """Input to ForgeTaskWorkflow."""

    task: TaskDefinition
    repo_root: str
    max_attempts: int = 2
    plan: bool = Field(default=False, description="Enable planning mode.")
    max_step_attempts: int = Field(
        default=2,
        description="Max retry attempts per step in planning mode.",
    )
    max_sub_task_attempts: int = Field(
        default=2,
        description="Max retry attempts per sub-task in fan-out steps.",
    )
    max_exploration_rounds: int = Field(
        default=10,
        description="Max rounds of LLM-guided context exploration (0 disables).",
    )
    max_fan_out_depth: int = Field(
        default=1,
        description="Maximum recursive fan-out depth. 1 = flat fan-out only (default).",
    )
    sanity_check_interval: int = Field(
        default=0,
        description="Run sanity check every N steps (0 = disabled).",
    )
    resolve_conflicts: bool = Field(
        default=True,
        description="Attempt LLM-based conflict resolution for fan-out file conflicts.",
    )
    model_routing: ModelConfig = Field(default_factory=ModelConfig)
    thinking: ThinkingPolicy = Field(default_factory=ThinkingPolicy)
    sync_mode: bool = Field(
        default=False,
        description="Use synchronous Messages API. False enables batch mode (default).",
    )
    log_messages: bool = Field(
        default=False,
        description="Save full API request/response JSON to messages/ in the worktree.",
    )
    batch_poll_interval_seconds: int = Field(
        default=600,
        ge=300,
        description=(
            "Seconds between batch status polls in the timer-loop transport (D88: "
            "configurable, never below 300 to protect the provider batch API)."
        ),
    )


def _batch_execution_timeout(waits: int) -> timedelta:
    """Wall-clock ceiling for ``waits`` sequential 25h batch waits, plus orchestration.

    Each wait may block up to :data:`BATCH_WAIT_CEILING`; the surrounding non-batch
    activities add a per-wait allowance, and one-time setup adds a flat base.
    """
    return waits * BATCH_WAIT_CEILING + waits * _PER_WAIT_ORCHESTRATION + _EXECUTION_TIMEOUT_BASE


def derive_execution_timeout(task_input: ForgeTaskInput) -> timedelta:
    """Derive a workflow execution timeout from the permitted batch-wait budget.

    Pure: no Temporal imports, importable by the CLI and by tests (T4.1 ST3c). The
    flat sync default is preserved; batch mode is derived from the maximum number of
    sequential 25h batch waits a run can legally perform under its input knobs, so a
    legitimately slow batch is never killed by the execution timeout.

    - **sync mode** → flat 48h (no batch waits).
    - **batch single-step** → ``max_attempts * (max_exploration_rounds + 1)`` waits:
      each attempt runs the exploration loop (≤ ``max_exploration_rounds`` waits) then
      one generation wait.
    - **batch planned** → one planner phase
      (``max_exploration_rounds + MAX_PLANNER_ATTEMPTS`` waits: exploration runs
      once, then up to :data:`MAX_PLANNER_ATTEMPTS` planner calls, since a plan
      rejected by the T5.6 preflight gate is re-planned), then
      ≤ :data:`MAX_PLAN_STEPS` steps. A step is either a regular step
      (≤ ``max_step_attempts`` generation waits) or a fan-out step whose parent-side
      budget is the depth-0 child budget (``max_sub_task_attempts + max_fan_out_depth``
      waits, which the parent blocks on inside its own execution timeout) plus one
      conflict-resolution wait. Sanity checks add ≤ one wait per interval.
    """
    if task_input.sync_mode:
        return _SYNC_EXECUTION_TIMEOUT
    if not task_input.plan:
        waits = task_input.max_attempts * (task_input.max_exploration_rounds + 1)
        return _batch_execution_timeout(waits)
    planner_waits = task_input.max_exploration_rounds + MAX_PLANNER_ATTEMPTS
    # Depth-0 child budget: max_sub_task_attempts leaf waits + one wait per nesting
    # level (remaining = max_fan_out_depth - 0). The parent adds one conflict-resolution
    # wait after the child gathers.
    child_budget = task_input.max_sub_task_attempts + task_input.max_fan_out_depth
    per_step_waits = max(task_input.max_step_attempts, child_budget + 1)
    sanity_waits = (
        MAX_PLAN_STEPS // task_input.sanity_check_interval
        if task_input.sanity_check_interval > 0
        else 0
    )
    waits = planner_waits + MAX_PLAN_STEPS * per_step_waits + sanity_waits
    return _batch_execution_timeout(waits)


# ---------------------------------------------------------------------------
# Git activity I/O models
# ---------------------------------------------------------------------------


class CreateWorktreeInput(BaseModel):
    """Input to create_worktree_activity."""

    repo_root: str
    task_id: str
    base_branch: str = "main"


class CreateWorktreeOutput(BaseModel):
    """Output from create_worktree_activity."""

    worktree_path: str
    branch_name: str


class RemoveWorktreeInput(BaseModel):
    """Input to remove_worktree_activity."""

    repo_root: str
    task_id: str
    force: bool = True


class CommitChangesInput(BaseModel):
    """Input to commit_changes_activity."""

    repo_root: str
    task_id: str
    status: str
    file_paths: list[str] | None = None
    message: str | None = Field(
        default=None,
        description="Override the auto-generated commit message.",
    )


class CommitChangesOutput(BaseModel):
    """Output from commit_changes_activity."""

    commit_sha: str


class ResetWorktreeInput(BaseModel):
    """Input to reset_worktree_activity."""

    repo_root: str
    task_id: str


# ---------------------------------------------------------------------------
# Fan-out activity I/O models
# ---------------------------------------------------------------------------


class SubTaskInput(BaseModel):
    """Input to ForgeSubTaskWorkflow."""

    parent_task_id: str
    parent_description: str = Field(description="Parent task description for context assembly.")
    sub_task: SubTask
    repo_root: str
    parent_branch: str = Field(description="e.g. 'forge/my-task'")
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    max_attempts: int = 2
    model_name: str = ""
    domain: TaskDomain = Field(default=TaskDomain.CODE_GENERATION)
    depth: int = Field(default=0, description="Current fan-out depth.")
    max_depth: int = Field(default=1, description="Maximum allowed fan-out depth.")
    resolve_conflicts: bool = Field(
        default=True,
        description=(
            "Attempt LLM-based conflict resolution for nested fan-out file "
            "conflicts (D71). Inherited from parent workflow; False falls back "
            "to the D27 terminal error."
        ),
    )
    model_routing: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Capability-tier model routing. Inherited from parent workflow.",
    )
    thinking: ThinkingPolicy = Field(
        default_factory=ThinkingPolicy,
        description="Extended thinking policy. Inherited from parent workflow.",
    )
    sync_mode: bool = Field(
        default=False,
        description="Use synchronous Messages API. Inherited from parent workflow.",
    )
    log_messages: bool = False
    batch_poll_interval_seconds: int = Field(
        default=600,
        ge=300,
        description=(
            "Seconds between batch status polls (D88); inherited from the parent workflow."
        ),
    )


class WriteFilesInput(BaseModel):
    """Input to write_files activity."""

    task_id: str
    worktree_path: str
    files: dict[str, str] = Field(description="Mapping of relative path to content.")


class AssembleSubTaskContextInput(BaseModel):
    """Input to assemble_sub_task_context activity."""

    parent_task_id: str
    parent_description: str
    sub_task: SubTask
    worktree_path: str = Field(description="Parent worktree (for reading context files).")
    repo_root: str = Field(default="", description="Repo root for reading project instructions.")
    context_config: ContextConfig = Field(
        default_factory=ContextConfig,
        description="Controls auto-discovery for the sub-task (auto-discovers by default).",
    )
    prior_errors: list[ValidationResult] = Field(default_factory=list)
    attempt: int = Field(default=1)
    max_attempts: int = Field(default=2)
    domain: TaskDomain = Field(default=TaskDomain.CODE_GENERATION)


# ---------------------------------------------------------------------------
# Planning activity I/O models
# ---------------------------------------------------------------------------


class PlannerInput(BaseModel):
    """Input to the call_planner activity."""

    task_id: str
    system_prompt: str
    user_prompt: str
    model_name: str = ""
    thinking: ThinkingPolicy = Field(default_factory=ThinkingPolicy)
    log_messages: bool = False
    worktree_path: str = ""


class PlanCallResult(LLMStats):
    """Output of call_planner."""

    task_id: str
    plan: Plan


class AssembleStepContextInput(BaseModel):
    """Input to assemble_step_context activity."""

    task_id: str
    task_description: str
    context_config: ContextConfig
    step: PlanStep
    step_index: int
    total_steps: int
    completed_steps: list[StepResult] = Field(default_factory=list)
    repo_root: str
    worktree_path: str
    prior_errors: list[ValidationResult] = Field(default_factory=list)
    attempt: int = Field(default=1)
    max_attempts: int = Field(default=2)


# ---------------------------------------------------------------------------
# Sanity check activity I/O models
# ---------------------------------------------------------------------------


class SanityCheckResponse(BaseModel):
    """LLM structured output from the sanity check."""

    verdict: SanityCheckVerdict
    explanation: str
    revised_steps: list[PlanStep] | None = Field(
        default=None,
        description="Replacement steps when verdict is 'revise'.",
    )


class SanityCheckInput(BaseModel):
    """Input to the call_sanity_check activity."""

    task_id: str
    system_prompt: str
    user_prompt: str
    model_name: str = ""
    thinking: ThinkingPolicy = Field(default_factory=ThinkingPolicy)
    log_messages: bool = False
    worktree_path: str = ""


class SanityCheckCallResult(LLMStats):
    """Output of call_sanity_check."""

    task_id: str
    response: SanityCheckResponse


class AssembleSanityCheckContextInput(BaseModel):
    """Input to assemble_sanity_check_context activity."""

    task_id: str
    task_description: str
    plan: Plan
    completed_steps: list[StepResult]
    remaining_steps: list[PlanStep]
    repo_root: str
    worktree_path: str


# ---------------------------------------------------------------------------
# Extraction activity I/O models (Phase 6)
# ---------------------------------------------------------------------------


class FetchExtractionInput(BaseModel):
    """Input to the fetch_extraction_input activity."""

    limit: int = Field(default=10, description="Max runs to extract from.")
    since_hours: int = Field(default=24, description="Look-back window in hours.")


class ExtractionInput(BaseModel):
    """Output of fetch_extraction_input, input to call_extraction_llm."""

    system_prompt: str
    user_prompt: str
    source_workflow_ids: list[str] = Field(
        description="Workflow IDs being processed.",
    )
    model_name: str = ""


class ExtractionCallResult(LLMStats):
    """Output of call_extraction_llm."""

    result: ExtractionResult
    source_workflow_ids: list[str]


class SaveExtractionInput(BaseModel):
    """Input to save_extraction_results activity."""

    entries: list[PlaybookEntry]
    source_workflow_ids: list[str]
    extraction_workflow_id: str


class ExtractionWorkflowInput(BaseModel):
    """Parameters for a knowledge-extraction run."""

    limit: int = Field(default=10, description="Max runs to extract from.")
    since_hours: int = Field(default=24, description="Look-back window in hours.")
    model_routing: ModelConfig = Field(default_factory=ModelConfig)


class ExtractionWorkflowResult(BaseModel):
    """Result of a knowledge-extraction run."""

    entries_created: int
    source_workflow_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Batch processing models (Phase 14)
# ---------------------------------------------------------------------------


class BatchSubmitInput(BaseModel):
    """Input to the submit_batch_request activity."""

    context: AssembledContext
    output_type_name: str = Field(description="Key in forge.output_types.OUTPUT_TYPES.")
    workflow_id: str = Field(description="Temporal workflow ID for audit linkage.")
    thinking: ThinkingPolicy = Field(default_factory=ThinkingPolicy)
    max_tokens: int = Field(default=4096, description="Max output tokens.")
    request_id: str = Field(
        description=(
            "Workflow-minted custom_id for the batch request (D88: "
            "``workflow.uuid4()`` in the workflow closes the submit-retry orphan "
            "window). A retried submit reuses the same custom_id, so the provider "
            "dedupes to one paid batch."
        ),
    )


class BatchSubmitResult(BaseModel):
    """Outcome of a forge batch submission (the ``submit_batch_request`` activity).

    Forge-internal since T4.2 ST3: it no longer crosses a task queue (ocr owns
    its own submit with its own typed output). Forge submits anthropic only, so
    ``provider`` is threaded back — always ``"anthropic"`` here — and the
    workflow persists the submission survivably with it (honest transport).
    """

    request_id: str = Field(description="Provider custom_id == batch_jobs PK, minted once.")
    batch_id: str = Field(description="Provider batch ID.")
    provider: str = Field(
        default="anthropic",
        description="Provider name, threaded back so the workflow can record the submission.",
    )


# ---------------------------------------------------------------------------
# Timer-loop batch transport models (T4.1, D88) — forge-internal activity I/O.
# These are NOT cross-workflow signal payloads: the requester is the recipient,
# so status/fetch results travel as ordinary activity returns.
# ---------------------------------------------------------------------------


class BatchStatusInput(BaseModel):
    """Input to the batch_status activity — one provider status poll."""

    batch_id: str = Field(description="Provider batch ID to poll.")
    provider: str = Field(
        description=(
            "Provider name, threaded from ``BatchSubmitResult.provider`` (no "
            "default — the caller always knows which provider it submitted to)."
        ),
    )


class BatchStatusResult(BaseModel):
    """Normalized, provider-agnostic snapshot of a batch's lifecycle state."""

    batch_id: str
    state: Literal["in_progress", "ended", "failed", "expired", "canceled"] = Field(
        description="Normalized lifecycle state, mapped from the provider's own status.",
    )


class FetchBatchResultInput(BaseModel):
    """Input to the fetch_batch_result activity — download one waiter's result line."""

    batch_id: str = Field(description="Provider batch ID to fetch results from.")
    request_id: str = Field(
        description="This waiter's custom_id; selects its line among the batch's results.",
    )
    provider: str = Field(
        description=(
            "Provider name, threaded from ``BatchSubmitResult.provider`` (no "
            "default — required, matching BatchStatusInput)."
        ),
    )


class BatchFetchResult(BaseModel):
    """Claim-check result of fetch_batch_result: exactly one field is set.

    ``raw_response_json`` carries the result body inline when it is small
    (<=256KB) and image-free; ``s3_key`` points at a stashed result envelope when
    the body is large or carries images; ``error`` is set when this waiter's line
    failed at the provider or its custom_id was absent from the finished batch.
    Mutually exclusive by construction.
    """

    raw_response_json: str | None = None
    s3_key: str | None = None
    error: str | None = None


class ParseResponseInput(BaseModel):
    """Input to a parse activity that deserializes a batch response (14b).

    The result body travels either inline (``raw_response_json``) or by reference
    (``s3_key``, fetched by the activity); exactly one is set. ``s3_key`` points at
    a result envelope, so the activity unwraps it before parsing.
    """

    raw_response_json: str | None = None
    s3_key: str | None = None
    output_type_name: str | None = None
    task_id: str
    provider: str = Field(default="anthropic", description="LLM provider name for parsing.")
    log_messages: bool = False
    worktree_path: str = ""
    max_tokens: int = Field(
        default=4096,
        description=(
            "The max_tokens cap the originating submit requested — carried through "
            "so the parse activity can name it in the max_tokens truncation warning."
        ),
    )


class ParsedLLMResponse(LLMStats):
    """Generic parsed LLM response from sync or batch path."""

    parsed_json: str = Field(description="JSON of the parsed Pydantic model (tool_use input).")
    latency_ms: float = 0.0
