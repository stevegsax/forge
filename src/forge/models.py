"""Core data models for Forge."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


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


class ValidationConfig(BaseModel):
    """Configuration for deterministic validation checks."""

    auto_fix: bool = True
    run_ruff_lint: bool = True
    run_ruff_format: bool = True
    run_tests: bool = False
    test_command: str | None = None


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


class Plan(BaseModel):
    """A decomposed plan for a task."""

    task_id: str
    steps: list[PlanStep] = Field(min_length=1)
    explanation: str = Field(description="Brief explanation of the decomposition strategy.")


class SubTaskResult(BaseModel):
    """The outcome of a single sub-task execution."""

    sub_task_id: str
    status: TransitionSignal
    output_files: dict[str, str] = Field(default_factory=dict)
    validation_results: list[ValidationResult] = Field(default_factory=list)
    digest: str = Field(default="", description="From LLMResponse.explanation (D18).")
    error: str | None = None
    llm_stats: LLMStats | None = None


class StepResult(BaseModel):
    """The outcome of executing a single plan step."""

    step_id: str
    status: TransitionSignal
    output_files: dict[str, str] = Field(default_factory=dict)
    validation_results: list[ValidationResult] = Field(default_factory=list)
    commit_sha: str | None = None
    error: str | None = None
    sub_task_results: list[SubTaskResult] = Field(default_factory=list)
    llm_stats: LLMStats | None = None


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
    worktree_path: str | None = None
    worktree_branch: str | None = None
    step_results: list[StepResult] = Field(default_factory=list)
    plan: Plan | None = None
    llm_stats: LLMStats | None = None
    planner_stats: LLMStats | None = None
    context_stats: ContextStats | None = None


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


def build_llm_stats(llm_result: LLMCallResult) -> LLMStats:
    """Build LLMStats from an LLMCallResult."""
    return LLMStats(
        model_name=llm_result.model_name,
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
        latency_ms=llm_result.latency_ms,
        cache_creation_input_tokens=llm_result.cache_creation_input_tokens,
        cache_read_input_tokens=llm_result.cache_read_input_tokens,
    )


def build_planner_stats(planner_result: PlanCallResult) -> LLMStats:
    """Build LLMStats from a PlanCallResult."""
    return LLMStats(
        model_name=planner_result.model_name,
        input_tokens=planner_result.input_tokens,
        output_tokens=planner_result.output_tokens,
        latency_ms=planner_result.latency_ms,
        cache_creation_input_tokens=planner_result.cache_creation_input_tokens,
        cache_read_input_tokens=planner_result.cache_read_input_tokens,
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
    """Structured output model for pydantic-ai Agent."""

    files: list[FileOutput] = Field(
        default_factory=list,
        description="New files to create with complete content.",
    )
    edits: list[FileEdit] = Field(
        default_factory=list,
        description="Search/replace edits for existing files.",
    )
    explanation: str = Field(description="Brief explanation of what was produced.")


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

    task: TaskDefinition
    available_providers: list[ContextProviderSpec]
    accumulated_context: list[ContextResult] = Field(default_factory=list)
    round_number: int
    max_rounds: int
    repo_root: str = Field(default="", description="Repo root for reading project instructions.")


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


class ExtractionResult(BaseModel):
    """Structured output from the knowledge extraction LLM call."""

    entries: list[PlaybookEntry] = Field(
        description="Extracted playbook entries from the completed work.",
    )
    summary: str = Field(
        description="Brief summary of what was extracted and why.",
    )


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


class LLMCallResult(BaseModel):
    """Output of call_llm, input to write_output."""

    task_id: str
    response: LLMResponse
    model_name: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


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

    task: TaskDefinition
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


class TransitionInput(BaseModel):
    """Input to the evaluate_transition activity."""

    validation_results: list[ValidationResult]
    attempt: int
    max_attempts: int = 2


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
    prior_errors: list[ValidationResult] = Field(default_factory=list)
    attempt: int = Field(default=1)
    max_attempts: int = Field(default=2)


# ---------------------------------------------------------------------------
# Planning activity I/O models
# ---------------------------------------------------------------------------


class PlannerInput(BaseModel):
    """Input to the call_planner activity."""

    task_id: str
    system_prompt: str
    user_prompt: str


class PlanCallResult(BaseModel):
    """Output of call_planner."""

    task_id: str
    plan: Plan
    model_name: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class AssembleStepContextInput(BaseModel):
    """Input to assemble_step_context activity."""

    task: TaskDefinition
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


class ExtractionCallResult(BaseModel):
    """Output of call_extraction_llm."""

    result: ExtractionResult
    source_workflow_ids: list[str]
    model_name: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class SaveExtractionInput(BaseModel):
    """Input to save_extraction_results activity."""

    entries: list[PlaybookEntry]
    source_workflow_ids: list[str]
    extraction_workflow_id: str


class ExtractionWorkflowInput(BaseModel):
    """Input to ForgeExtractionWorkflow."""

    limit: int = Field(default=10, description="Max runs to extract from.")
    since_hours: int = Field(default=24, description="Look-back window in hours.")


class ExtractionWorkflowResult(BaseModel):
    """Output of ForgeExtractionWorkflow."""

    entries_created: int
    source_workflow_ids: list[str] = Field(default_factory=list)
