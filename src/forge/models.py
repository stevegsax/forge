"""Core data models for Forge."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


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


class StepResult(BaseModel):
    """The outcome of executing a single plan step."""

    step_id: str
    status: TransitionSignal
    output_files: dict[str, str] = Field(default_factory=dict)
    validation_results: list[ValidationResult] = Field(default_factory=list)
    commit_sha: str | None = None
    error: str | None = None
    sub_task_results: list[SubTaskResult] = Field(default_factory=list)


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


# ---------------------------------------------------------------------------
# LLM structured output
# ---------------------------------------------------------------------------


class FileOutput(BaseModel):
    """A single file produced by the LLM."""

    file_path: str = Field(description="Relative path within the worktree.")
    content: str = Field(description="Complete file content.")


class LLMResponse(BaseModel):
    """Structured output model for pydantic-ai Agent."""

    files: list[FileOutput] = Field(description="Files to create or modify.")
    explanation: str = Field(description="Brief explanation of what was produced.")


# ---------------------------------------------------------------------------
# Inter-activity transport
# ---------------------------------------------------------------------------


class AssembledContext(BaseModel):
    """Output of assemble_context, input to call_llm."""

    task_id: str
    system_prompt: str
    user_prompt: str


class LLMCallResult(BaseModel):
    """Output of call_llm, input to write_output."""

    task_id: str
    response: LLMResponse
    model_name: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class WriteResult(BaseModel):
    """Output of write_output."""

    task_id: str
    files_written: list[str]


# ---------------------------------------------------------------------------
# Activity input models (Temporal single-arg convention)
# ---------------------------------------------------------------------------


class AssembleContextInput(BaseModel):
    """Input to the assemble_context activity."""

    task: TaskDefinition
    repo_root: str
    worktree_path: str


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


class AssembleStepContextInput(BaseModel):
    """Input to assemble_step_context activity."""

    task: TaskDefinition
    step: PlanStep
    step_index: int
    total_steps: int
    completed_steps: list[StepResult] = Field(default_factory=list)
    repo_root: str
    worktree_path: str
