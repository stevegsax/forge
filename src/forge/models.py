"""Core data models for Forge."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class TransitionSignal(str, Enum):
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

    run_ruff_lint: bool = True
    run_ruff_format: bool = True
    run_tests: bool = False
    test_command: str | None = None


class TaskDefinition(BaseModel):
    """A single unit of work to be executed by the workflow."""

    task_id: str
    description: str = Field(description="What the task should produce.")
    target_files: list[str] = Field(description="Files to create or modify.")
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
