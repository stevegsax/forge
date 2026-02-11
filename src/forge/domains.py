"""Domain configuration registry for Forge.

Each TaskDomain maps to a DomainConfig that parameterizes prompts and
validation defaults without changing the pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from forge.models import TaskDomain, ValidationConfig

# ---------------------------------------------------------------------------
# The code-generation output requirements string — preserved exactly from
# context.py so the code_generation domain is byte-identical.
# ---------------------------------------------------------------------------

_CODE_OUTPUT_REQUIREMENTS = (
    "You MUST respond with a valid LLMResponse containing an `explanation` string "
    "and either `files`, `edits`, or both.\n\n"
    "- **`files`**: Use for NEW files that don't exist yet. Each entry needs "
    "`file_path` and `content` (complete file content).\n"
    "- **`edits`**: Use for EXISTING files that need changes. Each entry needs "
    "`file_path` and a list of `edits`, where each edit has `search` (exact text "
    "to find, must match exactly once) and `replace` (replacement text).\n\n"
    "A file path must NOT appear in both `files` and `edits`. "
    "Do NOT return an empty object."
)

_PROSE_OUTPUT_REQUIREMENTS = (
    "You MUST respond with a valid LLMResponse containing an `explanation` string "
    "and a `files` list.\n\n"
    "Write your findings as one or more markdown files using the `files` list. "
    "Each entry needs `file_path` and `content` (complete file content).\n\n"
    "Do NOT return an empty object."
)


# ---------------------------------------------------------------------------
# DomainConfig model
# ---------------------------------------------------------------------------


class DomainConfig(BaseModel, frozen=True):
    """Prompt fragments and validation defaults for a task domain."""

    role_prompt: str = Field(description="Opening role sentence for system prompts.")
    output_requirements: str = Field(description="Output format instructions.")
    user_prompt_template: str = Field(
        description="User prompt for single-step execution.",
    )
    step_user_prompt_template: str = Field(
        description="User prompt for a plan step. Placeholders: {step_id}, {step_description}.",
    )
    sub_task_user_prompt_template: str = Field(
        description=(
            "User prompt for a sub-task. Placeholders: {sub_task_id}, {sub_task_description}."
        ),
    )
    exploration_task_noun: str = Field(
        description="Noun phrase for the task in exploration prompts (e.g. 'coding task').",
    )
    exploration_completion_noun: str = Field(
        description="Noun phrase for completion in exploration prompts (e.g. 'code generation').",
    )
    planner_domain_instruction: str = Field(
        description="Instruction block appended to planner system prompt.",
    )
    validation_defaults: ValidationConfig = Field(
        description="Domain-specific validation defaults.",
    )


# ---------------------------------------------------------------------------
# Domain configurations
# ---------------------------------------------------------------------------

_CODE_GENERATION_CONFIG = DomainConfig(
    role_prompt="You are a code generation assistant.",
    output_requirements=_CODE_OUTPUT_REQUIREMENTS,
    user_prompt_template=(
        "Generate the code described above. "
        "For existing files, use `edits` with search/replace operations. "
        "For new files, use `files` with complete content."
    ),
    step_user_prompt_template=(
        "Execute step '{step_id}': {step_description}\n\n"
        "For existing files, use `edits` with search/replace operations. "
        "For new files, use `files` with complete content."
    ),
    sub_task_user_prompt_template=(
        "Execute sub-task '{sub_task_id}': {sub_task_description}\n\n"
        "For existing files, use `edits` with search/replace operations. "
        "For new files, use `files` with complete content."
    ),
    exploration_task_noun="coding task",
    exploration_completion_noun="code generation",
    planner_domain_instruction=(
        "This is a **code generation** task. Each step should produce or modify "
        "source code files. Steps are validated with ruff lint and format checks."
    ),
    validation_defaults=ValidationConfig(
        auto_fix=True,
        run_ruff_lint=True,
        run_ruff_format=True,
        run_tests=False,
        test_command=None,
    ),
)

_RESEARCH_CONFIG = DomainConfig(
    role_prompt="You are a research assistant.",
    output_requirements=_PROSE_OUTPUT_REQUIREMENTS,
    user_prompt_template=(
        "Conduct the research described above. "
        "Write your findings as markdown files using the `files` list."
    ),
    step_user_prompt_template=(
        "Execute step '{step_id}': {step_description}\n\n"
        "Write your findings as markdown files using the `files` list."
    ),
    sub_task_user_prompt_template=(
        "Execute sub-task '{sub_task_id}': {sub_task_description}\n\n"
        "Write your findings as markdown files using the `files` list."
    ),
    exploration_task_noun="research task",
    exploration_completion_noun="report writing",
    planner_domain_instruction=(
        "This is a **research** task. Each step should produce markdown files "
        "containing findings, analysis, or recommendations. Code linting is disabled."
    ),
    validation_defaults=ValidationConfig(
        auto_fix=False,
        run_ruff_lint=False,
        run_ruff_format=False,
        run_tests=False,
        test_command=None,
    ),
)

_CODE_REVIEW_CONFIG = DomainConfig(
    role_prompt="You are a code review assistant.",
    output_requirements=_PROSE_OUTPUT_REQUIREMENTS,
    user_prompt_template=(
        "Conduct the code review described above. "
        "Write your review as markdown files using the `files` list."
    ),
    step_user_prompt_template=(
        "Execute step '{step_id}': {step_description}\n\n"
        "Write your review as markdown files using the `files` list."
    ),
    sub_task_user_prompt_template=(
        "Execute sub-task '{sub_task_id}': {sub_task_description}\n\n"
        "Write your review as markdown files using the `files` list."
    ),
    exploration_task_noun="code review task",
    exploration_completion_noun="review writing",
    planner_domain_instruction=(
        "This is a **code review** task. Each step should produce markdown files "
        "containing review comments, issues, and suggestions. Code linting is disabled."
    ),
    validation_defaults=ValidationConfig(
        auto_fix=False,
        run_ruff_lint=False,
        run_ruff_format=False,
        run_tests=False,
        test_command=None,
    ),
)

_DOCUMENTATION_CONFIG = DomainConfig(
    role_prompt="You are a documentation assistant.",
    output_requirements=_PROSE_OUTPUT_REQUIREMENTS,
    user_prompt_template=(
        "Write the documentation described above. Produce markdown files using the `files` list."
    ),
    step_user_prompt_template=(
        "Execute step '{step_id}': {step_description}\n\n"
        "Produce markdown files using the `files` list."
    ),
    sub_task_user_prompt_template=(
        "Execute sub-task '{sub_task_id}': {sub_task_description}\n\n"
        "Produce markdown files using the `files` list."
    ),
    exploration_task_noun="documentation task",
    exploration_completion_noun="documentation writing",
    planner_domain_instruction=(
        "This is a **documentation** task. Each step should produce markdown files "
        "containing documentation, guides, or reference material. Code linting is disabled."
    ),
    validation_defaults=ValidationConfig(
        auto_fix=False,
        run_ruff_lint=False,
        run_ruff_format=False,
        run_tests=False,
        test_command=None,
    ),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_DOMAIN_REGISTRY: dict[TaskDomain, DomainConfig] = {
    TaskDomain.CODE_GENERATION: _CODE_GENERATION_CONFIG,
    TaskDomain.RESEARCH: _RESEARCH_CONFIG,
    TaskDomain.CODE_REVIEW: _CODE_REVIEW_CONFIG,
    TaskDomain.DOCUMENTATION: _DOCUMENTATION_CONFIG,
}


def get_domain_config(domain: TaskDomain) -> DomainConfig:
    """Look up the domain configuration for a given TaskDomain."""
    return _DOMAIN_REGISTRY[domain]
