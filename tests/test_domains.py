"""Tests for forge.domains — domain configuration registry."""

from __future__ import annotations

from forge.domains import DomainConfig, get_domain_config
from forge.models import TaskDomain, ValidationConfig

# ---------------------------------------------------------------------------
# get_domain_config
# ---------------------------------------------------------------------------


class TestGetDomainConfig:
    def test_every_domain_has_config(self) -> None:
        for domain in TaskDomain:
            config = get_domain_config(domain)
            assert isinstance(config, DomainConfig)

    def test_returns_frozen_model(self) -> None:
        config = get_domain_config(TaskDomain.CODE_GENERATION)
        assert config.model_config.get("frozen") is True


# ---------------------------------------------------------------------------
# Code generation config
# ---------------------------------------------------------------------------


class TestCodeGenerationConfig:
    def test_role_prompt(self) -> None:
        config = get_domain_config(TaskDomain.CODE_GENERATION)
        assert config.role_prompt == "You are a code generation assistant."

    def test_output_requirements_match_original(self) -> None:
        """Output requirements must match the original _OUTPUT_REQUIREMENTS constant."""
        from forge.activities.context import _OUTPUT_REQUIREMENTS

        config = get_domain_config(TaskDomain.CODE_GENERATION)
        assert config.output_requirements == _OUTPUT_REQUIREMENTS

    def test_user_prompt_template(self) -> None:
        config = get_domain_config(TaskDomain.CODE_GENERATION)
        assert "Generate the code" in config.user_prompt_template

    def test_step_user_prompt_has_placeholders(self) -> None:
        config = get_domain_config(TaskDomain.CODE_GENERATION)
        assert "{step_id}" in config.step_user_prompt_template
        assert "{step_description}" in config.step_user_prompt_template

    def test_sub_task_user_prompt_has_placeholders(self) -> None:
        config = get_domain_config(TaskDomain.CODE_GENERATION)
        assert "{sub_task_id}" in config.sub_task_user_prompt_template
        assert "{sub_task_description}" in config.sub_task_user_prompt_template

    def test_validation_ruff_on(self) -> None:
        config = get_domain_config(TaskDomain.CODE_GENERATION)
        assert config.validation_defaults.run_ruff_lint is True
        assert config.validation_defaults.run_ruff_format is True

    def test_exploration_nouns(self) -> None:
        config = get_domain_config(TaskDomain.CODE_GENERATION)
        assert config.exploration_task_noun == "coding task"
        assert config.exploration_completion_noun == "code generation"


# ---------------------------------------------------------------------------
# Research config
# ---------------------------------------------------------------------------


class TestResearchConfig:
    def test_role_prompt_contains_research(self) -> None:
        config = get_domain_config(TaskDomain.RESEARCH)
        assert "research" in config.role_prompt.lower()

    def test_ruff_off(self) -> None:
        config = get_domain_config(TaskDomain.RESEARCH)
        assert config.validation_defaults.run_ruff_lint is False
        assert config.validation_defaults.run_ruff_format is False

    def test_exploration_nouns(self) -> None:
        config = get_domain_config(TaskDomain.RESEARCH)
        assert "research" in config.exploration_task_noun


# ---------------------------------------------------------------------------
# Code review config
# ---------------------------------------------------------------------------


class TestCodeReviewConfig:
    def test_role_prompt_contains_review(self) -> None:
        config = get_domain_config(TaskDomain.CODE_REVIEW)
        assert "code review" in config.role_prompt.lower()

    def test_ruff_off(self) -> None:
        config = get_domain_config(TaskDomain.CODE_REVIEW)
        assert config.validation_defaults.run_ruff_lint is False
        assert config.validation_defaults.run_ruff_format is False


# ---------------------------------------------------------------------------
# Documentation config
# ---------------------------------------------------------------------------


class TestDocumentationConfig:
    def test_role_prompt_contains_documentation(self) -> None:
        config = get_domain_config(TaskDomain.DOCUMENTATION)
        assert "documentation" in config.role_prompt.lower()

    def test_ruff_off(self) -> None:
        config = get_domain_config(TaskDomain.DOCUMENTATION)
        assert config.validation_defaults.run_ruff_lint is False
        assert config.validation_defaults.run_ruff_format is False


# ---------------------------------------------------------------------------
# Generic config
# ---------------------------------------------------------------------------


class TestGenericConfig:
    def test_role_prompt(self) -> None:
        config = get_domain_config(TaskDomain.GENERIC)
        assert config.role_prompt == "You are a helpful assistant."

    def test_output_requirements_files_and_explanation(self) -> None:
        config = get_domain_config(TaskDomain.GENERIC)
        assert "explanation" in config.output_requirements
        assert "files" in config.output_requirements

    def test_ruff_off(self) -> None:
        config = get_domain_config(TaskDomain.GENERIC)
        assert config.validation_defaults.run_ruff_lint is False
        assert config.validation_defaults.run_ruff_format is False

    def test_exploration_nouns(self) -> None:
        config = get_domain_config(TaskDomain.GENERIC)
        assert config.exploration_task_noun == "task"
        assert config.exploration_completion_noun == "response generation"


# ---------------------------------------------------------------------------
# Validation defaults
# ---------------------------------------------------------------------------


class TestValidationDefaults:
    def test_all_domains_return_validation_config(self) -> None:
        for domain in TaskDomain:
            config = get_domain_config(domain)
            assert isinstance(config.validation_defaults, ValidationConfig)
