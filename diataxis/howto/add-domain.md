# How to Add a Domain

This guide shows you how to add a new task domain to Forge. Adding a domain requires changes to exactly four files.

For background on what domains are and what they control, see [Task Domains](../explanation/task-domains.md). For `DomainConfig` field descriptions and per-domain examples, see the [Task Domains Reference](../reference/task-domains.md).

## Step 1: Add the enum value to `models.py`

Open `src/forge/models.py` and add your domain to the `TaskDomain` enum:

```python
class TaskDomain(StrEnum):
    CODE_GENERATION = "code_generation"
    RESEARCH = "research"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    GENERIC = "generic"
    MY_DOMAIN = "my_domain"  # ← add here
```

The string value must be lowercase with underscores. It is what the user passes to `--domain` on the CLI.

## Step 2: Create and register the config in `domains.py`

Open `src/forge/domains.py` and define a `DomainConfig` for your domain:

```python
_MY_DOMAIN_CONFIG = DomainConfig(
    role_prompt="You are a ... assistant.",
    output_requirements=_PROSE_OUTPUT_REQUIREMENTS,
    user_prompt_template="...",
    step_user_prompt_template="Execute step '{step_id}': {step_description}\n\n...",
    sub_task_user_prompt_template="Execute sub-task '{sub_task_id}': {sub_task_description}\n\n...",
    exploration_task_noun="... task",
    exploration_completion_noun="...",
    planner_domain_instruction="This is a **...** task. Each step should produce ...",
    validation_defaults=ValidationConfig(
        auto_fix=False,
        run_ruff_lint=False,
        run_ruff_format=False,
        run_tests=False,
        test_command=None,
    ),
)
```

Use `_CODE_OUTPUT_REQUIREMENTS` if the domain produces Python files (enabling ruff makes sense). Use `_PROSE_OUTPUT_REQUIREMENTS` for prose output. Write custom `output_requirements` text if neither fits.

Then add the new config to the registry:

```python
_DOMAIN_REGISTRY: dict[TaskDomain, DomainConfig] = {
    TaskDomain.CODE_GENERATION: _CODE_GENERATION_CONFIG,
    TaskDomain.RESEARCH: _RESEARCH_CONFIG,
    TaskDomain.CODE_REVIEW: _CODE_REVIEW_CONFIG,
    TaskDomain.DOCUMENTATION: _DOCUMENTATION_CONFIG,
    TaskDomain.GENERIC: _GENERIC_CONFIG,
    TaskDomain.MY_DOMAIN: _MY_DOMAIN_CONFIG,  # ← add here
}
```

### Checklist for the config

Before moving on, verify:

- `role_prompt` clearly frames the LLM's persona for this task type.
- `output_requirements` specifies which `LLMResponse` fields the LLM should populate (`files`, `edits`, `explanation`, or a combination).
- All three user prompt templates (`user_prompt_template`, `step_user_prompt_template`, `sub_task_user_prompt_template`) give consistent instructions.
- `step_user_prompt_template` includes `{step_id}` and `{step_description}` placeholders; `sub_task_user_prompt_template` includes `{sub_task_id}` and `{sub_task_description}`.
- `exploration_task_noun` reads naturally in "gather context needed to complete a \_\_\_".
- `exploration_completion_noun` reads naturally in "ready for the \_\_\_ phase".
- `planner_domain_instruction` tells the planner what output each step should produce.
- `validation_defaults` only enables ruff if the domain produces Python files.

## Step 3: Update the CLI option in `cli.py`

Open `src/forge/cli.py` and update the `--domain` option to include your new value:

```python
@click.option(
    "--domain",
    type=click.Choice([
        "code_generation",
        "research",
        "code_review",
        "documentation",
        "generic",
        "my_domain",  # ← add here
    ]),
    help="Task domain: code_generation, research, code_review, documentation, generic, my_domain.",
)
```

## Step 4: Add tests in `tests/test_domains.py`

The existing `test_every_domain_has_config` test automatically picks up new `TaskDomain` enum values and verifies each has a registered config. You still need to add a domain-specific test class to verify the content of your config:

```python
class TestMyDomainConfig:
    def test_role_prompt(self) -> None:
        config = get_domain_config(TaskDomain.MY_DOMAIN)
        assert "..." in config.role_prompt.lower()

    def test_validation_defaults(self) -> None:
        config = get_domain_config(TaskDomain.MY_DOMAIN)
        assert config.validation_defaults.run_ruff_lint is False

    def test_step_template_has_placeholders(self) -> None:
        config = get_domain_config(TaskDomain.MY_DOMAIN)
        assert "{step_id}" in config.step_user_prompt_template
        assert "{step_description}" in config.step_user_prompt_template

    def test_sub_task_template_has_placeholders(self) -> None:
        config = get_domain_config(TaskDomain.MY_DOMAIN)
        assert "{sub_task_id}" in config.sub_task_user_prompt_template
        assert "{sub_task_description}" in config.sub_task_user_prompt_template
```

Run the tests:

```
uv run pytest tests/test_domains.py -v
```

All existing tests should still pass, and your new test class should pass as well.

## Verification

After completing all four steps:

1. Confirm the domain appears in the CLI help:

    ```
    forge run --help
    ```

    The `--domain` option should list `my_domain` as a valid choice.

2. Submit a minimal task using the new domain:

    ```
    forge run --domain my_domain --description "Test the new domain" --target-file scratch.md
    ```

3. Inspect the assembled context to confirm the role prompt and output requirements from your config appear in the system prompt:

    ```
    forge status --workflow-id <id> --verbose
    ```
