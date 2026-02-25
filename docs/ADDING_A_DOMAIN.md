# Adding a Task Domain

Forge parameterizes LLM behavior through task domains. A domain controls the system prompt role, output requirements, user prompt templates, exploration vocabulary, planner guidance, and validation defaults — without changing any pipeline logic.

## How domains work

Every domain is a `DomainConfig` instance registered in `src/forge/domains.py`. The pipeline queries the config through `get_domain_config(domain)` at each stage:

- **Context assembly** (`src/forge/activities/context.py`) — reads `role_prompt`, `output_requirements`, and `user_prompt_template` (or the step/sub-task variants)
- **Exploration** (`src/forge/activities/exploration.py`) — reads `exploration_task_noun` and `exploration_completion_noun` to phrase exploration prompts
- **Planner** (`src/forge/activities/planner.py`) — reads `planner_domain_instruction` to guide step decomposition
- **CLI** (`src/forge/cli.py`) — reads `validation_defaults` to set default validation flags

No other files reference domain configuration directly. The output writer, validator, transition evaluator, and workflow orchestrator are all domain-agnostic.

## DomainConfig fields

| Field | Purpose | Example (code_generation) |
|-------|---------|---------------------------|
| `role_prompt` | Opening sentence of the system prompt | `"You are a code generation assistant."` |
| `output_requirements` | Tells the LLM how to structure its response within the `LLMResponse` schema | `"You MUST respond with a valid LLMResponse containing..."` |
| `user_prompt_template` | User message for single-step execution | `"Generate the code described above..."` |
| `step_user_prompt_template` | User message for a plan step; placeholders: `{step_id}`, `{step_description}` | `"Execute step '{step_id}': {step_description}..."` |
| `sub_task_user_prompt_template` | User message for a fan-out sub-task; placeholders: `{sub_task_id}`, `{sub_task_description}` | `"Execute sub-task '{sub_task_id}': {sub_task_description}..."` |
| `exploration_task_noun` | Noun phrase used in exploration prompts | `"coding task"` |
| `exploration_completion_noun` | Noun phrase for the post-exploration phase | `"code generation"` |
| `planner_domain_instruction` | Appended to the planner system prompt under "## Task Domain" | `"This is a **code generation** task..."` |
| `validation_defaults` | Default `ValidationConfig` (can be overridden by CLI flags) | `auto_fix=True, run_ruff_lint=True, ...` |

## Output schema

All domains share the same `LLMResponse` schema (`src/forge/models.py`):

```python
class LLMResponse(BaseModel):
    files: list[FileOutput] = Field(default_factory=list)
    edits: list[FileEdit] = Field(default_factory=list)
    explanation: str
```

The `output_requirements` text steers the LLM toward the appropriate fields:

- **File-producing domains** (code generation, research, documentation) instruct the LLM to populate `files` and/or `edits`
- **Explanation-only domains** can instruct the LLM to put its answer in `explanation` and leave `files`/`edits` empty — both default to empty lists, so no schema change is needed

## Files to modify

Adding a domain requires changes to exactly four files:

### 1. `src/forge/models.py` — add the enum value

```python
class TaskDomain(StrEnum):
    CODE_GENERATION = "code_generation"
    RESEARCH = "research"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    MY_DOMAIN = "my_domain"  # ← add here
```

### 2. `src/forge/domains.py` — create and register the config

```python
_MY_DOMAIN_CONFIG = DomainConfig(
    role_prompt="You are a ... assistant.",
    output_requirements=_PROSE_OUTPUT_REQUIREMENTS,  # or _CODE_OUTPUT_REQUIREMENTS, or custom
    user_prompt_template="...",
    step_user_prompt_template="Execute step '{step_id}': {step_description}\n\n...",
    sub_task_user_prompt_template="Execute sub-task '{sub_task_id}': {sub_task_description}\n\n...",
    exploration_task_noun="... task",
    exploration_completion_noun="...",
    planner_domain_instruction="This is a **...** task. ...",
    validation_defaults=ValidationConfig(
        auto_fix=False,
        run_ruff_lint=False,
        run_ruff_format=False,
        run_tests=False,
        test_command=None,
    ),
)
```

Then add it to the registry:

```python
_DOMAIN_REGISTRY: dict[TaskDomain, DomainConfig] = {
    ...
    TaskDomain.MY_DOMAIN: _MY_DOMAIN_CONFIG,
}
```

### 3. `src/forge/cli.py` — update the CLI option

Update the `--domain` choice list and help text:

```python
@click.option(
    "--domain",
    type=click.Choice(["code_generation", "research", "code_review", "documentation", "my_domain"]),
    help="Task domain: code_generation, research, code_review, documentation, my_domain.",
)
```

### 4. `tests/test_domains.py` — add tests

The existing `test_every_domain_has_config` test auto-covers new enum values. Add a domain-specific test class:

```python
class TestMyDomainConfig:
    def test_role_prompt(self) -> None:
        config = get_domain_config(TaskDomain.MY_DOMAIN)
        assert "..." in config.role_prompt.lower()

    def test_validation_defaults(self) -> None:
        config = get_domain_config(TaskDomain.MY_DOMAIN)
        assert config.validation_defaults.run_ruff_lint is False
```

## Design checklist

When writing a new `DomainConfig`, consider:

- [ ] Does the `role_prompt` clearly set the LLM's persona for this task type?
- [ ] Does `output_requirements` tell the LLM which `LLMResponse` fields to use (`files`, `edits`, `explanation`, or a combination)?
- [ ] Do the three user prompt templates (`user_prompt_template`, `step_user_prompt_template`, `sub_task_user_prompt_template`) give consistent instructions?
- [ ] Do the step/sub-task templates include `{step_id}`/`{step_description}` and `{sub_task_id}`/`{sub_task_description}` placeholders?
- [ ] Are `exploration_task_noun` and `exploration_completion_noun` natural-sounding in the phrases "gather the context needed to complete a ___" and "ready for the ___ phase"?
- [ ] Does `planner_domain_instruction` tell the planner what kind of output each step should produce?
- [ ] Are `validation_defaults` appropriate? (Code linting only makes sense for domains that produce Python files.)
