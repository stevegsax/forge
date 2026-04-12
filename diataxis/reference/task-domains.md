# Task Domains Reference

This reference covers the `TaskDomain` enum, `DomainConfig` fields, pipeline touchpoints, and per-domain validation defaults. For background on how domains work and why they are designed this way, see [Task Domains](../explanation/task-domains.md). To add a new domain, see [How to Add a Domain](../howto/add-domain.md).

## TaskDomain enum

Defined in `src/forge/models.py`.

| Value | String | Description |
|-------|--------|-------------|
| `TaskDomain.CODE_GENERATION` | `"code_generation"` | Code generation and editing tasks. |
| `TaskDomain.RESEARCH` | `"research"` | Research and information synthesis tasks. |
| `TaskDomain.CODE_REVIEW` | `"code_review"` | Code review and commentary tasks. |
| `TaskDomain.DOCUMENTATION` | `"documentation"` | Documentation authoring tasks. |
| `TaskDomain.GENERIC` | `"generic"` | Fallback for tasks that do not fit other domains. |

## DomainConfig fields

Each field is described with its type, which pipeline activity reads it, and an example drawn from an existing domain.

| Field | Type | Read by | Description |
|-------|------|---------|-------------|
| `role_prompt` | str | Context assembly | Opening sentence of the system prompt; sets the LLM's persona. |
| `output_requirements` | str | Context assembly | Instructions for how to populate the `LLMResponse` schema (`files`, `edits`, `explanation`). |
| `user_prompt_template` | str | Context assembly | User-turn message for single-step execution. |
| `step_user_prompt_template` | str | Context assembly | User-turn message for a named plan step. Placeholders: `{step_id}`, `{step_description}`. |
| `sub_task_user_prompt_template` | str | Context assembly | User-turn message for a fan-out sub-task. Placeholders: `{sub_task_id}`, `{sub_task_description}`. |
| `exploration_task_noun` | str | Exploration | Fills "gather the context needed to complete a \_\_\_". |
| `exploration_completion_noun` | str | Exploration | Fills "ready for the \_\_\_ phase". |
| `planner_domain_instruction` | str | Planner | Appended to the planner system prompt under "## Task Domain". |
| `validation_defaults` | ValidationConfig | CLI | Default `ValidationConfig` used when the user does not supply validation flags. |

### Examples by domain

The table below shows the value of each field for each built-in domain.

#### `role_prompt`

| Domain | Value |
|--------|-------|
| `code_generation` | `"You are a code generation assistant."` |
| `research` | `"You are a technical research assistant."` |
| `code_review` | `"You are a code review assistant."` |
| `documentation` | `"You are a technical documentation assistant."` |
| `generic` | `"You are a general-purpose assistant."` |

#### `exploration_task_noun`

| Domain | Value |
|--------|-------|
| `code_generation` | `"coding task"` |
| `research` | `"research task"` |
| `code_review` | `"code review"` |
| `documentation` | `"documentation task"` |
| `generic` | `"task"` |

#### `exploration_completion_noun`

| Domain | Value |
|--------|-------|
| `code_generation` | `"code generation"` |
| `research` | `"research"` |
| `code_review` | `"code review"` |
| `documentation` | `"documentation"` |
| `generic` | `"completion"` |

## Pipeline touchpoints

The table below maps each pipeline stage to the `DomainConfig` fields it reads.

| Pipeline stage | File | Fields read |
|----------------|------|-------------|
| Context assembly | `src/forge/activities/context.py` | `role_prompt`, `output_requirements`, `user_prompt_template` (or `step_user_prompt_template` / `sub_task_user_prompt_template` depending on execution mode) |
| Exploration | `src/forge/activities/exploration.py` | `exploration_task_noun`, `exploration_completion_noun` |
| Planner | `src/forge/activities/planner.py` | `planner_domain_instruction` |
| CLI | `src/forge/cli.py` | `validation_defaults` |

No other files read `DomainConfig` directly. The output writer, transition evaluator, observability store, and Temporal workflows are all domain-agnostic.

## ValidationConfig defaults per domain

The `validation_defaults` field sets initial values for the `ValidationConfig` used when CLI flags are not supplied. User-supplied flags override these defaults.

| Domain | `auto_fix` | `run_ruff_lint` | `run_ruff_format` | `run_tests` | `test_command` |
|--------|------------|-----------------|-------------------|-------------|----------------|
| `code_generation` | `True` | `True` | `True` | `False` | `None` |
| `research` | `False` | `False` | `False` | `False` | `None` |
| `code_review` | `False` | `False` | `False` | `False` | `None` |
| `documentation` | `False` | `False` | `False` | `False` | `None` |
| `generic` | `False` | `False` | `False` | `False` | `None` |

`code_generation` enables ruff linting and formatting with auto-fix because its output is Python code. All other domains disable ruff because their output is prose or commentary. Test execution is disabled by default across all domains; it must be enabled explicitly with `--run-tests` and `--test-command`.

## Domain registry

Domains are registered in `_DOMAIN_REGISTRY` in `src/forge/domains.py`:

```python
_DOMAIN_REGISTRY: dict[TaskDomain, DomainConfig] = {
    TaskDomain.CODE_GENERATION: _CODE_GENERATION_CONFIG,
    TaskDomain.RESEARCH: _RESEARCH_CONFIG,
    TaskDomain.CODE_REVIEW: _CODE_REVIEW_CONFIG,
    TaskDomain.DOCUMENTATION: _DOCUMENTATION_CONFIG,
    TaskDomain.GENERIC: _GENERIC_CONFIG,
}
```

Access a domain config at runtime via `get_domain_config(domain: TaskDomain) -> DomainConfig`. Raises `KeyError` if the domain is not registered.
