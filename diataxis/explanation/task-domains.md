+++
title = "Task Domains"
weight = 141
description = "How Forge parameterizes LLM behavior through task domains — role prompts, output requirements, user prompt templates, and validation defaults."
topic = "task-domains"
covers = [
    "What a domain is and what it controls",
    "How domains achieve task-agnostic orchestration — same pipeline, different prompts",
    "The four pipeline touchpoints: context assembly, exploration, planner, CLI validation defaults",
    "Why all domains share the same LLMResponse schema",
    "The existing domains: code_generation, research, code_review, documentation, generic",
]
detail = "Concise explanation connecting domains to the task-agnostic principle. Show how the same workflow step produces different behavior purely through prompt parameterization."
+++
Prerequisites: [Context Assembly](context-assembly/), [Validation and Retries](validation-and-retries/).

Forge is described as task-agnostic: the same orchestration pipeline handles code generation, research, documentation, and code review. That agnosticism is not a property of the pipeline itself — the pipeline knows nothing about what the LLM is being asked to produce. It is a property of task domains.

A domain is a named configuration bundle that parameterizes the pipeline's prompts and validation defaults for a specific category of work. The pipeline reads domain configuration at each stage where the task type matters. Everything else — context assembly mechanics, batch submission, edit application, observability writes, Temporal workflow structure — is domain-agnostic by design.

For the exact fields, enum values, and per-domain defaults, see the [Task Domains Reference](../reference/task-domains/). To add a new domain, see [How to Add a Domain](../howto/add-domain/).

## What a domain controls

A domain is a `DomainConfig` instance, registered in `src/forge/domains.py` and looked up by `get_domain_config(domain)`. It contains prompt text and a default `ValidationConfig`. The prompts cover four different message positions:

The **role prompt** opens the system prompt and establishes the LLM's persona for the task type. "You are a code generation assistant" reads differently from "You are a technical research assistant" — the framing shapes how the model interprets ambiguity and structures its response.

The **output requirements** tell the LLM how to use the `LLMResponse` schema for this domain. Code generation tasks expect the model to produce `files` and `edits`. Prose tasks (research, documentation) expect `files` only. The output requirements section is what steers the model toward the appropriate fields.

The **user prompt templates** are the actual user-turn messages. There are three variants: one for single-step execution, one for a named step within a plan, and one for a fan-out sub-task. The step and sub-task templates include `{step_id}` and `{step_description}` (or the sub-task equivalents) as placeholders, which the pipeline fills in at execution time.

The **exploration noun phrases** shape how the exploration prompts read: `exploration_task_noun` fills in "gather context for a \_\_\_" and `exploration_completion_noun` fills in "ready for the \_\_\_ phase." These are minor but affect the naturalness of the exploration prompts, which the LLM reads to decide what context to request.

The **planner domain instruction** is appended to the planner's system prompt under a "Task Domain" header. It tells the planner what kind of output each step should produce — for code generation, this means naming specific files; for research, it means producing structured notes.

The **validation defaults** set the initial `ValidationConfig` for CLI invocations that do not specify validation flags. Code generation defaults enable ruff linting and formatting; prose domains disable them (running a Python linter on markdown output would be meaningless).

## How domains achieve task-agnostic orchestration

The pipeline reads domain configuration at exactly four points: context assembly, exploration, planning, and CLI validation defaults. No other code references domain configuration directly. The output writer, transition evaluator, observability writes, and Temporal workflow structure are all completely unaware of which domain is active.

This boundary is what makes the architecture extensible. Adding a new domain requires registering a `DomainConfig`, adding an enum value, updating the CLI option list, and writing tests. It does not require touching any pipeline logic. The planner, the context assembler, the exploration loop, and the validator all continue to work exactly as before — they just receive different prompt text.

The consequence is also a constraint: domain configuration can only influence the pipeline through text (prompts) and through `ValidationConfig` booleans and flags. If a new task type requires different orchestration behavior — a different number of exploration rounds, a different retry strategy, a different edit application mode — that is not a domain concern. That would require a pipeline change, not a domain addition.

## The four pipeline touchpoints

Understanding where domain configuration is consumed is important for both using existing domains correctly and designing new ones.

**Context assembly** (`src/forge/activities/context.py`) reads `role_prompt`, `output_requirements`, and the appropriate user prompt template (`user_prompt_template`, `step_user_prompt_template`, or `sub_task_user_prompt_template` depending on execution mode). These become sections of the assembled system prompt. This is the most impactful touchpoint: the role prompt and output requirements appear near the top of the system prompt, in a stable cache tier, and the model reads them before everything else.

**Exploration** (`src/forge/activities/exploration.py`) reads `exploration_task_noun` and `exploration_completion_noun` to construct the exploration request prompts. These affect only the phrasing of the instructions the LLM reads when deciding what context to request.

**Planner** (`src/forge/activities/planner.py`) reads `planner_domain_instruction` and appends it to the planning system prompt. This is the touchpoint that shapes the structure of the emitted plan — what kinds of steps the planner produces, what outputs it expects from each step.

**CLI** (`src/forge/cli.py`) reads `validation_defaults` to set the initial values of validation flags before parsing any CLI arguments the user provides. User-supplied flags override these defaults.

## Why all domains share the same LLMResponse schema

Every domain uses the same `LLMResponse` schema: `files`, `edits`, and `explanation`. This is not a limitation — it is a deliberate boundary between prompt configuration and pipeline mechanics.

The output writer, which applies edits and creates files, does not know which domain produced the response. It works on `LLMResponse` objects. If different domains required different response schemas, the output writer would need to know which domain produced the response and handle each case differently. That coupling would make the pipeline more complex and make domain addition more invasive.

Instead, the `output_requirements` text in each domain configuration steers the LLM toward the fields that make sense for that domain. Research and documentation tasks are instructed to use `files` for their output. Code generation tasks are instructed to use both `files` for new files and `edits` for modifications to existing ones. The schema is shared; the usage varies by instruction.

## The existing domains

Forge ships with five domains:

**`code_generation`** is the primary domain. It enables ruff linting and formatting by default, uses both `files` and `edits`, and directs the planner to name specific files for each step. Context assembly and the exploration loop are tuned for code tasks.

**`research`** produces prose output (markdown files). Validation is disabled by default. The planner instruction directs steps to produce structured notes and summaries.

**`code_review`** produces review output as files. Ruff is disabled since the output is commentary, not code. The role prompt positions the LLM as a reviewer rather than an implementer.

**`documentation`** produces documentation files. Ruff is disabled. The planner instruction directs steps to produce self-contained documentation sections.

**`generic`** applies minimal prompt framing and disables all validation by default. It is a starting point for tasks that do not fit the other categories and a fallback when the domain is not specified.
