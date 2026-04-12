# Reference

Technical descriptions and specifications.

## System Overview

[System Overview](system-overview.html)
: Architecture principles table (principle, consequence, enforcement point), Module map — every package and top-level module with a one-line description, Technology stack table (component, technology, version/source), ...

## The Universal Workflow Step

[The Universal Workflow Step](workflow-step.html)
: Activity definitions: name, input type, output type, timeout, retry policy, Workflow definitions: name, input type, output type, signals, queries, TransitionSignal enum values and their meanings, ...

## Prompt Construction

[Prompt Construction](prompt-construction.html)
: System prompt section table: section number, name, source module/config, stability class, cache behavior and breakpoint placement, Cache breakpoint summary table, Error injection format: what the 'Previous Errors' section contains, ...

## Context Assembly

[Context Assembly](context-assembly.html)
: Context priority ordering table (1=highest to 6=lowest) with what each level contains, Exploration provider table: provider name, description, parameters, return format, Token budget algorithm: inputs, priority order, truncation behavior, ...

## Task Decomposition and Execution

[Task Decomposition and Execution](task-decomposition.html)
: Plan data model: Plan, PlanStep, SubTask fields and constraints, Planner prompt structure and inputs, Execution mode selection logic (plan flag, sub_tasks presence), ...

## Output Processing

[Output Processing](output-processing.html)
: LLMResponse, FileOutput, FileEdit, EditOperation model fields, Edit matching algorithm: exact, whitespace-normalized, indentation-normalized, fuzzy, Fuzzy match parameters: similarity threshold (60%), uniqueness gap (5%), ...

## Validation and Retries

[Validation and Retries](validation-and-retries.html)
: ValidationResult and ValidationConfig model fields, Validation checks: name, what it runs, pass/fail criteria, TransitionSignal values and mapping rules, ...

## Model Routing and Batch Processing

[Model Routing and Batch Processing](llm-dispatch.html)
: Capability tier table: tier name, default model, use cases, LLM call sites, Model override mechanism: per-step capability_tier in plans, Batch processing: batch_jobs table schema, batch states, signal names, ...

## Observability and Debugging

[Observability and Debugging](observability.html)
: SQLite store schema: interactions table, runs table, batch_jobs table, Alembic migration management, CLI commands: forge status (all flags and output formats), ...

## Forge Run Extraction

[Forge Run Extraction](forge-run-extraction.html)
: Playbook table schema (title, content, tags, source task/workflow IDs) — the forge.db playbooks table specifically, Tag inference rules: file extension mapping, keyword mapping, defaults, PlaybookEntry and ExtractionResult data models (forge's, not pbook's), ...

## Transcript Ingestion

[Transcript Ingestion](transcript-ingestion.html)
: TranscriptIngestionWorkflow: input JSON shape, output dict, signals, task queue, BatchIngestionWorkflow: input JSON shape, output dict, fan-out behavior, task queue, prepare_transcript activity: input and output JSON shapes, ...

## Task Domains

[Task Domains](task-domains.html)
: DomainConfig field table: field name, type, purpose, example for each existing domain, TaskDomain enum values, Pipeline touchpoints: which code reads which DomainConfig field, ...

## OCR Pipeline

[OCR Pipeline](ocr-pipeline.html)
: OCR workflow definitions: OcrSyncWorkflow, OcrSubmitWorkflow, OcrStoreWorkflow, OcrGatherWorkflow, OcrExportWorkflow, OcrListJobsWorkflow, OCR data models: OcrInput, OcrResult, OcrImage, OcrListJobsInput/Result, OcrJobEntry fields, Database tables: ocr_images, batch_jobs (OCR-specific columns), ...

## Planner Evaluation

[Planner Evaluation](planner-eval.html)
: Eval corpus format: directory structure, case file schema, Deterministic check definitions: check name, what it verifies, pass/fail criteria, LLM-as-judge scoring criteria and score scale, ...

## Nushell Integration

[Nushell Integration](nushell-module.html)
: Module location and loading, ocr submit: parameters, flags, return type, ocr list: parameters, flags, return type and column schema, ...
