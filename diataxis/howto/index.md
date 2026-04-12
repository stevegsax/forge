# How-to Guides

Practical directions for accomplishing specific tasks.

## Context Assembly

[Context Assembly](control-context-assembly.html)
: How to include specific files as context (--context-file), How to include dependency contents upfront (--include-deps), How to disable or limit exploration (--no-explore, --max-exploration-rounds), ...

## Task Decomposition and Execution

[Task Decomposition and Execution](submit-tasks.html)
: How to submit a single-step task, How to submit a planned multi-step task, How to use a JSON task file for complex task definitions, ...

## Validation and Retries

[Validation and Retries](configure-validation.html)
: How to enable or disable specific validation checks, How to add custom test execution to validation, How to configure retry limits, ...

## Model Routing and Batch Processing

[Model Routing and Batch Processing](configure-llm-dispatch.html)
: How to run a task in sync mode (--sync), How to check batch job status, How to configure the batch poll interval, ...

## Observability and Debugging

[Observability and Debugging](debug-workflow.html)
: How to inspect a completed workflow's full history, How to find what prompt was sent for a specific step, How to diagnose why a step failed validation, ...

## Forge Run Extraction

[Forge Run Extraction](manage-playbooks.html)
: How to run knowledge extraction on completed forge runs, How to list and inspect existing forge playbook entries, How to manually add a playbook entry to forge's store, ...

## Transcript Ingestion

[Transcript Ingestion](ingest-transcripts.html)
: How to ingest a single Claude Code session file, How to discover and ingest all sessions from ~/.claude/projects/, How to filter discovered sessions by project, ...

## Task Domains

[Task Domains](add-domain.html)
: How to add a new task domain to Forge, The four files to modify (models.py, domains.py, tests, CLI), How to write domain-specific prompts, ...

## OCR Pipeline

[OCR Pipeline](run-ocr.html)
: How to run synchronous OCR on a document, How to submit a document for batch OCR, How to check batch OCR job status, ...

## Planner Evaluation

[Planner Evaluation](run-evaluations.html)
: How to run planner evaluations against the eval corpus, How to create a new eval case, How to run evaluations with LLM-as-judge scoring, ...

## Nushell Integration

[Nushell Integration](use-nushell-module.html)
: How to load the OCR module, How to submit documents for OCR, How to list and filter OCR jobs, ...
