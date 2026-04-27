+++
title = "Reference"
weight = 40
description = "Technical descriptions and lookup tables for Forge's modules, models, schemas, and configuration."
+++

Reference documents are for lookup, not learning. Each one mirrors the structure of the underlying code — modules, data models, schemas, environment variables, CLI flags. Skim the tables to find what you need; read the matching explanation for the reasoning and how-to guides for the task-focused counterpart.

- [System Overview](system-overview/) — Architecture principles, module map, technology stack, environment variables, and file system paths.
- [The Universal Workflow Step](workflow-step/) — Activity and workflow definitions, signal and query names, transition enum values.
- [Prompt Construction](prompt-construction/) — Section table, cache breakpoint summary, and error injection format.
- [Context Assembly](context-assembly/) — Context priority ordering, exploration provider table, and the token budget algorithm.
- [Task Decomposition and Execution](task-decomposition/) — Plan data models, planner prompt structure, and execution mode selection logic.
- [Output Processing](output-processing/) — LLMResponse model fields, edit matching algorithm, and fuzzy match parameters.
- [Validation and Retries](validation-and-retries/) — ValidationResult/ValidationConfig fields, validation checks, and transition mapping rules.
- [Model Routing and Batch Processing](llm-dispatch/) — Capability tier table, override mechanism, and batch_jobs schema.
- [Observability and Debugging](observability/) — SQLite store schema, Alembic migrations, and `forge status` CLI surface.
- [Forge Run Extraction](forge-run-extraction/) — Playbook table schema, tag inference rules, and data models for the forge playbook store.
- [Transcript Ingestion](transcript-ingestion/) — Workflow input/output shapes, activity signatures, and task queues.
- [Task Domains](task-domains/) — DomainConfig field table, TaskDomain enum values, and pipeline touchpoints.
- [OCR Pipeline](ocr-pipeline/) — OCR workflow and data model definitions, and the `ocr_images` / `batch_jobs` schema.
- [Planner Evaluation](planner-eval/) — Eval corpus format, deterministic check definitions, and LLM-as-judge scoring criteria.
- [Nushell Module](nushell-module/) — Module location, function signatures, and column schemas for the Nushell OCR pipeline.
