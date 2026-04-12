# Explanation

Background, context, and deeper understanding.

## System Overview

[System Overview](system-overview.html)
: What problem Forge solves and why it exists, The batch-first, document-completion paradigm vs. chat-based orchestrators, The seven architecture principles and their practical consequences, ...

## The Universal Workflow Step

[The Universal Workflow Step](workflow-step.html)
: Why every operation uses the same five-phase pattern, How this enables task-agnostic orchestration — new task types need new prompts, not new workflows, The relationship between Temporal workflows and activities — what runs where and why, ...

## Prompt Construction

[Prompt Construction](prompt-construction.html)
: The system prompt as a structured document — not a chat transcript, The eleven sections and their cache-optimized ordering, Cache breakpoints and their placement, Error injection on retry: structured lint/test output plus AST-derived code context, ...

## Context Assembly

[Context Assembly](context-assembly.html)
: How import graph analysis discovers relevant files without manual specification, How PageRank ranks files by structural importance, Token budget packing (knapsack-style priority algorithm), The exploration loop: separate document completions for batch compatibility, ...

## Task Decomposition and Execution

[Task Decomposition and Execution](task-decomposition.html)
: The three execution modes and when to use each (single-step, planned, fan-out), How the planner decomposes a task into ordered steps — what inputs it receives, what it produces, Why planning gets the most expensive models and highest token budgets, ...

## Output Processing

[Output Processing](output-processing.html)
: Why Forge uses tool-use for structured output instead of parsing free-form text, The LLMResponse schema: explanation, files, and edits, Why edits use search/replace instead of full-file replacement (the D50 decision), ...

## Validation and Retries

[Validation and Retries](validation-and-retries.html)
: Why deterministic validation runs before LLM-based review, The validation pipeline: ruff lint, ruff format, optional test execution, How validation results map to transition signals (SUCCESS, FAILURE_RETRYABLE, FAILURE_TERMINAL), ...

## Model Routing and Batch Processing

[Model Routing and Batch Processing](llm-dispatch.html)
: Capability tiers: why abstract tiers instead of concrete model names, The four tiers (reasoning, generation, summarization, classification) and their use cases, Why planning and conflict resolution get the reasoning tier, ...

## Observability and Debugging

[Observability and Debugging](observability.html)
: The observability strategy: SQLite store for heavyweight data, Temporal for lightweight stats, What is stored and why: full prompts, token usage, latency, context stats, Best-effort writes: why store failures never block workflow execution, ...

## Forge Run Extraction

[Forge Run Extraction](forge-run-extraction.html)
: The playbook concept: structured lessons Forge generates from its own completed runs, Why extraction runs as an independent workflow (not on the critical path), The extraction pipeline: fetch unextracted runs from the observability store, call LLM, save entries, ...

## Transcript Ingestion

[Transcript Ingestion](transcript-ingestion.html)
: Why transcript ingestion lives in forge and not in pbook, The two workflows: TranscriptIngestionWorkflow (single session) and BatchIngestionWorkflow (fan-out parent), Why the analysis call uses the SUMMARIZATION tier and batch mode, ...

## Learning Loops

[Learning Loops](learning-loops.html)
: The problem: LLM orchestrators don't learn between runs by default, The two-loops reality today: forge's self-learning loop vs. the pbook-backed cross-project loop, Side-by-side field comparison of forge's playbooks table and pbook's entries table, ...

## Task Domains

[Task Domains](task-domains.html)
: What a domain is and what it controls, How domains achieve task-agnostic orchestration — same pipeline, different prompts, The four pipeline touchpoints: context assembly, exploration, planner, CLI validation defaults, ...

## OCR Pipeline

[OCR Pipeline](ocr-pipeline.html)
: How the OCR pipeline builds on Forge's core workflow and batch primitives, The two execution paths: synchronous (OcrSyncWorkflow) and batch (OcrSubmitWorkflow), Image extraction and the ocr-image:// URI scheme, ...

## Planner Evaluation

[Planner Evaluation](planner-eval.html)
: Why planner evaluation matters — plan quality bounds everything downstream, The two evaluation modes: deterministic structural checks and LLM-as-judge, What deterministic checks verify: file coverage, step ordering, constraint adherence, ...
