+++
title = "Explanation"
weight = 10
description = "Background, context, and deeper understanding of how Forge works and why."
+++

Explanation documents discuss the why — the reasoning, design rationale, and trade-offs behind Forge's architecture. Read these to build a mental model of the system, not to accomplish a specific task. Each explanation links to the matching reference for lookup details and to how-to guides for the task-focused counterpart.

- [System Overview](system-overview/) — What problem Forge solves, the seven architecture principles, and how the major components fit together.
- [The Universal Workflow Step](workflow-step/) — The five-phase pattern (construct, send, receive, serialize, transition) that every operation in Forge follows.
- [Prompt Construction](prompt-construction/) — The eleven-section system prompt, cache-optimized ordering, and error injection on retry.
- [Context Assembly](context-assembly/) — Import graph discovery, PageRank ranking, token budget packing, and the exploration loop.
- [Task Decomposition and Execution](task-decomposition/) — The three execution modes and how fan-out/gather achieves parallelism.
- [Output Processing](output-processing/) — The LLMResponse schema, edit application with fuzzy matching, and file writing.
- [Validation and Retries](validation-and-retries/) — Deterministic validation and error-aware retries that feed validation errors back to the LLM.
- [Model Routing and Batch Processing](llm-dispatch/) — Capability tiers, model routing decisions, and how batch mode decouples execution from LLM latency.
- [Observability and Debugging](observability/) — The observability store, logging, tracing, and CLI inspection commands.
- [Forge Run Extraction](forge-run-extraction/) — Forge's self-learning loop: extracting playbooks from completed runs and injecting them into future contexts.
- [Transcript Ingestion](transcript-ingestion/) — Reading Claude Code session transcripts, analyzing them via batch, and handing results to pbook cross-queue.
- [Learning Loops](learning-loops/) — The two parallel learning pipelines, why they are separate today, and the convergence refactor that would unify them.
- [Task Domains](task-domains/) — How Forge parameterizes LLM behavior through domain configs — role prompts, output requirements, and validation defaults.
- [OCR Pipeline](ocr-pipeline/) — How the OCR pipeline builds on the core workflow primitives, with sync and batch execution paths.
- [Planner Evaluation](planner-eval/) — Why plan quality bounds downstream quality, and the two evaluation modes (deterministic checks and LLM-as-judge).
