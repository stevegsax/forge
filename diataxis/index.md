+++
title = "Forge"
description = "Human-facing documentation for Forge, a batch-first LLM task orchestrator. Covers the core orchestration pipeline end-to-end, plus extensions (OCR, evaluation, knowledge extraction, transcript ingestion) that build on the core."

[cascade]
  type = "docs"
+++

Most LLM orchestrators are thin wrappers around a chat loop: the model decides what to do next, the orchestrator serves the conversation, and every call depends on the conversation history that precedes it. That approach ties execution to a live connection, leaves the LLM in control of the workflow, and is incompatible with batch APIs. Forge inverts the relationship. The orchestrator owns the control loop, the LLM is a stateless function, and every call is a self-contained document completion that a batch API can process at half the cost. This documentation explains how that inversion works in practice: how tasks are decomposed and planned, how context is assembled fresh for every call, how structured output is validated and retried with error feedback, how knowledge extraction and transcript ingestion feed lessons back into future task contexts, and how the whole pipeline fits together around Temporal workflows. The audience is developers who need to create, debug, or extend Forge workflows — not end users of a deployed Forge instance.

**Audience**: Internal developers who create or debug Forge workflows

## Sections

- [Explanation](explanation/) — Background and context
- [Tutorials](tutorials/) — Learn by doing
- [How-to Guides](howto/) — Accomplish specific tasks
- [Reference](reference/) — Technical descriptions
