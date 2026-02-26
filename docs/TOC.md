# Documentation Table of Contents

## User Guides

- [playbooks.md](user/playbooks.md) — How playbooks capture lessons from completed tasks and inject them into future ones.

## Planning & Design

- [DESIGN.md](planning/DESIGN.md) — Full architecture and design document: batch-first LLM orchestrator with document completion and core architectural principles.
- [ARCHITECTURE.md](planning/ARCHITECTURE.md) — Architecture overview: universal workflow step, execution modes, context assembly, and data models.
- [DECISIONS.md](planning/DECISIONS.md) — Key design decisions and rationale (batch mode, universal workflow step, Temporal, orchestrator control, etc.).
- [USAGE.md](planning/USAGE.md) — Guide for submitting code development and research tasks to Forge with CLI examples.
- [DEBUGGING.md](planning/DEBUGGING.md) — Logging, observability store, API message logs, OTel tracing, and environment variables for debugging.
- [ADDING_A_DOMAIN.md](planning/ADDING_A_DOMAIN.md) — How to parameterize LLM behavior through task domains without changing pipeline logic.
- [test-strategy.md](planning/test-strategy.md) — Practical guide for writing Python tests with emphasis on testing pyramid and signal quality.
- [LSP_INTEGRATION_PLAN.md](planning/LSP_INTEGRATION_PLAN.md) — Investigation of Language Server Protocol support for context generation.

## Phase Specifications

- [PHASE1.md](planning/PHASE1.md) — The Minimal Loop: universal workflow step, Temporal activity boundaries, git worktree lifecycle, OpenTelemetry tracing.
- [PHASE2.md](planning/PHASE2.md) — Planning and Multi-Step Execution: planner decomposes tasks into ordered sub-steps with sequential execution.
- [PHASE3.md](planning/PHASE3.md) — Fan-Out / Gather: parallel sub-task execution within planned steps via Temporal child workflows.
- [PHASE4.md](planning/PHASE4.md) — Intelligent Context Assembly: automatic context discovery, importance ranking, and token budget management.
- [PHASE5.md](planning/PHASE5.md) — Observability Store: persist LLM interaction data to SQLite with CLI inspection commands.
- [PHASE6.md](planning/PHASE6.md) — Knowledge Extraction: extract lessons from completed work and inject as playbook entries into future contexts.
- [PHASE7.md](planning/PHASE7.md) — LLM-Guided Context Exploration: exploration loop where the LLM requests context before code generation.
- [PHASE8.md](planning/PHASE8.md) — Error-Aware Retries: feed validation errors back to the LLM on retry with code context.
- [PHASE9.md](planning/PHASE9.md) — Prompt Caching: leverage Anthropic's prompt caching to reduce input token costs.
- [PHASE10.md](planning/PHASE10.md) — Fuzzy Edit Matching: make search/replace edits resilient to minor LLM output discrepancies.
- [PHASE11.md](planning/PHASE11.md) — Model Routing: route LLM calls to appropriate models based on task capability requirements.
- [PHASE12.md](planning/PHASE12.md) — Extended Thinking for Planning: enable Claude's extended thinking mode for planner calls.
- [PHASE13.md](planning/PHASE13.md) — Tree-Sitter Multi-Language Support: replace Python `ast` with tree-sitter for multi-language code analysis (deferred to Release 2).
- [PHASE14.md](planning/PHASE14.md) — Batch Processing: replace synchronous LLM calls with async batch processing via Anthropic Batch API.

## Research

- [planner-prompt-research.md](research/planner-prompt-research.md) — Survey of planning approaches in Claude Code, Aider, and other AI coding tools.
- [attractor-analysis.md](research/attractor-analysis.md) — Analysis of strongDM's "software factory" specification and comparison with Forge architecture.
- [system-prompts-claude.md](research/system-prompts-claude.md) — Catalog of system prompt components from Claude Code v2.1.50.

## Unmerged Reference Material

### Batch Completion Documents

- [merged-batch-completion-guide.md](to-merge/completion-documents/merged-batch-completion-guide.md) — Consolidated guide from five source reports with shared conclusions on batch completion best practices.
- [claude-01-batch-completion-guide.md](to-merge/completion-documents/claude-01-batch-completion-guide.md) — Best practices for structuring documents in batch-completion LLM APIs.
- [codex-01-batch-completion-document-best-practices.md](to-merge/completion-documents/codex-01-batch-completion-document-best-practices.md) — Document structure, model tuning, and multi-pass pipelines.
- [codex-02-completion-documents-batch-best-practices.md](to-merge/completion-documents/codex-02-completion-documents-batch-best-practices.md) — Recommended scaffold with goal/constraints/context/schema/rubric.
- [codex-03-batch-completion-document-best-practices.md](to-merge/completion-documents/codex-03-batch-completion-document-best-practices.md) — Document structure and model-family differences.
- [codex-04-batch-completion-document-best-practices.md](to-merge/completion-documents/codex-04-batch-completion-document-best-practices.md) — Fixed scaffold with objective, constraints, context, schema.

### Code Reviews

- [codex-01-code-review.md](to-merge/code-review/codex-01-code-review.md) — Architecture review: reliability, safety, observability, and operational concerns.
- [codex-02-code-review.md](to-merge/code-review/codex-02-code-review.md) — Assessment vs. OpenHands, SWE-agent, LangGraph, Aider.
- [codex-03-code-review.md](to-merge/code-review/codex-03-code-review.md) — Production readiness: queue backpressure, resource isolation, SLOs.
- [codex-04-code-review.md](to-merge/code-review/codex-04-code-review.md) — Core thesis assessment: runtime safety, conflict semantics, governance.
