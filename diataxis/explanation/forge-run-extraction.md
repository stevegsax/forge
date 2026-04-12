# Forge Run Extraction

Prerequisites: [Context Assembly](context-assembly.md).

Forge has two separate learning pipelines. This document covers the first one — Forge's self-learning loop, scoped entirely to Forge's own completed run history. It does not cover transcript ingestion from Claude Code sessions; that pipeline is a separate topic, documented in [Transcript Ingestion](transcript-ingestion.md), and it writes to a different database. For the cross-cutting story of why there are two loops and how they relate, see [Learning Loops](learning-loops.md).

Over time, Forge accumulates experience from its own runs: completed tasks, retried steps, failed validations, and the interventions that resolved them. Without a mechanism to capture and reuse that experience, the same context gaps and error patterns recur in every new run. Forge run extraction is the system's answer. It processes completed workflow results from Forge's observability store, asks an LLM to identify reusable lessons, and stores them as structured entries called playbooks in Forge's own `playbooks` table. Future Forge tasks that resemble previous ones receive those lessons as part of their assembled context.

For technical details on schemas, tag rules, and CLI commands, see the [Forge Run Extraction Reference](../reference/forge-run-extraction.md). For practical steps, see [How to Manage Playbooks](../howto/manage-playbooks.md).

## The playbook concept

A playbook entry is a titled, tagged lesson derived from prior work. It is not a log entry or a raw interaction dump — it is a synthesized, human-readable note of the form "when doing X, remember Y." A typical entry might say: "When generating Pydantic models for SQLAlchemy tables, include the `from __future__ import annotations` import; the ORM mapper fails without it." Another might document a validation pattern that consistently catches a particular category of error.

Each entry carries a small set of tags that describe the context where it applies: the programming language, the task type (test-writing, refactoring, migration), the domain. Tags are how the system decides which playbooks to surface for a given task. The tags on a new task are compared to the tags on stored entries; overlapping entries are included in the context assembly.

## Why extraction runs independently

Knowledge extraction is not on the critical path of task execution. It runs as a separate Temporal workflow — `ForgeExtractionWorkflow` — triggered manually via `forge extract`. This is a deliberate design choice, not an omission.

The reason is priority. The task execution pipeline is already doing expensive, time-sensitive work: assembling context, calling LLMs, applying edits, running validation. Adding extraction to that pipeline would extend every task's execution time, introduce additional failure modes, and block the workflow on extraction success. None of that is acceptable when the task result is what the user actually wants.

Extraction is also naturally batchy: it benefits from processing multiple completed runs at once, identifying patterns across tasks, and producing a consolidated set of entries rather than one per task. This matches the workflow-level granularity of `ForgeExtractionWorkflow`, which fetches a configurable batch of unextracted runs in a single call.

## The extraction pipeline

The extraction workflow has three activities, executed in sequence.

**`fetch_extraction_input`** queries the observability store for runs that have not yet been processed. It applies a configurable limit and a lookback window (default: runs within the last N hours), then assembles an extraction prompt from the run data — task descriptions, assembled contexts, LLM responses, validation results, and any error feedback that was injected on retry.

**`call_extraction_llm`** sends the assembled prompt to a summarization-tier model and expects a structured `ExtractionResult` response. The prompt instructs the model to identify lessons worth capturing, each with a title, content, and suggested tags. This is the same document-completion pattern used everywhere in Forge: a complete prompt in, structured output out.

**`save_extraction_results`** writes each `PlaybookEntry` to the `playbooks` table in the SQLite observability store. It also marks the processed runs as extracted, so they are not reprocessed on the next `forge extract` call.

The workflow short-circuits cleanly: if there are no unextracted runs, it returns immediately. If the LLM returns no entries (nothing worth capturing), it skips the save step.

## Tag inference

Tags are derived deterministically from task metadata — they are not produced by the LLM. This keeps the tagging consistent and queryable without relying on the model's judgment about which vocabulary to use.

The inference rules operate on two kinds of signals. First, file extensions in the task's target files: `.py` maps to `python`, `.ts` and `.tsx` to `typescript`, `.js` and `.jsx` to `javascript`. Second, keywords in the task description: the word "test" maps to `test-writing`, "refactor" to `refactoring`, "api" to `api`, "database" to `database`, "migration" to `migration`, "cli" to `cli`, "validate" to `validation`, and "bug" or "fix" to `bug-fix`. If none of these match, the default tag `code-generation` is applied.

The same inference logic runs in two places: during extraction (to tag the new entries) and during retrieval (to compute the query tags for the current task). This symmetry is what makes the matching work. A playbook entry tagged `python, test-writing` will be retrieved for any subsequent task whose metadata yields those same tags.

## Playbook injection into context assembly

Playbooks are a context source. During context assembly, the `assemble_context` activity retrieves playbooks whose tags overlap with the current task's inferred tags, then wraps each as a `ContextItem` with `Representation.PLAYBOOK` at priority 5. For the token budget packing order, priority 5 sits between deterministic analysis results (priority 4) and broader project context (priority 6).

The consequence of that position is that playbooks are included when the budget allows, but are the first items dropped when the budget is tight. This is intentional: a playbook is an optimization, not a correctness requirement. The task can succeed without the playbook; it may succeed more efficiently with it. If the budget is already full with target files, interface context, and validation results, dropping the playbooks is the right tradeoff.

For how context assembly uses the token budget packer and its priority ordering, see [Context Assembly](context-assembly.md).

## The self-learning feedback loop

Forge run extraction creates a feedback loop: the system learns from its own history. A Forge task that fails due to a missing type stub produces a validation error, which is recorded in the observability store. Extraction processes that run, identifies the pattern, and creates a playbook entry noting the dependency. The next similar Forge task receives that entry in its context, before it makes the same mistake.

The loop works for successes too, not just failures. When a task succeeds and the assembled context included an unusually helpful file or a specific import that the LLM used immediately, extraction can note that pattern and surface it for future tasks.

The loop is not instantaneous — it requires running `forge extract` between the original task and the future task. In practice, running extraction periodically (after a batch of tasks or at the end of a work session) keeps the playbook store current without adding overhead to individual task runs.

This loop does not learn from anything outside Forge's own observability store. A lesson captured from a Claude Code conversation transcript, a lesson added manually through pbook's CLI, or a lesson extracted from another project's runs — none of those can flow into Forge's playbook table through this pipeline. Those sources are handled by [Transcript Ingestion](transcript-ingestion.md), which writes to a separate store (pbook's). For the full picture of how these two pipelines relate, and what a unified pipeline might look like, see [Learning Loops](learning-loops.md).
