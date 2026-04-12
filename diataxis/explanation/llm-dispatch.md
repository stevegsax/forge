# Model Routing and Batch Processing

Forge makes two kinds of decisions about every LLM call: which model to send it to, and how to send it. These decisions are made by different mechanisms for different reasons, but they share a common root — the batch-first, document-completion design that runs through the whole system.

This document explains how those mechanisms work and why they are designed the way they are. For the configuration details, see the [reference](../reference/llm-dispatch.md). For step-by-step recipes, see [how to configure LLM dispatch](../howto/configure-llm-dispatch.md).

---

## Capability Tiers

### Why abstract tiers instead of model names

A naive approach to model selection would be to pick a model name at the point in the code that needs a model. The planner would call `claude-opus-4-6`, code generation would call `claude-sonnet-4-5`, and so on. This works, but it couples two concerns that should be separate: the capability requirement (how much reasoning does this task need?) and the vendor resolution (which model satisfies that requirement today?).

Forge separates these concerns through a capability tier system. Call sites declare which tier they need — `REASONING`, `GENERATION`, `SUMMARIZATION`, or `CLASSIFICATION` — without naming a specific model. A `ModelConfig` object, passed in with the workflow input, maps each tier to a concrete model string at execution time.

This separation has practical benefits. When Anthropic releases a new model and the performance-to-cost ratio shifts, you update the tier mapping in one place rather than hunting through activity code. If you want to route a specific run to a different model for testing, you pass a custom `ModelConfig` without touching the workflow definition. And you can override individual tiers from the CLI — promoting exploration to a more capable model while leaving code generation at the default — without rebuilding anything.

The tier boundaries also make capability requirements explicit and reviewable. When a developer looks at a call site and sees `CapabilityTier.CLASSIFICATION`, that is a design claim: this call only needs a model that can classify, not one that can reason. That claim can be challenged, documented, and updated as understanding improves.

### The four tiers and their use cases

The four tiers reflect four qualitatively different task types that Forge executes.

**Reasoning** covers tasks where output quality is the binding constraint on everything downstream. The primary use case is planning: when the planner decomposes a task into ordered steps, every error in that plan propagates through all subsequent execution. A step that targets the wrong file, a dependency ordering that creates conflicts, a sub-task decomposition that misses an interface — these failures are expensive to recover from. Conflict resolution is also in the reasoning tier for the same reason: it is a judgment call where being wrong creates incorrect merged code.

**Generation** covers the primary output-producing calls: code generation, test writing, documentation. These calls produce the artifacts that the user actually wanted. Generation tasks are complex — the model must hold the full context of the task, the relevant code, the validation requirements, and the edit format simultaneously — but they are not as judgment-intensive as planning.

**Summarization** covers extraction and synthesis tasks where the input structure is clear and the output is a distillation rather than a judgment. Knowledge extraction from completed workflows is the main use case: the model reads a completed run's prompts and outputs and writes playbook entries that capture lessons learned. A less capable model can do this reliably, and routing it to a cheaper model reduces cost without meaningful quality loss.

**Classification** covers tasks that are structurally simple: making a discrete choice from a bounded option set. The exploration loop — where the model decides which context providers to call — is a classification task: given a task description and a list of available providers with descriptions, which providers are relevant? Transition evaluation — deciding whether a validation result maps to `SUCCESS`, `FAILURE_RETRYABLE`, or `FAILURE_TERMINAL` — is also in this tier.

### Why planning gets the reasoning tier

The architecture principle that planning is the hard part is not rhetorical. It has a direct consequence for model routing: the most capable available model goes to the planner.

The planner's output is a plan — an ordered list of steps, each with target files, descriptions, and optional sub-task decompositions. Everything that runs after the planner is bounded by this plan. If the plan contains a bad step ordering, execution will fail at that step. If the plan misses a file, the generated code will have missing imports. If the plan's sub-task boundaries create conflicts, conflict resolution must clean them up.

Better planning inputs produce better plans, and better plans reduce total work — fewer retries, fewer conflicts, less post-hoc repair. Spending more tokens on the planner is a leveraged investment. The cost of running a reasoning-tier model on the planner is typically dominated by the cost savings from shorter, cleaner execution downstream.

The same logic applies to conflict resolution. When two sub-tasks modify the same file, a reasoning-tier model receives both versions and produces a merged result. Getting this wrong produces incorrect code that then fails validation and requires another retry cycle — a much more expensive outcome than paying for the initial quality.

---

## Batch Mode

### Why submit-wait-resume instead of synchronous calls

Every LLM call in Forge is stateless and self-contained: the orchestrator assembles a complete prompt, sends it, and expects a single structured response. No streaming, no mid-turn tool calls, no conversation history. This is the document-completion paradigm described in [The Universal Workflow Step](../explanation/workflow-step.md).

This property is what makes batch mode possible. Because each call is independent, it can be submitted to a queue, processed at some future time, and the result retrieved later. The workflow does not need to hold an activity slot open while waiting; it can yield, let Temporal persist its state, and resume when the result arrives.

The [Anthropic Message Batches API](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) processes submitted requests asynchronously, typically within an hour, and charges 50% of the standard input token price. For Forge's workloads — where prompts can be 50K+ tokens and run across many steps — this is a significant cost reduction.

There are two other benefits beyond cost. First, worker slot consumption: in synchronous mode, each in-flight LLM call holds an activity slot for the duration of the API call. In fan-out scenarios with ten parallel sub-tasks, ten slots are held simultaneously. Batch mode releases the slot immediately after submission. Second, latency coupling: in synchronous mode, API slowdowns and rate limits stall the workflow directly. In batch mode, the workflow is insulated — it submitted its request and is waiting for a signal, unaffected by API response time.

### How the batch poller delivers results via Temporal signals

Three components work together to make batch mode work.

The workflow, when making an LLM call in batch mode, calls a `batch_submit` activity that formats the request and submits it to the Anthropic Batch API. It records the submission in the `batch_jobs` table and sets a Temporal Search Attribute (`forge_batch_id`) on the running workflow. It then calls `workflow.wait_condition()`, suspending execution until a signal arrives.

The batch poller is a `BatchPollerWorkflow` that runs on a Temporal Schedule (schedule ID: `forge-batch-poller`). It executes at a configurable interval (default 60 seconds, configurable via `--batch-poll-interval` on the worker). On each run, it queries the Anthropic API for completed batch jobs, matches results to waiting workflows using the `forge_batch_id` search attribute, and sends a `batch_result_received` signal to each matching workflow.

When the signal arrives, the workflow resumes. It calls a `batch_parse` activity to extract and validate the structured response from the batch result, then continues with the normal write-output and validate sequence.

The `batch_jobs` table records every submission and outcome. This provides an audit trail and enables anomaly detection: if a job appears in the local database as submitted but is absent from the Anthropic API, the poller logs it as a missing job. If a job appears on the Anthropic API but has no corresponding local record, it is logged as an unknown job.

### Why document completion enables batch

The connection between document completion and batch mode is not incidental. Batch APIs process each request as an independent unit. They do not support conversation sessions, tool calls that produce intermediate results to be appended to a running context, or stateful interactions of any kind.

If Forge used a chat-loop pattern — where the LLM decides when to call tools, accumulates results mid-conversation, and determines when to stop — batch mode would be structurally impossible. The request would need to stay open while the model called tools, received results, and continued reasoning. Batch APIs cannot support this.

The document-completion design eliminates this constraint. Each LLM call receives a fully assembled context document and returns a complete structured response. The orchestrator, not the LLM, manages the iteration. This means every call can be submitted as a standalone batch request, the model processes it in isolation, and the result is complete and self-contained.

This is why the architecture principle — the LLM call is the universal primitive — has a direct cost consequence. Designing every operation as a document completion is what makes the 50% cost reduction available.

---

## Prompt Caching

### Cache-efficient ordering

Forge's prompts are large. A system prompt with auto-discovered context, a repo map, playbooks, and exploration results can reach 50K tokens or more. Without prompt caching, every LLM call re-processes those tokens at full cost.

Anthropic's prompt caching allows a prefix of the input to be cached after the first call. Subsequent calls that share the same prefix pay a reduced cache read rate rather than the full input rate. The tradeoff is a modest write surcharge on the first call, which is recovered within one or two cache hits.

The key requirement is prefix stability: cached content must appear at the beginning of the prompt, and it must be identical across calls that should share the cache. Volatile content — content that changes between calls — must appear at the end.

Forge orders system prompt sections with stability as the primary criterion. The role statement, output requirements, and project instructions (from `CLAUDE.md`) are maximally stable: identical across all calls in a workflow and across many workflows in the same repository. Repository structure and playbooks change slowly. Target file contents change when a step commits. Exploration results change each round. Error context from a retry is unique to that attempt.

This ordering — stable first, volatile last — maximizes the cached prefix length. On a retry, the entire prompt up to the error section is potentially cached. On a subsequent planned step that targets different files, the entire prompt up to the target file contents section is potentially cached.

### Cache control headers

Cache breakpoints are placed at strategic positions in the prompt using Anthropic's `cache_control` header mechanism. The `call_llm` activity applies these headers when constructing the API request. Breakpoints are placed after the stable preamble, after the context section, and after the exploration results section — creating a tiered caching structure where the outermost layer (the stable preamble) has the longest cache lifetime and the innermost layer (exploration results) is only cached when it repeats across rounds.

Cache token counts are tracked in the `LLMStats` model and persisted to the observability store. The `interactions` table records `cache_creation_input_tokens` and `cache_read_input_tokens` for each call. The `forge status --verbose` output surfaces these per-interaction.
