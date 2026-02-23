# Attractor Analysis: Pyramid Document

**Repository:** [strongdm/attractor](https://github.com/strongdm/attractor)
**Date:** 2026-02-18
**Status:** First Draft

---

## Executive Summary

Attractor is a **specification-only** repository (no implementation code) from strongDM that describes a "software factory" built from three layered specs: a **DOT-based pipeline runner** for orchestrating multi-stage AI workflows, a **coding agent loop** library, and a **unified multi-provider LLM client SDK**. The specs are written as "NLSpecs" (natural language specifications) intended to be directly implementable by coding agents.

Attractor and Forge solve the same fundamental problem -- orchestrating multi-step LLM workflows for code generation -- but make different architectural bets. Forge is batch-first with Temporal as the workflow engine and the orchestrator controlling the loop. Attractor is agent-first with a declarative DOT graph defining the workflow and an agentic loop giving the LLM real-time tool access. The two approaches are complementary, and several Attractor ideas could strengthen Forge.

**Top integration opportunities (ranked by impact):**

1. **Multi-provider LLM abstraction** -- Forge is Anthropic-only; Attractor's unified SDK pattern would enable OpenAI/Gemini
2. **Context fidelity modes** -- Explicit control over how much prior state carries between steps
3. **Declarative pipeline visualization** -- DOT graph rendering for workflow inspection
4. **Goal gate enforcement** -- Mandatory success checkpoints before pipeline completion
5. **Execution environment abstraction** -- Run tools in Docker/K8s instead of only locally

---

## Table of Contents

- [1. What Attractor Is](#1-what-attractor-is)
- [2. Document Map](#2-document-map)
- [3. Core Architecture](#3-core-architecture)
    - [3.1 Pipeline Orchestration (attractor-spec)](#31-pipeline-orchestration-attractor-spec)
    - [3.2 Coding Agent Loop (coding-agent-loop-spec)](#32-coding-agent-loop-coding-agent-loop-spec)
    - [3.3 Unified LLM SDK (unified-llm-spec)](#33-unified-llm-sdk-unified-llm-spec)
- [4. Concept-by-Concept Comparison with Forge](#4-concept-by-concept-comparison-with-forge)
- [5. Ideas Worth Integrating](#5-ideas-worth-integrating)
- [6. Ideas That Don't Fit](#6-ideas-that-dont-fit)
- [7. Topic Index](#7-topic-index)

---

## 1. What Attractor Is

Attractor is **three specification documents** totaling ~270KB of markdown. There is no implementation code. The specs are designed as "NLSpecs" -- detailed enough that a coding agent (Claude Code, Codex, etc.) can implement them directly from the document.

The three specs form a layered stack:

```
┌─────────────────────────────────────────────────┐
│  attractor-spec.md (Pipeline Orchestration)      │  ← Highest level
│  DOT graphs define multi-stage AI workflows      │
├─────────────────────────────────────────────────┤
│  coding-agent-loop-spec.md (Agentic Loop)        │  ← Middle layer
│  Programmable agent with tool execution          │
├─────────────────────────────────────────────────┤
│  unified-llm-spec.md (LLM Client SDK)           │  ← Foundation
│  Multi-provider LLM abstraction                  │
└─────────────────────────────────────────────────┘
```

**Key insight:** Attractor's pipeline spec is backend-agnostic. The `CodergenBackend` interface accepts any LLM execution strategy -- the coding agent loop, direct API calls, CLI subprocess spawning, or something else entirely. This is a deliberate decoupling.

> Source: [README.md](https://github.com/strongdm/attractor/blob/main/README.md) lines 1-24

---

## 2. Document Map

### attractor-spec.md (~90KB)

The pipeline orchestration spec. Defines how directed graphs (in Graphviz DOT syntax) describe multi-stage AI workflows.

| Section | Topic | Lines (approx) |
|---------|-------|----------------|
| [1. Overview and Goals](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#1-overview-and-goals) | Problem statement, design principles | 1-60 |
| [2. DOT DSL Schema](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#2-dot-dsl-schema) | Grammar, node/edge attributes, shape-to-handler mapping | 64-310 |
| [3. Pipeline Execution Engine](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#3-pipeline-execution-engine) | Core loop, edge selection, retry, goal gates | 315-570 |
| [4. Node Handlers](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#4-node-handlers) | Codergen, human gate, parallel, fan-in, tool, manager loop | 574-990 |
| [5. State and Context](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#5-state-and-context) | Context store, outcomes, checkpoints, fidelity modes, artifacts | 994-1240 |
| [6. Human-in-the-Loop](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#6-human-in-the-loop-interviewer-pattern) | Interviewer interface, question model, built-in implementations | 1243-1370 |
| [7. Validation and Linting](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#7-validation-and-linting) | Diagnostic model, built-in lint rules | 1374-1445 |
| [8. Model Stylesheet](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#8-model-stylesheet) | CSS-like per-node LLM configuration | 1448-1525 |
| [9. Transforms and Extensibility](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#9-transforms-and-extensibility) | AST transforms, HTTP server, events, tool hooks | 1526-1660 |
| [10. Condition Expression Language](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#10-condition-expression-language) | Edge routing conditions | 1662-1775 |
| [11. Definition of Done](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#11-definition-of-done) | Validation checklists, smoke tests | 1776-1975 |

### coding-agent-loop-spec.md (~69KB)

Spec for a programmable agentic loop library. This is the "inner loop" that gives an LLM real-time tool access.

| Section | Topic | Lines (approx) |
|---------|-------|----------------|
| [1. Overview and Goals](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#1-overview-and-goals) | Problem statement, design principles, architecture | 1-120 |
| [2. Agentic Loop](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#2-agentic-loop) | Session, turn types, core loop, steering, stop conditions, events, loop detection | 124-450 |
| [3. Provider-Aligned Toolsets](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#3-provider-aligned-toolsets) | Per-provider tool profiles (OpenAI/Anthropic/Gemini), tool registry | 454-700 |
| [4. Tool Execution Environment](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#4-tool-execution-environment) | ExecutionEnvironment abstraction, local/Docker/K8s/WASM/SSH | 710-836 |
| [5. Tool Output and Context Management](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#5-tool-output-and-context-management) | Truncation (head/tail), output size limits, timeouts, context awareness | 839-975 |
| [6. System Prompts and Environment Context](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#6-system-prompts-and-environment-context) | Layered prompt construction, project doc discovery | 976-1048 |
| [7. Subagents](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#7-subagents) | Spawn/send/wait/close interface, depth limiting | 1049-1115 |
| [8. Out of Scope](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#8-out-of-scope-nice-to-haves) | MCP, skills, sandbox, compaction, approvals | 1117-1135 |
| [9. Definition of Done](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#9-definition-of-done) | Checklists, cross-provider parity matrix | 1136-1300 |
| [Appendix A](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#appendix-a-apply_patch-v4a-format-reference) | apply_patch v4a format reference (OpenAI) | 1300-1390 |
| [Appendix B](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#appendix-b-error-handling) | Error handling taxonomy | 1391-1435 |
| [Appendix C](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#appendix-c-design-decision-rationale) | Design decision rationale | 1437-1452 |

### unified-llm-spec.md (~108KB)

Spec for a multi-provider LLM client library with a unified interface.

| Section | Topic | Lines (approx) |
|---------|-------|----------------|
| [1. Overview and Goals](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#1-overview-and-goals) | Problem statement, design principles, reference projects | 1-50 |
| [2. Architecture](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#2-architecture) | Four-layer architecture, client config, middleware, provider adapters, model catalog, prompt caching | 52-345 |
| [3. Data Model](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#3-data-model) | Message, Role, ContentPart, Request, Response, Usage, StreamEvent | 346-790 |
| [4. Generation and Streaming](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#4-generation-and-streaming) | complete(), stream(), generate(), generate_object(), cancellation | 794-1048 |
| [5. Tool Calling](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#5-tool-calling) | Tool definitions, execute handlers, tool choice, multi-step loop, parallel execution | 1050-1270 |
| [6. Error Handling and Retry](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#6-error-handling-and-retry) | Error taxonomy, retry policy, rate limiting | 1274-1466 |
| [7. Provider Adapter Contract](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#7-provider-adapter-contract) | Per-provider message/tool/response/streaming translation details | 1469-1780 |
| [8. Definition of Done](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#8-definition-of-done) | Validation checklists | 1967-end |

---

## 3. Core Architecture

### 3.1 Pipeline Orchestration (attractor-spec)

**Core idea:** Workflows are defined as **directed graphs in Graphviz DOT syntax**. Nodes are tasks, edges are transitions with conditions, and attributes configure behavior. An execution engine traverses the graph deterministically.

**Node handler types** (resolved by node `shape` attribute):

| Shape | Handler | Forge Equivalent |
|-------|---------|-----------------|
| `Mdiamond` | `start` (no-op entry point) | Workflow start |
| `Msquare` | `exit` (no-op terminal) | Workflow completion |
| `box` | `codergen` (LLM task) | `call_llm` activity |
| `hexagon` | `wait.human` (human gate) | Human-gated merge to main |
| `diamond` | `conditional` (route by condition) | `evaluate_transition` activity |
| `component` | `parallel` (fan-out) | `ForgeSubTaskWorkflow` fan-out |
| `tripleoctagon` | `parallel.fan_in` (gather) | Sub-task gathering + conflict resolution |
| `parallelogram` | `tool` (external command) | `validate_output` activity |
| `house` | `stack.manager_loop` (supervisor) | No direct equivalent |

> Source: [attractor-spec.md Section 2.8](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#28-shape-to-handler-type-mapping)

**Edge selection algorithm** (5-step priority):

1. Condition-matching edges (boolean expression evaluated against context)
2. Preferred label match (handler suggests which edge to take)
3. Suggested next IDs (handler suggests specific target nodes)
4. Highest weight among unconditional edges
5. Lexical tiebreak on target node ID

> Source: [attractor-spec.md Section 3.3](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#33-edge-selection-algorithm)

**Execution lifecycle:**

```
PARSE → VALIDATE → INITIALIZE → EXECUTE → FINALIZE
```

The core loop: resolve start node → execute handler → record outcome → apply context updates → save checkpoint → select edge → advance → repeat.

> Source: [attractor-spec.md Section 3.2](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#32-core-execution-loop)

**Key mechanisms:**

- **Goal gates** -- Nodes marked `goal_gate=true` must reach SUCCESS before the pipeline can exit. If unsatisfied at exit, the engine jumps to a `retry_target` node.
    > Source: [attractor-spec.md Section 3.4](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#34-goal-gate-enforcement)

- **Context fidelity** -- Controls how much prior conversation state carries to the next node's LLM session: `full` (reuse session), `truncate` (minimal), `compact` (structured summary), `summary:low/medium/high` (varying detail levels with token budgets).
    > Source: [attractor-spec.md Section 5.4](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#54-context-fidelity)

- **Model stylesheet** -- CSS-like rules for assigning LLM models to nodes by class or ID. `* { llm_model: claude-sonnet-4-5 }` sets default; `.code { llm_model: claude-opus-4-6 }` overrides for code tasks; `#critical_review { llm_model: gpt-5.2 }` pins a specific node.
    > Source: [attractor-spec.md Section 8](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#8-model-stylesheet)

- **Condition expression language** -- Minimal boolean expressions on edges: `outcome=success`, `context.tests_passed=true && outcome!=fail`. Only `=` and `!=` operators with `&&` conjunction.
    > Source: [attractor-spec.md Section 10](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#10-condition-expression-language)

- **Manager loop handler** -- A supervisor node that orchestrates observe/steer/wait cycles over a child pipeline. Ingests child telemetry, evaluates progress, optionally steers the child via intervention instructions.
    > Source: [attractor-spec.md Section 4.11](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#411-manager-loop-handler)

- **Interviewer pattern** -- All human interaction goes through an `Interviewer` interface with pluggable implementations: `ConsoleInterviewer`, `AutoApproveInterviewer`, `QueueInterviewer`, `CallbackInterviewer`, `RecordingInterviewer`.
    > Source: [attractor-spec.md Section 6](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#6-human-in-the-loop-interviewer-pattern)

### 3.2 Coding Agent Loop (coding-agent-loop-spec)

**Core idea:** A **programmable agentic loop library** (not a CLI) that gives a host application full control over the agent's execution. The host can submit input, observe every event, steer mid-task, change configuration, swap execution environments, and compose agents.

**Key distinction from Forge:** The coding agent loop gives the LLM real-time tool access (read files, edit files, run commands). Forge's model is batch document completion -- the LLM produces a structured response and the orchestrator applies it. The agent loop is interactive; Forge is declarative.

**Core loop pseudocode** (simplified):

```
LOOP:
    check limits (max_turns, max_rounds, abort)
    build LLM request (system prompt + history + tools)
    response = client.complete(request)
    record assistant turn
    IF no tool calls: BREAK (natural completion)
    execute tool calls through ExecutionEnvironment
    drain steering messages
    check loop detection
END LOOP
```

> Source: [coding-agent-loop-spec.md Section 2.5](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#25-the-core-agentic-loop)

**Provider-aligned toolsets** -- Each LLM provider gets tools matching its native agent:

- OpenAI: `apply_patch` (v4a diff format) replaces `edit_file`
- Anthropic: `edit_file` (old_string/new_string search/replace)
- Gemini: gemini-cli-aligned tools including `web_search`

> Source: [coding-agent-loop-spec.md Section 3.1](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#31-the-provider-alignment-principle)

**Execution environment abstraction** -- All tool operations go through an `ExecutionEnvironment` interface:

- `LocalExecutionEnvironment` (default)
- `DockerExecutionEnvironment` (sandboxed)
- `KubernetesExecutionEnvironment` (cloud)
- `WASMExecutionEnvironment` (browser)
- `RemoteSSHExecutionEnvironment` (remote dev)

Environments are composable with decorators: `LoggingExecutionEnvironment(inner)`, `ReadOnlyExecutionEnvironment(inner)`.

> Source: [coding-agent-loop-spec.md Section 4](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#4-tool-execution-environment)

**Tool output truncation** -- Head/tail split preserving both beginning (imports, headers) and end (results):

```
output[0..half] + "[WARNING: truncated, N chars removed]" + output[-half..]
```

Default limits: read_file 50K, shell 30K, grep 20K, glob 20K. Character-based truncation runs first (handles pathological cases), then line-based.

> Source: [coding-agent-loop-spec.md Section 5](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#5-tool-output-and-context-management)

**Steering** -- The host can inject messages mid-task via `session.steer(message)`. Messages are queued and injected between tool rounds as user-role messages.

> Source: [coding-agent-loop-spec.md Section 2.6](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#26-steering)

**Loop detection** -- Tracks tool call signatures (name + args hash). If the last N calls (default 10) contain a repeating pattern of length 1-3, injects a warning telling the model to try a different approach.

> Source: [coding-agent-loop-spec.md Section 2.10](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#210-loop-detection)

### 3.3 Unified LLM SDK (unified-llm-spec)

**Core idea:** A **four-layer client library** providing a single interface across OpenAI, Anthropic, and Gemini. Each provider adapter speaks the provider's **native API** (not a compatibility shim).

**Four layers:**

```
Layer 4: High-Level API         generate(), stream(), generate_object()
Layer 3: Core Client            Client, provider routing, middleware
Layer 2: Provider Utilities     HTTP helpers, SSE parsing, retry
Layer 1: Provider Specification  ProviderAdapter interface, shared types
```

> Source: [unified-llm-spec.md Section 2.1](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#21-four-layer-architecture)

**Critical design decisions:**

- **Native API usage** -- OpenAI Responses API (not Chat Completions), Anthropic Messages API, Gemini GenerateContent API. Using compatibility layers loses reasoning tokens, thinking blocks, prompt caching, etc.
    > Source: [unified-llm-spec.md Section 2.7](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#27-native-api-usage-critical)

- **Provider options escape hatch** -- `provider_options` dict for provider-specific params (Anthropic beta headers, Gemini safety settings) that don't fit the unified model.
    > Source: [unified-llm-spec.md Section 3.6](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#36-request)

- **Prompt caching** -- OpenAI and Gemini: automatic. Anthropic: requires explicit `cache_control` annotations (90% cost reduction when properly configured).
    > Source: [unified-llm-spec.md Section 2.10](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#210-prompt-caching-critical-for-cost)

- **Model catalog** -- Ships a data file of known models with capabilities, context windows, and costs. Advisory, not restrictive -- unknown model strings pass through.
    > Source: [unified-llm-spec.md Section 2.9](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#29-model-catalog)

- **Middleware pattern** -- Onion/chain-of-responsibility for logging, caching, cost tracking, rate limiting. Works for both blocking and streaming calls.
    > Source: [unified-llm-spec.md Section 2.3](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#23-middleware--interceptor-pattern)

---

## 4. Concept-by-Concept Comparison with Forge

| Concept | Forge | Attractor | Notes |
|---------|-------|-----------|-------|
| **Workflow engine** | Temporal (durable, distributed) | Custom graph traversal engine | Forge's Temporal provides durability, retries, and distribution for free. Attractor rolls its own with checkpoints. |
| **Workflow definition** | Python code (workflow + activities) | Graphviz DOT files (declarative) | Attractor's DOT graphs are visual, diffable, version-controllable. Forge's code-defined workflows are more flexible but harder to visualize. |
| **LLM interaction model** | Batch document completion (orchestrator controls loop) | Agentic loop (LLM controls tools in real-time) | Fundamental architectural difference. Forge optimizes for cost via batch API; Attractor optimizes for capability via tool use. |
| **LLM providers** | Anthropic only | OpenAI + Anthropic + Gemini (unified SDK) | Forge's single-provider focus means deep integration. Attractor's multi-provider approach enables model selection per task. |
| **Multi-step execution** | Planner decomposes into PlanSteps, executed sequentially | DOT graph defines step ordering with edge conditions | Forge's planner is LLM-generated; Attractor's graph is human-authored. Both support conditional branching. |
| **Fan-out / parallel** | ForgeSubTaskWorkflow with child workflows per sub-task | `parallel` handler with join policies (wait_all, first_success, k_of_n) | Both support parallel branches. Attractor has richer join policies. |
| **Context between steps** | Assembled fresh per step (progressive disclosure + exploration) | Context fidelity modes (full/truncate/compact/summary:*) | Attractor's fidelity system is more explicit. Forge rebuilds context each step. |
| **Human-in-the-loop** | Human-gated merge to main (external to workflow) | Interviewer pattern with pluggable frontends (CLI, web, queue) | Attractor's approach is richer -- human gates are first-class workflow nodes. |
| **Retry logic** | TransitionSignal-based (retryable vs terminal) | Configurable retry policies with backoff presets | Both use exponential backoff. Attractor has named presets (standard, aggressive, patient). |
| **Error feedback** | Phase 8: prior errors + AST context fed back on retry | Outcome-based routing to retry targets | Forge's error-aware retries are more sophisticated (AST-enriched error context). |
| **Validation** | ruff lint/format + optional test command | DOT graph linting (structural validation) | Different domains -- Forge validates generated code; Attractor validates workflow definitions. |
| **Model routing** | CapabilityTier enum (REASONING/GENERATION/SUMMARIZATION/CLASSIFICATION) | Model stylesheet (CSS-like rules) + per-node attributes | Attractor's stylesheet is more declarative and visual. Forge's tier system is more programmatic. |
| **Observability** | OTel spans + SQLite store + Temporal history | Typed event stream (pipeline, stage, parallel, human, checkpoint events) | Both comprehensive. Forge persists to SQLite; Attractor streams events. |
| **Prompt caching** | Phase 9: cache_control headers on system prompts + tool defs | SDK-level: automatic for OpenAI/Gemini, explicit annotations for Anthropic | Similar approach for Anthropic. Attractor handles caching for all providers. |
| **Git strategy** | Worktrees per task, branched from main | Not specified (pipeline-level concern, not in spec) | Forge's git worktree isolation is a strength not addressed by Attractor. |
| **Knowledge extraction** | Phase 6: playbook generation from completed tasks | Not specified | Forge-unique capability. |
| **Context exploration** | Phase 7: LLM requests context from providers before generating | Not specified (agent loop has real-time file access) | Different approaches. Forge's exploration is pre-generation; Attractor's agent reads files in-loop. |
| **Tool execution** | Activities execute tools (ruff, git, tests) | ExecutionEnvironment abstraction (local/Docker/K8s/WASM/SSH) | Attractor's environment abstraction is more flexible. |
| **Checkpointing** | Temporal provides durable execution | JSON checkpoint after each node | Temporal is more robust; Attractor's checkpoints are simpler but manual. |

---

## 5. Ideas Worth Integrating

### 5.1 Multi-Provider LLM Abstraction

**What:** Attractor's unified LLM SDK provides a single interface across OpenAI, Anthropic, and Gemini, using each provider's native API.

**Why it matters for Forge:** Forge is locked to Anthropic. Supporting OpenAI (especially GPT-5.2-codex for code tasks) and Gemini (for cost-effective classification) would enable per-step model selection based on capability and cost. The planner could use one provider while generation steps use another.

**Integration approach:** Create a provider adapter layer in Forge's `llm_client.py` that wraps provider-specific SDKs behind a common interface. Forge already has `CapabilityTier` → model mapping; extend this to include provider selection.

**Complexity:** High. Requires new provider adapters, response normalization, and testing across providers.

> Source: [unified-llm-spec.md Sections 2.1, 2.7, 7](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#21-four-layer-architecture)

### 5.2 Context Fidelity Modes

**What:** Explicit control over how much prior context carries between pipeline steps:

- `full` -- Reuse LLM session (full conversation history)
- `truncate` -- Minimal (only goal and run ID)
- `compact` -- Structured bullet-point summary
- `summary:low/medium/high` -- Varying detail (600/1500/3000 tokens)

**Why it matters for Forge:** Forge currently assembles context fresh for each step. For long pipelines, explicitly controlling context carryover could reduce token costs and improve coherence. A `full` mode would let consecutive steps share a conversation, while `summary:medium` would give later steps awareness of earlier work without the full token cost.

**Integration approach:** Add a `fidelity` field to `PlanStep`. In `assemble_context`, use fidelity to decide how much prior step history to include. `full` could reuse an Anthropic conversation via prompt caching; `summary:*` could generate a compressed summary of prior steps.

**Complexity:** Medium. Requires changes to context assembly and planner prompts.

> Source: [attractor-spec.md Section 5.4](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#54-context-fidelity)

### 5.3 Goal Gate Enforcement

**What:** Nodes marked `goal_gate=true` must reach SUCCESS before the pipeline can exit. If any goal gate is unsatisfied when reaching the exit node, the engine jumps to a retry target.

**Why it matters for Forge:** Forge's planner can decompose tasks into steps, but there's no mechanism to enforce that critical steps (e.g., "tests must pass") succeed before the workflow completes. Goal gates would provide a declarative way to express "this step is mandatory" and automatically re-route on failure.

**Integration approach:** Add `goal_gate: bool` to `PlanStep`. In the workflow, after all steps complete, check if all goal-gate steps succeeded. If not, re-execute from a specified recovery point.

**Complexity:** Low-Medium. Straightforward to add to the workflow logic.

> Source: [attractor-spec.md Section 3.4](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#34-goal-gate-enforcement)

### 5.4 Model Stylesheet Pattern

**What:** CSS-like rules for assigning LLM models to pipeline nodes:

```css
* { llm_model: claude-sonnet-4-5; }
.code { llm_model: claude-opus-4-6; }
#critical_review { llm_model: gpt-5.2; reasoning_effort: high; }
```

Selectors match by universal (`*`), class (`.code`), or node ID (`#critical_review`). Specificity: universal < class < ID. Later rules of equal specificity win.

**Why it matters for Forge:** Forge already has `CapabilityTier` mapping and per-step model overrides, but the configuration is programmatic. A stylesheet-like approach would be more declarative and easier to tune -- especially for operations teams who want to change model assignments without modifying Python code.

**Integration approach:** This could be a configuration layer on top of Forge's existing model routing. A YAML/TOML config file with selector-based rules that map to `CapabilityTier` or direct model names.

**Complexity:** Medium. Requires a selector/specificity engine and a config format.

> Source: [attractor-spec.md Section 8](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#8-model-stylesheet)

### 5.5 Execution Environment Abstraction

**What:** All tool execution goes through an `ExecutionEnvironment` interface with methods for file I/O, command execution, and search. Implementations exist for local, Docker, Kubernetes, WASM, and SSH.

**Why it matters for Forge:** Forge currently runs all validation and git operations locally. An environment abstraction would enable sandboxed execution (Docker) for untrusted code generation, remote execution (K8s) for scaling, and SSH-based execution for deploying to target machines.

**Integration approach:** Define an `ExecutionEnvironment` protocol that Forge's activities use for file and command operations. Start with `LocalExecutionEnvironment` (current behavior), then add `DockerExecutionEnvironment` for sandboxed validation.

**Complexity:** High. Requires refactoring activities to use the environment abstraction.

> Source: [coding-agent-loop-spec.md Section 4](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#4-tool-execution-environment)

### 5.6 Pipeline Visualization via DOT

**What:** Attractor's workflows are DOT graphs that can be rendered to SVG/PNG with standard Graphviz tooling, giving immediate visual feedback on workflow structure.

**Why it matters for Forge:** Forge's planned workflows (sequences of `PlanStep` with optional sub-tasks) have an implicit graph structure that is hard to visualize. Generating a DOT representation of a plan and rendering it would help with debugging, documentation, and human review of plans before execution.

**Integration approach:** Add a `plan_to_dot()` function that converts a `Plan` to DOT syntax. Render with `graphviz` Python package. Add a CLI command like `forge plan --visualize`.

**Complexity:** Low. Pure function, no architectural changes needed.

> Source: [attractor-spec.md Section 1.2](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#12-why-dot-syntax)

### 5.7 Interviewer Pattern for Human Interaction

**What:** All human interaction goes through a pluggable `Interviewer` interface. Implementations include console (CLI), auto-approve (CI/CD), queue (testing), callback (web/Slack), and recording (audit).

**Why it matters for Forge:** Forge's human interaction is limited to gated merges. For more complex workflows, structured human gates (approve/reject/redirect) within the workflow would be valuable. The interviewer pattern also enables deterministic testing of human-interaction paths.

**Integration approach:** Define an `Interviewer` protocol. Add a `wait.human` activity type that presents choices and blocks. Use `QueueInterviewer` in tests for deterministic execution.

**Complexity:** Medium. Requires a new activity type and interviewer implementations.

> Source: [attractor-spec.md Section 6](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#6-human-in-the-loop-interviewer-pattern)

### 5.8 Loop Detection

**What:** Track tool call signatures (name + args hash). If the last N calls contain a repeating pattern of length 1-3, inject a warning telling the model to try a different approach.

**Why it matters for Forge:** Forge's exploration loop (Phase 7) could get stuck requesting the same context repeatedly. Adding loop detection to the exploration round would catch this pattern and either break the loop or escalate.

**Integration approach:** Add a `detect_loop()` function that checks exploration request patterns. Apply it in the exploration loop before each new round.

**Complexity:** Low. Small utility function.

> Source: [coding-agent-loop-spec.md Section 2.10](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#210-loop-detection)

### 5.9 Tool Output Truncation Strategy

**What:** Head/tail split preserving both beginning and end of output, with explicit truncation markers telling the model what was removed. Character-based truncation runs first (handles pathological cases), then line-based.

**Why it matters for Forge:** Forge's context assembly already has token budgets and packing, but the truncation strategy for individual files could be improved. The head/tail approach preserves imports (beginning) and recent code (end), which is more useful than simple truncation from the end.

**Integration approach:** Apply head/tail truncation in Forge's `pack_context()` when a file exceeds its token budget.

**Complexity:** Low. Algorithm change in context packing.

> Source: [coding-agent-loop-spec.md Section 5.1](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#51-tool-output-truncation)

---

## 6. Ideas That Don't Fit

### 6.1 Agentic Loop as Primary Execution Model

Attractor's coding agent loop gives the LLM real-time tool access (read, edit, run commands). This is fundamentally incompatible with Forge's batch-first architecture. Forge deliberately chooses batch document completion for cost efficiency -- every LLM call is a structured response, not an interactive tool loop. Switching to an agentic model would sacrifice Forge's batch API compatibility and cost advantages.

**However:** A hybrid approach could work for specific phases. The planner or exploration loop could use an agentic session (small, bounded) while keeping batch document completion for the main generation steps.

### 6.2 DOT as Primary Workflow Definition

While DOT visualization is valuable (see 5.6), replacing Forge's Temporal-based workflow definition with DOT graphs would sacrifice:

- Temporal's durable execution guarantees
- Built-in retry, timeout, and error handling
- Distributed execution across workers
- Activity-level checkpointing

DOT is better suited as a visualization layer on top of existing workflows, not a replacement.

### 6.3 Rolling Your Own Checkpoint/Resume

Attractor specifies manual JSON checkpointing after each node. Forge gets this for free from Temporal's durable execution model. No reason to reimplement.

### 6.4 Provider-Aligned Tool Formats

Attractor specifies different edit formats per provider (apply_patch for OpenAI, old_string/new_string for Anthropic). Forge uses search/replace edits exclusively (Anthropic-aligned). If Forge adds multi-provider support, provider-aligned tool formats would add complexity without clear benefit -- Forge's edits list is a structured output model, not a tool call format.

---

## 7. Topic Index

Quick-reference index for revisers to locate specific topics across the three Attractor specs.

### A

- **Abort signals** -- [coding-agent-loop-spec Section 2.8](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#28-stop-conditions), [unified-llm-spec Section 4.7](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#47-cancellation-and-timeouts)
- **Accelerator keys (human gates)** -- [attractor-spec Section 4.6](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#46-wait-for-human-handler)
- **apply_patch (v4a format)** -- [coding-agent-loop-spec Appendix A](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#appendix-a-apply_patch-v4a-format-reference)
- **Artifact store** -- [attractor-spec Section 5.5](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#55-artifact-store)
- **AST transforms** -- [attractor-spec Section 9.1](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#91-ast-transforms)

### B

- **Backoff (exponential with jitter)** -- [attractor-spec Section 3.6](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#36-retry-policy), [unified-llm-spec Section 6.6](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#66-retry-policy)
- **Beta headers (Anthropic)** -- [unified-llm-spec Section 2.8](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#28-provider-beta-headers-and-feature-flags)
- **BNF grammar (DOT subset)** -- [attractor-spec Section 2.2](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#22-bnf-style-grammar)

### C

- **Cache control (prompt caching)** -- [unified-llm-spec Section 2.10](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#210-prompt-caching-critical-for-cost)
- **Checkpoint and resume** -- [attractor-spec Section 5.3](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#53-checkpoint)
- **Class attribute (CSS-like)** -- [attractor-spec Section 2.12](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#212-class-attribute)
- **CodergenBackend interface** -- [attractor-spec Section 4.5](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#45-codergen-handler-llm-task)
- **Compaction (out of scope)** -- [coding-agent-loop-spec Section 8](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#8-out-of-scope-nice-to-haves)
- **Condition expression language** -- [attractor-spec Section 10](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#10-condition-expression-language)
- **Conditional handler** -- [attractor-spec Section 4.7](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#47-conditional-handler)
- **Context (key-value store)** -- [attractor-spec Section 5.1](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#51-context)
- **Context fidelity modes** -- [attractor-spec Section 5.4](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#54-context-fidelity)
- **Context window awareness** -- [coding-agent-loop-spec Section 5.5](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#55-context-window-awareness)
- **ContentPart (tagged union)** -- [unified-llm-spec Section 3.3](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#33-contentpart-tagged-union)
- **Cross-provider parity matrix** -- [coding-agent-loop-spec Section 9.12](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#912-cross-provider-parity-matrix)

### D

- **Definition of Done (pipeline)** -- [attractor-spec Section 11](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#11-definition-of-done)
- **Definition of Done (agent loop)** -- [coding-agent-loop-spec Section 9](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#9-definition-of-done)
- **Definition of Done (LLM SDK)** -- [unified-llm-spec Section 8](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#8-definition-of-done)
- **Design decision rationale** -- [coding-agent-loop-spec Appendix C](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#appendix-c-design-decision-rationale), [unified-llm-spec Appendix C](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#appendix-c-design-decision-rationale)
- **Docker execution environment** -- [coding-agent-loop-spec Section 4.3](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#43-alternative-environments-extension-points)
- **DOT DSL schema** -- [attractor-spec Section 2](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#2-dot-dsl-schema)

### E

- **Edge attributes** -- [attractor-spec Section 2.7](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#27-edge-attributes)
- **Edge selection algorithm** -- [attractor-spec Section 3.3](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#33-edge-selection-algorithm)
- **Environment context block** -- [coding-agent-loop-spec Section 6.3](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#63-environment-context-block)
- **Environment variable filtering** -- [coding-agent-loop-spec Section 4.2](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#42-localexecutionenvironment-required-implementation)
- **Error handling (agent loop)** -- [coding-agent-loop-spec Appendix B](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#appendix-b-error-handling)
- **Error taxonomy (SDK)** -- [unified-llm-spec Section 6.1](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#61-error-taxonomy)
- **Event system (agent)** -- [coding-agent-loop-spec Section 2.9](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#29-event-system)
- **Event system (pipeline)** -- [attractor-spec Section 9.6](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#96-observability-and-events)
- **ExecutionEnvironment interface** -- [coding-agent-loop-spec Section 4.1](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#41-the-execution-environment-abstraction)

### F

- **Failure routing** -- [attractor-spec Section 3.7](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#37-failure-routing)
- **Fan-in handler** -- [attractor-spec Section 4.9](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#49-fan-in-handler)
- **Fidelity modes** -- see Context fidelity modes
- **FinishReason mapping** -- [unified-llm-spec Section 3.8](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#38-finishreason)
- **Follow-up queue** -- [coding-agent-loop-spec Section 2.6](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#26-steering)
- **Four-layer architecture (SDK)** -- [unified-llm-spec Section 2.1](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#21-four-layer-architecture)

### G

- **generate() function** -- [unified-llm-spec Section 4.3](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#43-high-level-generate)
- **generate_object() (structured output)** -- [unified-llm-spec Section 4.5](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#45-high-level-generate_object)
- **Gemini profile** -- [coding-agent-loop-spec Section 3.6](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#36-gemini-profile-gemini-cli-aligned)
- **Goal gate enforcement** -- [attractor-spec Section 3.4](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#34-goal-gate-enforcement)
- **Graceful shutdown** -- [coding-agent-loop-spec Appendix B](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#appendix-b-error-handling)

### H

- **Handler interface** -- [attractor-spec Section 4.1](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#41-handler-interface)
- **Handler registry** -- [attractor-spec Section 4.2](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#42-handler-registry)
- **Head/tail truncation** -- [coding-agent-loop-spec Section 5.1](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#51-tool-output-truncation)
- **HTTP server mode** -- [attractor-spec Section 9.5](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#95-http-server-mode)
- **Human-in-the-loop (Interviewer)** -- [attractor-spec Section 6](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#6-human-in-the-loop-interviewer-pattern)

### I-J

- **Image upload handling** -- [unified-llm-spec Section 3.5](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#35-content-data-structures)
- **Interviewer implementations** -- [attractor-spec Section 6.4](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#64-built-in-interviewer-implementations)
- **Join policies (parallel)** -- [attractor-spec Section 4.8](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#48-parallel-handler)

### K-L

- **Kubernetes execution** -- [coding-agent-loop-spec Section 4.3](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#43-alternative-environments-extension-points)
- **Lint rules (pipeline validation)** -- [attractor-spec Section 7.2](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#72-built-in-lint-rules)
- **Loop detection** -- [coding-agent-loop-spec Section 2.10](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#210-loop-detection)
- **loop_restart (edge attribute)** -- [attractor-spec Section 2.7](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#27-edge-attributes)

### M

- **Manager loop handler** -- [attractor-spec Section 4.11](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#411-manager-loop-handler)
- **Message translation (per-provider)** -- [unified-llm-spec Section 7.3](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#73message-translation-details)
- **Middleware pattern** -- [unified-llm-spec Section 2.3](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#23-middleware--interceptor-pattern)
- **Model catalog** -- [unified-llm-spec Section 2.9](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#29-model-catalog)
- **Model stylesheet** -- [attractor-spec Section 8](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#8-model-stylesheet)

### N-O

- **Native API usage (critical)** -- [unified-llm-spec Section 2.7](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#27-native-api-usage-critical)
- **Node attributes** -- [attractor-spec Section 2.6](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#26-node-attributes)
- **Observability events** -- [attractor-spec Section 9.6](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#96-observability-and-events)
- **OpenAI-compatible endpoints** -- [unified-llm-spec Section 7.10](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#710openai-compatible-endpoints)
- **OpenAI profile (codex-rs-aligned)** -- [coding-agent-loop-spec Section 3.4](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#34-openai-profile-codex-rs-aligned)
- **Outcome model** -- [attractor-spec Section 5.2](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#52-outcome)

### P

- **Parallel handler** -- [attractor-spec Section 4.8](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#48-parallel-handler)
- **Parallel tool execution** -- [unified-llm-spec Section 5.7](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#57-parallel-tool-execution)
- **Pipeline examples (DOT)** -- [attractor-spec Section 2.13](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#213-minimal-examples)
- **Project document discovery** -- [coding-agent-loop-spec Section 6.5](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#65-project-document-discovery)
- **Prompt caching** -- [unified-llm-spec Section 2.10](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#210-prompt-caching-critical-for-cost)
- **Provider adapter interface** -- [unified-llm-spec Section 2.4](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#24-provider-adapter-interface)
- **Provider alignment principle** -- [coding-agent-loop-spec Section 3.1](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#31-the-provider-alignment-principle)
- **Provider options (escape hatch)** -- [unified-llm-spec Section 3.6](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#36-request)
- **Provider quirks reference** -- [unified-llm-spec Section 7.8](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#78-provider-quirks-reference)
- **ProviderProfile interface** -- [coding-agent-loop-spec Section 3.2](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#32-providerprofile-interface)

### Q-R

- **Rate limit handling** -- [unified-llm-spec Section 6.7](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#67-rate-limit-handling)
- **Reasoning effort** -- [coding-agent-loop-spec Section 2.7](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#27-reasoning-effort)
- **Reasoning token handling** -- [unified-llm-spec Section 3.9](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#39-usage)
- **Reference projects** -- [coding-agent-loop-spec Section 1.4](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#14-reference-projects), [unified-llm-spec Section 1.3](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#13-reference-open-source-projects)
- **Retry policies (named presets)** -- [attractor-spec Section 3.6](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#36-retry-policy)
- **Retry-After header** -- [unified-llm-spec Section 6.6](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#66-retry-policy)
- **Run directory structure** -- [attractor-spec Section 5.6](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#56-run-directory-structure)

### S

- **Session lifecycle** -- [coding-agent-loop-spec Section 2.3](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#23-session-lifecycle)
- **Session configuration** -- [coding-agent-loop-spec Section 2.2](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#22-session-configuration)
- **Shape-to-handler mapping** -- [attractor-spec Section 2.8](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#28-shape-to-handler-type-mapping)
- **Steering (mid-task injection)** -- [coding-agent-loop-spec Section 2.6](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#26-steering)
- **Stop conditions** -- [coding-agent-loop-spec Section 2.8](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#28-stop-conditions)
- **StreamEvent types** -- [unified-llm-spec Section 3.14](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#314-streameventtype)
- **Streaming translation** -- [unified-llm-spec Section 7.7](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#77-streaming-translation)
- **Structured output (generate_object)** -- [unified-llm-spec Section 4.5](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#45-high-level-generate_object)
- **Subagents** -- [coding-agent-loop-spec Section 7](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#7-subagents)
- **Subgraphs (scoping)** -- [attractor-spec Section 2.10](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#210-subgraphs)
- **System prompts (layered construction)** -- [coding-agent-loop-spec Section 6.1](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#61-layered-system-prompt-construction)

### T

- **Thinking blocks (Anthropic)** -- [unified-llm-spec Section 3.5](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#35-content-data-structures)
- **Thread ID (session reuse)** -- [attractor-spec Section 5.4](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#54-context-fidelity)
- **Timeout handling (commands)** -- [coding-agent-loop-spec Section 5.4](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#54-default-command-timeouts)
- **Tool call validation and repair** -- [unified-llm-spec Section 5.8](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#58-tool-call-validation-and-repair)
- **Tool execution pipeline** -- [coding-agent-loop-spec Section 3.8](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#38-tool-registry)
- **Tool handler (external commands)** -- [attractor-spec Section 4.10](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#410-tool-handler)
- **Tool hooks (pre/post)** -- [attractor-spec Section 9.7](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#97-tool-call-hooks)
- **Tool output truncation** -- [coding-agent-loop-spec Section 5](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#5-tool-output-and-context-management)
- **Tool registry** -- [coding-agent-loop-spec Section 3.8](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#38-tool-registry)
- **Transforms (AST)** -- [attractor-spec Section 9.1](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#91-ast-transforms)
- **Truncation (character-based first)** -- [coding-agent-loop-spec Section 5.3](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#53-truncation-order-important)
- **Turn types** -- [coding-agent-loop-spec Section 2.4](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#24-turn-types)

### U-V

- **Unified LLM SDK overview** -- [unified-llm-spec Section 1](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#1-overview-and-goals)
- **Usage tracking** -- [unified-llm-spec Section 3.9](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md#39-usage)
- **Validation (pipeline lint rules)** -- [attractor-spec Section 7](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#7-validation-and-linting)
- **Variable expansion ($goal)** -- [attractor-spec Section 9.2](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#92-built-in-transforms)

### W-Z

- **WASM execution environment** -- [coding-agent-loop-spec Section 4.3](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md#43-alternative-environments-extension-points)
- **Wait for human handler** -- [attractor-spec Section 4.6](https://github.com/strongdm/attractor/blob/main/attractor-spec.md#46-wait-for-human-handler)
