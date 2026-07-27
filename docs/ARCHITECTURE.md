# Forge Architecture Overview

Forge is a batch-first LLM task orchestrator. It decomposes work into independent units, executes each as a structured document-completion request, validates the results with deterministic checks, and decides what to do next. Temporal provides the workflow engine; the LLM never owns the control loop.

This document explains how the system works end-to-end: the universal workflow step, how prompts are assembled, how the LLM "calls tools" without an agentic loop, and how fan-out parallelism works.

---

## Table of Contents

1. [Core Idea: Document Completion, Not Chat](#core-idea-document-completion-not-chat)
2. [The Universal Workflow Step](#the-universal-workflow-step)
3. [Execution Modes](#execution-modes)
4. [Context Assembly: Building the Document](#context-assembly-building-the-document)
5. [Structured Output: How the LLM Responds](#structured-output-how-the-llm-responds)
6. [Exploration: How the LLM "Calls Tools"](#exploration-how-the-llm-calls-tools)
7. [Error-Aware Retries](#error-aware-retries)
8. [Fan-Out / Gather](#fan-out--gather)
9. [Model Routing](#model-routing)
10. [Batch vs. Sync Execution](#batch-vs-sync-execution)
11. [Key Data Models](#key-data-models)
12. [Module Map](#module-map)

---

## Core Idea: Document Completion, Not Chat

Most LLM orchestrators run the model in a chat loop where the LLM decides when to call tools, when to stop, and what to do next. Forge inverts this. Every LLM call is a **single document completion**: the orchestrator assembles a complete prompt (system + user), sends it, receives a structured response, and evaluates the outcome. The orchestrator—not the LLM—decides whether to retry, what context to gather next, or when to fan out to parallel sub-tasks.

This design has a concrete benefit: every LLM call is stateless and self-contained, which makes it compatible with **batch APIs** (where requests are queued and results arrive later) and **local LLM tools** (where there is no streaming chat session). It also makes each call independently testable and observable.

```mermaid
flowchart LR
    O[Orchestrator<br/>Temporal] -->|1. Assemble prompt| P[Prompt]
    P -->|2. Send| LLM[LLM API]
    LLM -->|3. Structured response| O
    O -->|4. Validate & decide| N{Next action}
    N -->|success| Done[Commit]
    N -->|retry| O
    N -->|fan-out| Children[Child workflows]
```

---

## The Universal Workflow Step

Every operation in Forge—code generation, planning, exploration, conflict resolution—follows the same five-phase pattern:

```text
Construct → Send → Receive → Serialize → Transition
```

The first four phases are Temporal **activities** (each a unit of work with its own timeout and retry policy); the fifth, Transition, is a pure decision inlined into the **workflow** (`forge.step_logic.determine_transition`, T5.1/D95) — deterministic work is not dressed up as external work. The workflow orchestrates them in sequence.

```mermaid
flowchart TD
    A["① Construct<br/><i>assemble_context</i>"]
    B["② Send<br/><i>call_llm</i>"]
    C["③ Receive + Serialize<br/><i>write_output</i>"]
    D["④ Validate<br/><i>validate_output</i>"]
    E["⑤ Transition<br/><i>determine_transition (inlined, pure)</i>"]

    A --> B --> C --> D --> E

    E -->|SUCCESS| F[Commit changes]
    E -->|FAILURE_RETRYABLE| A
    E -->|FAILURE_TERMINAL| G[Report failure]
```

### What happens at each phase

| # | Phase | Activity | What it does |
| --- | ------- | ---------- | ------------- |
| 1 | **Construct** | `assemble_context` | Builds the system prompt and user prompt. Discovers relevant files via import graph analysis and PageRank ranking. Reads target file contents from the worktree. Injects project instructions, playbooks, repo maps, and—on retries—previous error output with AST-derived context. |
| 2 | **Send** | `call_llm` | Packages the assembled prompt into an Anthropic API structured-outputs call (`sax_platform.llm.AnthropicLLM.complete`; `messages.parse` on the sync lane, `output_config.format` on the batch lane — D90/T3.5). Sends it. Records latency and token usage. |
| 3 | **Receive + Serialize** | `write_output` | Extracts the Pydantic-validated response returned by the structured-outputs call. Writes new files to the worktree. Applies search/replace edits to existing files using a four-level matching fallback chain (exact → whitespace-normalized → indentation-normalized → fuzzy). |
| 4 | **Validate** | `validate_output` | Runs deterministic checks: `ruff` lint, `ruff` format, and optionally a test suite. Produces a list of pass/fail results with error details. |
| 5 | **Transition** | `determine_transition` (pure, inlined — not an activity) | Maps validation results plus the attempt number to a signal: `SUCCESS` (all checks pass), `FAILURE_RETRYABLE` (checks failed but attempts remain), or `FAILURE_TERMINAL` (no retries left or unrecoverable). Runs in-workflow with no round-trip (T5.1/D95). |

---

## Execution Modes

Forge supports three execution modes. All three use the same universal workflow step internally.

```mermaid
flowchart TD
    Input[Task Input] --> Mode{plan=?}

    Mode -->|"plan=False"| Single["<b>Single-step</b><br/>One LLM call per attempt"]
    Mode -->|"plan=True<br/>no sub-tasks"| Planned["<b>Planned multi-step</b><br/>Planner → sequential steps"]
    Mode -->|"plan=True<br/>steps have sub_tasks"| FanOut["<b>Fan-out / gather</b><br/>Parallel child workflows"]

    Single --> UWS1["Universal Workflow Step"]
    Planned --> UWS2["Universal Workflow Step<br/>(per step)"]
    FanOut --> UWS3["Universal Workflow Step<br/>(per sub-task, in parallel)"]
```

### Single-step (`plan=False`)

The simplest mode. One worktree is created per attempt. The universal workflow step runs once. On success, changes are committed. On retryable failure, the worktree is destroyed and a fresh one is created for the next attempt, with the previous error output injected into the prompt.

### Planned multi-step (`plan=True`)

A planning LLM (using a higher-capability model) decomposes the task into an ordered list of steps. One shared worktree is created at the start. Each step executes the universal workflow step and commits on success. On retryable failure, uncommitted changes are reset (prior committed steps are preserved) and the step is retried with error context.

A **sanity check** can run periodically between steps: a reasoning-tier LLM reviews progress and remaining steps, and can issue `CONTINUE`, `REVISE` (rewrite remaining steps), or `ABORT`. By design the driver skips the check after a fan-out step and after the final step (verified T5.3, `_drive_plan`).

### Fan-out / gather (`plan=True`, steps with `sub_tasks`)

When a planned step contains sub-tasks, each sub-task is dispatched as a **child workflow** running in its own git worktree. All children execute in parallel. After all children complete, their output files are merged into the parent worktree. If multiple sub-tasks modify the same file, a conflict resolution LLM resolves the differences. The merged result is validated and committed.

Sub-tasks can themselves contain nested sub-tasks, bounded by a configurable `max_fan_out_depth`. Since T5.3 both levels run the one gather in `forge.blocks.gather` — the top-level step borrows the plan's worktree and commits, a nested node creates its own and removes it on every exit without committing (D16) — and child failure is isolated: a child that *raises* (its execution timeout, an escaping activity failure) becomes a failed sub-task result while its siblings run to completion, so the parent still returns a `TaskResult` and writes its run record. Children are started with an explicit `ParentClosePolicy.TERMINATE`, which now fires only if the parent itself dies.

---

## Context Assembly: Building the Document

The prompt sent to the LLM is not a chat transcript—it is a structured document assembled from multiple sources. The `assemble_context` activity constructs a system prompt and a user prompt.

### System prompt structure

The system prompt is assembled in a specific order optimized for **prompt caching** (stable content first, volatile content last):

```mermaid
flowchart TD
    subgraph "System Prompt (top to bottom)"
        direction TB
        R["① Role statement<br/><i>'You are a code generation assistant.'</i>"]
        O["② Output requirements<br/><i>Describes the files/edits response format</i>"]
        PI["③ Project instructions<br/><i>From CLAUDE.md in the repo root</i>"]
        RM["④ Repository structure<br/><i>PageRank-ranked file tree</i>"]
        PB["⑤ Playbooks<br/><i>Lessons from past tasks</i>"]
        T["⑥ Task description + target files"]
        TF["⑦ Target file contents<br/><i>Current content from worktree</i>"]
        DD["⑧ Direct dependencies<br/><i>Full content (if --include-deps)</i>"]
        IC["⑨ Interface context<br/><i>Signatures of transitive imports</i>"]
        EX["⑩ Exploration results<br/><i>Context gathered by the exploration loop</i>"]
        ER["⑪ Previous errors<br/><i>Only on retry attempts</i>"]
    end

    R --- O --- PI --- RM --- PB --- T --- TF --- DD --- IC --- EX --- ER
```

Sections ①–⑤ are **stable** across calls for the same repository and task, allowing the Anthropic API's prompt caching to avoid re-processing them on retries. Section ⑪ only appears on retry attempts.

### How context discovery works

When auto-discovery is enabled (the default), Forge builds context through these steps:

1. **Import graph analysis** — Uses `grimp` to build the project's import dependency graph.
2. **PageRank ranking** — Runs `networkx.pagerank` on the import graph to rank files by structural importance.
3. **Symbol extraction** — Uses Python's `ast` module to extract function signatures, class definitions, and constants from each file.
4. **Token budget packing** — A knapsack-style algorithm packs files into the token budget by priority: target files first (full content), then direct dependencies, then interface signatures of transitive imports, and finally the repo map.

By default, only target file contents and the repo map are included upfront. Dependency contents are omitted to keep prompts lean—the LLM can pull them on demand through the exploration loop.

### User prompt

The user prompt is short and domain-specific. For code generation:

> *"Generate the requested code changes. Use `edits` for existing files and `files` for new ones."*

For planned steps, it includes the step ID and description. The user prompt is intentionally minimal—all the substantive context is in the system prompt.

---

## Structured Output: How the LLM Responds

Forge gets structured output through Anthropic's native **structured outputs** (D90/T3.5). Each call passes the target Pydantic model's JSON schema as `output_config.format`, and the provider guarantees the response conforms. The client is `sax_platform.llm.AnthropicLLM` — `complete[T]` (`messages.parse`) on the sync lane and the batch request builder (`output_config.format`) on the batch lane — returning a validated model or a typed refusal/truncation/mismatch outcome. Forced tool use as the structured-output mechanism (`tool_choice={type: "tool", ...}`) is retired.

```mermaid
sequenceDiagram
    participant F as Forge
    participant A as Anthropic API

    F->>A: messages.parse(<br/>  output_config={format: {schema: ...}}<br/>)
    A->>F: message conforming to the schema
    F->>F: Pydantic model returned (or typed refusal / truncation / mismatch)
```

The `LLMResponse` schema has three fields:

- **`explanation`** — Free-text reasoning about what was done.
- **`files`** — New files to create (path + complete content).
- **`edits`** — Changes to existing files (path + list of search/replace pairs).

A file path may appear in `files` or `edits`, but not both.

### Edit application

Edits use a search/replace model: the LLM provides an exact string to find and a replacement string. To tolerate minor formatting discrepancies from the LLM, the edit engine uses a four-level fallback chain:

1. **Exact match** — The search string appears exactly once in the file.
2. **Whitespace-normalized** — Trailing whitespace is stripped from each line before matching.
3. **Indentation-normalized** — The search string is dedented and re-indented at each indentation level found in the file.
4. **Fuzzy match** — `difflib.SequenceMatcher` finds the best match above a 60% similarity threshold, with a uniqueness check (best match must be at least 5% ahead of the second-best).

At every level, ambiguity (multiple matches) is an error. Edits are applied sequentially—each edit sees the result of the previous one.

---

## Exploration: How the LLM "Calls Tools"

In a traditional agentic loop, the LLM calls tools mid-conversation and the results are appended to the chat history. Forge achieves the same effect—letting the LLM request additional context—while keeping every LLM call as a standalone document completion.

The exploration loop runs **before** the generation call:

```mermaid
flowchart TD
    Start[Start exploration] --> ECall["Call exploration LLM<br/><i>What context do you need?</i>"]
    ECall --> Check{Requests<br/>empty?}
    Check -->|Yes| Gen["Proceed to generation"]
    Check -->|No| Fulfill["Fulfill context requests<br/><i>Dispatch to providers</i>"]
    Fulfill --> Acc["Accumulate results"]
    Acc --> Round{Round limit<br/>reached?}
    Round -->|No| ECall
    Round -->|Yes| Gen
```

### How it works

1. A lightweight exploration LLM (classification-tier) is shown the task description, target files, and a list of **available providers** with their parameters.
2. It returns a list of context requests (e.g., `{provider: "read_file", params: {path: "src/forge/models.py"}}`).
3. The orchestrator dispatches each request to the matching provider handler, which runs the operation and returns text.
4. The results are accumulated and shown to the exploration LLM in the next round.
5. When the LLM returns an empty request list, or the round limit is reached, exploration ends.
6. All accumulated context is appended to the system prompt as an "Exploration Results" section before the generation call.

### Available providers

| Provider | Description |
| ---------- | ------------- |
| `read_file` | Read full contents of a file from the worktree |
| `search_code` | Regex pattern search across files (up to 100 matches) |
| `symbol_list` | Extract public API (functions, classes, constants) from a module |
| `import_graph` | Show what a module imports and what imports it |
| `run_tests` | Execute `pytest` and return results (30s timeout) |
| `lint_check` | Run `ruff` linter on specified files |
| `git_log` | Show recent commit history |
| `git_diff` | Diff against a base branch |
| `repo_map` | Generate PageRank-ranked project structure |
| `discover_context` | Run full auto-discovery for target files |
| `past_runs` | Show recent workflow run results from the observability store |
| `playbooks` | Retrieve playbook entries by tag |

This is not a traditional tool-calling loop. The LLM is **not** making tool calls within a single conversation turn. Each exploration round is a separate, complete document-completion request. The orchestrator manages the iteration.

---

## Error-Aware Retries

When validation fails and the step is retried, the orchestrator doesn't just retry blind. It builds an error section that includes:

1. **Structured error output** — The `ruff` lint or format errors, or test failure output.
2. **AST-derived context** — For lint/format errors, Forge parses the error's file path and line number, then uses Python's `ast` module to find the enclosing function or class. A code snippet showing the scope header and the error line (marked with `# <-- ERROR`) is included so the LLM has immediate visual context around each failure.

This error section is appended to the end of the system prompt (the most volatile position, preserving cache efficiency for all preceding content).

```mermaid
flowchart LR
    V[Validation fails] --> Parse["Parse error<br/>locations"]
    Parse --> AST["AST: find enclosing<br/>function/class"]
    AST --> Build["Build error section<br/>with code snippets"]
    Build --> Inject["Inject into retry<br/>prompt"]
    Inject --> LLM["LLM sees exactly<br/>what went wrong"]
```

---

## Fan-Out / Gather

Fan-out allows a single planned step to split into parallel sub-tasks, each running in its own git worktree. This is how Forge achieves parallelism for tasks where multiple independent files or components can be worked on simultaneously.

```mermaid
flowchart TD
    Plan["Plan step with sub_tasks"] --> Spawn

    subgraph "Parallel execution"
        direction LR
        Spawn["Start child workflows"] --> C1["Sub-task A<br/><i>own worktree</i>"]
        Spawn --> C2["Sub-task B<br/><i>own worktree</i>"]
        Spawn --> C3["Sub-task C<br/><i>own worktree</i>"]
    end

    C1 --> Gather
    C2 --> Gather
    C3 --> Gather

    Gather["Gather results"] --> Conflict{File<br/>conflicts?}
    Conflict -->|No| Merge["Merge files into<br/>parent worktree"]
    Conflict -->|Yes| Resolve["LLM conflict<br/>resolution"]
    Resolve --> Merge
    Merge --> Validate["Validate merged output"]
    Validate --> Commit["Commit"]
```

### The mechanics

1. **Dispatch** — The parent workflow starts one Temporal child workflow per sub-task. Each child gets its own worktree branched from the parent's branch. All children start concurrently.

2. **Execute** — Each child runs the universal workflow step independently (assemble context → call LLM → write output → validate → transition), with its own retry budget. Children do not commit to git—they just produce output files.

3. **Gather** — The parent awaits all children. If any child fails, the step fails.

4. **Conflict detection** — The parent checks whether multiple sub-tasks produced different content for the same file path. Non-conflicting files are collected directly.

5. **Conflict resolution** — If conflicts exist and resolution is enabled, a reasoning-tier LLM receives all conflicting versions alongside the task and step descriptions, and produces a merged version of each file.

6. **Merge and validate** — All files (non-conflicting + resolved) are written to the parent worktree. Validation runs on the merged output. If validation passes, the step is committed.

### Nested fan-out

Sub-tasks can themselves contain `sub_tasks`, creating recursive fan-out bounded by `max_fan_out_depth` (default 1 = flat fan-out only). The child workflow checks whether it should execute as a leaf (single-step) or recurse (nested fan-out) based on its current depth vs. the maximum.

> **Temporal reference:** child-workflow fan-out follows the SDK [child-workflow sample](https://github.com/temporalio/samples-python/blob/4d453de6adce21be822a02e2dc553138b684945d/hello/hello_child_workflow.py) ([concept docs](https://docs.temporal.io/develop/python/workflows/child-workflows)).

---

## Model Routing

Forge routes different LLM calls to different model tiers based on the capability required:

| Capability Tier | Default Model | Used For |
| ---------------- | --------------- | ---------- |
| **Reasoning** | `anthropic:claude-opus-4-8` | Planning, sanity checks, conflict resolution, eval-as-judge |
| **Generation** | `anthropic:claude-sonnet-5` | Code/content generation |
| **Summarization** | `anthropic:claude-sonnet-5` | Knowledge extraction |
| **Classification** | `anthropic:claude-haiku-4-5` | Exploration |

The tier registry (`CapabilityTier`, `ModelConfig`, `resolve_model`) and the `ThinkingPolicy` are single-sourced in `sax_platform.llm.tiers` and re-exported by `forge.models` (D94, T3.2). Transition evaluation is **not** in this table: it is a deterministic pure function (`step_logic.determine_transition`, inlined into the workflows since T5.1/D95 — there is no transition activity) that makes no LLM call.

The plan can override the tier for individual steps via `capability_tier`, so a particularly complex step can use the reasoning tier while simpler steps use the generation tier.

---

## Batch vs. Sync Execution

Every LLM call in Forge can run in two modes:

- **Batch mode** (default) — The activity submits the request to the Anthropic Batch API; the workflow then polls `batch_status` on a `workflow.sleep` loop until the batch ends and fetches its own result line via `fetch_batch_result` (per-workflow timer-loop transport, D88/T4.1 — no shared poller, no signal). (Forge's batch transport is anthropic-only, T4.2; Mistral batch lives entirely in the `apps/ocr` app.)
- **Sync mode** (opt-in via `--sync`) — The activity calls the provider's messages API directly and waits for the response.

`ForgeTaskInput.sync_mode` defaults to `False`, so batch is the default path. The prompt construction is identical in both modes. Since T5.3 the lane fork exists once, in `forge.blocks.dispatch`: the five arms (generation, planner, sanity check, conflict resolution, exploration) differ only by the row they occupy in the pure `ARMS` table — sync activity, its timeout, batch output type, and batch `max_tokens` — and each arm's call, either lane, writes one interaction record. This is why every LLM call must be a self-contained document completion—batch APIs don't support multi-turn conversations.

```mermaid
flowchart TD
    Call["LLM call needed"] --> Mode{sync_mode?}

    Mode -->|Sync| Direct["client.messages.create()"]
    Direct --> Result["Parse response"]

    Mode -->|Batch| Submit["Submit to Batch API"]
    Submit --> Poll["Workflow polls batch_status (sleep loop)"]
    Poll --> Fetch["fetch_batch_result"]
    Fetch --> Parse["Parse response"]
```

---

## Key Data Models

These are the core Pydantic models that flow through the system:

| Model | Purpose |
| ------- | --------- |
| `TaskDefinition` | Input: task ID, description, target files, domain, validation config |
| `ForgeTaskInput` | Workflow input: task + execution settings (plan, retries, model routing) |
| `AssembledContext` | The fully constructed prompt: system prompt + user prompt + metadata |
| `LLMResponse` | LLM output: explanation + files (new) + edits (modifications) |
| `LLMCallResult` | LLM output + token usage + latency metrics |
| `Plan` | Planner output: ordered list of `PlanStep` objects |
| `PlanStep` | A single step: ID, description, target files, optional sub-tasks |
| `SubTask` | A parallelizable unit within a fan-out step |
| `ValidationResult` | A single check result: check name, passed/failed, details |
| `TransitionSignal` | The outcome: `SUCCESS`, `FAILURE_RETRYABLE`, or `FAILURE_TERMINAL` |
| `TaskResult` | Final workflow output: status, output files, validation results, stats |
| `ExplorationResponse` | Exploration LLM output: list of context provider requests |
| `ContextResult` | Result from a single provider: provider name, content, token estimate |

---

## Module Map

```text
src/forge/
├── workflows/                 # Temporal workflow drivers (T5.4)
│   ├── task.py                # ForgeTaskWorkflow — single-step and planned runs
│   └── subtask.py             # ForgeSubTaskWorkflow — one fan-out node
├── blocks/                    # The shapes the drivers compose (T5.2-T5.4)
│   ├── step.py                # The universal step pipeline (MODE_POLICIES)
│   ├── gather.py              # The fan-out gather (GATHER_POLICIES)
│   ├── dispatch.py            # Typed LLM dispatch, five arms (ARMS)
│   ├── exploration.py         # LLM-guided context exploration loop
│   ├── transport.py           # Batch submit + timer-loop wait (D88)
│   ├── worktree.py            # Worktree removal + post-failure cleanup
│   └── host.py                # DispatchHostBase / RunSettings (per-run state)
├── presets.py                 # Activity timeouts, retry policies, token caps
├── step_logic.py              # Pure step decisions (zero temporalio imports)
├── ingestion_workflow.py      # Transcript ingestion → pbook (cross-queue)
├── manual_playbook_workflow.py # Manual playbook add with LLM review
├── export_playbook_workflow.py # Playbook export
├── models.py                  # All Pydantic data models (tiers/ThinkingPolicy re-exported from sax_platform)
├── providers.py               # Context provider registry (12 providers)
├── domains.py                 # Domain configs (code_generation, research, etc.)
├── cli.py                     # CLI entry point (forge run, forge worker, ...)
├── worker.py                  # Temporal worker process
├── git.py                     # Git operations and worktree management
├── store.py                   # SQLite/Postgres observability store (+ playbooks)
├── persist_models.py          # Survivable-write request models
├── temporal_client.py         # Temporal connect + (dormant) mTLS config
├── path_safety.py             # Path-traversal guards
├── subprocess_env.py          # Subprocess environment construction
├── subprocess_result.py       # Subprocess result models
├── tracing.py                 # OpenTelemetry instrumentation
├── logging_config.py          # Log file paths and rotation
├── message_log.py             # API message logging utilities
│
├── activities/
│   ├── context.py             # Prompt assembly (system + user prompts)
│   ├── llm.py                 # LLM call execution (structured outputs via sax_platform.llm)
│   ├── output.py              # File writing + edit application
│   ├── validate.py            # Deterministic validation (ruff, tests)
│   ├── planner.py             # Planning LLM call
│   ├── exploration.py         # Exploration loop LLM calls
│   ├── extraction.py          # Knowledge extraction + playbook generation
│   ├── sanity_check.py        # Mid-plan sanity checks
│   ├── conflict_resolution.py # Fan-out file conflict resolution
│   ├── git_activities.py      # Worktree create/remove/reset/commit
│   ├── ingestion.py           # Transcript ingestion activities
│   ├── persist.py             # Survivable idempotent store writes
│   ├── playbook_export.py     # Playbook export activity
│   ├── playbook_review.py     # Manual-playbook LLM review activity
│   ├── batch_submit.py        # Anthropic Batch API submission (anthropic-only, T4.2)
│   ├── batch_parse.py         # Batch response parsing
│   └── batch_fetch.py         # Batch status polling + result fetch (timer loop; no signal)
│
├── code_intel/
│   ├── graph.py               # Import graph analysis (grimp + networkx)
│   ├── parser.py              # Symbol extraction (ast-based)
│   ├── budget.py              # Token budget packing
│   └── repo_map.py            # Repository structure mapping
│
├── eval/
│   ├── runner.py              # Evaluation harness
│   ├── deterministic.py       # Deterministic plan checks
│   ├── judge.py               # LLM-as-judge scoring (reasoning tier)
│   ├── models.py              # Evaluation data models
│   └── corpus.py              # Test corpus management
│
└── alembic/                   # Store migrations (env.py + versions/001–003)
```

The LLM client — `AnthropicLLM` (structured outputs, both lanes), the tier registry, `ThinkingPolicy`, and the Mistral OCR capability — lives in the `sax-platform` workspace member (`libs/sax-platform`), not under `src/forge/` (the former `sax-llm` provider layer was deleted at T3.5). The OCR pipeline is no longer in this tree either: it is the separate `apps/ocr` consumer app, which owns its full Mistral batch lifecycle via `sax_platform.ocr.MistralOcr` and shares only the cross-queue batch ledger with forge (T4.2 deleted the batch SPI).

> The map above covers the core loop. Major shipped subsystems — batch execution (the default), transcript ingestion, the knowledge/playbook lifecycle, planner evaluation, store externalization (Postgres + S3 with survivable writes), the OCR consumer app (`apps/ocr`, its own Mistral batch lifecycle), and mTLS remote access (infrastructure since removed, D99) — are summarized in [Subsystems Beyond the Core Loop](#subsystems-beyond-the-core-loop).

---

## End-to-End Example: Planned Task with Exploration

Here is the complete flow for a planned code generation task:

```mermaid
sequenceDiagram
    participant CLI as CLI
    participant W as Workflow
    participant Ctx as Context Assembly
    participant Exp as Exploration Loop
    participant LLM as LLM API
    participant Out as Write Output
    participant Val as Validate
    participant Git as Git

    CLI->>W: ForgeTaskInput (plan=True)
    W->>Git: Create worktree

    Note over W: Planning phase
    W->>Ctx: Assemble planner context
    W->>Exp: Exploration loop (for planner)
    loop Until LLM returns empty requests
        Exp->>LLM: What context do you need?
        LLM->>Exp: [read_file, search_code, ...]
        Exp->>Exp: Fulfill requests via providers
    end
    W->>LLM: Call planner (reasoning tier)
    LLM->>W: Plan with ordered steps

    Note over W: Execution phase (per step)
    loop For each step in plan
        W->>Ctx: Assemble step context
        Ctx->>W: System prompt + user prompt
        W->>LLM: Call generation LLM
        LLM->>W: LLMResponse (edits + files)
        W->>Out: Apply edits, write files
        W->>Val: Run ruff lint + format
        Val->>W: ValidationResult[]

        alt All checks pass
            W->>Git: Commit step
        else Retryable failure
            W->>Git: Reset worktree
            Note over W: Retry with error context
        end
    end

    W->>CLI: TaskResult (success/failure)
```

---

## Subsystems Beyond the Core Loop

The universal workflow step is the spine, but several shipped subsystems run alongside it. Each is documented here at a pointer level; module paths are under `src/forge/` unless noted. Full status and known gaps: [OVERVIEW.md](OVERVIEW.md).

### Batch execution (default path)

All five LLM call sites (generation, planner, exploration, sanity check, conflict resolution) submit to the Anthropic Batch API by default (`sync_mode=False`). Each workflow then runs its own timer loop: it records the submission in `batch_jobs`, polls `batch_status` on a `workflow.sleep` interval (min 300s, D88) until the batch ends or the 25h ceiling passes, fetches its own result line via `fetch_batch_result`, and records the final outcome. Submit/status/fetch/parse: `activities/batch_submit.py`, `activities/batch_fetch.py`, `activities/batch_parse.py`; the wait loop lives in `blocks/transport.py`; lifecycle states: `models.py::BatchJobStatus`. Durable `workflow.sleep` timers keep all workflow state alive across the wait rather than terminate-and-restart (per-workflow timer-loop transport, D88/T4.1 — no shared poller, no signal).

### The OCR consumer app

Forge's batch transport is anthropic-only (T4.2): every forge batch is an Anthropic batch, and `batch_fetch` rejects any non-anthropic provider — forge itself runs zero `@workflow.signal`. OCR lives entirely in the separate `apps/ocr` consumer app — it submits its own Mistral batches through `sax_platform.ocr.MistralOcr` (`submit_ocr_batch`, a single `fetch_and_store_ocr_result`), extracts images, rewrites `ocr-image://` URIs, chunks PDFs, and stores blobs in S3. Completion detection is a stateless status tracker (T4.4/D101): a Temporal Schedule fires `OcrBatchTrackerWorkflow` every 120s, each run sweeps Mistral's batch **list** endpoint once, and it broadcasts `ocr_status_hint` signals to the waiting `OcrStoreWorkflow` state machines — the sanctioned hint pattern, where a hint advances a receiver's state but never carries a payload and each child still fetches its own result keyed by its own `batch_id`/`request_id`. `OcrSubmitWorkflow` awaits its store children inline (no ABANDON) and reassembles the document; if any chunk fails, the document fails once all chunks have settled — a failed chunk no longer strands the document for the old 26h gather timeout. The app imports `sax_platform` only (never `forge`); the sole cross-queue link is the shared batch ledger — ocr writes `batch_jobs` rows via `persist_block` activity calls on `forge-task-queue` and reads that ledger read-only, both from `sax_platform.contracts` (the former `forge_contracts` package, absorbed at T3.4).

### Transcript ingestion

`forge ingest` reads Claude Code JSONL sessions, analyzes them via the batch API, and hands extracted experiences to pbook's `ExtractionWorkflow` cross-queue on `pbook-task-queue` (`ingestion_workflow.py`, `activities/ingestion.py`). Guarded: if pbook is not installed, ingestion workflows are not registered.

### Knowledge / playbook lifecycle

Playbook entries are injected into future contexts (priority 5, D47). The scheduled re-extraction loop that mined completed runs was removed in T1.8; the extraction activities remain (`activities/extraction.py`) and are now driven by manual add with LLM review (`manual_playbook_workflow.py`, which reuses the extraction save activity). Export via `export_playbook_workflow.py`. Forge's `playbooks` table is separate from pbook's `entries`.

### Planner evaluation

`eval/` scores planner output with deterministic plan-structure checks (`eval/deterministic.py`) and LLM-as-judge (`eval/judge.py`), with baseline/candidate comparison (`eval/runner.py`). CLI: `forge eval-planner`. Not yet wired as a release gate — see OVERVIEW tech debt.

### Store externalization & survivable writes

The observability store runs on SQLite (dev/test) or Postgres (production) behind one SQLAlchemy interface (`store.py`, `alembic/`). Writes funnel through a single idempotent, retried `persist_to_store` activity (`activities/persist.py`, `persist_models.py`) that fails the workflow loudly on prolonged DB outage rather than silently dropping data (supersedes the original best-effort policy, D42).

### mTLS remote access (infrastructure removed, D99)

`temporal_client.py` (`build_tls_config`) builds the mutual-TLS config for connecting the worker and CLI to a remote Temporal server. The remote-access infrastructure (EC2 gateway, certs) was removed by D99 — Temporal is loopback-only on the deployment desktop — so this path is dormant; the code stays should remote access return.
