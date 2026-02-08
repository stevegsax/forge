# Forge: Design Document

## Overview

Forge is an LLM task orchestrator built around batch mode with document completion rather than iterative streaming. The core insight is that by investing heavily in upfront planning, we can identify parallelizable work units and submit them as independent batch requests. Each request is a single step in a state machine, not a turn in a conversation.

Forge is suitable for any task that benefits from structured decomposition, parallel execution, and deterministic validation: code generation, research, analysis, content production, data processing, and more. The architecture is task-agnostic -- the differentiation between use cases lives entirely in prompts, context, and validation criteria.

Git and worktrees serve as the general-purpose data store and isolation mechanism. Just as worktrees isolate parallel code branches, they equally isolate parallel research threads, analysis tracks, or any body of work that benefits from independent progress with controlled reconciliation.

## Principles

**Deterministic work should be deterministic.** Never ask the LLM to figure out something you can compute. Pre-calculate facts and include them in context. Use the LLM for reasoning and generation, not information gathering. If an answer won't change, compute it client-side.

**Context isolation is a feature.** Each task gets a tightly constrained definition of "done" and a customized context. Agents don't inherit conversation history. They receive exactly the information they need, assembled fresh for each request.

**Planning is the hard part.** Invest the most expensive models and highest token budgets in planning and conflict resolution. A good plan reduces conflicts, improves validation coverage, and produces better results. Everything downstream is bounded by plan quality.

**Halt when confused.** When the orchestrator encounters a situation it cannot classify, it stops and escalates to a human. The failure modes of continuing with a bad plan are worse than the cost of pausing.

**The LLM call is the universal primitive.** Every task -- code generation, review, conflict resolution, knowledge extraction, sanity checking -- is an instance of the same workflow step with different prompts and context.

## Architecture

### The Universal Workflow Step

Every operation in Forge follows the same pattern:

1. **Construct message.** Assemble the prompt and context for the LLM call.
2. **Send.** Submit the request to the selected model.
3. **Receive.** Get the response.
4. **Serialize.** Store the structured result.
5. **Transition.** Evaluate the result and determine the next state.

The differentiation between "types" of work lives entirely in the prompt and context, not in the workflow machinery.

### Temporal Orchestration

Temporal provides the workflow engine. The orchestrator owns the task DAG, assigns tasks to agents, and manages state transitions.

**Activity boundaries:**

- The LLM call is one activity (idempotent via caching).
- The transition evaluation is a separate activity.
- Context assembly is a separate activity.
- Deterministic validation (lint, type check, tests) is a separate activity.

**Fan-out/gather** uses Temporal child workflows. A parent workflow creates child workflows for each sub-task, awaits their completion signals, and processes the gathered results. This is hub-and-spoke by construction: the parent is always the coordination point.

**Recursion** is tree-shaped. Any workflow step can fan out to child workflows, and those children can fan out again. Recursion depth is configurable per prompt and should be bounded. Fan out when sub-tasks are genuinely independent and can be validated independently; keep work inline when sub-tasks share context that would be expensive to reconstruct.

### Task Structure

There are two levels of agent:

**Task-level agents** receive work directly from the orchestrator. They decompose their assigned task into internal steps, execute those steps in order, and commit to their git worktree after each step completes. Task-level agents are the primary unit of work.

**Sub-agents** are transitory. They are spawned by task-level agents (or by other sub-agents) via fan-out. They produce artifacts consumed by their parent but do not commit to git. Their output is gathered by the parent and incorporated into the parent's work.

### Git Strategy

Each task-level agent works in its own git worktree, branched from the current state of `main`.

**Merge policy:** All merges to `main` are human-gated. The system never auto-merges.

**Conflict avoidance:** Task ordering from the plan is the primary mechanism for preventing conflicts. The planner draws explicit boundaries around each task's write scope ("what not to touch"). For the initial version, we trade parallelism for conflict avoidance -- tasks execute in a sequence that minimizes overlap.

**Worktree lifecycle:** Worktrees are disposable. On task failure, document the problem, create a fresh worktree, and start over. On repeated failure, halt and escalate.

**Future (deferred):** Parallel execution of tasks with optimistic concurrency. Conflict resolution treated as another instance of the universal workflow step with a specialized prompt and larger context budget.

### Model Routing

The planner specifies a **capability tier** per task rather than a concrete model. The orchestrator resolves tiers to specific models at dispatch time.

**Capability tiers** (indicative, not final):

| Tier | Use Cases | Examples |
|------|-----------|---------|
| Reasoning | Planning, conflict resolution, complex architectural decisions | Claude Opus, GPT-4o |
| Generation | Code generation, test writing, documentation | Claude Sonnet, GPT-4o-mini |
| Summarization | Leaf result summaries, progress digests, knowledge extraction | Flash models, local LLMs |
| Classification | Transition signal extraction, simple validation | Cheapest available model |

**Model selection** is part of the task definition emitted by the planner, not a global setting. As cheaper models improve, update the tier-to-model mapping without changing prompts or workflow logic.

**Evaluation data:** Track task success rates per model tier. Tasks that fail on a light model and succeed on a heavy one reveal capability boundaries. Over time, shift tasks downward as cheaper models improve. The knowledge extraction workflow can produce these insights automatically.

### Context Assembly

Context assembly is deterministic, budget-aware, and pre-computed. It is the system's core competency alongside prompt construction.

**Sources:**

- Task-specific analysis tools: tree-sitter and LSPs for code tasks; other domain-appropriate tools for non-code tasks.
- Git: Diff context, file history, branch state, prior work products.
- Playbooks: Structured lessons from prior tasks, indexed by task type and domain.
- Task metadata: The plan node, task description, boundaries, validation criteria.
- Domain context: Reference documents, specifications, prior research, external data.

**Budget management:** Each model has a token limit. Context assembly is a packing problem with a priority ordering:

1. Task description and definition of "done"
2. Immediate working context (files the task will read/write, reference material)
3. Interface context (type signatures, API contracts, schemas, specifications)
4. Deterministic analysis results (existing validation output, computed facts)
5. Relevant playbooks
6. Broader project context (structure, conventions, related work)

Graceful truncation: lower-priority items are dropped first. The token budget for the target model is known at assembly time.

### Validation

**Deterministic checks first.** The specific checks depend on the task domain:

- Code tasks: linting (ruff for Python), type checking, test execution, import resolution.
- Research tasks: citation verification, format validation, completeness checks against requirements.
- Any task: schema validation of structured outputs, constraint checking, task-specific scripts (agents can write and run their own).

**LLM-based review** for subjective quality: coherence, consistency with prior work, whether the output matches the intent. Routed to an appropriate model tier.

**Disagreement resolution:** If two review agents disagree, escalate to a more expensive model to break the tie.

**Feedback format:** Validation results sent back to agents should be concise summaries with pointers to details, not raw dumps. Pre-compute aggregate statistics. Respect output discipline to avoid context pollution.

### Knowledge Extraction

Runs as an independent workflow on its own schedule, not on the critical path of task execution.

**Inputs:** Completed task results, execution traces, human escalation resolutions.

**Outputs:** Structured playbooks indexed by task type, domain, error pattern, and model tier.

**Feedback loop:** Playbooks are injected into future task contexts by the assembly step. The latest available playbooks at assembly time are included, with a timestamp indicating freshness.

**Human escalation learning:** When a human resolves an escalation, capture not just the escalation but the resolution. The knowledge extraction workflow turns these into playbook entries. The system learns from its own failures.

### Observability

**OpenTelemetry** traces covering the full hierarchy:

- Pipeline run
- Workflow instance
- Activity (LLM call, validation, context assembly)
- Individual LLM request/response

**Execution journal:** Records decisions (why this task was assigned to this model, why this transition was chosen), not just events (timing, status codes). The journal makes traces legible to human reviewers during escalation.

**Plan-to-execution linking:** Execution spans are linked back to the plan DAG nodes they correspond to. A human reviewing a trace can see "this subtree corresponds to plan task 7: implement authentication middleware."

**Escalation reports:** When the system halts, it produces a structured summary: the current plan, completed task summaries, the triggering task and failure reason, and the orchestrator's understanding of what went wrong. This is a summarization task that can run on a cheap model.

### Human Interaction

**Merge gating:** Humans review and merge task worktrees to `main`.

**Escalation handling:** When the system pauses, the human retrieves the state summary, works with external tools (Claude Code, etc.) to correct course, updates the task list, clarifies requirements, and unpauses the affected tasks.

**Escalation types:**

- *Confused halt:* The orchestrator receives a result it can't classify. Immediate full stop.
- *Degraded halt:* Performance metrics suggest something is wrong (high retry rates, increasing failures). Softer escalation: notification with option to intervene or continue.

### Recurring Meta-Tasks

**Sanity check the task list.** Periodically re-evaluate whether the current plan still makes sense given completed work. Triggered by time (every N completed tasks), events (new-tasks-discovered transitions), or thresholds (failure rate exceeds limit). Consumes knowledge extraction summaries rather than raw task outputs. Output: either "plan is valid, continue" or a revised task DAG.

## Plan Format

The planner's output is the critical design surface. It must be machine-readable (for the orchestrator to create workflow instances) and rich enough (for executing agents to work independently).

Each task in the plan specifies:

- **Objective:** What the task should produce (expected outputs, acceptance criteria, behavioral description).
- **Boundaries:** What the task should not touch (explicit scope limits to reduce conflicts).
- **Context requirements:** Which files, types, tests, and playbooks the task needs.
- **Validation criteria:** Deterministic checks first, then LLM-based if needed.
- **Dependencies:** Other task IDs that must complete before this task can start.
- **Capability tier:** The minimum model capability required.
- **Recursion budget:** How many levels of fan-out the task is allowed.

## Transition Vocabulary

The orchestrator needs a finite set of outcome signals to act on:

| Signal | Meaning | Action |
|--------|---------|--------|
| `success` | Task completed, validation passed | Proceed to next task or gather |
| `failure_retryable` | Task failed, worth retrying | Fresh worktree, retry (up to limit) |
| `failure_terminal` | Task failed, cannot recover | Halt and escalate |
| `new_tasks_discovered` | Agent identified work not in the plan | Trigger sanity check / re-planning |
| `blocked_on_human` | Needs clarification or decision | Pause and escalate |
| `blocked_on_sibling` | Discovered unexpected dependency | Re-evaluate task ordering |

## Retry and Failure

1. On failure: document where the task got stuck (structured failure report).
2. Create a new worktree and start the task from scratch.
3. If the second attempt also fails, halt and escalate to a human.
4. The failure documentation from both attempts is included in the escalation report.

## Technology Stack

**Core:**

| Component | Technology |
|-----------|-----------|
| Workflow orchestration | Temporal |
| Observability | OpenTelemetry |
| Output validation | Pydantic |
| Package management | uv |
| LLM providers | Anthropic, OpenAI, Mistral, local models |
| LLM client library | pydantic-ai |
| Data store / isolation | git, git worktrees |

**Code generation domain (initial use case):**

| Component | Technology |
|-----------|-----------|
| Code analysis | tree-sitter, LSPs |
| Python linting/formatting | ruff |

## Development Phases

### Phase 0: Project Skeleton and Design Document (complete)

- This design document.
- Project structure, dependencies, and configuration.
- Temporal running locally.
- Basic tracing infrastructure.

### Phase 1: The Minimal Loop (complete)

Single workflow executing one LLM call with hardcoded context. One model (Anthropic), one task domain (Python code generation as the initial use case). No fan-out, no planning step, no model routing.

Proves out: the universal workflow step, activity boundaries, git worktree lifecycle, tracing infrastructure.

Deliverable: Describe a small task, Forge executes it in a worktree, validates the output, and presents the result for human review.

### Phase 2: Planning and Multi-Step (complete)

Add a planning step that decomposes a task into ordered sub-steps. Task-level agent executes steps sequentially, committing after each. Still single-model, no fan-out.

Proves out: plan format, transition logic, incremental commit strategy.

Deliverable: Describe a larger task, Forge plans the steps, executes them in order, with a reviewable commit history showing incremental progress.

### Phase 3+ (Future)

- Fan-out / gather (parallel sub-agents).
- Model routing (capability tiers mapped to concrete models).
- Knowledge extraction (playbook generation and injection).
- Multi-provider support.
- Conflict resolution workflow.
- Additional task domains (TypeScript code generation, research, analysis).
- Evaluation and reporting.

Each phase uses the previous version to build the next iteration.
