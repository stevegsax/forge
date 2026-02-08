# Design Decisions

This document captures key design decisions and their rationale. Decisions are numbered for reference.

## D1: Batch Mode Over Conversational Streaming

**Decision:** Forge uses LLM batch mode with document completion rather than iterative streaming (REPL-style) interaction.

**Rationale:** Batch mode enables true parallelism. Each request is self-contained with no dependency on conversation history, allowing multiple requests to execute simultaneously. The system invests in upfront planning to identify parallelizable work, then submits independent requests. Temporal provides the execution loop rather than a chat-style REPL.

## D2: Universal Workflow Step

**Decision:** Every task type (code generation, review, conflict resolution, knowledge extraction, research) uses the same workflow primitive: construct, send, receive, serialize, transition.

**Rationale:** A single primitive means the orchestration engine has no special-case logic for different task types. The differentiation lives entirely in prompts, context assembly, and validation criteria. This keeps the workflow machinery simple and makes adding new task types a matter of writing new prompts.

## D3: Temporal for Orchestration

**Decision:** Use Temporal for workflow orchestration rather than a custom scheduler.

**Rationale:** Temporal provides durable execution, retry semantics, child workflows (for fan-out/gather), signal handling (for human interaction), and visibility into workflow state. These are hard problems that Temporal solves well. The LLM call and transition evaluation are separate Temporal activities, giving us independent retry and timeout control for each.

## D4: Orchestrator Assigns Tasks (Not Self-Selection)

**Decision:** The orchestrator selects tasks and creates agents to complete them, rather than having agents self-select from a task pool.

**Rationale:** Eliminates the entire class of lock-contention and duplicate-work problems. There is no race condition because there is no race. The tradeoff is that decomposition quality depends entirely on the planner, but we accept this because we are investing heavily in planning.

## D5: Hub-and-Spoke Coordination (Not Peer-to-Peer)

**Decision:** Sub-agents report results to their parent workflow, which coordinates. No direct agent-to-agent messaging.

**Rationale:** Fan-out/gather via Temporal child workflows is a natural hub-and-spoke pattern. The parent is always the coordination point. When an agent discovers information that affects siblings (e.g., a schema change is needed), the response goes back to the orchestrator, which decides whether to reprioritize. This avoids the complexity of peer-to-peer messaging between batch jobs.

## D6: Git Worktrees as General-Purpose Isolation

**Decision:** Git worktrees serve as the isolation mechanism for all task types, not just code.

**Rationale:** Worktrees provide independent working directories with controlled reconciliation via merge. This applies equally to code branches, research threads, analysis tracks, or any work product that benefits from parallel progress. Git also provides history, diffing, and a natural audit trail.

## D7: Human-Gated Merges

**Decision:** All merges to `main` require human approval. The system never auto-merges.

**Rationale:** Safety. Until the system has proven its planning and conflict resolution capabilities, a human in the loop at the merge point prevents compounding errors. This is a conservative choice for v1 that can be relaxed later.

## D8: Conflict Avoidance Through Planning, Not Locking

**Decision:** The primary conflict avoidance mechanism is task ordering and explicit write-scope boundaries in the plan, not file locks or branch coordination.

**Rationale:** Good planning should minimize conflicts. Each task specifies what it will touch and what it must not touch. For v1, we trade parallelism for conflict avoidance (sequential execution). Parallel execution with optimistic concurrency is deferred. When conflicts do arise, resolution is another instance of the universal workflow step.

## D9: Halt and Escalate as Default Policy

**Decision:** When the orchestrator encounters a situation it cannot classify, it halts and escalates to a human rather than attempting to continue.

**Rationale:** The failure modes of continuing with a bad plan (wasted tokens, divergent branches, wrong results) are worse than the cost of pausing. Two escalation types: "confused halt" (unclassifiable result, immediate stop) and "degraded halt" (anomalous metrics, softer notification).

## D10: Model Routing via Capability Tiers

**Decision:** The planner specifies a capability tier per task (reasoning, generation, summarization, classification). The orchestrator resolves tiers to concrete models at dispatch time.

**Rationale:** Decouples task definitions from specific models. As cheaper models improve, update the tier-to-model mapping without changing prompts or workflows. Enables tracking success rates per tier to discover where capability boundaries actually are.

## D11: Planning Gets Premium Resources

**Decision:** Dedicate the most expensive models and highest token budgets to planning and conflict resolution.

**Rationale:** Plan quality bounds everything downstream. A good plan reduces conflicts, improves validation, and produces better results. Planning is the hard part of the problem; the more we figure out early, the easier the rest of the process is.

## D12: Deterministic Before LLM-Based Validation

**Decision:** Always run deterministic validation (lint, type check, schema validation, tests) before LLM-based review.

**Rationale:** Deterministic checks are cheap, fast, and reliable. LLM-based evaluation is reserved for subjective or complex judgments that cannot be computed. When two LLM reviewers disagree, escalate to a more expensive model to break the tie.

## D13: Knowledge Extraction as Independent Workflow

**Decision:** Playbook generation runs on its own schedule, independent of the task execution critical path.

**Rationale:** Avoids adding latency to task execution. Playbooks are an optimization, not a correctness requirement. New tasks use the latest available playbooks at context assembly time. Staleness is acceptable.

## D14: Context Assembly is a Packing Problem

**Decision:** Context assembly is deterministic and budget-aware. It prioritizes context elements by importance and truncates gracefully to fit the target model's token limit.

**Rationale:** Different model tiers have different token limits. Including too much context wastes budget; including too little produces poor results. Priority ordering (task description > immediate context > interfaces > validation results > playbooks > broad context) ensures the most important information is always present.

## D15: Retry From Clean State

**Decision:** On task failure, document the problem, create a fresh worktree, and start over. On repeated failure, halt and escalate.

**Rationale:** Clean-state retries are simpler and more predictable than attempting to continue from a partially completed state. Worktrees are disposable by design. The failure documentation from both attempts is included in the escalation report.

## D16: Task-Level vs. Sub-Agent Commit Behavior

**Decision:** Task-level agents (assigned directly by the orchestrator) commit after each internal step. Sub-agents are transitory and do not commit.

**Rationale:** Task-level agents represent meaningful units of work whose progress should be tracked. Sub-agents are implementation details of a task -- their outputs are consumed by the parent and incorporated into the parent's commits.

## D17: Recursion Budget Per Task

**Decision:** Each task in the plan specifies a recursion budget (maximum fan-out depth). This is configurable per prompt.

**Rationale:** Unbounded recursion creates deep trees that are expensive and hard to reason about. The planner, which has the most context about task complexity, decides how much decomposition each task warrants.

## D18: Dual Representation of Task Results

**Decision:** Each task produces both its full output (for storage and merge) and a compact digest (for consumption by parent gather steps and sanity checks).

**Rationale:** Gather steps need to understand all children's outputs within a single context window. Full outputs may not fit. A digest designed for parent consumption keeps gather steps tractable without losing detail in the stored artifacts.

## D19: Sanity Check as Recurring Meta-Task

**Decision:** Periodically re-evaluate the plan against completed work. Triggered by time, events (new tasks discovered), or thresholds (failure rate).

**Rationale:** Plans can become stale as work reveals new information. The sanity check consumes knowledge extraction summaries and determines whether the plan is still valid or needs revision. This is the system's self-correction mechanism.

## D20: Self-Bootstrapping Development

**Decision:** Each phase of Forge's development uses the previous version to build the next iteration, starting with a minimal manual foundation.

**Rationale:** The system is its own best test case. Early phases are simple (single-step, single-model, hardcoded context, human review). Each subsequent phase adds one dimension of complexity. Early halts become training data for improving the system.

## D21: Auto-Fix Before Validation

**Decision:** Run `ruff check --fix` and `ruff format` on generated files before running validation checks. Enabled by default via `ValidationConfig.auto_fix`. All ruff invocations use `--config tool-config/ruff.toml` to ensure consistent rules across worktrees (see D22).

**Rationale:** Deterministic cosmetic cleanup (unused imports, `typing.List` → `list`, formatting) is not the LLM's job. Burning retry budget on issues that `ruff` can fix in milliseconds wastes tokens and time. Reserve validation failures for issues the LLM actually needs to think about: logic errors, test failures, architectural violations. The `auto_fix` flag exists for opt-out in debugging or evaluation scenarios.

## D22: Standardized Tool Configuration Location

**Decision:** Tool configurations live in `tool-config/` in the repository root. Ruff invocations reference `--config tool-config/ruff.toml` rather than relying on config file discovery.

**Rationale:** Git worktrees share the repository's file tree, so `tool-config/` is available in every worktree automatically. Explicit `--config` avoids ambiguity from ruff's config file discovery, which can pick up different files depending on the working directory. A dedicated directory keeps tool configs separate from project metadata (`pyproject.toml`) and makes it obvious where to look.

## D23: Step-Level Retry Over Task-Level Retry for Planned Execution

**Decision:** In multi-step planned execution, retry at the step level using `git reset --hard HEAD && git clean -fd` rather than destroying and recreating the worktree. If a step exhausts retries, the task fails terminal. Task-level retry (re-plan from scratch) is deferred to Phase 3.

**Rationale:** Phase 1 retries destroy and recreate the entire worktree. In multi-step execution, prior steps are already committed — destroying the worktree would lose that progress. Step-level reset preserves committed work while giving the current step a clean slate. This is a natural consequence of the incremental commit strategy: each committed step is a checkpoint.

## D24: Planning as a Separate Activity

**Decision:** Planning (task decomposition) is a separate Temporal activity from the step-level LLM call, with its own prompt construction, output schema (`Plan` vs `LLMResponse`), and agent configuration.

**Rationale:** Planning and execution have different concerns. The planner needs to see the full task description and produce a structured plan (ordered steps with dependencies and file targets). Step execution needs to see the current step's context, completed step history, and produce file outputs. Sharing the `call_llm` activity for both would conflate two distinct prompting strategies. Separate activities also allow different timeout and retry configurations.

## D25: Single Worktree for the Entire Plan

**Decision:** A single git worktree is created before planning and persists across all steps. Steps commit incrementally to this worktree.

**Rationale:** Steps in a plan are sequential and build on each other. Step 2 may read files that step 1 created. A shared worktree with incremental commits makes prior step outputs available as regular files via `context_files`. This also produces a linear commit history in the worktree branch, making human review straightforward — each commit corresponds to one plan step.

## D26: Compound Task IDs for Sub-Task Isolation

**Decision:** Sub-tasks use compound IDs in the form `{parent_task_id}.sub.{sub_task_id}`. These IDs are used for worktree paths and branch names via the existing git functions.

**Rationale:** Dots are already valid in the task ID regex (`^[A-Za-z0-9][A-Za-z0-9._-]*$`), so all existing git functions (`worktree_path`, `branch_name`, `create_worktree`, `remove_worktree`) work unchanged. The compound ID makes the parent-child relationship visible in git branch names (e.g., `forge/my-task.sub.analyze-schema`) and worktree paths, aiding debugging and cleanup.

## D27: File Conflict as Terminal Error

**Decision:** When two sub-tasks produce files at the same path, the fan-out step fails with a terminal error rather than attempting resolution.

**Rationale:** File conflicts between sub-tasks indicate a planning error — the planner should have assigned non-overlapping file targets. Automatic resolution (last-writer-wins, merge, or LLM-based) adds complexity and risks silent data loss. Failing terminal surfaces the planning error for human review. LLM-based conflict resolution is deferred to a future phase.

## D28: Sub-Task Context Reads From Parent Worktree

**Decision:** Sub-task context files are read from the parent worktree, not the sub-task's own worktree.

**Rationale:** Sub-task worktrees start as copies of the parent branch state. Context files referenced by sub-tasks are files from prior steps that are already committed to the parent worktree. Reading from the parent is explicit about the data source and avoids confusion about which worktree's state is being used.

## D29: No Recursive Fan-Out in Phase 3

**Decision:** Sub-tasks execute the universal workflow step but cannot themselves fan out into further sub-tasks. Fan-out depth is limited to one level.

**Rationale:** Recursive fan-out introduces tree-shaped execution that is harder to reason about, debug, and bound. Phase 3 proves out the fan-out/gather primitive at a single level. Recursive fan-out can be added in a future phase by allowing `ForgeSubTaskWorkflow` to detect sub-tasks in its step, but the additional complexity is not justified until single-level fan-out is proven.
