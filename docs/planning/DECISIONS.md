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

## D30: Python `ast` Over tree-sitter for Phase 4

**Decision:** Use Python's stdlib `ast` module for code analysis in Phase 4. Defer tree-sitter to a future multi-language phase.

**Rationale:** Phase 4 targets Python only, and Forge analyzes valid committed code (not in-progress edits). For this use case, `ast` is simpler (zero dependencies), provides full access to Python type annotations, and has a cleaner API for extracting structured information. tree-sitter's advantages — error-tolerant parsing, incremental re-parsing, multi-language support — are not needed yet. When non-Python language support is added, tree-sitter becomes the natural choice, and the `code_intel` package's interface can accommodate both backends behind the same API.

## D31: grimp Over Custom Import Resolution

**Decision:** Use `grimp` for import graph analysis rather than building import resolution from scratch.

**Rationale:** Import resolution in Python is surprisingly complex: relative imports, namespace packages, `src/` layouts, `__init__.py` re-exports, editable installs. `grimp` handles all of these correctly with Rust-backed performance and provides a rich query API (`find_upstream_modules`, `find_downstream_modules`, `find_shortest_chain`, `get_import_details` with line numbers). Building this from scratch would be a significant effort with many edge cases. `grimp` is actively maintained (used as the engine for `import-linter`).

## D32: PageRank for Importance Ranking

**Decision:** Rank files by importance using PageRank on the import graph, following Aider's approach.

**Rationale:** Not all files in the import graph are equally important for context. A utility module imported by 30 files is more important than a leaf module imported by one. PageRank naturally surfaces "hub" files (heavily imported utilities, base classes, shared types) that provide the most context value per token. Personalized PageRank with seed weights on target files biases the ranking toward files relevant to the current task. This approach is validated by Aider's production usage and by a 2025 benchmark showing AST-derived deterministic graphs achieving 15/15 correctness vs. 6/15 for vector-only RAG — at 70x lower cost than LLM-extracted graphs.

## D33: Character-Based Token Estimation

**Decision:** Estimate tokens at 4 characters per token rather than using a tokenizer library.

**Rationale:** The purpose of token budgeting is overflow prevention, not exact accounting. A 4:1 character-to-token ratio is conservative for English and code. It avoids adding a dependency on `tiktoken` (OpenAI-specific) or model-specific tokenizers. If estimation accuracy becomes a problem, a tokenizer can be substituted behind the same interface.

## D34: Automatic Discovery Augments, Does Not Replace

**Decision:** Automatic context discovery supplements manual `context_files`. If both are specified, manual files are included at priority 6 (packed if budget allows).

**Rationale:** There are context files that import graph analysis cannot discover: configuration files, documentation, test fixtures, data samples, non-Python files. Manual specification remains the escape hatch for these. The planner already produces `context_files` lists — these continue to work. Automatic discovery fills the gap when the caller or planner doesn't know what to include.

## D35: Signature Extraction as Graceful Degradation

**Decision:** When a file is too large to include in full within the token budget, fall back to including its extracted signatures instead of omitting it entirely.

**Rationale:** A file's interface (function signatures, class definitions, type annotations) is almost always more useful than nothing. This implements the "graceful truncation" principle from D14: lower-priority items are reduced before being dropped. A 500-line module's 20-line signature summary fits easily and gives the LLM enough to produce correct imports and type-compatible code. Validated by repominify's finding of 78-82% token reduction while preserving the information LLMs need to produce correct output.

## D36: Repo Map as Standard Context

**Decision:** Include a compressed repo map (file paths + top-ranked signatures) in every context assembly, sized to a configurable token budget (default: 2048 tokens).

**Rationale:** The repo map gives the LLM structural orientation — which modules exist, what their public interfaces are, and how they relate. This is especially valuable for the planner, which must decompose tasks across a codebase it hasn't seen. Aider, Continue.dev, Kiro, and Augment Code all include some form of repo map. The fixed token budget (with binary search sizing) ensures the map never dominates the context window. Following Aider's approach: PageRank-ranked symbols are included in descending order until the repo map budget is filled.

## D37: Import Depth Limit

**Decision:** Trace imports to a configurable depth (default: 2). Direct imports get full content; deeper imports get signatures only.

**Rationale:** Import graphs in real projects can be deep and wide. Unbounded traversal would pull in the entire project. Depth 2 captures the immediate dependency neighborhood — the files that target files import, and the files those import. Beyond that, signatures provide sufficient interface information. The depth limit is configurable for tasks that need broader or narrower context.

## D38: Defer LSP to a Future Phase

**Decision:** Do not integrate LSP in Phase 4. Defer to a future phase.

**Rationale:** Research confirms the hybrid ast + LSP pattern is the consensus for production tools (Claude Code, Kiro, OpenCode all use it). However, LSP requires managing language server lifecycles across multiple git worktrees, which adds significant operational complexity. The `ast` module + `grimp` combination provides sufficient analysis for Phase 4's goals (import discovery, symbol extraction, importance ranking). LSP becomes valuable when Forge needs precise cross-file reference tracking ("find all callers of this function"), type inference beyond annotations, or diagnostic feedback loops — natural additions once Phase 4's foundation is proven.

## D39: SQLite Observability Store Outside Temporal Payloads

**Decision:** Full LLM interaction data (prompts, context, responses) is persisted to a local SQLite database rather than stored in Temporal workflow results. Temporal payloads carry only lightweight statistics (model name, token counts, latency).

**Rationale:** Temporal has a ~2MB payload limit for workflow results. A planned workflow with 5 steps could produce 2MB+ of prompts alone (each step assembles up to 100k tokens of context). Storing prompts in the Temporal result would hit this limit. A local SQLite database has no such constraint and provides rich queryability (filter by task, step, date range). The Temporal result remains lean and fast to retrieve, while the full observability data is available via CLI queries against SQLite.

## D40: SQLAlchemy for Database Access

**Decision:** Use SQLAlchemy (Core + ORM) for all database access rather than raw `sqlite3`.

**Rationale:** SQLAlchemy provides a well-tested abstraction over database connections, transactions, and schema definition. It handles connection pooling, thread safety, and migration support (via Alembic) that would require manual implementation with raw `sqlite3`. The ORM layer maps directly to Pydantic models, reducing boilerplate. SQLAlchemy also enables future migration to PostgreSQL or another backend without changing application code — important if Forge eventually runs with a shared database in a multi-worker deployment.

## D41: Alembic for Schema Management

**Decision:** Use Alembic for database schema migrations rather than ad-hoc `CREATE TABLE IF NOT EXISTS` statements.

**Rationale:** The observability schema will evolve as Forge adds features (tool calling, cost tracking, evaluation scores). Alembic provides versioned migrations, rollback support, and autogeneration from SQLAlchemy models. This prevents schema drift between development and production databases and makes schema changes reviewable in version control. The initial migration creates the baseline schema; subsequent features add migrations incrementally.

## D42: Best-Effort Store Writes in Activities

**Decision:** Store writes in LLM activities are wrapped in try/except and log warnings on failure. The store never blocks or fails the main workflow.

**Rationale:** Observability is secondary to task execution. If the database is unavailable (disk full, permissions, corruption), the LLM call should still succeed and return its result to the workflow. The store is a side effect, not a dependency. This also simplifies testing — activities can be tested without a database by setting `FORGE_DB_PATH` to empty string.

## D43: Playbooks as Flat Tagged Entries

**Decision:** Playbook entries are flat rows in a `playbooks` table, indexed by JSON tag arrays. No hierarchy or categorization beyond tags.

**Rationale:** Flat entries are simple to store, query, and display. Tags provide flexible categorization without requiring a taxonomy. SQLite's `json_each()` function enables efficient tag-based queries. Hierarchy (e.g., grouping by project or domain) can be added later as a view over the same data without schema changes.

## D44: Extraction as a Temporal Workflow

**Decision:** Knowledge extraction is a Temporal workflow with three activities: fetch input, call LLM, save results. It follows D2 (universal workflow step).

**Rationale:** Extraction is another instance of the universal workflow step: construct context, call LLM, serialize result. Using a Temporal workflow provides the same durability, retry, and observability guarantees as task execution. The three-activity decomposition separates I/O (store reads/writes) from the LLM call, enabling independent retry and timeout configuration.

## D45: Relevance by Tag Overlap

**Decision:** Playbook retrieval uses deterministic tag matching rather than semantic similarity (vector embeddings).

**Rationale:** Tag inference from file extensions and description keywords is deterministic, fast, and transparent. It requires no embedding model, no vector database, and no similarity threshold tuning. The tags used for retrieval are the same tags used during extraction, so the matching is consistent. Semantic similarity can be added later if tag overlap proves insufficient.

## D46: Playbooks Share the Observability Store

**Decision:** Playbooks are stored in the same SQLite database as interactions and runs, managed by the same Alembic migration system.

**Rationale:** A single database simplifies deployment, backup, and migration. The observability store already handles connection management, WAL mode, and migration on worker startup. Adding a table is a single Alembic migration. The `playbooks` table references `source_workflow_id` from the `runs` table, making cross-table queries straightforward.

## D47: PLAYBOOK Representation Type

**Decision:** Add a `PLAYBOOK` value to the `Representation` enum to distinguish playbook context items from repo map items, both of which sit at priority 5.

**Rationale:** Before Phase 6, priority 5 was exclusively used by repo map items. Playbook items also belong at priority 5 (lower priority than file content, higher than manual context). Adding a distinct representation type allows `build_system_prompt_with_context` to filter and render each type in its own section without ambiguity.

## D48: LLM-Guided Context Exploration

**Decision:** The LLM chooses what context to request from a menu of available providers. Context requests are fulfilled by Temporal activities, not inline tool calls. The LLM iterates until it signals readiness to generate.

**Rationale:** The LLM knows what information it needs better than a deterministic heuristic. Fulfillment via Temporal activities provides durability, retries, and observability for each analysis step. A configurable round limit bounds token spend. The exploration results feed into the generation prompt, keeping the exploration and generation phases cleanly separated.

## D49: Progressive Disclosure for Context Assembly

**Decision:** By default, only target file contents and the repo map are assembled upfront. Dependency file contents (direct imports at priority 3) and transitive symbol signatures (priority 4) are omitted unless `--include-deps` is passed. The `discover_context()` function accepts an `include_dependencies` parameter (default `False`), and the `ContextConfig` model exposes this as `include_dependencies`.

**Rationale:** Dumping all auto-discovered dependency code into the system prompt overwhelms the model. For a simple task like "add a docstring to main()" targeting `cli.py`, the upfront approach produced a 161K-character prompt where 69% was dependency code the model did not need. This caused the model to return empty structured output instead of proper `LLMResponse` tool calls. The planner already follows the lean pattern (task + repo map only, no dependency dumps). Phase 7's exploration providers (`read_file`, `symbol_list`, `discover_context`) let the model pull dependencies on demand when it actually needs them. The `--include-deps` flag restores the old behavior for tasks that benefit from upfront dependency context.

## D50: Diff-Based Output Over Full File Replacement

**Decision:** The LLM produces diffs (search/replace blocks) rather than complete file contents. The `write_output` activity applies diffs to existing files instead of overwriting them. New files still use full content.

**Rationale:** Full file replacement has two compounding problems. First, the LLM must reproduce every line of an existing file to change a few — a 370-line file modified with a 10-line addition requires outputting all 380 lines, wasting output tokens and inviting errors. Second, and more critically, if the LLM doesn't receive the full file content (as happened in planned mode's step-level execution, which assembled only 1333 input tokens for a 370-line target file), it writes the file from scratch, silently destroying all existing code. Diff-based output eliminates both problems: the LLM specifies only what changes, and the system applies those changes to the existing file. This is the standard approach used by Claude Code, Aider, and other production AI coding tools.

## D51: Error Feedback in Retry Context

**Decision:** Include validation errors from the previous attempt in the retry prompt as a dedicated section, with optional AST-derived code context around error locations.

**Rationale:** Blind retries waste tokens repeating the same mistakes. Both Aider and Claude Code feed error output back to the LLM. The error section is placed before Output Requirements to ensure the LLM sees it before generating. AST-based context enrichment (showing the enclosing function around an error line) helps the LLM understand errors without requiring it to re-read the entire file. Test failure output is included verbatim since it typically already contains sufficient context.

## D52: Backward-Compatible Retry Fields

**Decision:** All retry-related fields (`prior_errors`, `attempt`, `max_attempts`) use defaults (`[]`, `1`, `2`), so existing serialized payloads and callers work unchanged.

**Rationale:** Temporal workflows may have in-flight executions when this change deploys. Default values ensure old payloads deserialize correctly. The first attempt always has `prior_errors=[]`, so the prompt is unchanged for non-retry calls.

## D53: Stable-First Prompt Ordering

**Decision:** Reorder prompt sections so the most stable content (role, output requirements, repo map) comes before volatile content (target file contents, exploration results, retry errors).

**Rationale:** Anthropic's prompt caching computes cache keys cumulatively — the hash for each block depends on all preceding content. Changes at any point invalidate that point and everything after it. By placing stable content first, the longest possible prefix remains cached across retries, exploration rounds, and (partially) across steps. This is the same pattern Aider and Claude Code use.

## D54: Automatic Cache Control via pydantic-ai

**Decision:** Use pydantic-ai's Anthropic model settings to enable prompt caching rather than manually constructing Anthropic API messages.

**Rationale:** pydantic-ai abstracts the Anthropic API. Bypassing it to inject `cache_control` headers would couple Forge to Anthropic's wire format and break the model abstraction. If pydantic-ai's automatic placement is suboptimal, manual breakpoints can be added as a refinement.

## D55: Fallback Chain Over Single Strategy

**Decision:** Use an ordered fallback chain (exact → whitespace → indentation → fuzzy) rather than jumping directly to fuzzy matching for all edits.

**Rationale:** Exact matching should remain the fast path — it's O(n) and unambiguous. Fuzzy matching is O(n*m) and introduces a confidence threshold. The fallback chain preserves the performance and correctness guarantees of exact matching for the common case while recovering gracefully when minor discrepancies occur. Each level is more expensive and less certain than the previous, so trying them in order minimizes cost and maximizes confidence. Logging which level matched enables monitoring match quality degradation over time.

## D56: Similarity Threshold at 0.6

**Decision:** Default fuzzy matching threshold is 0.6 (configurable).

**Rationale:** Aider uses 0.6 as its default and reports good results — it catches most whitespace and minor content differences while avoiding false matches. A threshold below 0.5 risks matching unrelated code blocks. The threshold is configurable per-task if needed, but the default should be conservative enough to avoid incorrect matches.

## D57: Uniqueness Required at All Levels

**Decision:** Even fuzzy matching requires a unique best match. If two blocks score within 0.05 of each other above the threshold, the edit fails as ambiguous.

**Rationale:** A non-unique match means the system cannot confidently determine which code block the LLM intended to modify. Applying the edit to the wrong block would silently corrupt the file — worse than failing and retrying. The 0.05 gap requirement ensures the best match is clearly distinguishable from alternatives.

## D58: Capability Tiers Over Direct Model Names

**Decision:** Route LLM calls via abstract capability tiers (REASONING, GENERATION, SUMMARIZATION, CLASSIFICATION) rather than passing concrete model names through the system.

**Rationale:** This follows D10 from the design document. Abstract tiers decouple the "what capability is needed" question from the "which model provides it" question. When a cheaper model improves enough to handle GENERATION tasks, a single config change upgrades the entire system. When a new provider is added, it slots into the tier mapping without changing any workflow code. The planner specifies capability requirements, not provider-specific model IDs.

## D59: Planner Gets Premium Model

**Decision:** Default the planner to the REASONING tier (Opus-class model).

**Rationale:** D11 states "Planning is the hard part" and "invest the most expensive models in planning." Plan quality bounds everything downstream — a better decomposition reduces conflicts, improves context_files accuracy, and produces fewer retries. The planner is called once per task, so the incremental cost is bounded. Moving from Sonnet to Opus for planning is the single highest-value model routing change.

## D60: Backward-Compatible Defaults

**Decision:** All `ModelConfig` fields have defaults, and `ForgeTaskInput.model_config_` defaults to `ModelConfig()`. Existing callers and serialized payloads work unchanged.

**Rationale:** Temporal workflows may have in-flight executions when this change deploys. Default values ensure old payloads deserialize correctly. The field alias avoids collision with Pydantic's reserved `model_config` attribute.

## D61: Thinking for Planning Only

**Decision:** Enable extended thinking for planner calls only, not for code generation or exploration.

**Rationale:** Planning is the highest-leverage use of extended thinking. The planner must reason about task decomposition, dependencies, and ordering — a complex multi-step reasoning task where thinking mode shows the most improvement. Code generation benefits less: the LLM already sees the target file contents and context, and produces edits in a relatively straightforward way. Exploration is classification (what context to request), which thinking mode can actually slow down. Enabling thinking selectively keeps costs bounded while maximizing quality impact.

## D62: Adaptive Thinking for Opus, Budget for Sonnet

**Decision:** Use adaptive thinking (`{'type': 'adaptive'}`) for Opus 4.6+ and budget-based thinking (`{'type': 'enabled', 'budget_tokens': N}`) for Sonnet.

**Rationale:** Opus 4.6 supports adaptive thinking where the model dynamically decides how much to think based on problem complexity. This is more efficient than a fixed budget — simple tasks get less thinking, complex tasks get more. Sonnet requires explicit budget-based thinking. The implementation detects the model and applies the appropriate configuration, so upgrading the planner model (Phase 11) automatically gets the right thinking mode.

## D63: Silent Degradation for Non-Anthropic Models

**Decision:** Thinking configuration is silently ignored for non-Anthropic models (no error, no warning).

**Rationale:** Forge's model abstraction (via pydantic-ai) supports multiple providers. If the planner is configured to use an OpenAI model, thinking configuration simply doesn't apply. This matches D54's approach to caching — provider-specific features degrade gracefully.

## D64: Tree-Sitter Over Language-Specific Parsers

**Decision:** Use tree-sitter as the universal parsing backend for all languages, replacing Python's `ast` for Python and providing new support for other languages.

**Rationale:** D30 deferred tree-sitter because Phase 4 targeted Python only and `ast` was simpler. Phase 13 extends to multiple languages where `ast` is not an option. Rather than maintaining two parsing backends (ast for Python, tree-sitter for everything else), a single tree-sitter backend reduces maintenance and ensures consistent behavior across languages. tree-sitter's error-tolerant parsing also handles malformed code that `ast.parse()` rejects.

## D65: Tag Query Pattern (Following Aider)

**Decision:** Use language-specific `.scm` tag query files to define which AST nodes represent definitions, following Aider's proven pattern.

**Rationale:** Aider has battle-tested tag queries for 20+ languages. The pattern separates language-specific knowledge (what constitutes a "function definition" in Go vs TypeScript) from the extraction logic (walk the tree, extract matching nodes, format signatures). Adding a new language requires only writing a `.scm` file, not modifying Python code.

## D66: Stable Output Interface

**Decision:** Retain the existing `SymbolSummary` and `ExtractedSymbol` output models. Only the extraction backend changes.

**Rationale:** The output interface is consumed by repo map generation, budget packing, and context assembly. Changing the interface would cascade into many modules. The existing models are language-agnostic — `SymbolKind` (function, class, type_alias, constant) covers the common definition types across languages. Language-specific refinements (e.g., Go interfaces, Rust traits) map to the closest existing kind.

## D67: Graceful Degradation for Unsupported Languages

**Decision:** Files with unrecognized extensions produce an empty `SymbolSummary` (path only, no symbols). No error.

**Rationale:** Multi-language codebases often include configuration files, data files, or files in niche languages. The system should not fail on encountering them. File-path-only entries in the repo map still provide structural orientation. New languages can be added incrementally by writing a tag query file.

## D68: CLI Test Output Stream Assertions

**Decision:** CLI tests use `result.stdout` for structured data assertions (JSON parsing), `result.stderr` for error/warning message assertions, and `result.output` for general "user sees this" assertions where the stream doesn't matter.

**Rationale:** Click 8.2+ provides separate `result.stdout` and `result.stderr` properties alongside the mixed `result.output`. Using `result.stdout` for JSON-parsing tests ensures they cannot be broken by warnings or errors written to stderr via `click.echo(..., err=True)`. Using `result.stderr` for error assertions verifies messages go to the correct stream. This makes output separation robust by construction rather than relying on print ordering. The CLI's use of `click.echo()` for stdout and `click.echo(..., err=True)` for stderr follows standard Unix convention.

## D69: Recursive Fan-Out With Depth Budget

**Decision:** `ForgeSubTaskWorkflow` can recursively fan out into child workflows when nested `sub_tasks` are present and `depth < max_depth`. A configurable `max_fan_out_depth` (default 1) bounds recursion. This supersedes D29's restriction of single-level fan-out.

**Rationale:** Single-level fan-out (Phase 3) is proven stable. Some decomposition patterns naturally require hierarchy — e.g. a sub-task for "implement all API endpoints" that itself decomposes into independent endpoint sub-tasks. The depth budget (D17) prevents unbounded recursion. Default `max_fan_out_depth=1` preserves existing behavior; users opt in to recursive fan-out with `--max-fan-out-depth N`. Child workflow timeouts scale with remaining depth to allow orchestration overhead at each nesting level.

## D70: Plan-Level Sanity Check With Step Interval Trigger

**Decision:** Opt-in sanity check runs after every N completed steps during planned execution. Uses REASONING tier with extended thinking. Three verdicts: continue, revise (replace remaining steps), abort.

**Rationale:** Plans can become stale as work reveals new information (D19). Periodic re-evaluation allows self-correction without waiting for failure. Step-interval trigger is simple and predictable. Revision replaces only remaining steps, preserving committed work. Disabled by default (interval=0) to avoid overhead for simple tasks.

## D71: LLM-Based Fan-Out Conflict Resolution

**Decision:** Replace D27 terminal error with REASONING-tier LLM resolution when multiple sub-tasks produce the same file. If the merged result fails validation, escalate to FAILURE_TERMINAL with a detailed conflict report. Supersedes D27.

**Rationale:** File conflicts between sub-tasks are not always planning errors — legitimate parallel work can produce overlapping changes to the same file (e.g., two sub-tasks adding different functions to a shared module). LLM-based resolution preserves the intent of all sub-tasks while keeping the workflow alive. The original D27 terminal error remains available via `--no-resolve-conflicts` for cases where conflicts should surface as planning errors.

## D72: Conflict Resolution as Universal Workflow Step

**Decision:** Conflict resolution follows the same construct-send-receive-serialize-transition pattern as all other LLM calls. Uses a specialized output type (`ConflictResolutionResponse`) but reuses the agent infrastructure, thinking settings, and store persistence patterns.

**Rationale:** Consistency with the universal workflow step (D2) reduces cognitive overhead and ensures conflict resolution benefits from the same observability, caching, and thinking infrastructure as other LLM calls.

## D73: REASONING Tier for Conflict Resolution

**Decision:** Always use REASONING tier (per D11 model routing) for conflict resolution. Merging conflicting code requires understanding the intent of multiple sub-tasks and producing correct combined output.

**Rationale:** Conflict resolution is a high-stakes operation — an incorrect merge can silently break functionality. The REASONING tier provides the best chance of producing correct merged output, and the cost is justified since conflicts are infrequent.

## D74: Parallel Task Execution Deferred

**Decision:** Parallel task execution with optimistic concurrency (Phase B) is deferred to a future phase. Fan-out conflict resolution (Phase A) proves the conflict resolution primitive first.

**Rationale:** Fan-out conflicts are simpler — competing file versions are already in memory, and the workflow engine handles orchestration. Parallel task execution adds git merge conflict detection, `depends_on` scheduling, and concurrent step management. Building Phase A first validates the LLM resolution approach before adding that complexity.

## D75: Remove pydantic-ai, Use Anthropic SDK + Plain Pydantic

**Decision:** Remove the pydantic-ai dependency. Replace `pydantic_ai.Agent` with direct Anthropic SDK calls (`client.messages.create`) for synchronous LLM calls and `client.messages.batches.create` for batch calls. Use plain Pydantic models for structured output via tool definitions (`Model.model_json_schema()`) and response validation (`Model.model_validate_json()`).

**Rationale:** pydantic-ai bundles request construction, API call, and response parsing into a single `agent.run()` call. Batch mode requires splitting this into submit (construct + send) and resume (receive + parse), which pydantic-ai does not support. Beyond the batch requirement, Forge does not use pydantic-ai's main features (dependency injection, conversation management, tool orchestration) — Temporal provides the workflow orchestration and retry logic. pydantic-ai adds a layer of abstraction over the Anthropic SDK that provides no value and prevents direct access to SDK features like batch submission. Plain Pydantic provides the same structured output capability (schema generation, validation) without the wrapper.

## D76: Batch Mode as Default Execution Path

**Decision:** All LLM calls use the Anthropic Message Batches API by default. Synchronous execution via the Messages API is available via `--sync` flag.

**Rationale:** Batch mode provides a 50% cost reduction on all token usage. For a system that processes many LLM calls per task (planner, exploration rounds, generation, validation), this halves the marginal cost of every operation. The latency tradeoff (polling interval vs. immediate response) is acceptable for an automated system that does not require interactive response times. The `--sync` flag provides an escape hatch for time-sensitive operations or Batch API outages.

## D77: Signal-Based Wait Over Terminate-and-Restart

**Decision:** Workflows wait for batch results via Temporal signals (`workflow.wait_condition`), keeping the workflow alive and all state in Temporal's durable execution. The alternative — terminating the workflow after submission and starting a new one when results arrive — was rejected.

**Rationale:** Temporal's durable execution model is designed for exactly this pattern: a workflow that does work, waits for an external event, then continues. Signal-based waiting preserves all workflow state (plan steps, retry counts, worktree paths, exploration context) without serializing it to an external database. If the worker crashes, Temporal replays the workflow history and resumes waiting. The terminate-and-restart alternative would require serializing the full continuation context to a batch job database, reconstructing it in a new workflow, and managing the handoff — duplicating what Temporal already provides.

## D78: Temporal Search Attributes for Batch Result Routing

**Decision:** Workflows set a custom search attribute (`forge_batch_id`) when waiting for a batch result. The batch poller queries Temporal's visibility API to find workflows waiting for a specific batch, then sends signals directly.

**Rationale:** Search attributes let the poller route results to workflows without maintaining a separate routing table. Temporal's visibility API supports efficient queries by indexed attributes. This eliminates the batch jobs table as a coordination mechanism — it becomes purely an audit log. For Release 1 (single-request batches), `forge_batch_id` uniquely identifies both the batch and the waiting workflow. Release 2 (multi-request batches) will add `forge_batch_request_id` for per-request routing within a batch.

## D79: Single-Request Batches for Release 1

**Decision:** Each LLM call is submitted as its own single-request batch. Multi-request batching (grouping calls into a single batch) is deferred to Release 2.

**Rationale:** Single-request batches are the simplest implementation: one batch per LLM call, one result per batch, one-to-one mapping between batch_id and workflow. This avoids the complexity of accumulation windows, batch formation strategies, and per-request routing within a batch. The 50% cost savings apply regardless of batch size. Multi-request batching adds throughput optimization but not cost optimization.

## D80: Batch Jobs Table as Audit Log

**Decision:** The `batch_jobs` SQLite table records all batch submissions and outcomes for observability and anomaly detection. It is not used for coordination — all coordination uses Temporal signals and search attributes.

**Rationale:** Consistent with D39 (SQLite store as observability layer) and D42 (best-effort store writes). The table provides a persistent record of batch lifecycle events for debugging, cost analysis, and anomaly detection (missing jobs, status regressions). Separating the audit function from coordination prevents the store from becoming a single point of failure.

## D81: Temporal Schedules for Batch Polling and Knowledge Extraction

**Decision:** Use Temporal Schedules for the batch poller (configurable interval, default 60s) and for knowledge extraction (configurable interval, default 4 hours). These replace the need for custom cron jobs or polling loops.

**Rationale:** Temporal Schedules provide durable, managed scheduling with visibility, pause/resume, and automatic retry. The batch poller must run reliably regardless of worker restarts — a Schedule guarantees this. Knowledge extraction (currently manual CLI invocation) is a natural fit for periodic scheduling: check for unextracted runs and process them. Using Temporal's built-in scheduling avoids duplicating functionality.

## D82: All LLM Call Sites Through Batch

**Decision:** All five LLM call sites (generation, planner, exploration, sanity check, conflict resolution) go through the batch path by default. No call site is exempted for Release 1.

**Rationale:** Consistency simplifies the implementation — one code path for all LLM calls, switchable between batch and sync via a single flag. The latency cost for sequential call sites (exploration rounds, planner) is bounded by the poll interval (default 60s per round). For a 10-round exploration at 60s polling, the worst case adds ~10 minutes — acceptable for an automated system saving 50% on token costs. Release 2 can introduce per-call routing for latency-sensitive call sites if needed.

## D83: Decompose Phase 14 into Sub-Phases (14a/14b/14c)

**Decision:** Split Phase 14 (batch processing) into three independently deliverable sub-phases rather than implementing it as a single phase.

**Rationale:** Phase 14 touches models, activities, workflows, scheduling, and CLI — too many concerns for a single deliverable. Decomposing into sub-phases provides incremental testability and reduces risk:

- **Phase 14a — Batch infrastructure**: Data models, `batch_jobs` audit table, `submit_batch_request` activity, `parse_llm_response` activities. No workflow changes. Everything testable in isolation.
- **Phase 14b — Workflow batch integration**: Signal handler, `_call_llm` helper, `forge_batch_id` search attribute, `--sync` flag, wire up all 5 call sites. Defaults to sync mode until the poller exists.
- **Phase 14c — Batch poller + scheduling**: `poll_batch_results` activity, `BatchPollerWorkflow`, Temporal Schedule registration, anomaly detection, knowledge extraction schedule migration, flip default to batch mode.

Each sub-phase is independently committable with all tests passing. 14a is pure plumbing with no behavior change. 14b adds workflow wiring but defaults to sync. 14c completes the loop and enables batch as default.

## D84: No Speculation — Ask First, Then Investigate

**Decision:** When encountering unexpected behavior (test failures, hangs, errors), do not speculate about the cause. Ask the user first. If the user doesn't know, offer to investigate.

**Rationale:** Speculative explanations presented as fact erode trust. A wrong guess wastes time and can send debugging down the wrong path. Asking the user is cheap and often yields the answer immediately (e.g., "the Temporal server is already running"). If the user doesn't know either, a concrete offer to investigate sets the right expectation and leads to an evidence-based answer.

## D85: File-Based Application Logging at XDG State Directory

**Decision:** Application logs are written to files under `$XDG_STATE_HOME/forge/` (defaulting to `~/.local/state/forge/`). Console output controlled by `-v` flags remains for interactive use, but the canonical log destination is the filesystem.

**Rationale:** Console-only logging is ephemeral — once the terminal scrolls or the session ends, the output is gone. File-based logging enables post-hoc debugging, support triage, and correlation across Temporal activities and workflows. The XDG Base Directory Specification designates `$XDG_STATE_HOME` for persistent state data such as logs, which keeps them separate from configuration (`$XDG_CONFIG_HOME`) and cached data (`$XDG_CACHE_HOME`).
