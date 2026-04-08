# Grill Session: playbook-service

Started: 2026-04-08
Last updated: 2026-04-08
Status: in-progress
Domain: Software architecture — extracting a subsystem from a monolithic Python/Temporal application into an independent service

## Summary

Extract the playbook subsystem from Forge into an independent, general-purpose knowledge service usable by any project — not just Forge. Exposed via SKILL.md for interactive Claude sessions and Temporal workflows for programmatic access. Own database (SQLite/SQLAlchemy/Alembic), own Temporal worker and queue, own CLI (`pbook`), packaged as a separate Python module. Three content types: pitfall entries (unexpected + actionable, extracted from experience), curated advice (human-submitted general knowledge), and API doc records (brief summary, typed signature, known-good examples from official docs, doc pointer — built incrementally on first use). Namespaced tags with controlled vocabulary. Intent-based retrieval (`create`/`fix`) with token budget. Quality bar: minimal and accurate — better to miss than mislead.

## Decision Log

### DECIDED: Multi-project scope
- **Decision**: The playbook service must be usable by any project, not just Forge. It should work with Claude generally.
- **Rationale**: The knowledge store is valuable beyond Forge's orchestration context
- **Date**: 2026-04-08

### DECIDED: Discovery mechanism
- **Decision**: Expose the service via SKILL.md or MCP so other projects can discover and use it
- **Rationale**: Standard Claude integration points make adoption frictionless
- **Date**: 2026-04-08

### DECIDED: Data storage
- **Decision**: Own dedicated SQLite database with SQLAlchemy ORM and Alembic migrations
- **Rationale**: Clean separation from Forge's database
- **Date**: 2026-04-08

### DECIDED: Temporal topology
- **Decision**: Separate Temporal task queue and separate worker process
- **Rationale**: Independent deployment and scaling
- **Date**: 2026-04-08

### DECIDED: Code organization
- **Decision**: Separate Python module that Forge can import. Not a fork — a dependency.
- **Rationale**: Clean boundary while allowing Forge to continue using playbooks
- **Date**: 2026-04-08

### DECIDED: CLI
- **Decision**: Separate `pbook` script as a thin wrapper calling module methods
- **Rationale**: Independent CLI identity, thin shell over library code
- **Date**: 2026-04-08

### DECIDED: Extraction ownership
- **Decision**: Extraction workflow moves to the playbook service. Client projects push completed work. The service runs the LLM extraction.
- **Rationale**: Push model decouples the service from any specific client schema.
- **Date**: 2026-04-08

### DECIDED: Project-level tagging
- **Decision**: Playbooks store which project pushed the information. Project is available as a filter for queries.
- **Rationale**: Different projects may want isolated or blended playbook views
- **Date**: 2026-04-08

### DECIDED: Push API format
- **Decision**: Structured input with a JSONSchema contract shared between client and server.
- **Rationale**: Schema ensures client/server agreement on keys.
- **Date**: 2026-04-08

### DECIDED: Two ingestion paths
- **Decision**: Two paths into the same store: (1) push raw experience for LLM extraction, (2) submit pre-formed lessons directly with LLM review.
- **Date**: 2026-04-08

### DECIDED: Namespaced tags with controlled vocabulary
- **Decision**: Tags use `namespace:value` format. Service defines taxonomy and accepted values.
- **Date**: 2026-04-08

### DECIDED: Extraction doesn't need to produce namespaced tags
- **Decision**: LLM extraction may produce imperfect tags. Controlled vocabulary enforced on read side.
- **Date**: 2026-04-08

### DECIDED: Two-tier tag taxonomy
- **Decision**: Five namespaces: `lang:`, `lib:`, `domain:` (general), `project:`, `pattern:` (extracted).
- **Date**: 2026-04-08

### DECIDED: Drop `error:` namespace
- **Decision**: No `error:` namespace. Error info found via `project:` and `pattern:` tags.
- **Date**: 2026-04-08

### DECIDED: Retrieval via Temporal workflow
- **Decision**: Playbook queries are Temporal workflows on the playbook service's queue.
- **Date**: 2026-04-08

### DECIDED: Token-budgeted retrieval
- **Decision**: Retrieval accepts a token budget (default 5,000 tokens). Service packs within budget.
- **Date**: 2026-04-08

### DECIDED: Extract LLM provider into shared module
- **Decision**: Extract `forge.llm_providers` and `forge.llm_client` into a shared package.
- **Date**: 2026-04-08

### DECIDED: SKILL.md as primary Claude interface
- **Decision**: Start with SKILL.md, not MCP. Interactive sessions where LLM helps refine entries.
- **Date**: 2026-04-08

### DECIDED: Server-provided instructions
- **Decision**: SKILL.md is thin bootstrap. Service provides detailed instructions on demand.
- **Date**: 2026-04-08

### DECIDED: Progressive disclosure with sub-agent and ReAct
- **Decision**: Skill runs as sub-agent. ReAct pattern. Information fetched on demand.
- **Date**: 2026-04-08

### DECIDED: SKILL.md bootstrap scope
- **Decision**: Bootstrap contains only: trigger conditions, sub-agent invocation, `pbook` as allowed tool, instruction to call `pbook skill-prompt`.
- **Date**: 2026-04-08

### DECIDED: Quality bar — minimal and accurate
- **Decision**: Better to miss than mislead. Minimal and accurate. Highest-priority constraint.
- **Date**: 2026-04-08

### DECIDED: Duplicate checking as separate step with refactoring support
- **Decision**: Separate `pbook` command. Near-duplicates may trigger refactoring.
- **Date**: 2026-04-08

### DECIDED: Optimistic review with fallback
- **Decision**: Extracted entries tagged `needs-review`, included by default, `--approved-only` to exclude.
- **Date**: 2026-04-08

### DECIDED: Testable playbook entries
- **Decision**: Eval framework analogous to skill-creator.
- **Date**: 2026-04-08

### DECIDED: LLM-proposed, human-confirmed evals
- **Decision**: Skill agent proposes evals during add sessions. Human confirms. Stored alongside entry.
- **Date**: 2026-04-08

### DECIDED: Intent-based retrieval ranking
- **Decision**: `mode`: `create` or `fix`. `create` boosts general + API docs; `fix` boosts project-specific pitfalls.
- **Date**: 2026-04-08

### DECIDED: Extraction targets unexpected + actionable situations only
- **Decision**: Only entries for unexpected situations with actionable advice.
- **Date**: 2026-04-08

### DECIDED: No `outcome` field in push schema
- **Decision**: Push schema describes what happened. No success/failure label.
- **Date**: 2026-04-08

### DECIDED: Three content types
- **Decision**: The service stores three distinct types of entries:
    1. **Pitfalls** — extracted from experience when unexpected things happen. Reactive. "Here's what goes wrong and how to fix it." Tagged with `project:`, `pattern:`. Created via extraction path.
    2. **Curated advice** — human-submitted general knowledge. Proactive. "Here's a best practice." Tagged with `lang:`, `lib:`, `domain:`. Created via direct submission with LLM review.
    3. **API doc records** — brief summary, typed method signature, known-good examples from official docs, doc URL pointer. Proactive. "Here's what correct usage looks like." Tagged with `lib:`. Created incrementally on first use.
- **Rationale**: Pitfalls course-correct away from mistakes. Advice provides best practices. API docs give reference implementations that help the LLM produce correct code on the first try. Good examples are paramount to getting the right answer faster — more valuable per token than explanatory text.
- **Date**: 2026-04-08

### DECIDED: API doc records are a distinct type, not just entries with code
- **Decision**: API doc records have structured fields (summary, signature, examples, doc_url) separate from freeform pitfall/advice content. The `type` field on entries distinguishes them for retrieval packing.
- **Rationale**: API docs serve a fundamentally different purpose from pitfalls. Pitfalls say "don't do X." API docs say "here's how to do Y correctly." They surface at different times (API docs prioritized in `create` mode, pitfalls in `fix` mode) and have different structural needs (signature + examples vs. narrative advice).
- **Date**: 2026-04-08

### DECIDED: Incremental API doc population
- **Decision**: API docs built incrementally on first encounter with a method, not bulk-indexed. Only methods actually used get documented. Created manually or semi-automated, not auto-created by extraction unless the default behavior was wrong.
- **Rationale**: Aligns with quality bar. Self-prioritizing — frequently used methods get documented first. Most of a large library will never be encountered.
- **Date**: 2026-04-08

## Open Threads

### 1. API doc creation workflow
- **Decided**: Incremental, on first meaningful encounter. Not auto-created unless default failed.
- **Open**: What triggers creation? Candidates:
    - Human says "document sqlalchemy create_engine" during a skill session
    - Skill agent recognizes during task context assembly that a method lacks a doc record and offers to create one
    - `pbook doc add --lib sqlalchemy --method create_engine` CLI command
- **Open**: Where do the examples and signature come from? Fetched from official docs URL, provided by the human, or generated by the LLM with human verification?
- **Open**: What's the structured schema for an API doc record? Proposed:
    - `library` (required) — e.g., "sqlalchemy"
    - `method` (required) — fully qualified, e.g., "sqlalchemy.create_engine"
    - `summary` (required) — 1-2 sentences
    - `signature` (required) — method signature with type hints
    - `examples` (required, list) — small set of working code examples
    - `doc_url` (optional) — pointer to official docs
    - `version` (optional) — library version the docs apply to

### 2. Retrieval packing priority
- Three content types competing for token budget
- **Open**: What's the priority order within a budget? In `create` mode: API docs > curated advice > pitfalls? In `fix` mode: pitfalls > API docs > curated advice?
- **Open**: Should the consumer be able to override priority (e.g., "give me only pitfalls")?

### 3. Eval storage and execution
- **Open**: Where stored? How executed? Batch vs individual?

### 4. `pbook` CLI command surface
- **Open**: Full list TBD. Includes doc management commands.

### 5. Extraction push schema
- Proposed: `project`, `problem`, `resolution`, `context`, `attempts`, `metadata`

## Parking Lot

- Migration strategy (Forge from inline to dependency)
- Existing playbook data migration from forge.db
- Testing strategy for the new module
- Project filter UX in the CLI
- Post-hoc tag editing
- `pattern:` namespace values
- Forge's `_load_playbooks_for_task` rewrite
- Shared LLM provider package name and repo location
- MCP interface (future)
- House-style examples (deferred)
- API doc versioning (library major version changes)
