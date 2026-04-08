# Grill Session: playbook-service

Started: 2026-04-08
Last updated: 2026-04-08
Status: in-progress
Domain: Software architecture — extracting a subsystem from a monolithic Python/Temporal application into an independent service

## Summary

Extract the playbook subsystem from Forge into an independent, general-purpose knowledge service usable by any project — not just Forge. Exposed via SKILL.md for interactive Claude sessions and Temporal workflows for programmatic access. Own database (SQLite/SQLAlchemy/Alembic), own Temporal worker and queue, own CLI (`pbook`), packaged as a separate Python module. Three content types: pitfall entries (unexpected + actionable, extracted from experience), curated advice (human-submitted general knowledge), and API doc records (brief summary, typed signature, known-good examples from official docs, doc pointer — built incrementally via Firecrawl). Library onboarding produces a progressive disclosure navigation tree from docs. Doc lookup runs as a separate agent thread with potentially cheaper LLM for cost efficiency. Namespaced tags with controlled vocabulary. Intent-based retrieval (`create`/`fix`) with token budget. Quality bar: minimal and accurate — better to miss than mislead.

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
- **Decision**: Pitfalls (reactive, extracted), curated advice (proactive, human-submitted), API doc records (proactive, reference implementations with examples). Distinct `type` field on entries.
- **Date**: 2026-04-08

### DECIDED: API doc records are a distinct type
- **Decision**: Structured fields: summary, signature, examples, doc_url. Separate from freeform pitfall/advice content.
- **Date**: 2026-04-08

### DECIDED: Incremental API doc population
- **Decision**: API docs built incrementally on first encounter. Only methods actually used get documented.
- **Date**: 2026-04-08

### DECIDED: User-initiated doc creation only
- **Decision**: The LLM never decides on its own to create API doc records. The user explicitly requests it and provides the doc home page URL.
- **Date**: 2026-04-08

### DECIDED: Firecrawl for doc retrieval
- **Decision**: The LLM uses the existing Firecrawl MCP to crawl and retrieve documentation pages in markdown format.
- **Date**: 2026-04-08

### DECIDED: Library onboarding workflow
- **Decision**: Three-phase: (1) Map — check for llms.txt, build TOC, identify types/enums/exceptions. (2) Seed — crawl tutorials, extract examples, create initial method records. (3) Incremental — add methods as encountered, user can say "read example X from page Y."
- **Date**: 2026-04-08

### DECIDED: User-directed example extraction
- **Decision**: User can tell the LLM "read example X from page Y" to extract specific examples.
- **Date**: 2026-04-08

### DECIDED: Cache page content in database
- **Decision**: Full Firecrawl markdown content cached in `doc_pages` table. Cache checked before re-crawling. Refresh command available for updates.
- **Rationale**: Avoids redundant Firecrawl calls, enables local search, instant retrieval. Staleness managed by user-controlled refresh.
- **Date**: 2026-04-08

### DECIDED: Progressive disclosure navigation tree
- **Decision**: During ingestion, raw doc pages are processed into a multi-level navigation tree. Each level is a self-contained summary that tells the LLM what it's looking at and how it fits in the larger picture. Levels get progressively more detailed, terminating in the cached page content. LLM-assisted ingestion produces the summaries at each level.
- **Rationale**: Raw web pages are structured for human scanning, not LLM navigation. The tree lets the LLM efficiently navigate to exactly what it needs without reading irrelevant content. Fits the token budget naturally — higher levels are cheap, full content only loaded when needed.
- **Date**: 2026-04-08

### DECIDED: Ingestion cost is acceptable
- **Decision**: LLM calls during ingestion (one per page to produce summaries) are acceptable and not a concern. The investment pays for itself through reduced token cost during later retrieval.
- **Rationale**: Bounded task — only a small set of libraries will be onboarded (sqlalchemy, pydantic, opentelemetry, temporal, etc.). Same libraries used repeatedly. Full re-scan only needed on major version release. Upfront cost is amortized across many future lookups.
- **Date**: 2026-04-08

### DECIDED: Doc lookup as separate agent thread
- **Decision**: Documentation lookup runs as a separate agent thread from the developer thread that needs the answer. The developer thread poses a question ("I'm seeing this error, help!"), the playbook lookup thread navigates the tree, performs multiple round-trips with the database as needed, and returns a consolidated answer. The lookup thread may use a cheaper LLM than the main thinking model for cost efficiency.
- **Rationale**: Finding the right answer may require several round-trips through the navigation tree — reading level 0, drilling into a section, reading examples, cross-referencing pitfalls. This navigation work shouldn't consume the main thread's context or use the expensive thinking model. The lookup thread's job is retrieval and synthesis, not novel reasoning. Separation keeps the developer thread focused and cost-efficient.
- **Date**: 2026-04-08

## Open Threads

### 1. Navigation tree depth and structure
- **Decided**: Progressive disclosure tree, LLM-produced summaries
- **Open**: How many levels? Proposed three (library → section → method/content), but large libraries might need four. Is this fixed or adaptive per library?
- **Open**: Data model for the tree — a `doc_nodes` table with parent references and depth? Or a simpler two-table model (library + doc_pages) with the tree structure encoded in a JSON column?

### 2. Lookup agent model routing
- **Decided**: Separate thread, potentially cheaper model
- **Open**: Which model tier? Forge uses `CapabilityTier` (CLASSIFICATION, SUMMARIZATION, etc.). Does the playbook service define its own tiers, or reuse the shared LLM provider's routing?
- **Open**: What's the interface between the developer thread and the lookup thread? The developer sends a question + context tags; the lookup thread returns a consolidated answer with citations (which entries/pages it drew from)?

### 3. Retrieval packing priority
- **Open**: Priority order within token budget across content types and modes

### 4. Eval storage and execution
- **Open**: Where stored, how executed, batch vs individual

### 5. `pbook` CLI command surface
- **Open**: Full list TBD. Includes: `skill-prompt`, `add`, `check-duplicate`, `list`, `query`, `update`, `test`, `review`, `approve`, `reject`, `lib init`, `lib refresh`, `doc add`, `doc read`

### 6. Extraction push schema
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
- Lookup thread failure handling (what if the cheaper model can't find the answer?)
