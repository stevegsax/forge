# Phase 6: Knowledge Extraction

## Goal

Extract structured lessons from completed workflow results and inject them as playbook entries into future task contexts. The extraction workflow runs independently from task execution, producing tagged entries that are retrieved by relevance at context assembly time.

The deliverable: run `forge extract` to process completed workflow results, then verify that subsequent tasks include relevant playbook entries in their assembled context via `forge playbooks`.

## Problem Statement

Phases 1-5 execute tasks and store full interaction data in SQLite, but lessons from completed work are discarded. When a task fails due to a missing type stub, a lint rule, or a context gap, the same failure can recur in future tasks. There is no mechanism to capture "always include type stubs for Pydantic models" and surface it when a similar task runs later.

## Scope

**In scope:**

- Playbook storage in the existing SQLite observability store (new `playbooks` table)
- Extraction workflow as an independent Temporal workflow with three activities
- Tag-based playbook retrieval using deterministic tag inference
- Playbook injection into context assembly at priority 5
- `forge extract` CLI command to trigger extraction
- `forge playbooks` CLI command to list and inspect entries

**Out of scope:**

- Semantic similarity search (vector embeddings). Tag overlap is sufficient for v1.
- Automatic extraction scheduling (cron or Temporal schedules). Manual trigger via CLI.
- Playbook deduplication or merging. Flat entries with potential overlap are acceptable.
- Playbook editing or deletion via CLI. Direct SQLite access for maintenance.
- Cross-project playbook sharing. Single database per deployment.

## Architecture

### Extraction Workflow (`src/forge/extraction_workflow.py`)

A Temporal workflow (`ForgeExtractionWorkflow`) with three activities:

1. **`fetch_extraction_input`** — Queries unextracted runs from the store, builds the extraction prompt.
2. **`call_extraction_llm`** — Calls a pydantic-ai agent with structured output (`ExtractionResult`).
3. **`save_extraction_results`** — Writes extracted playbook entries to the store.

The workflow short-circuits if no unextracted runs are found, and skips the save step if the LLM returns no entries.

### Playbook Storage

New `playbooks` table in the existing SQLite database (`$XDG_STATE_HOME/forge/forge.db`):

```
class Playbook(Base):
    __tablename__ = "playbooks"

    id: int (PK, autoincrement)
    title: str (NOT NULL)
    content: str (NOT NULL)
    tags_json: str (NOT NULL)           # JSON array of tag strings
    source_task_id: str (NOT NULL, indexed)
    source_workflow_id: str (NOT NULL)
    extraction_workflow_id: str (NOT NULL)
    created_at: datetime (server default: now UTC)
```

Tag-based retrieval uses SQLite's `json_each()` to unnest the `tags_json` array and match against input tags.

### Playbook Injection (`src/forge/activities/context.py`)

During context assembly, playbooks are loaded and injected as `ContextItem` objects with `Representation.PLAYBOOK` at priority 5. The token budget packer determines whether playbooks fit alongside other context items. If the budget is full, playbooks are dropped (they are an optimization, not a correctness requirement per D13).

### Tag Inference

Tags are inferred deterministically from task metadata:

- **File extensions:** `.py` -> `python`, `.ts`/`.tsx` -> `typescript`
- **Description keywords:** `test` -> `test-writing`, `refactor` -> `refactoring`, `api` -> `api`, `bug`/`fix` -> `bug-fix`
- **Default:** `code-generation` when no matches

The same inference runs during extraction (to tag entries) and during retrieval (to query).

## Data Models

New models in `src/forge/models.py`:

- `PlaybookEntry` — A structured lesson with title, content, tags, and source references
- `ExtractionResult` — Structured LLM output containing entries and a summary
- `FetchExtractionInput` — Input to the fetch activity (limit, since_hours)
- `ExtractionInput` — Output of fetch, input to the LLM call (prompts + source IDs)
- `ExtractionCallResult` — LLM call output with usage statistics
- `SaveExtractionInput` — Input to the save activity (entries + metadata)
- `ExtractionWorkflowInput` — Workflow input (limit, since_hours)
- `ExtractionWorkflowResult` — Workflow output (entries_created, source_workflow_ids)

## Project Structure

New and modified files:

```
src/forge/
├── models.py                        # Modified: Phase 6 data models
├── store.py                         # Modified: Playbook ORM, store functions
├── extraction_workflow.py           # New: Temporal extraction workflow
├── code_intel/
│   └── budget.py                    # Modified: PLAYBOOK representation
├── alembic/
│   └── versions/
│       └── 002_playbooks.py         # New: playbooks table migration
├── activities/
│   ├── __init__.py                  # Modified: export extraction activities
│   ├── extraction.py                # New: extraction activities
│   └── context.py                   # Modified: playbook injection
├── worker.py                        # Modified: register extraction workflow
└── cli.py                           # Modified: extract + playbooks commands
```

## Key Design Decisions

- **D43: Playbooks as Flat Tagged Entries.** Flat rows indexed by JSON tag arrays. No hierarchy.
- **D44: Extraction as a Temporal Workflow.** Three activities following D2 (universal workflow step).
- **D45: Relevance by Tag Overlap.** Deterministic tag matching for retrieval, not semantic similarity.
- **D46: Playbooks Share the Observability Store.** Same SQLite database, new Alembic migration.
- **D47: PLAYBOOK Representation Type.** New enum value distinguishes playbook items from repo map items at priority 5.

## CLI Usage

### `forge extract`

Trigger knowledge extraction from completed runs:

```bash
forge extract                          # Extract from last 24h, up to 10 runs
forge extract --limit 50 --since-hours 168  # Last week, up to 50 runs
forge extract --dry-run                # List unextracted runs without processing
forge extract --json                   # Machine-readable JSON output
```

### `forge playbooks`

List and inspect playbook entries:

```bash
forge playbooks                        # List recent playbooks
forge playbooks --tag python           # Filter by tag
forge playbooks --tag python --tag api # Multiple tags (OR match)
forge playbooks --task-id my-task      # Filter by source task
forge playbooks --json                 # Machine-readable JSON output
```

## Edge Cases

- **No unextracted runs:** Workflow short-circuits, returns 0 entries.
- **LLM returns empty entries:** Save step is skipped.
- **Store unavailable:** Playbook loading returns empty list (D42 pattern). Context assembly proceeds without playbooks.
- **Budget exceeded:** Playbook items at priority 5 are dropped by the token packer, preserving higher-priority context.
- **Empty `FORGE_DB_PATH`:** Store is disabled. Extraction and playbook commands report "no store available."
- **Missing `source_workflow_id` from LLM:** Falls back to first source workflow ID from the input.

## Implementation Order

1. Data models (`models.py`)
2. Store layer + migration (`store.py`, `002_playbooks.py`)
3. Representation enum (`budget.py`)
4. Extraction activities (`activities/extraction.py`)
5. Extraction workflow (`extraction_workflow.py`)
6. Playbook injection (`activities/context.py`)
7. Worker registration (`worker.py`, `activities/__init__.py`)
8. CLI commands (`cli.py`)
9. Tests
10. Documentation

## Definition of Done

Phase 6 is complete when:

- `forge extract` processes unextracted runs and creates playbook entries
- `forge extract --dry-run` lists unextracted runs without processing
- `forge playbooks` lists entries, filterable by tag and task ID
- Playbook entries are injected into context assembly for matching tasks
- Playbooks respect the token budget (dropped when budget is exceeded)
- Context assembly gracefully handles missing store (empty playbooks)
- Alembic migration `002` creates the `playbooks` table
- All Phase 1-5 tests continue to pass
- New tests cover: extraction activities, extraction workflow, playbook injection, store CRUD, CLI commands
