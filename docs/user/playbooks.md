# Playbooks

Playbooks are reusable lessons that Forge extracts from completed tasks and injects into future ones. They capture patterns like "always include type stubs for Pydantic models" or "validate input before type coercion" so the LLM avoids repeating the same mistakes.

Each playbook entry has:

- **Title** — Short description of the lesson.
- **Content** — Actionable guidance (2–4 sentences).
- **Tags** — Index tags for matching playbooks to future tasks.

## Creating Playbooks

There are two ways to create playbook entries: automatic extraction from completed workflows and manual addition via the CLI.

### Automatic Extraction

Playbooks are created by running the `forge extract` command, which reviews completed workflow results and uses an LLM to extract actionable lessons. Extraction also runs automatically on a schedule (every 4 hours by default) when the worker is running. The interval is configurable via the `--extraction-interval` flag on `forge worker`.

```bash
forge extract                                    # Last 24 hours, up to 10 runs
forge extract --limit 50 --since-hours 168       # Last week, up to 50 runs
forge extract --dry-run                          # Preview which runs would be processed
forge extract --json                             # Machine-readable output
```

Extraction skips runs that have already been processed. Each run is only extracted once.

### Manual Addition

Add entries manually with `forge playbooks add`. This is useful for recording techniques discovered outside of Forge workflows — for example, a query pattern found during an interactive debugging session.

```bash
forge playbooks add --file entry.json            # Add from JSON file (with LLM review)
forge playbooks add --schema                     # Print the PlaybookEntry JSON schema
```

The `--file` option accepts a JSON file matching the `PlaybookEntry` schema. Before saving, the entry is sent to an LLM for review. The reviewer checks clarity, correctness, completeness, and duplication against existing entries. If the reviewer rejects the entry, the reason is printed to stderr and nothing is saved. If approved, the reviewer may suggest improvements to the title, content, or tags — these are applied automatically.

#### PlaybookEntry JSON Schema

```json
{
  "properties": {
    "title": {
      "description": "Short descriptive title of the lesson.",
      "type": "string"
    },
    "content": {
      "description": "The actionable lesson or pattern.",
      "type": "string"
    },
    "tags": {
      "description": "Index tags: task type, domain, error pattern, etc.",
      "items": { "type": "string" },
      "type": "array"
    },
    "source_task_id": {
      "description": "Task ID this was extracted from.",
      "type": "string"
    },
    "source_workflow_id": {
      "description": "Workflow ID this was extracted from.",
      "default": "",
      "type": "string"
    }
  },
  "required": ["title", "content", "source_task_id"]
}
```

**Required fields:** `title`, `content`, `source_task_id`. The `tags` array defaults to empty. The `source_workflow_id` defaults to empty and is set to `"manual"` for manually added entries.

#### Example

```json
{
  "title": "Query successfully OCR'd documents from the store",
  "content": "To list all documents with successful OCR results, query the ocr_results table filtering on marked_for_removal = 0. Use get_db_path() and get_engine() from forge.store to connect.",
  "tags": ["domain:database", "pattern:success-pattern"],
  "source_task_id": "manual-ocr-query"
}
```

## Exporting Playbooks

Export playbook entries as `PlaybookEntry`-compatible JSON with `forge playbooks export`. The output is directly compatible with `forge playbooks add --file` for round-trip backup and sharing.

```bash
forge playbooks export                          # Export all playbooks as JSON to stdout
forge playbooks export --tag python             # Filter by tag
forge playbooks export --tag python --tag api   # Multiple tags (OR match)
forge playbooks export --task-id my-task        # Filter by source task ID
forge playbooks export --limit 10               # Limit entries
forge playbooks export -o backup.json           # Write to file instead of stdout
```

The export runs as a Temporal workflow (`ExportPlaybookWorkflow`) that fans out one activity per playbook row for parallel conversion. Each row is converted from DB format to a clean `PlaybookEntry` dict (tags are deserialized from JSON, DB-only fields like `id`, `created_at`, and `extraction_workflow_id` are dropped).

### Round-trip Example

```bash
forge playbooks export -o playbooks-backup.json
# ... later, on another machine or after a fresh install ...
# Re-import each entry (one file per entry, or iterate with jq):
cat playbooks-backup.json | jq -c '.[]' | while read entry; do
  echo "$entry" > /tmp/entry.json
  forge playbooks add --file /tmp/entry.json
done
```

### What Automatic Extraction Reviews

Extraction reads from the `runs` table in the observability database. Each row represents a completed Forge task workflow execution and contains:

| Column | Description |
|---|---|
| `task_id` | The logical task name (e.g., `add-forge-clean`) |
| `workflow_id` | The Temporal workflow ID |
| `status` | Final status of the run |
| `result_json` | Serialized JSON with the full task result |
| `created_at` | Timestamp |

The extraction LLM receives a summary of each run built from `result_json`, including:

- Task ID and workflow ID
- Final status (success/failure)
- Error messages (if any)
- Per-step results — step IDs, status, and errors
- Validation results — check name, pass/fail, and summary (e.g., ruff lint failures)
- Output files — list of files that were generated

Extraction learns from **task execution outcomes** — what succeeded, what failed validation, what required retries — not from the generated code itself or the LLM conversation content.

### What the Extraction LLM Looks For

The LLM analyzes these run summaries looking for:

- Context that was needed for success
- Validation failures and how they were resolved
- Retry patterns
- File organization insights

Each extracted entry is tagged automatically based on file extensions (e.g., `.py` maps to `python`) and description keywords (e.g., "test" maps to `test-writing`, "api" maps to `api`).

## How Playbooks Are Used

Forge injects relevant playbooks into task context automatically. When a task starts, Forge infers tags from the task's target files and description, queries the playbook store for matching entries, and includes them in the prompt.

Playbooks are also available during exploration rounds. The LLM can request playbooks on demand by tag (e.g., `python,api`) to pull in relevant lessons while analyzing the task.

Injection is best-effort — playbooks are subject to the token budget and are dropped if the budget is exceeded.

## Reading Playbooks

List playbook entries with the `forge playbooks` command:

```bash
forge playbooks                          # List recent playbooks (up to 20)
forge playbooks --tag python             # Filter by tag
forge playbooks --tag python --tag api   # Multiple tags (OR match)
forge playbooks --task-id my-task        # Filter by source task
forge playbooks --limit 5               # Limit results
forge playbooks --json                  # Full JSON with content
```

The default text output shows the title, tags, source task, and creation date:

```
Playbooks (2):

  [1] Include type stubs for Pydantic models
    Tags: python, code-generation, bug-fix
    Source: my-task-1 (workflow-abc-123)
    Created: 2025-02-26 14:30:45

  [2] Validate input before type coercion
    Tags: api, validation, bug-fix
    Source: my-task-2 (workflow-def-456)
    Created: 2025-02-26 12:00:00
```

Use `--json` to see the full content of each entry.

## Temporal Workflows

Playbook extraction runs as a Temporal workflow (`ForgeExtractionWorkflow`) with three activities executed in sequence:

```
ForgeExtractionWorkflow
├── fetch_extraction_input    (30s timeout)
│   Query the runs table for unprocessed workflow results.
│   Build extraction prompts from run summaries.
│   Return ExtractionInput with source_workflow_ids.
│
├── call_extraction_llm       (5m timeout, 60s heartbeat)
│   Send prompts to the SUMMARIZATION tier model.
│   Parse structured ExtractionResult (list of PlaybookEntry).
│   Record interaction in the observability store.
│
└── save_extraction_results   (30s timeout)
    Convert PlaybookEntry models to database rows.
    Bulk insert into the playbooks table.
```

The workflow short-circuits if `fetch_extraction_input` finds no unprocessed runs. The `call_extraction_llm` activity uses heartbeats so Temporal can detect stalled LLM calls.

Manual additions via `forge playbooks add` bypass Temporal entirely. They run synchronously in the CLI process: validate JSON, call the CLASSIFICATION tier model for review, and write directly to the playbooks table with `extraction_workflow_id="manual"`.

### Data Models

Key Pydantic models used by the extraction workflow:

| Model | Purpose |
|---|---|
| `ExtractionWorkflowInput` | Workflow input: `limit`, `since_hours`, `model_routing` |
| `ExtractionInput` | Activity payload: system/user prompts, source workflow IDs |
| `ExtractionResult` | LLM structured output: list of `PlaybookEntry` + summary |
| `ExtractionCallResult` | LLM response metadata: tokens, latency, model name |
| `FetchExtractionInput` | `fetch_extraction_input` params: `limit`, `since_hours` |
| `SaveExtractionInput` | `save_extraction_results` params: entries, workflow IDs |
| `PlaybookEntry` | Single lesson: title, content, tags, source references |
| `PlaybookReviewResult` | Manual add review output: approved, suggestions |

### Scheduling

When the Forge worker is running, extraction is scheduled automatically at a configurable interval (default: every 4 hours). The `forge extract` CLI command triggers an on-demand extraction by starting a `ForgeExtractionWorkflow` on the Temporal server.

## Storage

Playbooks are stored in the observability database at `$XDG_STATE_HOME/forge/forge.db` (default `~/.local/state/forge/forge.db`). Override the path with the `FORGE_DB_PATH` environment variable.

The `playbooks` table schema:

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Auto-incrementing primary key |
| `title` | TEXT | Short descriptive title |
| `content` | TEXT | Actionable lesson content |
| `tags_json` | TEXT | JSON array of tag strings |
| `source_task_id` | TEXT | Task that produced this lesson |
| `source_workflow_id` | TEXT | Temporal workflow ID (or empty) |
| `extraction_workflow_id` | TEXT | Extraction workflow ID, or `"manual"` |
| `created_at` | TIMESTAMP | Row creation time |
