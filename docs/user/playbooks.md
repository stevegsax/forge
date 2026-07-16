# Playbooks

Playbooks are reusable lessons captured in Forge and injected into future tasks. They capture patterns like "always include type stubs for Pydantic models" or "validate input before type coercion" so the LLM avoids repeating the same mistakes.

Each playbook entry has:

- **Title** — Short description of the lesson.
- **Content** — Actionable guidance (2–4 sentences).
- **Tags** — Index tags for matching playbooks to future tasks.

## Creating Playbooks

Playbook entries are added with the `forge playbooks add` CLI command.

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

```text
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

## Storage

Playbooks are stored in the observability database configured by the required `FORGE_DB_URL` environment variable — a `sqlite:///<path>` URL for local use (e.g. `sqlite:///~/.local/state/forge/forge.db`) or a `postgresql+psycopg2://...` URL in production.

The `playbooks` table schema:

| Column | Type | Description |
| --- | --- | --- |
| `id` | INTEGER | Auto-incrementing primary key |
| `title` | TEXT | Short descriptive title |
| `content` | TEXT | Actionable lesson content |
| `tags_json` | TEXT | JSON array of tag strings |
| `source_task_id` | TEXT | Task that produced this lesson |
| `source_workflow_id` | TEXT | Temporal workflow ID (or empty) |
| `extraction_workflow_id` | TEXT | Extraction workflow ID, or `"manual"` |
| `created_at` | TIMESTAMP | Row creation time |
