---
name: forge-playbooks
description: Manage Forge playbook entries — list, add, and export. Use when the user asks about playbooks, wants to view stored lessons, add new entries, export for backup/sharing, or check what playbooks exist.
allowed-tools: Bash(forge playbooks *)
---

Use the `forge playbooks` CLI to manage playbook entries in the Forge knowledge base. Playbooks are reusable lessons extracted from completed tasks or added manually.

## Commands

### List playbooks

```bash
forge playbooks                          # List recent playbooks (up to 20)
forge playbooks --tag python             # Filter by tag
forge playbooks --tag python --tag api   # Multiple tags (OR match)
forge playbooks --task-id my-task        # Filter by source task
forge playbooks --limit 5               # Limit results
forge playbooks --json                   # Full JSON with DB fields
```

### Add a playbook entry

```bash
forge playbooks add --file entry.json    # Add from JSON file (with LLM review)
forge playbooks add --schema             # Print the PlaybookEntry JSON schema
```

The `--file` option accepts a JSON file matching the `PlaybookEntry` schema. The entry is sent to an LLM for review before saving. If rejected, the reason is printed to stderr.

#### PlaybookEntry JSON format

```json
{
  "title": "Short descriptive title",
  "content": "The actionable lesson or pattern.",
  "tags": ["tag1", "tag2"],
  "source_task_id": "task-id-or-manual-label"
}
```

Required fields: `title`, `content`, `source_task_id`.

### Export playbooks

```bash
forge playbooks export                          # Export all as JSON to stdout
forge playbooks export --tag python             # Filter by tag
forge playbooks export --tag python --tag api   # Multiple tags (OR match)
forge playbooks export --task-id my-task        # Filter by source task ID
forge playbooks export --limit 10               # Limit entries
forge playbooks export -o backup.json           # Write to file instead of stdout
```

Export produces `PlaybookEntry`-compatible JSON (an array of objects). The output is directly compatible with `forge playbooks add --file` for round-trip import.

#### Round-trip example

```bash
forge playbooks export -o backup.json
# Re-import each entry:
cat backup.json | jq -c '.[]' | while read entry; do
  echo "$entry" > /tmp/entry.json
  forge playbooks add --file /tmp/entry.json
done
```

## Prerequisites

- The Forge worker must be running (`forge worker`) for `add` and `export` commands, which execute as Temporal workflows.
- The `list` command reads directly from the database and does not require the worker.

## Interpreting results

- **List output** shows title, tags, source task, and creation date for each entry.
- **Export output** is a JSON array of clean `PlaybookEntry` dicts (no DB-only fields like `id`, `created_at`, or `extraction_workflow_id`).
- **Add** prints the saved entry on success, or the rejection reason on failure.
