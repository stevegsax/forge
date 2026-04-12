# Forge Run Extraction Reference

Tabular reference for Forge's own playbook storage: the `playbooks` table schema in `forge.db`, the `ForgeExtractionWorkflow` data models, the tag inference rules, the CLI commands that manage forge-side playbooks, and the injection behavior inside context assembly. This reference is scoped to forge's self-learning loop only. It does not describe pbook's `entries` table or any aspect of the transcript ingestion pipeline — for those, see the [Transcript Ingestion Reference](transcript-ingestion.md).

For background on how Forge's self-learning loop is designed, see [Forge Run Extraction](../explanation/forge-run-extraction.md). For step-by-step instructions, see [How to Manage Playbooks](../howto/manage-playbooks.md). For the cross-cutting comparison between this pipeline and transcript ingestion, see [Learning Loops](../explanation/learning-loops.md).

## Playbook table schema

The `playbooks` table resides in the SQLite observability store at `$XDG_STATE_HOME/forge/forge.db`.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | integer | primary key, autoincrement | Row identifier. |
| `title` | text | not null | Short title describing the lesson. |
| `content` | text | not null | Full text of the playbook entry. |
| `tags_json` | text | not null | JSON array of tag strings, e.g. `["python", "test-writing"]`. |
| `source_task_id` | text | not null, indexed | Task ID of the run this entry was extracted from. |
| `source_workflow_id` | text | not null | Temporal workflow ID of the source run. |
| `extraction_workflow_id` | text | not null | Temporal workflow ID of the extraction run that produced this entry. |
| `created_at` | datetime | server default: now UTC | Timestamp when the entry was written. |

Tag-based retrieval uses SQLite's `json_each()` to unnest `tags_json` and match against a query tag set. A playbook is retrieved for a task if any of its tags overlap with the task's inferred tags.

## Tag inference rules

The same tag inference logic runs during extraction (tagging new entries) and during retrieval (computing query tags for the current task).

### File extension mapping

| Extension | Tag |
|-----------|-----|
| `.py` | `python` |
| `.ts`, `.tsx` | `typescript` |
| `.js`, `.jsx` | `javascript` |

### Description keyword mapping

| Keyword in task description | Tag |
|-----------------------------|-----|
| `test` | `test-writing` |
| `refactor` | `refactoring` |
| `api` | `api` |
| `database` | `database` |
| `migration` | `migration` |
| `cli` | `cli` |
| `validate` | `validation` |
| `bug`, `fix` | `bug-fix` |

### Default

If no file extension or keyword rules match, the tag `code-generation` is applied.

## Data models

### `PlaybookEntry`

Represents a single structured lesson, used as input to the save activity and as context item content.

| Field | Type | Description |
|-------|------|-------------|
| `title` | str | Short title for the entry. |
| `content` | str | Full lesson text. |
| `tags` | list[str] | Tags inferred or assigned at extraction time. |
| `source_task_id` | str | Task ID the entry was derived from. |
| `source_workflow_id` | str | Workflow ID the entry was derived from. |

### `ExtractionResult`

Structured LLM output from the `call_extraction_llm` activity.

| Field | Type | Description |
|-------|------|-------------|
| `entries` | list[PlaybookEntry] | Extracted playbook entries. May be empty. |
| `summary` | str | Brief summary of what was extracted and why. |

### `ExtractionWorkflowInput`

Input to `ForgeExtractionWorkflow`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | 50 | Maximum number of unextracted runs to process per invocation. |
| `since_hours` | int | 24 | Lookback window in hours; only runs within this window are fetched. |

### `ExtractionWorkflowResult`

Output from `ForgeExtractionWorkflow`.

| Field | Type | Description |
|-------|------|-------------|
| `entries_created` | int | Number of playbook entries written to the store. |
| `source_workflow_ids` | list[str] | Workflow IDs of runs that were processed. |

## CLI commands

### `forge extract`

Triggers `ForgeExtractionWorkflow` to process unextracted runs.

```
forge extract [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--limit N` | 50 | Maximum number of runs to process. |
| `--since-hours N` | 24 | Lookback window in hours. |

Prints the number of entries created and the workflow IDs processed. Returns without output if there are no unextracted runs.

### `forge playbooks list`

Lists stored playbook entries, optionally filtered by tag.

```
forge playbooks list [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--tag TAG` | (none) | Filter entries to those containing this tag. Repeatable. |
| `--limit N` | 20 | Maximum entries to display. |

Output columns: `id`, `title`, `tags`, `created_at`.

### `forge playbooks show`

Displays the full content of a single playbook entry.

```
forge playbooks show ID
```

Prints `title`, `tags_json`, `content`, `source_task_id`, `source_workflow_id`, and `created_at`.

### `forge playbooks add`

Manually inserts a playbook entry into the store.

```
forge playbooks add --title TITLE --content CONTENT [--tag TAG ...]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--title TITLE` | yes | Entry title. |
| `--content CONTENT` | yes | Entry content. |
| `--tag TAG` | no (repeatable) | Tag to apply. May be specified multiple times. |

Sets `source_task_id` and `source_workflow_id` to `manual`.

## Playbook injection

During context assembly, the `assemble_context` activity retrieves playbooks and injects them as context items.

| Property | Value |
|----------|-------|
| Representation type | `Representation.PLAYBOOK` |
| Priority level | 5 (out of 6) |
| Retrieval method | Tag overlap match against task inferred tags |
| Budget behavior | Dropped first when token budget is tight |

Priority 5 places playbooks below deterministic analysis results (priority 4) and above broader project context (priority 6). For the full priority table and token budget algorithm, see the [Context Assembly Reference](context-assembly.md).
