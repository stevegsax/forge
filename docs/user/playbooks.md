# Playbooks

Playbooks are reusable lessons that Forge extracts from completed tasks and injects into future ones. They capture patterns like "always include type stubs for Pydantic models" or "validate input before type coercion" so the LLM avoids repeating the same mistakes.

Each playbook entry has:

- **Title** — Short description of the lesson.
- **Content** — Actionable guidance (2–4 sentences).
- **Tags** — Index tags for matching playbooks to future tasks.

## Creating Playbooks

Playbooks are created by running the `forge extract` command, which reviews completed workflow results and uses an LLM to extract actionable lessons.

```bash
forge extract                                    # Last 24 hours, up to 10 runs
forge extract --limit 50 --since-hours 168       # Last week, up to 50 runs
forge extract --dry-run                          # Preview which runs would be processed
forge extract --json                             # Machine-readable output
```

Extraction skips runs that have already been processed. Each run is only extracted once.

The LLM looks for:

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

## Storage

Playbooks are stored in the observability database at `$XDG_STATE_HOME/forge/forge.db` (default `~/.local/state/forge/forge.db`). Override the path with the `FORGE_DB_PATH` environment variable.
