# CLI Reference

Entry point: `pbook`

## Worker

```
pbook worker [--temporal-address localhost:7233]
```

Start the pbook Temporal worker. Connects to the Temporal server, registers all workflows and activities, and runs until interrupted. Registers the Anthropic LLM provider via sax-llm at startup.

## Entry management

### List entries

```
pbook list [--tag TAG]... [--type TYPE] [--project PROJECT] [--limit 20] [--json]
```

List playbook entries ordered by creation time (newest first). Tags use OR matching — an entry matching any specified tag is included. `--type` filters by entry type (`pitfall`, `curated`, `api_doc`). `--project` filters by source project. `--json` outputs machine-readable JSON.

### Get a single entry

```
pbook get ENTRY_ID [--json]
```

Fetch and display a single entry by its database ID. Exits with error if not found.

### Add an entry

```
pbook add --file FILE [--schema]
```

Add a playbook entry from a JSON file containing a `PlaybookEntry` object. Tags are validated against the namespace vocabulary — invalid tags cause rejection. Use `--schema` to print the JSON schema without adding anything.

Example entry file:

```json
{
    "title": "Use dispose() in SQLAlchemy test fixtures",
    "content": "SQLAlchemy's create_engine caches connections by URL. Call engine.dispose() in test teardown to prevent connection leaks across tests.",
    "tags": ["lib:sqlalchemy", "domain:testing"],
    "entry_type": "curated"
}
```

### Update an entry

```
pbook update ENTRY_ID --file FILE
```

Update fields of an existing entry. The JSON file contains only the fields to change. Tags in the update are validated.

### Approve an entry

```
pbook approve ENTRY_ID
```

Clear the `needs_review` flag on an entry. Use after reviewing an LLM-extracted entry.

### Reject an entry

```
pbook reject ENTRY_ID
```

Delete an entry from the database. Use to remove entries that fail manual review.

## Duplicate checking

```
pbook check-duplicate --title TITLE [--tag TAG]...
```

Search for existing entries with similar titles (case-insensitive substring match). If tags are provided, results are sorted by tag overlap (most overlapping first). Returns up to 10 matches.

## Experience push

```
pbook push --file FILE [--temporal-address localhost:7233]
```

Push experience data for LLM extraction. The file contains a `PushExperienceInput` object (or an array of them). Submits an `ExtractionWorkflow` to the pbook Temporal worker and waits for completion.

Example experience file:

```json
{
    "project": "my-project",
    "problem": "SQLite check_same_thread error in Temporal activities",
    "resolution": "Pass check_same_thread=False to create_engine because Temporal activities run on different threads",
    "context": "Using SQLAlchemy with SQLite inside async Temporal activities"
}
```

## Review management

```
pbook review [--limit 20]
```

List entries that have `needs_review=True`. Use `pbook approve ID` or `pbook reject ID` to process them.

## Database

```
pbook migrate
```

Run Alembic migrations to bring the database schema up to date. This is also done automatically by most commands on first use.
