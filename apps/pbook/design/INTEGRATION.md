# Client Integration

pbook is designed to be consumed by other projects. Clients interact with pbook through three interfaces: the Temporal workflow API, the CLI, and (planned) a Claude Code SKILL.md.

## Temporal workflow API

Clients call pbook workflows via cross-queue workflow execution on `pbook-task-queue`. The pbook worker must be running.

### Retrieving playbook entries

Execute `RetrievalWorkflow` with a `RetrievalInput`:

```python
from pbook.models import RetrievalInput, RetrievalMode

result = await client.execute_workflow(
    "RetrievalWorkflow",
    RetrievalInput(
        tags=["lang:python", "lib:sqlalchemy"],
        mode=RetrievalMode.CREATE,
        token_budget=5000,
    ),
    id=f"pbook-retrieve-{uuid4().hex[:8]}",
    task_queue="pbook-task-queue",
)

# result.entries: list[dict] — packed within token budget
# result.token_count: int
# result.total_candidates: int
```

Use `mode=RetrievalMode.CREATE` when generating new code (boosts general knowledge and API docs). Use `mode=RetrievalMode.FIX` when debugging (boosts project-specific pitfalls).

Set `approved_only=True` to exclude LLM-extracted entries that haven't been manually reviewed.

### Pushing experience for extraction

Execute `ExtractionWorkflow` with experience data:

```python
import json

result = await client.execute_workflow(
    "ExtractionWorkflow",
    json.dumps({
        "experiences": [
            {
                "project": "my-project",
                "problem": "Base64 data included a data URI prefix",
                "resolution": "Strip the prefix before decoding",
                "context": "Mistral OCR API response",
            }
        ],
        "project": "my-project",
    }),
    id=f"pbook-extract-{uuid4().hex[:8]}",
    task_queue="pbook-task-queue",
)

# result["entries_created"]: int
```

The extraction LLM will only produce entries for situations that are both unexpected and actionable. It may return zero entries if nothing meets the quality bar.

### Submitting a manual entry

Execute `ManualEntryWorkflow` with a raw JSON `PlaybookEntry`:

```python
result = await client.execute_workflow(
    "ManualEntryWorkflow",
    json.dumps({
        "title": "Use dispose() in test fixtures",
        "content": "SQLAlchemy caches connections by URL.",
        "tags": ["lib:sqlalchemy", "domain:testing"],
    }),
    id=f"pbook-manual-{uuid4().hex[:8]}",
    task_queue="pbook-task-queue",
)

# result["approved"]: bool
# result["rejection_reason"]: str (if rejected)
# result["entry"]: dict (the final entry, with suggestions applied)
```

The review LLM may reject the entry if it's too generic, vague, or duplicates an existing entry. It may also suggest improvements to the title, content, or tags.

## CLI

All operations are available via the `pbook` command. See [CLI.md](CLI.md) for the full reference. Key commands for integration:

```bash
# Add an entry (direct write, no LLM review)
pbook add --file entry.json

# Push experience for LLM extraction
pbook push --file experience.json

# Query entries by tag
pbook list --tag lang:python --tag lib:sqlalchemy --json

# Review and approve extracted entries
pbook review
pbook approve 42
```

## Programmatic (library) access

pbook's store functions can be called directly as a Python library, bypassing Temporal:

```python
from pbook.store import get_db_path, get_engine, get_entries_by_tags, run_migrations

db_path = get_db_path()
run_migrations(db_path)
engine = get_engine(db_path)

entries = get_entries_by_tags(
    engine,
    ["lang:python", "lib:sqlalchemy"],
    limit=10,
    approved_only=True,
)
```

This is useful for lightweight queries that don't need workflow orchestration. The retrieval ranking logic (`rank_and_pack`) is a pure function that can be called directly:

```python
from pbook.activities.retrieval import rank_and_pack
from pbook.models import RetrievalMode

packed, token_count = rank_and_pack(
    candidates=entries,
    query_tags=["lang:python", "lib:sqlalchemy"],
    mode=RetrievalMode.CREATE,
    token_budget=5000,
)
```

## Tag inference

Clients can use pbook's tag inference to determine which tags to query with:

```python
from pbook.tags import infer_tags_from_context

tags = infer_tags_from_context(
    file_extensions=[".py"],
    description="Fix the database migration test",
)
# Returns: ["domain:bug-fix", "domain:database", "domain:migration", "domain:testing", "lang:python"]
```

## LLM provider setup

When using pbook through Temporal (the primary path), the worker handles LLM provider registration automatically. When using pbook as a library with LLM-dependent operations (extraction, review), register a provider first:

```python
from sax_llm import get_provider
from pbook.llm import set_provider

provider = get_provider("anthropic:claude-haiku-4-5-20251001")
set_provider(provider)
```

See the sax-llm project documentation for provider configuration and available models.

## Planned: SKILL.md interface

A Claude Code SKILL.md is planned that will provide an interactive interface for adding, querying, and reviewing playbook entries. The skill will run as a sub-agent using the ReAct pattern, calling `pbook` CLI commands and receiving server-provided instructions via `pbook skill-prompt`. This is not yet implemented.
