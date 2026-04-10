# Data Model

## Database schema

Single table, SQLite with WAL mode. Managed by Alembic (migration `001_initial`).

```sql
CREATE TABLE entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    tags_json   TEXT    NOT NULL,       -- JSON array: ["lang:python", "lib:sqlalchemy"]
    entry_type  TEXT    NOT NULL DEFAULT 'curated',  -- pitfall | curated | api_doc
    source_project  TEXT NOT NULL DEFAULT '',
    source_task_id  TEXT NOT NULL DEFAULT '',
    needs_review    INTEGER NOT NULL DEFAULT 0,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_entries_source_project ON entries(source_project);
CREATE INDEX ix_entries_entry_type ON entries(entry_type);
```

Tag queries use SQLite's `json_each()` to unnest the `tags_json` array and match against input tags with OR semantics.

## Entry types

### Pitfall (`entry_type = "pitfall"`)

Extracted from experience by the LLM. Represents a specific, unexpected situation where the default approach did not work.

```json
{
    "title": "Mistral OCR returns base64 with data URI prefix",
    "content": "The image_base64 field includes a 'data:image/jpeg;base64,' prefix. Strip it before calling base64.b64decode() or the prefix gets decoded into garbage bytes prepended to the image data.",
    "tags": ["lib:mistral", "domain:ocr", "project:forge"],
    "entry_type": "pitfall",
    "source_project": "forge",
    "needs_review": true
}
```

### Curated advice (`entry_type = "curated"`)

Submitted by a human, reviewed by the LLM. General best practices or library-specific guidance.

```json
{
    "title": "Use from __future__ import annotations with Pydantic",
    "content": "When using Pydantic v2 with forward references, add 'from __future__ import annotations' at the top of every file. This enables PEP 604 union syntax (X | Y) and avoids NameError on forward-referenced types.",
    "tags": ["lang:python", "lib:pydantic"],
    "entry_type": "curated",
    "needs_review": false
}
```

### API doc record (`entry_type = "api_doc"`)

Library documentation with method signature and working examples. The `content` field holds a serialized `ApiDocRecord`:

```json
{
    "title": "sqlalchemy.create_engine",
    "content": "{\"library\": \"sqlalchemy\", \"method\": \"sqlalchemy.create_engine\", \"summary\": \"Create a new Engine instance.\", \"signature\": \"def create_engine(url: str, *, echo: bool = False, pool_size: int = 5) -> Engine\", \"examples\": [\"engine = create_engine('sqlite:///app.db', echo=True)\"], \"doc_url\": \"https://docs.sqlalchemy.org/en/20/core/engines.html\"}",
    "tags": ["lib:sqlalchemy"],
    "entry_type": "api_doc",
    "needs_review": false
}
```

## Pydantic models

### PlaybookEntry (write model)

Used for all three content types. Validated on ingestion.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `title` | str | required | Short descriptive name |
| `content` | str | required | Lesson text or serialized ApiDocRecord |
| `tags` | list[str] | `[]` | Namespaced tags |
| `entry_type` | EntryType | `curated` | `pitfall`, `curated`, or `api_doc` |
| `source_project` | str | `""` | Which project submitted this |
| `source_task_id` | str | `""` | Client's task identifier |
| `needs_review` | bool | `False` | Set `True` for LLM-extracted entries |

## Tag system


What clients send when pushing experience data for LLM extraction.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `project` | str | required | Source project |
| `problem` | str | required | What went wrong |
| `resolution` | str | required | What fixed it |
| `context` | str | `""` | Code, errors, environment details |
| `metadata` | dict | `{}` | Arbitrary key-value pairs |

No `outcome` field — the extraction LLM determines what is noteworthy from the problem/resolution narrative.

## Tag system

Tags use `namespace:value` format. Five namespaces in two tiers:

**General** (cross-project, human-curated):

- `lang:` — programming language. Inferred from file extensions: `.py` → `python`, `.ts`/`.tsx` → `typescript`, `.go` → `go`, `.rs` → `rust`, `.java` → `java`, `.rb` → `ruby`
- `lib:` — library or framework. Free-form values: `sqlalchemy`, `pydantic`, `temporal`, `anthropic`
- `domain:` — problem domain. Inferred from description keywords: `test` → `testing`, `refactor` → `refactoring`, `api`, `database`, `migration`, `cli`, `validate` → `validation`, `bug`/`fix` → `bug-fix`

**Extracted** (project-specific, LLM-produced):

- `project:` — source project identifier
- `pattern:` — lesson type: `failure-pattern`, `retry-pattern`, `success-pattern`

### Validation

- `parse_tag(tag)` raises `ValueError` if the tag lacks a colon, has an empty value, or uses an unrecognized namespace.
- `validate_tags(tags)` returns a list of error messages (empty list means all valid).
- The CLI rejects entries with invalid tags on `add` and `update`.

### Inference

`infer_tags_from_context(file_extensions, description)` derives tags automatically from file extensions and description keywords. Returns a sorted, deduplicated list.
 `success-pattern`

### Validation

- `parse_tag(tag)` raises `ValueError` if the tag lacks a colon, has an empty value, or uses an unrecognized namespace.
- `validate_tags(tags)` returns a list of error messages (empty list means all valid).
- The CLI rejects entries with invalid tags on `add` and `update`.

### Inference

`infer_tags_from_context(file_extensions, description)` derives tags automatically from file extensions and description keywords. Returns a sorted, deduplicated list.
