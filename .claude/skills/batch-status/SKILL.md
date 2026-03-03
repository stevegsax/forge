---
name: batch-status
description: Check the status of submitted batch jobs and OCR results. Use when the user asks about batch job status, OCR job progress, which jobs succeeded or failed, or downloaded results.
allowed-tools: Bash(sqlite3 *)
---

Query the Forge observability database to show batch job and OCR result status.

## Database location

Resolve the database path using the same precedence as Forge:

1. `$FORGE_DB_PATH` (if set and non-empty)
2. `$XDG_STATE_HOME/forge/forge.db`
3. `~/.local/state/forge/forge.db`

Set the resolved path as `DB` for use in queries below:

```bash
DB="${FORGE_DB_PATH:-${XDG_STATE_HOME:-$HOME/.local/state}/forge/forge.db}"
```

If the database file does not exist, tell the user no Forge database was found.

## What to do

### Default (no arguments): show recent batch jobs and OCR results

Run both queries and present the results together.

**Batch jobs** (most recent 20):

```bash
sqlite3 -header -column "$DB" "
SELECT id, batch_id, provider, status, error_message,
       datetime(created_at) AS created, datetime(updated_at) AS updated
FROM batch_jobs
ORDER BY updated_at DESC
LIMIT 20;
"
```

**OCR results** (most recent 20):

```bash
sqlite3 -header -column "$DB" "
SELECT document_id, file_path, page_count, model_name,
       input_tokens, output_tokens, batch_id,
       datetime(created_at) AS created
FROM ocr_results
ORDER BY created_at DESC
LIMIT 20;
"
```

### Summarise the results

After running the queries, provide a brief summary:

- Total batch jobs by status (submitted, succeeded, failed, expired, etc.)
- How many OCR results have been stored
- Any failed or errored jobs with their error messages
- Whether pending jobs exist that haven't completed yet

### Useful follow-up queries

If the user asks for more detail, use these:

**Batch jobs by status counts:**

```bash
sqlite3 -header -column "$DB" "
SELECT provider, status, COUNT(*) AS count
FROM batch_jobs
GROUP BY provider, status
ORDER BY provider, status;
"
```

**Failed batch jobs only:**

```bash
sqlite3 -header -column "$DB" "
SELECT id, batch_id, provider, error_message,
       datetime(updated_at) AS updated
FROM batch_jobs
WHERE status NOT IN ('submitted', 'succeeded')
ORDER BY updated_at DESC;
"
```

**OCR results for a specific batch:**

```bash
sqlite3 -header -column "$DB" "
SELECT document_id, file_path, page_count, length(text) AS text_chars,
       input_tokens, output_tokens
FROM ocr_results
WHERE batch_id = '<batch_id>';
"
```

**Batch jobs still pending (submitted but not completed):**

```bash
sqlite3 -header -column "$DB" "
SELECT id, batch_id, provider,
       datetime(created_at) AS created,
       ROUND((julianday('now') - julianday(created_at)) * 24, 1) AS hours_ago
FROM batch_jobs
WHERE status = 'submitted'
ORDER BY created_at;
"
```

**Cross-reference: batch jobs with their OCR results (if any):**

```bash
sqlite3 -header -column "$DB" "
SELECT b.id, b.batch_id, b.provider, b.status,
       CASE WHEN o.document_id IS NOT NULL THEN 'yes' ELSE 'no' END AS has_ocr_result,
       o.document_id, o.file_path
FROM batch_jobs b
LEFT JOIN ocr_results o ON b.batch_id = o.batch_id
ORDER BY b.updated_at DESC
LIMIT 30;
"
```
