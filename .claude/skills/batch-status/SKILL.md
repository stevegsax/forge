---
name: batch-status
description: Check the status of submitted batch jobs and OCR results. Use when the user asks about batch job status, OCR job progress, which jobs succeeded or failed, or downloaded results.
allowed-tools: Bash(*/batch-status.sh *)
---

# Batch Status

Query the Forge `batch_jobs` ledger and OCR projections using the companion script.

## Script location

```text
.claude/skills/batch-status/batch-status.sh
```

The shell script cds to the repo root and runs `.claude/skills/batch-status/batch_status.py`
via `uv run python`, so it resolves the workspace venv (SQLAlchemy is already a dependency
there — nothing extra to install).

## Prerequisites

The store is **Postgres**, not a local file — Supabase in production, or the local podman
stack's Postgres (port 5434 on this machine) for local dev. The script reads `FORGE_DB_URL`
and exits with a clear error if it is unset.

**Read-only against the production store — SELECTs only.** The ambient `FORGE_DB_URL` in this
environment points at the production Supabase database. Every query in `batch_status.py` is a
parameterized `SELECT`; the module must never grow an INSERT/UPDATE/DELETE/DDL statement.

## Status vocabulary

`batch_jobs.status` (`sax_platform.contracts.models.BatchJobStatus`):

| Status | Meaning |
| --- | --- |
| `submitted` | In flight at the provider. The only non-terminal state. |
| `ended` | Terminal **success** — the waiter fetched the finished batch's result. |
| `failed` | Terminal failure — submission rejected, or the provider reported FAILED/CANCELED. |
| `expired` | Terminal failure — provider TIMEOUT_EXCEEDED. |
| `missing` | Terminal failure — the waiter gave up at its 25h ceiling. |
| `processing` | **Legacy only.** Written by the retired shared poller (pre-T4.1); the current timer-loop transport never writes it. Read-tolerated, shown in a clearly labeled `legacy (poller-era)` bucket — never hidden or folded into failed/pending. |

`batch_jobs` is the platform's generic audit/spend ledger and carries no domain fields (no
`file_path`). Where OCR file/document context is wanted, the script LEFT JOINs OCR's own
`ocr_job_status` table on `batch_jobs.id = ocr_job_status.request_id` — the same join
`ocr.activities.execute_list_ocr_jobs` uses. LEFT JOIN because non-OCR `batch_jobs` rows (e.g.
forge's own Anthropic batches) have no `ocr_job_status` counterpart.

## Usage

### Show recent batch jobs and OCR results (default)

```bash
.claude/skills/batch-status/batch-status.sh
```

### Batch job counts grouped by provider and status

```bash
.claude/skills/batch-status/batch-status.sh summary
```

### Show terminal-failure batch jobs

```bash
.claude/skills/batch-status/batch-status.sh failed
```

Filters `status IN ('failed', 'expired', 'missing')`. `ended` rows (successes) never appear here.

### Show in-flight and legacy batch jobs

```bash
.claude/skills/batch-status/batch-status.sh pending
```

Filters `status IN ('submitted', 'processing')`; each row carries a `bucket` label
(`in flight` vs. `legacy (poller-era)`) and `hours_ago` so the two are never conflated.

### Show OCR results for a specific provider batch

```bash
.claude/skills/batch-status/batch-status.sh ocr <batch_id>
```

`<batch_id>` is the **provider's** batch id (`ocr_results.batch_id`/`batch_jobs.batch_id`), not
the request/correlation id (`batch_jobs.id`/`ocr_job_status.request_id`).

### Cross-reference batch jobs with OCR job status

```bash
.claude/skills/batch-status/batch-status.sh cross-ref
```

## Summarise the results

After running the script, provide a brief summary:

- Total batch jobs by status, using the vocabulary table above — `ended` is success, not failure.
- How many OCR results have been stored.
- Any failed/expired/missing jobs with their error messages.
- Whether `submitted` jobs are stuck (large `hours_ago` in `pending`).
- Call out any `processing` (legacy) rows separately — they predate the current transport and
    are not evidence of anything currently stuck.
