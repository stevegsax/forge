---
name: batch-status
description: Check the status of submitted batch jobs and OCR results. Use when the user asks about batch job status, OCR job progress, which jobs succeeded or failed, or downloaded results.
allowed-tools: Bash(*/batch-status.sh *)
---

# Batch Status

Query the Forge observability database using the companion script.

## Script location

```text
.claude/skills/batch-status/batch-status.sh
```

## Prerequisites

The Forge database must exist. The script resolves the path using:

1. `$FORGE_DB_PATH` (if set and non-empty)
2. `$XDG_STATE_HOME/forge/forge.db`
3. `~/.local/state/forge/forge.db`

## Usage

### Show recent batch jobs and OCR results (default)

```bash
.claude/skills/batch-status/batch-status.sh
```

### Batch job counts grouped by provider and status

```bash
.claude/skills/batch-status/batch-status.sh summary
```

### Show failed/errored batch jobs

```bash
.claude/skills/batch-status/batch-status.sh failed
```

### Show submitted-but-not-completed batch jobs

```bash
.claude/skills/batch-status/batch-status.sh pending
```

### Show OCR results for a specific batch

```bash
.claude/skills/batch-status/batch-status.sh ocr <batch_id>
```

### Cross-reference batch jobs with OCR results

```bash
.claude/skills/batch-status/batch-status.sh cross-ref
```

## Summarise the results

After running the script, provide a brief summary:

- Total batch jobs by status. Valid values are `submitted`, `storing`,
    `succeeded`, `errored`, `failed`, `expired`, `canceled`, `missing`.
    Note: `storing` means the provider completed the batch and the
    downstream store workflow is writing results — it is NOT terminal.
    A row still in `storing` long after creation is stuck.
- How many OCR results have been stored
- Any failed or errored jobs with their error messages
- Whether pending jobs exist that haven't completed yet
