---
name: mistral-batch
description: Check Mistral batch job status directly via the Mistral API. Use when the user asks to check a Mistral batch job, poll a running batch, or download batch results/errors from the Mistral server.
allowed-tools: Bash(*/mistral-batch.sh *)
---

# Mistral Batch

Query the Mistral Batch API using the companion script.

## Script location

```text
.claude/skills/mistral-batch/mistral-batch.sh
```

## Prerequisites

`MISTRAL_API_KEY` must be set in the environment. If the script exits with an error about it being unset, tell the user to export it.

## Usage

### List recent batch jobs (default)

```bash
.claude/skills/mistral-batch/mistral-batch.sh
```

### Get detailed status for a specific batch job

```bash
.claude/skills/mistral-batch/mistral-batch.sh <batch_id>
```

### Download error file contents

```bash
.claude/skills/mistral-batch/mistral-batch.sh errors <file_id>
```

### Download output file contents

```bash
.claude/skills/mistral-batch/mistral-batch.sh output <file_id>
```

## Summarise the results

After running the script, provide a brief summary:

- Job status (QUEUED, RUNNING, SUCCESS, FAILED, TIMEOUT_EXCEEDED, CANCELLED)
- Request counts: total, succeeded, failed
- Whether output and error files exist
- Error details if the job failed
- How long the job has been running (if still in progress)
