+++
title = "How to Ingest Transcripts"
weight = 123
description = "The pipeline that reads Claude Code session transcripts, analyzes them with forge's batch LLM path, and hands extracted experiences to pbook's ExtractionWorkflow cross-queue."
topic = "transcript-ingestion"
covers = [
    "How to ingest a single Claude Code session file",
    "How to discover and ingest all sessions from ~/.claude/projects/",
    "How to filter discovered sessions by project",
    "How to preview what would be ingested without submitting (dry-run)",
    "How to reprocess a session that was already ingested (--force)",
    "How to debug a stuck or failing ingestion",
]
detail = "Short CLI-focused recipes. Each recipe is a numbered sequence ending in a verification step."
+++
This guide shows you how to use `forge ingest` to submit Claude Code session files to the transcript ingestion pipeline. Ingested sessions are analyzed by forge's batch LLM path and handed off to pbook's extraction workflow for storage.

For how the pipeline is designed, see [Transcript Ingestion](../explanation/transcript-ingestion/). For full flag documentation, see the [Transcript Ingestion Reference](../reference/transcript-ingestion/).

## Prerequisites

1. The forge worker is running and connected to Temporal.
2. The pbook worker is running on the same Temporal server and listening on `pbook-task-queue`. If pbook is not installed or not running, `forge ingest` exits with an error.
3. The `BatchPollerWorkflow` schedule is active on the forge worker. Ingestion uses batch mode, so completion signals are delivered by the batch poller.

## Ingest a single session

1. Locate the JSONL file for the session you want to ingest. Claude Code stores sessions under `~/.claude/projects/<encoded-project-dir>/<session-id>.jsonl`.

2. Run `forge ingest` with the file path:

    ```
    forge ingest ~/.claude/projects/-Users-you-repos-myproj/sess-abc123.jsonl
    ```

    The CLI wraps the single path in a one-element batch and submits `BatchIngestionWorkflow`. The output confirms the submission and then reports the aggregate result once the workflow completes:

    ```
    Submitting 1 session(s) for ingestion...
    Ingestion complete: 1 sessions processed, 3 experiences found, 3 entries created.
    ```

3. If you want to override the project name (for example, because the directory encoding is obscure), pass `--project`:

    ```
    forge ingest ~/.claude/projects/-tmp-scratch/sess-abc123.jsonl --project myproj
    ```

    The override propagates to the experience metadata and to pbook's `source_project` field.

## Ingest all sessions from `~/.claude/projects/`

1. Run `forge ingest --all` to discover and ingest every session larger than 10 KB that has not already been processed:

    ```
    forge ingest --all
    ```

    Discovery scans `~/.claude/projects/` recursively, filters out files smaller than `--min-size` bytes (default 10240), and queries pbook's `ingested_sessions` table to skip sessions that have already been processed. The output reports how many sessions were skipped before submitting the rest:

    ```
    Skipping 8 already-ingested session(s).
    Submitting 14 session(s) for ingestion...
    Ingestion complete: 14 sessions processed, 31 experiences found, 27 entries created.
    ```

2. Expect the workflow to take minutes to hours, depending on how many sessions are submitted and the batch API's current turnaround time. For why ingestion uses batch mode, see [Transcript Ingestion](../explanation/transcript-ingestion/).

## Filter discovered sessions by project

1. To ingest only sessions belonging to a specific project, combine `--all` with `--project`:

    ```
    forge ingest --all --project forge
    ```

    This runs discovery as usual and then filters the result to sessions whose inferred project name matches the flag value. Project inference takes the last path segment of the Claude Code directory name, so `~/.claude/projects/-Users-you-repos-forge/` yields `forge`.

2. If nothing matches, the CLI exits cleanly with `No sessions found.` — use `--dry-run` if you want to check discovery before submitting.

## Preview what would be ingested

1. Add `--dry-run` to see which sessions would be submitted without actually calling the workflow:

    ```
    forge ingest --all --dry-run
    ```

    The output groups sessions by project with total size, and lists per-session details for projects with three or fewer sessions:

    ```
    Found 14 session(s) to ingest (18.4 MB):

      forge: 6 session(s), 9.2 MB
      pbook: 4 session(s), 5.1 MB
      sandbox: 2 session(s), 2.8 MB
        abc12345...  1420 KB
        def67890...  1380 KB
      scratch: 2 session(s), 1.3 MB
        11112222...  720 KB
        33334444...  580 KB
    ```

2. The dry-run still queries pbook for already-ingested sessions and subtracts them, so the preview reflects what `--all` without `--dry-run` would actually submit.

## Reprocess a session that was already ingested

1. Use `--force` to bypass the already-ingested filter:

    ```
    forge ingest --all --force
    ```

    All discovered sessions are submitted regardless of whether they have been processed before. Each reprocessed session gets a fresh workflow ID suffix, so there are no Temporal workflow ID collisions.

2. Reprocessing creates new entries in pbook's store; it does not update existing ones. If you want to replace old entries, delete them through pbook's CLI before running ingestion with `--force`. Running `--force` without cleaning up first may leave duplicate-looking entries in pbook's `entries` table.

## Debug a stuck ingestion

1. Check the forge worker log for the ingestion workflow activity:

    ```
    tail -f ~/.local/state/forge/worker.log
    ```

    Look for `prepare_transcript` completion lines and for any retry counters on the batch submission. If `prepare_transcript` is absent, the workflow never started — verify the worker has registered ingestion workflows (the startup log prints a warning if pbook is unavailable).

2. If the workflow started but is waiting on the batch API, confirm the batch poller schedule is running:

    ```
    forge status --verbose
    ```

    The output includes the `forge-batch-poller` schedule state. If the schedule is missing or paused, batch completion signals will not be delivered and ingestion workflows will hang until their 25-hour timeout.

3. If the analysis LLM returned malformed JSON, the workflow exits cleanly with an error marker and does *not* record the session as ingested. Check the workflow result:

    ```
    forge status --workflow-id <workflow-id>
    ```

    A result dict containing `"error": "malformed_llm_response"` means the LLM output could not be parsed. Re-run the single session with `forge ingest <path>` — the batch poller does not retry malformed responses automatically, but a fresh submission does get a fresh LLM call.

4. If pbook is unreachable, cross-queue calls to `ExtractionWorkflow` or `record_ingested_session` will block until they time out. Verify pbook's worker is running and listening on `pbook-task-queue` — check pbook's own worker log, or run `pbook worker` in a separate terminal. `forge ingest` itself does not surface pbook worker health; failures only appear after the cross-queue timeout fires.
