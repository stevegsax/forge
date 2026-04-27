+++
title = "How to Manage Playbooks"
weight = 113
description = "Forge's self-learning loop: extracting playbook entries from its own completed run history and injecting them into future task contexts."
topic = "forge-run-extraction"
covers = [
    "How to run knowledge extraction on completed forge runs",
    "How to list and inspect existing forge playbook entries",
    "How to manually add a playbook entry to forge's store",
    "How to verify forge playbooks appear in future task contexts",
]
detail = "Short CLI-focused recipes, scoped to forge's own playbook store."
+++
This guide shows you how to run extraction against completed Forge runs, inspect stored playbook entries in Forge's own store, manually add entries, and verify that playbooks appear in future task contexts. It is scoped to Forge's self-learning loop — for ingesting Claude Code transcripts into pbook, use [How to Ingest Transcripts](ingest-transcripts/) instead.

For background on how extraction works, see [Forge Run Extraction](../explanation/forge-run-extraction/). For full CLI option details, see the [Forge Run Extraction Reference](../reference/forge-run-extraction/).

## Run extraction on completed workflows

1. Ensure the Forge worker is running.

2. Run extraction against recently completed runs:

    ```
    forge extract --since-hours 24
    ```

    The command triggers `ForgeExtractionWorkflow`, which fetches unextracted runs from the last 24 hours, calls the summarization-tier LLM, and writes new playbook entries to the store.

3. Check the output for the number of entries created:

    ```
    Extraction complete. Entries created: 4. Runs processed: 7.
    ```

    If no runs were found in the window, the command returns with no entries created.

4. To process a larger backlog, increase the lookback window and limit:

    ```
    forge extract --since-hours 168 --limit 200
    ```

## List and inspect playbook entries

1. List recent entries:

    ```
    forge playbooks list
    ```

    Output shows `id`, `title`, `tags`, and `created_at` for the 20 most recent entries.

2. Filter by tag to see entries relevant to a specific context:

    ```
    forge playbooks list --tag python --tag test-writing
    ```

3. View the full content of a specific entry by its ID:

    ```
    forge playbooks show 12
    ```

    Output includes the full `content`, `source_task_id`, `source_workflow_id`, and timestamps.

## Manually add a playbook entry

1. Use `forge playbooks add` to insert an entry without running extraction:

    ```
    forge playbooks add \
      --title "Always import annotations for Pydantic models" \
      --content "When generating Pydantic models used with SQLAlchemy, include 'from __future__ import annotations' at the top of the file. The ORM mapper fails without it on Python 3.10 and earlier." \
      --tag python \
      --tag database
    ```

2. Confirm the entry was added:

    ```
    forge playbooks list --tag python
    ```

    The new entry appears with `source_task_id: manual`.

3. If you need to correct the content of a manually added entry, use direct SQLite access:

    ```
    sqlite3 ~/.local/state/forge/forge.db \
      "UPDATE playbooks SET content = '...' WHERE id = <id>;"
    ```

## Verify playbooks appear in future task contexts

1. Run a task with a description that should match existing playbook tags. For example, if you have entries tagged `python, test-writing`, run a task with "test" in the description targeting `.py` files.

2. After the task completes, inspect the assembled context for that run:

    ```
    forge status --workflow-id <id> --verbose
    ```

    Look for a `PLAYBOOK` section in the context assembly output. If entries were retrieved and the token budget had room, they appear there.

3. If no playbooks appear and you expect them, check that:

    - The task's inferred tags overlap with the stored entry's tags. The tag inference rules are documented in the [Forge Run Extraction Reference](../reference/forge-run-extraction/).
    - The token budget was not already full. Playbooks are dropped first when the budget is tight.
    - The entries exist. Run `forge playbooks list` to confirm.
