# How to Use the Nushell OCR Module

The `ocr.nu` module provides composable functions for OCR operations that talk
directly to the Temporal server. Every command returns structured data (records
or tables) that can be filtered, sorted, and piped into other commands.

For the full command reference, see
[Nushell Module Reference](../reference/nushell-module.md). For background on
the OCR pipeline, see [OCR Pipeline](../explanation/ocr-pipeline.md).


## Load the module

The module lives at `scripts/nushell/ocr.nu`. Set `NU_LIB_DIRS` so nushell
can find it:

```nu
NU_LIB_DIRS="scripts/nushell" nu
```

Then load the module:

```nu
use ocr.nu
```

To make this permanent, add the path to `NU_LIB_DIRS` in
`~/.config/nushell/env.nu`.


## Submit a document for OCR

1. Submit via the batch API (returns immediately):

    ```nu
    ocr submit ./contract.pdf
    ```

2. Submit and wait for the result with `--sync`:

    ```nu
    ocr submit --sync ./receipt.png
    ```

3. Re-submit a document that was already OCR'd:

    ```nu
    ocr submit --skip-duplicate-detection ./contract.pdf
    ```

4. Submit all PDFs in a directory in parallel:

    ```nu
    ls ~/scans/*.pdf | par-each { |f| ocr submit $f.name }
    ```


## List and filter OCR jobs

1. List all jobs (most recent first):

    ```nu
    ocr list
    ```

2. Show only specific columns:

    ```nu
    ocr list | select file_path document_id status created_at
    ```

3. Filter by status:

    ```nu
    ocr list | where status == "succeeded"
    ```

    Or use the server-side filter for efficiency:

    ```nu
    ocr list --status succeeded
    ```

4. Filter by filename pattern and sort:

    ```nu
    ocr list | where file_path =~ "signed" | sort-by file_path
    ```

5. Count jobs by status:

    ```nu
    ocr list --limit 1000 | group-by status | transpose status jobs | each { |r| { status: $r.status, count: ($r.jobs | length) } }
    ```


## Export OCR results

1. Export a document by its ID:

    ```nu
    ocr export doc a1b2c3d4-...
    ```

2. Export to a specific directory:

    ```nu
    ocr export doc a1b2c3d4-... --output-dir /tmp/my-export
    ```

3. Search by filename and export the matches:

    ```nu
    ocr list | where file_path =~ "signed" | each { |r| ocr export doc $r.document_id }
    ```

4. Export all succeeded jobs:

    ```nu
    ocr list | where status == "succeeded" | each { |r| ocr export doc $r.document_id }
    ```


## Mark and unmark documents for removal

1. Soft-delete a document:

    ```nu
    ocr mark a1b2c3d4-...
    ```

2. Undo a soft-delete:

    ```nu
    ocr unmark a1b2c3d4-...
    ```

3. Mark all jobs matching a pattern:

    ```nu
    ocr list | where file_path =~ "draft" | each { |r| ocr mark $r.document_id }
    ```


## Compose operations with pipelines

The module is designed so that one command's output feeds another's input.

1. Submit a document synchronously, then export:

    ```nu
    let result = ocr submit --sync ./doc.pdf
    ocr export doc $result.document_id
    ```

2. Find documents older than a week and mark for removal:

    ```nu
    ocr list | where created_at < ((date now) - 7day) | each { |r| ocr mark $r.document_id }
    ```

3. Batch submit and collect document IDs:

    ```nu
    let ids = ls *.pdf | par-each { |f| ocr submit $f.name } | get document_id
    ```
