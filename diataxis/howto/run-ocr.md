# How to Run OCR

This guide shows you how to submit documents for OCR processing, check job status,
export results, and handle multi-chunk documents. All OCR workflows are launched
via the `forge start` command.

For background on how the OCR pipeline works, see
[OCR Pipeline](../explanation/ocr-pipeline.md). For the full workflow and data
model reference, see the [OCR Pipeline reference](../reference/ocr-pipeline.md).


## Run synchronous OCR on a document

Use `OcrSyncWorkflow` for small documents where you need the result immediately.
The command blocks until OCR is complete when `--wait` is passed.

1. Submit the document and wait for the result:

    ```bash
    forge start OcrSyncWorkflow \
      '{"file_path": "/path/to/document.pdf"}' \
      --wait
    ```

    The result is printed as JSON:

    ```json
    {
      "document_id": "a1b2c3d4-...",
      "text_length": 8432,
      "page_count": 12,
      "stored": true,
      "skipped": false,
      "skip_reason": ""
    }
    ```

2. If the document was already OCR'd, the result shows `"skipped": true` and the
   `document_id` of the existing result. Pass `"skip_duplicate_detection": true`
   to force reprocessing:

    ```bash
    forge start OcrSyncWorkflow \
      '{"file_path": "/path/to/document.pdf", "skip_duplicate_detection": true}' \
      --wait
    ```


## Submit a document for batch OCR

Use `OcrSubmitWorkflow` for large documents or when you want the 50% cost
reduction from the Mistral batch API. The command returns a workflow ID
immediately; results are stored asynchronously.

1. Submit the document:

    ```bash
    forge start OcrSubmitWorkflow \
      '{"file_path": "/path/to/large-document.pdf"}'
    ```

    The command prints the Temporal workflow ID, for example:
    `ocrsubmitworkflow-3f2a1b7c`.

2. Note the workflow ID. You will use it to check status.


## Check batch OCR job status

Use the Temporal CLI to check whether a batch OCR workflow has completed.

1. Check the workflow status by workflow ID:

    ```bash
    temporal workflow describe --workflow-id ocrsubmitworkflow-3f2a1b7c
    ```

2. To check the status of a child `OcrStoreWorkflow`, use the ID pattern
   `ocr-store-{document_id}`:

    ```bash
    temporal workflow describe --workflow-id ocr-store-a1b2c3d4-...
    ```

3. To list all running OCR workflows on the task queue:

    ```bash
    temporal workflow list --query 'WorkflowType = "OcrStoreWorkflow" AND ExecutionStatus = "Running"'
    ```

If you want to check Mistral batch job status directly (outside of Temporal), see
`scripts/mistral-batch-jobs.sh` or the
[Mistral documentation](https://docs.mistral.ai/capabilities/batch).


## List OCR job submissions

Use `forge ocr-jobs` to see all OCR submissions with their status. This runs the
`OcrListJobsWorkflow` through Temporal and waits for the result.

1. List all jobs (most recent first):

    ```bash
    forge ocr-jobs
    ```

2. Filter by status:

    ```bash
    forge ocr-jobs --status processing
    ```

3. Limit the number of results:

    ```bash
    forge ocr-jobs --limit 10
    ```

4. Pipe to `nushell` for tabular output:

    ```bash
    forge ocr-jobs | nu -c 'from json | get jobs | select file_path document_id status created_at'
    ```

5. Filter by file path and sort the results. For example, show only documents
   with "signed" in the filename, sorted by path:

    ```bash
    forge ocr-jobs | nu -c 'from json | get jobs | where file_path =~ "signed" | each { |r| $r | update file_path ($r.file_path | path basename) } | sort-by file_path | select file_path document_id status created_at'
    ```

Each entry includes `file_path`, `document_id`, `status` (`processing`,
`succeeded`, or `errored`), `chunk_count`, and `created_at`. The `document_id`
is empty for jobs that have not yet completed.


## Export OCR results to disk

After results are stored (either synchronously or after a batch job completes),
export the markdown text and images to a directory.

1. Export using the document ID:

    ```bash
    forge start OcrExportWorkflow \
      '{"document_id": "a1b2c3d4-..."}' \
      --wait
    ```

    The result shows the export directory and file paths:

    ```json
    {
      "document_id": "a1b2c3d4-...",
      "export_dir": "/home/user/.local/share/forge/ocr-export/a1b2c3d4-...",
      "markdown_path": "/home/user/.local/share/forge/ocr-export/a1b2c3d4-.../document.md",
      "image_count": 7
    }
    ```

2. To export to a specific directory instead of the default XDG location:

    ```bash
    forge start OcrExportWorkflow \
      '{"document_id": "a1b2c3d4-...", "output_dir": "/tmp/my-export"}' \
      --wait
    ```


## Handle multi-chunk documents

Large PDFs are automatically split into chunks. No special handling is required —
the workflow manages chunking transparently.

For **synchronous** OCR, `OcrSyncWorkflow` processes each chunk in sequence and
reassembles the result before returning. The final `OcrStoreResult` reflects the
full document.

For **batch** OCR, `OcrSubmitWorkflow` starts a separate `OcrGatherWorkflow` that
waits for all chunks to complete before reassembling. To check when all chunks
have finished:

1. Look up the gather workflow by its ID pattern `ocr-gather-{document_id}`:

    ```bash
    temporal workflow describe --workflow-id ocr-gather-a1b2c3d4-...
    ```

2. When the gather workflow completes, the full document result is available in
   `ocr_results` under the original `document_id`.


## Mark a document for removal

To soft-delete a document (does not immediately remove data):

```bash
forge start OcrMarkForRemovalWorkflow \
  '{"document_id": "a1b2c3d4-..."}' \
  --wait
```

To clear a removal mark:

```bash
forge start OcrClearRemovalMarkWorkflow \
  '{"document_id": "a1b2c3d4-..."}' \
  --wait
```


## Backfill SHA-256 hashes

If you have OCR results stored before duplicate detection was added, backfill
their hashes so future submissions detect them as duplicates.

1. Preview what would be updated:

    ```bash
    forge backfill-hashes --dry-run
    ```

2. Apply the updates:

    ```bash
    forge backfill-hashes
    ```
