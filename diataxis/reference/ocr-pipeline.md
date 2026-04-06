# OCR Pipeline Reference

## Workflow Definitions

| Workflow | Input | Output | Purpose |
|---|---|---|---|
| `OcrSyncWorkflow` | `OcrSyncInput` | `OcrStoreResult` | Synchronous OCR via direct Mistral API call. Blocks until the document is processed and stored. |
| `OcrSubmitWorkflow` | `OcrSubmitInput` | `OcrSubmitResult` | Batch OCR submission. Returns immediately after submitting; child workflows store results asynchronously. |
| `OcrStoreWorkflow` | `OcrStoreInput` | `OcrStoreResult` | Waits for the `batch_result_received` signal, then parses and stores the OCR result for one chunk. |
| `OcrGatherWorkflow` | `OcrGatherInput` | `OcrStoreResult` | Waits for `chunk_completed` signals from all `OcrStoreWorkflow` children, then reassembles chunks into one result. |
| `OcrExportWorkflow` | `OcrExportInput` | `OcrExportResult` | Exports OCR text and images for a document to a directory on disk. |
| `OcrMarkForRemovalWorkflow` | `OcrMarkInput` | `OcrMarkResult` | Sets `marked_for_removal=True` on one document. |
| `OcrListJobsWorkflow` | `OcrListJobsInput` | `OcrListJobsResult` | Lists OCR job submissions grouped by file, with aggregate status. |
| `OcrClearRemovalMarkWorkflow` | `OcrMarkInput` | `OcrMarkResult` | Sets `marked_for_removal=False` on one document. |


### OcrStoreWorkflow Signals

| Signal | Payload | Description |
|---|---|---|
| `batch_result_received` | `BatchResult` | Delivered by the batch poller when the Mistral batch job completes. Contains the raw response JSON. |

### OcrGatherWorkflow Signals

| Signal | Payload | Description |
|---|---|---|
| `chunk_completed` | `str` (chunk document ID) | Sent by each `OcrStoreWorkflow` when it finishes storing its chunk result. |


## Data Models

### OcrSyncInput

| Field | Type | Required | Description |
|---|---|---|---|
| `file_path` | `str` | yes | Absolute path to the document file. Must be non-empty. |
| `model_name` | `str` | no | Mistral model identifier. Default: `mistral:mistral-ocr-latest`. |
| `document_id` | `str` | no | Document identifier. Auto-generated UUID if empty. |
| `skip_duplicate_detection` | `bool` | no | Skip SHA-256 hash check. Default: `False`. |

### OcrSubmitInput

| Field | Type | Required | Description |
|---|---|---|---|
| `file_path` | `str` | yes | Absolute path to the document file. Must be non-empty. |
| `model_name` | `str` | no | Mistral model identifier. Default: `mistral:mistral-ocr-latest`. |
| `max_tokens` | `int` | no | Maximum tokens for the OCR response. Default: `16384`. |
| `document_id` | `str` | no | Document identifier. Auto-generated UUID if empty. |
| `skip_duplicate_detection` | `bool` | no | Skip SHA-256 hash check. Default: `False`. |

### OcrSubmitResult

| Field | Type | Description |
|---|---|---|
| `document_id` | `str` | The document identifier assigned or matched. |
| `batch_refs` | `list[OcrBatchRef]` | Batch tracking references, one per chunk submitted. |
| `chunk_count` | `int` | Number of chunks the document was split into. |
| `skipped` | `bool` | True if duplicate detection short-circuited submission. |
| `skip_reason` | `str` | Reason for skipping, if `skipped` is True. |

### OcrStoreInput

| Field | Type | Required | Description |
|---|---|---|---|
| `batch_id` | `str` | yes | Batch job ID (empty until resolved from signal). |
| `request_id` | `str` | yes | Per-request ID within the batch. |
| `document_id` | `str` | yes | Document or chunk document ID. |
| `file_path` | `str` | yes | Original source file path (metadata only). |
| `gather_workflow_id` | `str` | no | If set, signal this `OcrGatherWorkflow` on completion. |

### OcrStoreResult

| Field | Type | Description |
|---|---|---|
| `document_id` | `str` | The document or chunk document ID. |
| `text_length` | `int` | Character length of the extracted text. |
| `page_count` | `int` | Number of pages in the document or chunk. |
| `stored` | `bool` | True if the result was written to the database. |
| `skipped` | `bool` | True if this was a duplicate. |
| `skip_reason` | `str` | Reason for skipping, if `skipped` is True. |

### OcrGatherInput

| Field | Type | Required | Description |
|---|---|---|---|
| `document_id` | `str` | yes | The parent document ID (not a chunk ID). |
| `chunk_document_ids` | `list[str]` | yes | Ordered list of chunk document IDs to await. |
| `store_workflow_ids` | `list[str]` | yes | `OcrStoreWorkflow` IDs (unused; completion is signaled). |
| `file_path` | `str` | yes | Original source file path. |
| `total_pages` | `int` | yes | Total page count across all chunks. |

### OcrExportInput

| Field | Type | Required | Description |
|---|---|---|---|
| `document_id` | `str` | yes | Document ID to export. |
| `output_dir` | `str` | no | Override export directory. Defaults to `$XDG_DATA_HOME/forge/ocr-export/{document_id}`. |

### OcrExportResult

| Field | Type | Description |
|---|---|---|
| `document_id` | `str` | The exported document ID. |
| `export_dir` | `str` | Absolute path to the export directory. |
| `markdown_path` | `str` | Absolute path to the exported markdown file. |
| `image_count` | `int` | Number of images written to disk. |

### OcrBatchRef

| Field | Type | Description |
|---|---|---|
| `batch_id` | `str` | Mistral batch job ID. |
| `request_id` | `str` | Per-request ID within the batch. |

### OcrMarkInput / OcrMarkResult

| Field | Type | Description |
|---|---|---|
| `document_id` (input) | `str` | Document to mark or unmark. |
| `document_id` (result) | `str` | Echo of the input document ID. |
| `found` (result) | `bool` | True if the document was found in the database. |

### ChunkRef

| Field | Type | Description |
|---|---|---|
| `content_id` | `str` | Foreign key into `file_content_blobs`. |
| `mime_type` | `str` | MIME type of the chunk. |
| `file_size_bytes` | `int` | Byte size of the chunk. |
| `chunk_index` | `int` | Zero-based sequence number. |
| `page_start` | `int` | First page in this chunk (1-based). |
| `page_end` | `int` | Last page in this chunk (1-based). |

### OcrDuplicateCheckResult

| Field | Type | Description |
|---|---|---|
| `is_duplicate` | `bool` | True if a matching hash was found. |
| `existing_document_id` | `str` | Document ID of the existing result, if `is_duplicate` is True. |

### OcrListJobsInput

| Field | Type | Required | Description |
|---|---|---|---|
| `limit` | `int` | no | Maximum number of jobs to return. Default: `50`. |
| `status_filter` | `str` | no | Filter by aggregate status (`processing`, `succeeded`, `errored`). Default: empty (all). |

### OcrListJobsResult

| Field | Type | Description |
|---|---|---|
| `jobs` | `list[OcrJobEntry]` | Matching job submissions. |
| `total` | `int` | Number of entries returned. |

### OcrJobEntry

| Field | Type | Description |
|---|---|---|
| `file_path` | `str` | Source file path of the submitted document. |
| `document_id` | `str` | Document ID from `ocr_results` (empty if result not yet stored). |
| `status` | `str` | Aggregate status: `processing`, `succeeded`, `errored`, or `unknown`. |
| `chunk_count` | `int` | Number of batch chunks for this submission. |
| `created_at` | `str` | ISO 8601 timestamp of the earliest chunk submission. |

Status is derived from the underlying `batch_jobs` rows for a given file path:
any chunk errored means `errored`; any chunk still submitted means `processing`;
all chunks succeeded means `succeeded`.


## Database Tables

### ocr_results

Stores the extracted text and metadata for each OCR document.

| Column | Type | Description |
|---|---|---|
| `id` | `Integer` (PK) | Auto-increment primary key. |
| `document_id` | `String` (unique, indexed) | Stable document identifier. |
| `file_path` | `String` | Original source file path. |
| `text` | `Text` | Extracted markdown text with `ocr-image://` URI references. |
| `page_count` | `Integer` | Number of pages. |
| `model_name` | `String` | Mistral model used (e.g. `mistral-ocr-latest`). |
| `input_tokens` | `Integer` | Tokens consumed in the OCR request. |
| `output_tokens` | `Integer` | Tokens produced in the OCR response. |
| `batch_id` | `String` | Mistral batch job ID (empty string for sync path). |
| `workflow_id` | `String` | Temporal workflow ID that stored the result. |
| `file_hash` | `String` (nullable, indexed) | SHA-256 hex digest of the source file. |
| `marked_for_removal` | `Boolean` | Soft-delete flag. Default: `False`. |
| `created_at` | `DateTime` | Row creation timestamp. |

### ocr_images

Stores images extracted from OCR responses, keyed by UUID.

| Column | Type | Description |
|---|---|---|
| `id` | `String` (PK) | UUID assigned at parse time. Used in `ocr-image://` URIs. |
| `document_id` | `String` (indexed) | Parent document ID. |
| `page_index` | `Integer` | Page number the image appeared on (0-based). |
| `original_image_id` | `String` | ID from the Mistral response (e.g. `img-0.jpeg`). |
| `data` | `LargeBinary` | Raw image bytes. |
| `mime_type` | `String` | Image MIME type (e.g. `image/jpeg`). |
| `file_size_bytes` | `Integer` | Byte size of the image. |
| `top_left_x` | `Integer` (nullable) | Bounding box, if provided by Mistral. |
| `top_left_y` | `Integer` (nullable) | Bounding box, if provided by Mistral. |
| `bottom_right_x` | `Integer` (nullable) | Bounding box, if provided by Mistral. |
| `bottom_right_y` | `Integer` (nullable) | Bounding box, if provided by Mistral. |
| `created_at` | `DateTime` | Row creation timestamp. |

### batch_jobs (OCR-specific columns)

The `batch_jobs` table is shared with the LLM batch path. See the
[Model Routing and Batch Processing reference](llm-dispatch.md) for the full
schema. OCR batch jobs are distinguished by the `provider` column value
`"mistral"`. The `file_path` column stores the source document path.


## ocr-image:// URI Format

Format: `ocr-image://{uuid}`

Where `{uuid}` is the `id` column of a row in the `ocr_images` table.

Example: `ocr-image://3f2a1b7c-9e4d-4c8a-b5f6-0a1d2e3c4b5a`

The URI appears in the `text` column of `ocr_results` as a markdown image
reference: `![alt text](ocr-image://3f2a1b7c-9e4d-4c8a-b5f6-0a1d2e3c4b5a)`.

The `OcrExportWorkflow` resolves these URIs to local filenames of the form
`{uuid}.jpg` (or the appropriate extension for the MIME type).


## CLI Commands

OCR workflows are launched via the generic `forge start` command, which submits any
named Temporal workflow with a JSON input.

**Run synchronous OCR:**

```
forge start OcrSyncWorkflow '{"file_path": "/path/to/doc.pdf"}'
```

**Submit batch OCR:**

```
forge start OcrSubmitWorkflow '{"file_path": "/path/to/doc.pdf"}'
```

**Export OCR results:**

```
forge start OcrExportWorkflow '{"document_id": "my-doc-id"}'
```

**Export to a specific directory:**

```
forge start OcrExportWorkflow \
  '{"document_id": "my-doc-id", "output_dir": "/tmp/ocr-out"}'
```

**List OCR job submissions:**

```
forge ocr-jobs
forge ocr-jobs --limit 20 --status processing
```

**Backfill SHA-256 hashes for existing OCR results:**

```
forge backfill-hashes
forge backfill-hashes --dry-run
```

**Start a workflow and wait for the result:**

```
forge start OcrSyncWorkflow \
  '{"file_path": "/path/to/doc.pdf"}' \
  --wait
```

See [How to Run OCR](../howto/run-ocr.md) for usage recipes. See
[OCR Pipeline](../explanation/ocr-pipeline.md) for design background.
