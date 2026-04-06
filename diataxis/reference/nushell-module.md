# Nushell Module Reference

## Module Location

| | |
|---|---|
| Path | `scripts/nushell/ocr.nu` |
| Load | `use ocr.nu` (requires `NU_LIB_DIRS` to include `scripts/nushell`) |

## Prerequisites

| Requirement | Description |
|---|---|
| `temporal` CLI | Must be on `$PATH`. Connects to `localhost:7233` by default; override with `TEMPORAL_ADDRESS` env var. |
| Forge worker | Must be running (`forge worker`) to serve workflow requests. |
| Nushell | Version 0.100+ recommended. |


## Commands

### ocr submit

Submit a document for OCR processing.

**Signature:** `nothing -> record`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file_path` | `path` | yes | Document to OCR. Automatically expanded to absolute path. |
| `--sync` | flag | no | Block until OCR completes. Without this flag, uses the batch API and returns immediately. |
| `--skip-duplicate-detection` | flag | no | Re-submit even if a document with the same SHA-256 hash already exists. |

**Returns:**

With `--sync` (`OcrStoreResult`):

| Field | Type | Description |
|---|---|---|
| `document_id` | `string` | Assigned document identifier. |
| `text_length` | `int` | Character count of extracted text. |
| `page_count` | `int` | Number of pages processed. |
| `stored` | `bool` | Whether the result was written to the database. |
| `skipped` | `bool` | True if duplicate detection short-circuited. |
| `skip_reason` | `string` | Reason for skipping, if applicable. |

Without `--sync` (`OcrSubmitResult`):

| Field | Type | Description |
|---|---|---|
| `document_id` | `string` | Assigned document identifier. |
| `batch_refs` | `list<record>` | Batch tracking references, one per chunk. |
| `chunk_count` | `int` | Number of chunks the document was split into. |
| `skipped` | `bool` | True if duplicate detection short-circuited. |
| `skip_reason` | `string` | Reason for skipping, if applicable. |


### ocr list

List OCR job submissions as a structured table.

**Signature:** `nothing -> table`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--limit` (`-l`) | `int` | no | `50` | Maximum results (applied server-side). |
| `--status` (`-s`) | `string` | no | all | Filter by status. Tab-completes to: `processing`, `succeeded`, `errored`. |

**Returns:** table with columns:

| Column | Type | Description |
|---|---|---|
| `file_path` | `string` | Basename of the submitted file (not the full path). |
| `document_id` | `string` | Document ID from OCR results. Empty if the job has not completed. |
| `status` | `string` | Aggregate status: `processing`, `succeeded`, `errored`, or `unknown`. |
| `chunk_count` | `int` | Number of batch chunks for this submission. |
| `created_at` | `datetime` | Submission timestamp. Rendered as relative time (e.g., "2 days ago") in tables. |


### ocr export doc

Export OCR results (markdown text and images) to disk.

**Signature:** `nothing -> record`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `document_id` | `string` | yes | Document to export. |
| `--output-dir` (`-o`) | `directory` | no | Override export directory. Defaults to `$XDG_DATA_HOME/forge/ocr-export/{document_id}`. |

**Returns:**

| Field | Type | Description |
|---|---|---|
| `document_id` | `string` | The exported document ID. |
| `export_dir` | `string` | Absolute path to the export directory. |
| `markdown_path` | `string` | Absolute path to the exported markdown file. |
| `image_count` | `int` | Number of images written to disk. |


### ocr mark

Mark a document for removal (soft-delete).

**Signature:** `nothing -> record`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `document_id` | `string` | yes | Document to mark. |

**Returns:**

| Field | Type | Description |
|---|---|---|
| `document_id` | `string` | Echo of the input document ID. |
| `found` | `bool` | True if the document was found in the database. |


### ocr unmark

Clear a removal mark on a document.

**Signature:** `nothing -> record`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `document_id` | `string` | yes | Document to unmark. |

**Returns:** Same as `ocr mark`.


## Internal Architecture

The module does not use the `forge` Python CLI. All commands call the `temporal`
CLI directly via `temporal workflow execute --output json`, parse the JSON
response, and extract the workflow result. The Temporal task queue is hardcoded
as `forge-task-queue`.

| Command | Temporal Workflow |
|---|---|
| `ocr submit` | `OcrSubmitWorkflow` or `OcrSyncWorkflow` |
| `ocr list` | `OcrListJobsWorkflow` |
| `ocr export doc` | `OcrExportWorkflow` |
| `ocr mark` | `OcrMarkForRemovalWorkflow` |
| `ocr unmark` | `OcrClearRemovalMarkWorkflow` |

See [How to Use the Nushell OCR Module](../howto/use-nushell-module.md) for
usage recipes. See [OCR Pipeline Reference](ocr-pipeline.md) for the underlying
workflow and data model details.
