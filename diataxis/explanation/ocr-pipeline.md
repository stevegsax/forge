# OCR Pipeline

The OCR pipeline applies Forge's core orchestration primitives to a specialized
domain: extracting text and images from documents using the Mistral OCR API. Rather
than introducing new patterns, it composes the primitives already described in
[The Universal Workflow Step](workflow-step.md) and
[Model Routing and Batch Processing](llm-dispatch.md). Understanding those two
topics first makes the OCR pipeline straightforward to reason about.


## Two Paths, One Set of Primitives

The pipeline offers two execution paths that mirror the distinction between
synchronous and batch LLM calls described in
[Model Routing and Batch Processing](llm-dispatch.md#batch-mode).

The **synchronous path** (`OcrSyncWorkflow`) calls the Mistral OCR API directly via
`client.ocr.process_async()` and waits for the response. It follows the same
workflow-activity structure as every other Forge workflow: one parent workflow
orchestrates a sequence of activities (read file, split into chunks, call OCR,
store result), each with defined timeouts and retry policies. For small documents
where batch latency is unacceptable, this path returns results in seconds.

The **batch path** (`OcrSubmitWorkflow`) uses the same submit-wait-signal pattern
as LLM batch processing. The parent workflow submits the document to the Mistral
batch endpoint (`/v1/ocr`), then returns immediately — fire-and-forget. Child
workflows (`OcrStoreWorkflow`) continue running independently with an ABANDON
parent-close policy. Each child waits for a `batch_result_received` signal, which
the batch poller delivers when Mistral reports the job complete. This is the same
signal pattern used for Anthropic batch calls; the only difference is the provider
and the API endpoint. The batch path offers a 50% cost reduction over synchronous
calls. For Mistral Batch API details, see the
[Mistral documentation](https://docs.mistral.ai/capabilities/batch).


## Document Splitting and Chunking

Both paths split large documents into chunks before submitting them. The
`split_file_into_chunks` activity divides a document (typically a large PDF) into
page ranges, storing each chunk as a binary blob in the `file_content_blobs` table.
Each chunk is processed as an independent OCR request and assigned a
`chunk_document_id` of the form `{document_id}__chunk_{N}`.

For single-chunk documents, no splitting overhead applies: the document is
submitted as a single request and the result is stored directly under the
`document_id`.


## Multi-Chunk Gathering

When a document produces multiple chunks, the batch path starts an additional
`OcrGatherWorkflow` before launching the per-chunk `OcrStoreWorkflow` children.
The gather workflow holds in memory the set of expected chunk IDs and waits (up to
26 hours) for each `OcrStoreWorkflow` to signal `chunk_completed` upon finishing.
Once all chunks have reported in, the gather workflow runs the
`reassemble_ocr_chunks` activity, which concatenates the per-chunk markdown texts
in page order and updates the main `ocr_results` row.

The synchronous path handles multi-chunk reassembly inline rather than through a
separate gather workflow, since all chunks are processed sequentially without
asynchronous signaling.


## Image Extraction and the ocr-image:// URI Scheme

The Mistral OCR API returns images embedded in its response alongside markdown
text. The pipeline extracts these images and stores them independently in the
`ocr_images` table, keyed by a UUID assigned at parse time. Image IDs returned by
the API (e.g. `img-0.jpeg`) are sequential within a single API call but not unique
across chunks of the same document. Assigning a UUID at storage time provides a
stable, globally unique identifier.

After storing the images, the pipeline rewrites all markdown image references in
the extracted text. References of the form `![alt](img-0.jpeg)` become
`![alt](ocr-image://{uuid})`. This rewriting happens in the `parse_ocr_result`
activity for the synchronous path and in the batch poller for the batch path. The
`ocr-image://` URI scheme is an internal identifier: it points to an image row in
the database rather than to a file path. When exporting results to disk, the
`OcrExportWorkflow` resolves these URIs back to local filenames.

This design allows the pipeline to strip raw base64 image data from the Mistral
response before signaling the store workflow. Temporal enforces a 2 MB signal size
limit; retaining embedded base64 data for a document with many images would exceed
it.


## SHA-256 Duplicate Detection

Before submitting any document, both paths compute a SHA-256 hash of the file
content and check the `ocr_results` table for a matching `file_hash`. If a match
is found, the workflow returns the existing `document_id` without re-submitting.
This prevents redundant OCR charges when the same file is submitted multiple times
under different paths or names.

Duplicate detection is based on content, not file path, so renaming a file does not
bypass it. The check can be disabled per-submission with `skip_duplicate_detection`
for cases where reprocessing is intentional.


## Export and Lifecycle Management

Once results are stored, `OcrExportWorkflow` writes the markdown text and all
associated images to a directory on disk. It resolves each `ocr-image://` URI to a
local filename and rewrites the markdown accordingly, producing a self-contained
export directory.

Two maintenance workflows, `OcrMarkForRemovalWorkflow` and
`OcrClearRemovalMarkWorkflow`, manage the soft-delete lifecycle. Documents are
flagged with `marked_for_removal=True` before actual deletion, allowing a separate
cleanup process to remove them at a controlled time.


## Relationship to the Core Pipeline

The OCR pipeline does not use the LLM task execution path (context assembly,
edit application, validation). It is a separate workflow family that happens to
share the same Temporal worker, observability store, and batch infrastructure.
The `batch_jobs` table used by the OCR batch path is the same table used by the
LLM batch path, with an additional `provider` column to distinguish Mistral from
Anthropic jobs. See the [Model Routing and Batch Processing reference](../reference/llm-dispatch.md)
for the full `batch_jobs` schema.

For technical details on the OCR workflows, data models, and database tables, see
the [OCR Pipeline reference](../reference/ocr-pipeline.md). For step-by-step
instructions, see [How to Run OCR](../howto/run-ocr.md).
