@ocr
Feature: OCR Pipeline
  The orchestrator processes documents through a three-workflow OCR pipeline:
  Submit, Store, and Gather. Large PDFs are automatically chunked. File content
  is stored as blobs to bypass the Temporal 2MB payload limit. Results are
  persisted to the ocr_results table.

  # --- Three-Tier Workflow Architecture ---

  @critical @temporal
  Scenario: OCR submit workflow orchestrates the pipeline
    Given a document file path and a document ID
    When the OcrSubmitWorkflow runs
    Then it reads the file, splits into chunks, starts a gather workflow, and submits batches

  @critical @temporal
  Scenario: OCR store workflow waits for batch result signal
    Given a batch has been submitted for an OCR chunk
    When the OcrStoreWorkflow runs
    Then it waits for a batch_result_received signal
    And parses the OCR result and stores it to the database

  @critical @temporal
  Scenario: OCR gather workflow waits for all chunks to complete
    Given a document split into 3 chunks
    When the OcrGatherWorkflow runs
    Then it waits for 3 chunk_completed signals
    And reassembles the OCR results into a single document

  # --- File Reading and MIME Detection ---

  @standard
  Scenario: MIME type is detected from file extension
    Given a file at "document.pdf"
    When the MIME type is detected
    Then the result is "application/pdf"

  @standard
  Scenario: Image files produce ImageContent blocks
    Given a file with MIME type "image/png"
    When the OCR messages are built
    Then the message contains an image content block

  @standard
  Scenario: Non-image files produce DocumentContent blocks
    Given a file with MIME type "application/pdf"
    When the OCR messages are built
    Then the message contains a document content block

  # --- File Blob Storage ---

  @critical
  Scenario: File content is stored as a blob to bypass Temporal payload limit
    Given a document file of 5 MB
    When the submit workflow reads the file
    Then the content is stored in the file_content_blobs table
    And a lightweight FileContentRef is passed between activities

  @standard
  Scenario: Blob stores data, MIME type, and file size
    When a file blob is saved
    Then the record includes binary data, mime_type, and file_size_bytes

  @standard
  Scenario: Original blob is deleted after successful chunking
    Given a PDF that is split into chunks
    When chunk blobs are created
    Then the original blob is deleted from the database

  # --- PDF Chunking ---

  @critical @deterministic
  Scenario: Small PDF is processed as a single chunk
    Given a PDF with 20 pages and 5 MB file size
    When the submit workflow splits the file
    Then a single chunk is produced containing all pages

  @critical @deterministic
  Scenario: Large PDF by page count is split into 25-page chunks
    Given a PDF with 60 pages and 8 MB file size
    When the submit workflow splits the file
    Then 3 chunks are produced: pages 1-25, 26-50, 51-60

  @critical @deterministic
  Scenario: Large PDF by file size is split into 25-page chunks
    Given a PDF with 25 pages and 15 MB file size
    When the submit workflow splits the file
    Then chunks are produced based on 25-page boundaries

  @standard @deterministic
  Scenario: PDF chunking thresholds
    When the chunking thresholds are checked
    Then MAX_FILE_SIZE_BYTES is 10485760 (10 MB)
    And MAX_PAGES is 30
    And CHUNK_SIZE_PAGES is 25

  @standard
  Scenario: Non-PDF files are not chunked
    Given an image file "photo.png"
    When the submit workflow splits the file
    Then a single chunk is produced

  @standard @edge-case
  Scenario: Non-PDF files exceeding 10MB are validated
    Given a non-PDF file exceeding 10 MB
    When the submit workflow attempts to process it
    Then the file size is validated against the 10 MB limit

  # --- Chunk Submission ---

  @critical @batch
  Scenario: Each chunk is submitted as a separate batch request
    Given a document split into 3 chunks
    When the submit workflow submits batches
    Then 3 batch requests are submitted

  @standard @batch
  Scenario: Chunk batches use Mistral OCR model by default
    When a chunk batch is submitted
    Then the model is "mistral:mistral-ocr-latest"

  @standard @batch
  Scenario: Chunk batches target the OCR endpoint
    When a chunk batch is submitted to Mistral
    Then the endpoint is "/v1/ocr" triggering file-based upload

  # --- OCR Result Parsing ---

  @critical
  Scenario: OCR response pages are parsed to text
    Given a batch result with pages containing markdown content
    When the OCR result is parsed
    Then the text is extracted from all pages joined together
    And the page_count equals the number of pages

  @standard
  Scenario: OCR result tracks token usage
    When an OCR result is parsed
    Then input_tokens and output_tokens are recorded

  # --- Chunk Reassembly ---

  @critical
  Scenario: Chunk results are reassembled in order
    Given 3 chunk OCR results with texts "Part A", "Part B", "Part C"
    When the gather workflow reassembles the results
    Then the combined text is "Part A" + newlines + "Part B" + newlines + "Part C"

  @standard
  Scenario: Reassembly sums token counts across chunks
    Given 3 chunk results with token counts 100, 150, and 200
    When the gather workflow reassembles
    Then the total input_tokens is the sum of all chunks

  @standard
  Scenario: Reassembly stores combined result under the real document ID
    Given chunk results stored under temporary chunk document IDs
    When the gather workflow reassembles
    Then the combined result is stored under the original document_id
    And the chunk rows are deleted

  @standard
  Scenario: Reassembly sums page counts across chunks
    Given chunk results with page counts 25, 25, and 10
    When the gather workflow reassembles
    Then the total page_count is 60

  # --- Result Persistence ---

  @critical
  Scenario: OCR result is persisted to the ocr_results table
    When the store workflow saves an OCR result
    Then a row is created with document_id, file_path, text, page_count, model_name, and tokens

  @standard
  Scenario: OCR result document_id is unique
    Given an OCR result already exists for document "doc-123"
    When a new result is saved for "doc-123"
    Then the existing row is replaced

  # --- Workflow Timeouts ---

  @standard @temporal
  Scenario: Submit workflow uses 30-second IO timeout
    When the submit workflow reads a file
    Then the activity timeout is 30 seconds

  @standard @temporal
  Scenario: Store workflow waits up to 25 hours for batch result
    Given a batch has been submitted
    When the store workflow waits for the signal
    Then the wait timeout is 25 hours

  @standard @temporal
  Scenario: Gather workflow waits up to 26 hours for all chunks
    Given chunks have been submitted
    When the gather workflow waits for completion signals
    Then the wait timeout is 26 hours
