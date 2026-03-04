@ocr-ui @web @api
Feature: OCR Web API
  A RESTful web API provides visibility into OCR job submissions, batch
  progress, and document results. The API follows the OpenAPI 3.1 specification
  so that additional clients (macOS, iOS, third-party integrations) can be
  generated from the schema. All endpoints are JSON-based and support pagination,
  filtering, and sorting.

  Background:
    Given the Forge API server is running
    And the observability store is available

  # --- OpenAPI Schema ---

  @critical @openapi
  Scenario: OpenAPI schema is served at a well-known endpoint
    When a client requests GET /openapi.json
    Then the response is a valid OpenAPI 3.1 document
    And it describes all OCR and human-in-the-loop endpoints

  @standard @openapi
  Scenario: Interactive API documentation is available
    When a client requests GET /docs
    Then an interactive API explorer is rendered (Swagger UI or Redoc)

  @standard @openapi
  Scenario: API version is included in the schema
    When a client requests GET /openapi.json
    Then the info.version field reflects the current Forge version

  # --- Job Listing ---

  @critical
  Scenario: List all OCR submissions
    Given 5 OCR jobs have been submitted
    When a client requests GET /api/v1/ocr/jobs
    Then the response contains a list of 5 job summaries
    And each summary includes job_id, document_name, status, submitted_at, and page_count

  @standard
  Scenario: Job listing is paginated
    Given 50 OCR jobs exist
    When a client requests GET /api/v1/ocr/jobs?limit=20&offset=0
    Then the response contains 20 job summaries
    And includes total_count of 50 and pagination metadata

  @standard
  Scenario: Job listing is sorted by submission time descending by default
    Given OCR jobs submitted at various times
    When a client requests GET /api/v1/ocr/jobs
    Then the jobs are ordered most-recent-first

  @standard
  Scenario: Job listing supports sorting by other fields
    When a client requests GET /api/v1/ocr/jobs?sort_by=document_name&order=asc
    Then the jobs are ordered alphabetically by document name

  @standard
  Scenario: Job listing can be filtered by status
    Given OCR jobs with statuses "processing", "completed", and "failed"
    When a client requests GET /api/v1/ocr/jobs?status=completed
    Then only jobs with status "completed" are returned

  @standard
  Scenario: Job listing can be filtered by date range
    When a client requests GET /api/v1/ocr/jobs?submitted_after=2026-03-01&submitted_before=2026-03-04
    Then only jobs submitted within that date range are returned

  # --- Job Detail ---

  @critical
  Scenario: Get detailed information for a single job
    Given an OCR job with document_id "doc-abc"
    When a client requests GET /api/v1/ocr/jobs/doc-abc
    Then the response includes:
      | field            | description                              |
      | document_id      | unique identifier for the document       |
      | document_name    | original file name                       |
      | file_path        | path to the source document              |
      | status           | current job status                       |
      | submitted_at     | ISO 8601 timestamp of submission         |
      | completed_at     | ISO 8601 timestamp of completion or null |
      | file_size_bytes  | size of the original file                |
      | mime_type        | detected MIME type                       |
      | total_pages      | total page count                         |
      | chunk_count      | number of chunks the document was split into |
      | model_name       | OCR model used                           |
      | workflow_id      | Temporal workflow ID                     |
      | metadata         | user-supplied metadata object            |

  @standard @error-handling
  Scenario: Job detail returns 404 for unknown document
    When a client requests GET /api/v1/ocr/jobs/nonexistent
    Then the response status is 404
    And the body contains an error message

  # --- Batch Drill-Down ---

  @critical
  Scenario: List batches for a specific job
    Given an OCR job "doc-abc" split into 3 chunks
    When a client requests GET /api/v1/ocr/jobs/doc-abc/batches
    Then the response contains 3 batch entries
    And each entry includes batch_id, chunk_index, status, page_range, and timestamps

  @standard
  Scenario: Batch detail shows provider-level information
    Given a batch "batch-xyz" for job "doc-abc"
    When a client requests GET /api/v1/ocr/jobs/doc-abc/batches/batch-xyz
    Then the response includes:
      | field            | description                              |
      | batch_id         | provider-specific batch identifier       |
      | request_id       | internal request UUID                    |
      | provider         | LLM provider name (e.g. "mistral")       |
      | status           | normalized batch status                  |
      | chunk_index      | position in the document chunk sequence  |
      | page_range       | start and end page numbers               |
      | submitted_at     | when the batch was submitted             |
      | updated_at       | last status update timestamp             |
      | input_tokens     | tokens consumed by this chunk            |
      | output_tokens    | tokens produced for this chunk           |
      | error_message    | error details if status is failed        |

  @standard
  Scenario: Batch listing shows aggregated token usage
    Given an OCR job with 3 completed batches
    When a client requests GET /api/v1/ocr/jobs/doc-abc/batches
    Then the response includes total_input_tokens and total_output_tokens across all batches

  # --- OCR Results ---

  @critical
  Scenario: Retrieve OCR text result for a completed job
    Given an OCR job "doc-abc" has completed successfully
    When a client requests GET /api/v1/ocr/jobs/doc-abc/result
    Then the response includes the extracted text and page_count

  @standard
  Scenario: Result endpoint returns 404 for incomplete job
    Given an OCR job "doc-abc" is still processing
    When a client requests GET /api/v1/ocr/jobs/doc-abc/result
    Then the response status is 404
    And the body explains the job has not yet completed

  @standard
  Scenario: Result can be retrieved as plain text
    Given a completed OCR job "doc-abc"
    When a client requests GET /api/v1/ocr/jobs/doc-abc/result with Accept: text/plain
    Then the response Content-Type is text/plain
    And the body is the raw extracted text

  # --- Job Submission ---

  @critical
  Scenario: Submit a new OCR job via the API
    Given a document file accessible at "/data/report.pdf"
    When a client sends POST /api/v1/ocr/jobs with body:
      """
      {
        "file_path": "/data/report.pdf",
        "document_id": "report-2026",
        "metadata": {"source": "scanner", "department": "legal"}
      }
      """
    Then a Temporal OcrSubmitWorkflow is started
    And the response status is 202 Accepted
    And the response includes the workflow_id and document_id

  @standard
  Scenario: Submit job with auto-generated document ID
    When a client sends POST /api/v1/ocr/jobs with body:
      """
      {
        "file_path": "/data/report.pdf"
      }
      """
    Then a document_id is auto-generated
    And returned in the response

  @standard @error-handling
  Scenario: Submit job with invalid file path returns 400
    When a client sends POST /api/v1/ocr/jobs with body:
      """
      {
        "file_path": "/nonexistent/file.pdf"
      }
      """
    Then the response status is 400
    And the body contains a validation error

  # --- Job Lifecycle Events ---

  @standard @sse
  Scenario: Client receives real-time status updates via SSE
    Given an OCR job "doc-abc" is in progress
    When a client connects to GET /api/v1/ocr/jobs/doc-abc/events
    Then the server sends Server-Sent Events as the job status changes
    And events include type, timestamp, and relevant data

  @standard @sse
  Scenario: SSE stream ends when job reaches terminal state
    Given a client is connected to the event stream for job "doc-abc"
    When the job completes
    Then a "completed" event is sent
    And the SSE connection is closed

  # --- Health and Metadata ---

  @standard
  Scenario: Health check endpoint
    When a client requests GET /api/v1/health
    Then the response includes status "ok", database connectivity, and Temporal connectivity

  @standard
  Scenario: Job statistics endpoint
    When a client requests GET /api/v1/ocr/stats
    Then the response includes counts by status, total pages processed, and total tokens used
