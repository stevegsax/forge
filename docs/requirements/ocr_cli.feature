@ocr-ui @cli
Feature: OCR CLI
  The CLI provides commands for inspecting OCR job status, drilling down into
  batches, viewing results, and responding to human-in-the-loop prompts. Output
  supports both human-readable tables and machine-readable JSON.

  # --- Job Listing ---

  @critical
  Scenario: List recent OCR jobs
    Given 5 OCR jobs have been submitted
    When the user runs "forge ocr jobs"
    Then a table is displayed with columns: document_id, document_name, status, submitted_at, pages
    And jobs are ordered most-recent-first

  @standard
  Scenario: List jobs with limit
    Given 50 OCR jobs exist
    When the user runs "forge ocr jobs --limit 10"
    Then at most 10 jobs are displayed

  @standard
  Scenario: List jobs filtered by status
    When the user runs "forge ocr jobs --status completed"
    Then only completed jobs are displayed

  @standard
  Scenario: List jobs with JSON output
    When the user runs "forge ocr jobs --json"
    Then the output is a JSON array of job summaries

  @standard
  Scenario: List jobs filtered by date
    When the user runs "forge ocr jobs --since 2026-03-01"
    Then only jobs submitted on or after that date are displayed

  # --- Job Detail ---

  @critical
  Scenario: Show detailed status for a specific job
    Given an OCR job with document_id "doc-abc"
    When the user runs "forge ocr status doc-abc"
    Then the output includes:
      | field           | example value                    |
      | Document ID     | doc-abc                          |
      | Document Name   | report.pdf                       |
      | Status          | completed                        |
      | Submitted       | 2026-03-04T10:30:00Z             |
      | Completed       | 2026-03-04T10:35:00Z             |
      | File Size       | 4.2 MB                           |
      | Pages           | 42                               |
      | Chunks          | 2                                |
      | Model           | mistral:mistral-ocr-latest       |
      | Workflow ID     | ocr-submit-abc123                |
      | Input Tokens    | 12,450                           |
      | Output Tokens   | 8,320                            |

  @standard
  Scenario: Job detail with JSON output
    When the user runs "forge ocr status doc-abc --json"
    Then the output is a JSON object with all job fields

  @standard @error-handling
  Scenario: Job detail for unknown document
    When the user runs "forge ocr status nonexistent"
    Then the error message is "No OCR job found for document ID: nonexistent"
    And the exit code is 1

  # --- Batch Drill-Down ---

  @critical
  Scenario: List batches for a job
    Given an OCR job "doc-abc" split into 3 chunks
    When the user runs "forge ocr batches doc-abc"
    Then a table is displayed with columns: batch_id, chunk, status, pages, submitted, updated
    And 3 rows are shown

  @standard
  Scenario: Batch listing with JSON output
    When the user runs "forge ocr batches doc-abc --json"
    Then the output is a JSON array of batch details

  @standard
  Scenario: Batch detail shows token usage
    When the user runs "forge ocr batches doc-abc"
    Then each row includes input_tokens and output_tokens
    And a summary row shows totals

  @standard
  Scenario: Batch detail shows error for failed batches
    Given a batch for job "doc-abc" has status "failed" with error "timeout"
    When the user runs "forge ocr batches doc-abc"
    Then the failed batch row includes the error message

  # --- OCR Results ---

  @critical
  Scenario: View OCR result text
    Given an OCR job "doc-abc" has completed
    When the user runs "forge ocr result doc-abc"
    Then the extracted text is printed to stdout

  @standard
  Scenario: Save OCR result to file
    When the user runs "forge ocr result doc-abc --output /tmp/result.txt"
    Then the extracted text is written to /tmp/result.txt

  @standard
  Scenario: Result with metadata
    When the user runs "forge ocr result doc-abc --json"
    Then the output includes text, page_count, model_name, and token counts as JSON

  @standard @error-handling
  Scenario: Result for incomplete job
    Given an OCR job "doc-abc" is still processing
    When the user runs "forge ocr result doc-abc"
    Then the error message indicates the job is not yet complete
    And the exit code is 1

  # --- Job Submission ---

  @critical
  Scenario: Submit a document for OCR via CLI
    When the user runs "forge ocr submit /data/report.pdf"
    Then an OcrSubmitWorkflow is started
    And the output shows the workflow_id and document_id

  @standard
  Scenario: Submit with custom document ID
    When the user runs "forge ocr submit /data/report.pdf --document-id report-2026"
    Then the document_id is "report-2026"

  @standard
  Scenario: Submit with metadata
    When the user runs "forge ocr submit /data/report.pdf --meta source=scanner --meta dept=legal"
    Then the metadata {"source": "scanner", "dept": "legal"} is associated with the job

  @standard
  Scenario: Submit with --wait blocks until completion
    When the user runs "forge ocr submit /data/report.pdf --wait"
    Then the command blocks until the OCR workflow completes
    And prints the result summary

  @standard
  Scenario: Submit with --json outputs machine-readable result
    When the user runs "forge ocr submit /data/report.pdf --json"
    Then the output is a JSON object with workflow_id and document_id

  # --- Watch Mode ---

  @standard
  Scenario: Watch a job in progress
    Given an OCR job "doc-abc" is processing
    When the user runs "forge ocr watch doc-abc"
    Then the display updates as batches complete
    And shows a progress indicator (e.g. "2/3 chunks complete")

  @standard
  Scenario: Watch exits when job completes
    Given a client is watching job "doc-abc"
    When all batches complete
    Then the final status is displayed
    And the command exits with code 0

  # --- Summary Statistics ---

  @standard
  Scenario: Show OCR processing statistics
    When the user runs "forge ocr stats"
    Then the output includes:
      | metric               | description                    |
      | Total jobs           | count of all OCR jobs           |
      | Completed            | count of completed jobs         |
      | In progress          | count of active jobs            |
      | Failed               | count of failed jobs            |
      | Total pages          | sum of all pages processed      |
      | Total tokens         | sum of all tokens used          |

  @standard
  Scenario: Statistics with JSON output
    When the user runs "forge ocr stats --json"
    Then the output is a JSON object with the statistics
