@batch @phase-14
Feature: Batch Processing
  The orchestrator submits LLM calls as batch requests for cost efficiency.
  A multi-provider poller monitors batch jobs and signals workflows on completion.
  Both Anthropic and Mistral batch APIs are supported with provider-specific
  submission modes and result parsing.

  # --- Batch Submission ---

  @critical
  Scenario: Batch request is submitted with provider-prefixed model
    Given a task using model "anthropic:claude-sonnet-4-5-20250929"
    When the orchestrator submits a batch request
    Then the request is routed to the Anthropic provider
    And the batch ID is returned

  @critical
  Scenario: Batch submission is recorded in the database
    When a batch is submitted successfully
    Then a batch_jobs row is created with status "submitted"
    And the provider name is recorded

  @standard
  Scenario: Failed batch submission is recorded before retry
    When a batch submission fails
    Then a batch_jobs row is created with status "failed" and the error message

  @standard
  Scenario: Anthropic batch uses inline request submission
    Given a list of batch requests for Anthropic
    When the batch is submitted
    Then the requests are passed inline to the Messages Batch API

  @standard @llm-providers
  Scenario: Mistral batch uses inline requests for chat completions
    Given a list of batch requests for Mistral chat completions
    When the batch is submitted
    Then the requests are passed inline to the Mistral batch API

  @critical @llm-providers @ocr
  Scenario: Mistral batch uses file-based upload for OCR endpoint
    Given a list of batch requests targeting the "/v1/ocr" endpoint
    When the Mistral provider submits the batch
    Then a JSONL file is uploaded with purpose "batch"
    And the batch job is created referencing the uploaded file ID

  # --- Batch Polling ---

  @critical @temporal
  Scenario: Poller queries pending batch jobs from the database
    Given batch jobs with status "submitted" exist in the database
    When the batch poller runs
    Then it queries all pending batch jobs

  @critical @temporal
  Scenario: Poller signals workflows on batch completion
    Given a batch job transitions to "ended" status
    When the poller detects the completion
    Then it signals the associated workflow with the batch results

  @standard @temporal
  Scenario: Poller handles both Anthropic and Mistral batches
    Given pending batch jobs from both Anthropic and Mistral providers
    When the batch poller runs
    Then it polls each job using the correct provider

  @standard
  Scenario: Default poll interval is 600 seconds
    When the worker starts with default settings
    Then the batch poll interval is 600 seconds

  @standard
  Scenario: Poll interval is configurable via CLI
    Given the CLI flag "--batch-poll-interval" is set to 300
    When the worker starts
    Then the batch poll interval is 300 seconds

  # --- Batch Status Mapping ---

  @critical
  Scenario Outline: Batch poll status is normalized across providers
    Given a batch with provider-native status "<native_status>" from "<provider>"
    When the poller normalizes the status
    Then the normalized status is "<normalized_status>"

    Examples:
      | provider  | native_status          | normalized_status |
      | mistral   | QUEUED                 | pending           |
      | mistral   | RUNNING                | in_progress       |
      | mistral   | SUCCESS                | ended             |
      | mistral   | FAILED                 | failed            |
      | mistral   | TIMEOUT_EXCEEDED       | expired           |
      | mistral   | CANCELLED              | canceled          |

  # --- Anomaly Detection ---

  @standard
  Scenario: Batch missing for over 24 hours is marked as missing
    Given a batch job submitted more than 24 hours ago
    And the provider cannot retrieve the batch
    When the poller checks the batch
    Then the job status is updated to "missing"
    And the error message is "Batch unretrievable after 24h"

  @standard @error-handling
  Scenario: Poll failure does not crash the poller
    Given a batch poll raises an unexpected exception
    When the poller processes the batch
    Then the error is logged as a warning
    And the poller continues to the next batch

  # --- Terminal Failure ---

  @critical @error-handling
  Scenario Outline: Terminal batch status signals error to workflow
    Given a batch reaches terminal status "<status>"
    When the poller processes the result
    Then it signals the workflow with an error

    Examples:
      | status   |
      | failed   |
      | expired  |
      | canceled |

  # --- Batch Result Parsing ---

  @critical
  Scenario: Successful batch entries are signaled individually
    Given a batch with 3 successful result entries
    When the poller processes the results
    Then 3 individual signals are sent to the workflow

  @standard
  Scenario: Failed batch entries include error details
    Given a batch with a failed result entry
    When the poller processes the results
    Then the signal includes the error message from the batch entry

  @standard @llm-providers
  Scenario: Mistral batch result parses output_file and error_file
    Given a completed Mistral batch with output_file and error_file
    When the provider retrieves batch results
    Then entries from both files are parsed
    And output_file entries take priority for duplicate custom_ids

  # --- Signal-Based Wait ---

  @critical @temporal
  Scenario: Workflow waits for batch result signal
    Given a batch has been submitted
    When the workflow enters the wait state
    Then it waits for a batch_result signal from the poller

  @standard @temporal @edge-case
  Scenario: Batch wait timeout after 25 hours
    Given a batch has been submitted
    And no signal is received within 25 hours
    When the wait timeout elapses
    Then an ApplicationError is raised

  # --- Sync Fallback ---

  @standard @edge-case
  Scenario: Sync fallback when batch is not supported
    Given a provider that does not support batch mode
    When the orchestrator dispatches a generation call
    Then it falls back to a synchronous LLM call
