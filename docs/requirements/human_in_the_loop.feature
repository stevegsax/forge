@hitl @ocr-ui
Feature: Human in the Loop
  Workflows can pause execution and request human input. The system presents
  structured prompts to operators via the web API and CLI. Humans respond
  through either interface, and the workflow resumes with the provided input.
  This enables quality review, ambiguity resolution, and approval gates
  within otherwise automated pipelines.

  # --- Workflow Pause and Signal ---

  @critical @temporal
  Scenario: Workflow pauses and emits a human input request
    Given an OCR workflow encounters an ambiguous result
    When the workflow requires human input
    Then it transitions to "awaiting_human_input" state
    And a HumanInputRequest is persisted to the database
    And the Temporal workflow waits for a "human_input_received" signal

  @critical @temporal
  Scenario: Workflow resumes when human input is received
    Given a workflow is in "awaiting_human_input" state
    When a "human_input_received" signal is delivered with the operator's response
    Then the workflow resumes execution with the provided input
    And the HumanInputRequest status is updated to "resolved"

  @standard @temporal
  Scenario: Human input request includes a timeout
    Given a workflow requests human input with a 4-hour timeout
    When 4 hours elapse without a response
    Then the workflow transitions to a configurable timeout action
    And the HumanInputRequest status is updated to "expired"

  @standard @temporal
  Scenario: Timeout action is configurable per request
    When a workflow creates a HumanInputRequest
    Then it specifies a timeout_action of "abort", "skip", or "retry"
    And the workflow follows that action if the timeout elapses

  # --- Human Input Request Model ---

  @critical
  Scenario: HumanInputRequest contains structured prompt information
    When a workflow creates a HumanInputRequest
    Then the request includes:
      | field            | description                                        |
      | request_id       | unique identifier for this input request            |
      | workflow_id      | Temporal workflow that is waiting                   |
      | document_id      | associated document (if applicable)                 |
      | prompt_type      | category: "review", "choice", "approval", "freeform"|
      | title            | short human-readable summary of what is needed      |
      | description      | detailed explanation and context                    |
      | options          | list of choices (for "choice" and "approval" types) |
      | context          | relevant data (e.g. OCR snippet, image URL)         |
      | priority         | "low", "normal", "high", "urgent"                   |
      | timeout_seconds  | how long to wait before timeout_action              |
      | timeout_action   | what to do on timeout: "abort", "skip", "retry"     |
      | created_at       | when the request was created                        |
      | status           | "pending", "resolved", "expired", "canceled"        |

  @standard
  Scenario: Approval prompt has accept/reject options
    When a workflow creates an "approval" type request
    Then the options include at least "approve" and "reject"
    And an optional "comment" field is available

  @standard
  Scenario: Choice prompt presents multiple options
    When a workflow creates a "choice" type request with 3 options
    Then each option has a label, value, and optional description

  @standard
  Scenario: Freeform prompt accepts arbitrary text input
    When a workflow creates a "freeform" type request
    Then the response schema accepts a text field with optional max_length

  @standard
  Scenario: Review prompt presents content for inspection
    When a workflow creates a "review" type request
    Then the context includes the content to be reviewed
    And the options include "approve", "reject", and "edit"

  # --- Persistence ---

  @critical
  Scenario: Human input requests are persisted to the database
    When a HumanInputRequest is created
    Then a row is inserted into the human_input_requests table
    And the row includes all request fields

  @standard
  Scenario: Human input response is persisted
    When an operator responds to a HumanInputRequest
    Then the response value, operator identity, and timestamp are stored
    And the request status transitions from "pending" to "resolved"

  @standard
  Scenario: Request history is preserved for auditing
    Given a HumanInputRequest has been resolved
    When the request is queried
    Then both the original request and the response are available
    And the time-to-resolution is calculable from timestamps

  # --- Web API Endpoints ---

  @critical @web @api
  Scenario: List pending human input requests
    Given 3 workflows are awaiting human input
    When a client requests GET /api/v1/hitl/requests?status=pending
    Then the response contains 3 pending requests
    And each request includes request_id, title, priority, prompt_type, and created_at

  @critical @web @api
  Scenario: Get details of a specific human input request
    Given a pending request with request_id "req-123"
    When a client requests GET /api/v1/hitl/requests/req-123
    Then the response includes all HumanInputRequest fields
    And includes the context data for rendering the prompt

  @critical @web @api
  Scenario: Submit a response to a human input request
    Given a pending request with request_id "req-123"
    When a client sends POST /api/v1/hitl/requests/req-123/respond with body:
      """
      {
        "value": "approve",
        "comment": "Looks correct after manual review"
      }
      """
    Then the request status transitions to "resolved"
    And a "human_input_received" signal is sent to the waiting workflow
    And the response status is 200

  @standard @web @api
  Scenario: Cannot respond to an already-resolved request
    Given a request with request_id "req-123" that has been resolved
    When a client sends POST /api/v1/hitl/requests/req-123/respond
    Then the response status is 409 Conflict
    And the body explains the request has already been resolved

  @standard @web @api
  Scenario: Cannot respond to an expired request
    Given a request with request_id "req-123" that has expired
    When a client sends POST /api/v1/hitl/requests/req-123/respond
    Then the response status is 410 Gone
    And the body explains the request has expired

  @standard @web @api
  Scenario: List requests filtered by document
    When a client requests GET /api/v1/hitl/requests?document_id=doc-abc
    Then only requests associated with document "doc-abc" are returned

  @standard @web @api
  Scenario: List requests filtered by priority
    When a client requests GET /api/v1/hitl/requests?priority=urgent
    Then only urgent requests are returned

  @standard @web @api
  Scenario: List requests sorted by priority then age
    When a client requests GET /api/v1/hitl/requests?status=pending
    Then requests are ordered by priority (urgent first) then by created_at (oldest first)

  @standard @web @api @sse
  Scenario: Client receives real-time notification of new requests via SSE
    Given a client is connected to GET /api/v1/hitl/events
    When a new HumanInputRequest is created
    Then the client receives a "new_request" event with the request summary

  @standard @web @api @sse
  Scenario: SSE notifies when a request is resolved by another operator
    Given a client is connected to GET /api/v1/hitl/events
    When request "req-123" is resolved
    Then the client receives a "request_resolved" event

  # --- CLI Commands ---

  @critical @cli
  Scenario: List pending human input requests in the CLI
    Given 3 workflows are awaiting human input
    When the user runs "forge hitl list"
    Then a table is displayed with columns: request_id, title, type, priority, waiting_since
    And requests are ordered by priority then age

  @critical @cli
  Scenario: View details of a human input request
    Given a pending request with request_id "req-123"
    When the user runs "forge hitl show req-123"
    Then the full prompt is displayed including title, description, context, and options

  @critical @cli
  Scenario: Respond to a human input request via CLI
    Given a pending "approval" request with request_id "req-123"
    When the user runs "forge hitl respond req-123 --value approve --comment 'Verified manually'"
    Then the response is submitted
    And the waiting workflow is signaled
    And a confirmation message is displayed

  @standard @cli
  Scenario: Interactive response mode
    Given a pending "choice" request with request_id "req-123" and 3 options
    When the user runs "forge hitl respond req-123"
    Then the options are displayed as a numbered list
    And the user is prompted to select an option interactively

  @standard @cli
  Scenario: Respond to a freeform request via CLI
    Given a pending "freeform" request with request_id "req-123"
    When the user runs "forge hitl respond req-123 --value 'The correct spelling is Acme Corp'"
    Then the text response is submitted

  @standard @cli
  Scenario: List requests with JSON output
    When the user runs "forge hitl list --json"
    Then the output is a JSON array of request summaries

  @standard @cli
  Scenario: Filter pending requests by document
    When the user runs "forge hitl list --document-id doc-abc"
    Then only requests for document "doc-abc" are shown

  @standard @cli
  Scenario: Show count of pending requests
    When the user runs "forge hitl count"
    Then the output shows the number of pending requests by priority

  @standard @cli @error-handling
  Scenario: Respond to nonexistent request
    When the user runs "forge hitl respond nonexistent --value approve"
    Then the error message is "No pending request found with ID: nonexistent"
    And the exit code is 1

  # --- Notification and Alerting ---

  @standard
  Scenario: Urgent requests trigger a notification
    When a workflow creates a HumanInputRequest with priority "urgent"
    Then a notification event is emitted
    And clients connected via SSE receive it immediately

  @standard
  Scenario: Approaching-timeout warning is emitted
    Given a request with a 4-hour timeout
    When 3 hours have elapsed without a response
    Then a "timeout_warning" event is emitted with time remaining

  # --- Concurrency and Safety ---

  @standard @edge-case
  Scenario: Only one response is accepted per request
    Given a pending request "req-123"
    When two operators submit responses simultaneously
    Then only the first response is accepted
    And the second receives a 409 Conflict response

  @standard @edge-case
  Scenario: Response validation rejects invalid values
    Given a "choice" request with options ["option-a", "option-b"]
    When a response is submitted with value "option-c"
    Then the response status is 422 Unprocessable Entity
    And the body lists the valid options

  @standard
  Scenario: Canceled workflow cleans up pending requests
    Given a workflow with a pending HumanInputRequest
    When the workflow is canceled
    Then the request status is updated to "canceled"
