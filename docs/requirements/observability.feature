@observability @phase-5
Feature: Observability Store
  The orchestrator persists all LLM interactions, task runs, batch jobs,
  playbook entries, OCR results, OCR images, and file content blobs to a
  SQLite database. Writes are best-effort to avoid blocking task execution.
  The database path follows the XDG Base Directory Specification.

  # --- Database Location ---

  @critical
  Scenario: Database path follows XDG specification
    Given no environment overrides are set
    When the orchestrator resolves the database path
    Then the path is "~/.local/state/forge/forge.db"

  @standard
  Scenario: XDG_STATE_HOME overrides default path
    Given XDG_STATE_HOME is set to "/custom/state"
    When the orchestrator resolves the database path
    Then the path is "/custom/state/forge/forge.db"

  @standard
  Scenario: FORGE_DB_PATH overrides XDG path
    Given FORGE_DB_PATH is set to "/data/forge.db"
    When the orchestrator resolves the database path
    Then the path is "/data/forge.db"

  @standard @edge-case
  Scenario: Empty FORGE_DB_PATH disables the store
    Given FORGE_DB_PATH is set to ""
    When the orchestrator resolves the database path
    Then the store is disabled and no database is created

  # --- Database Configuration ---

  @standard
  Scenario: SQLite uses WAL journal mode
    When the database engine is created
    Then the journal mode is set to WAL for crash recovery

  @standard
  Scenario: Alembic migrations run on startup
    When the database is initialized
    Then Alembic migrations are executed programmatically

  # --- Interactions Table ---

  @critical
  Scenario: LLM interaction is persisted after each call
    When the orchestrator completes an LLM call
    Then a row is inserted into the interactions table
    And it includes task_id, model_name, input_tokens, output_tokens, and latency_ms

  @standard
  Scenario: Interaction records step and sub-task context
    Given an LLM call for step "step-2" of sub-task "analyze"
    When the interaction is persisted
    Then the row includes step_id "step-2" and sub_task_id "analyze"

  @standard @phase-9
  Scenario: Interaction records cache token statistics
    Given an Anthropic LLM call with cache hits
    When the interaction is persisted
    Then the row includes cache_creation_input_tokens and cache_read_input_tokens

  @standard
  Scenario: Interaction includes context statistics
    When the interaction is persisted
    Then the context_stats_json field contains serialized context assembly stats

  # --- Runs Table ---

  @critical
  Scenario: Task run is recorded on completion
    Given a task completes with status "success"
    When the run is saved
    Then a row is created with task_id, workflow_id, status, and result_json

  @standard
  Scenario: Workflow ID is unique across runs
    When a run is saved
    Then the workflow_id column enforces uniqueness

  @standard
  Scenario: Recent runs can be listed
    Given multiple completed runs
    When recent runs are queried with limit 20
    Then up to 20 runs are returned ordered by creation date descending

  # --- Batch Jobs Table ---

  @critical @batch
  Scenario: Batch submission is recorded
    When a batch is submitted
    Then a batch_jobs row is created with batch_id, workflow_id, status "submitted", and provider name

  @standard @batch
  Scenario: Batch job status is updated on poll
    Given a batch job with status "submitted"
    When the poller updates the batch status
    Then the status and updated_at fields are modified

  @standard @batch
  Scenario: Provider column defaults to "anthropic"
    When a batch job is created without specifying a provider
    Then the provider defaults to "anthropic"

  @standard @batch
  Scenario: Pending batch jobs can be queried
    Given batch jobs with various statuses
    When pending batch jobs are queried
    Then only jobs with status "submitted" are returned

  # --- Playbooks Table ---

  @critical @knowledge
  Scenario: Playbook entries are saved in bulk
    Given a list of extracted playbook entries
    When the entries are saved
    Then rows are created with title, content, tags_json, source_task_id, and source_workflow_id

  @standard @knowledge
  Scenario: Playbooks can be queried by tags
    Given playbook entries with various tags
    When playbooks are queried with tags "python" and "validation"
    Then entries matching any of those tags are returned

  @standard @knowledge
  Scenario: Playbooks can be listed by recency
    When recent playbooks are queried with limit 20
    Then up to 20 entries are returned ordered by creation date descending

  # --- OCR Results Table ---

  @critical @ocr
  Scenario: OCR result is persisted
    When an OCR result is saved
    Then a row is created with document_id, file_path, text, page_count, model_name, input_tokens, and output_tokens

  @standard @ocr
  Scenario: OCR result document_id is unique
    When an OCR result is saved
    Then the document_id column enforces uniqueness

  @standard @ocr
  Scenario: OCR results can be deleted
    Given an OCR result for document "chunk-1"
    When the result is deleted
    Then the row is removed from the table

  # --- OCR Images Table ---

  @critical @ocr
  Scenario: OCR image is persisted
    When an OCR image is saved
    Then a row is created with id, document_id, page_index, original_image_id, data, mime_type, and file_size_bytes

  @standard @ocr
  Scenario: OCR image document_id defaults to empty string
    When an OCR image is stored during batch polling
    Then the document_id is empty because the real document ID is not yet known

  @standard @ocr
  Scenario: OCR image document_id is updated after result storage
    Given OCR images with empty document_id
    When the store activity saves the OCR result with image_ids
    Then the image rows are updated with the correct document_id

  @standard @ocr
  Scenario: OCR images are reassigned during chunk reassembly
    Given OCR images assigned to chunk document IDs
    When the gather workflow reassembles chunk results
    Then the image rows are reassigned to the final document_id

  @standard @ocr
  Scenario: OCR image metadata can be listed without blob data
    Given a document with 3 stored images
    When the images are listed for the document
    Then metadata is returned (id, page_index, mime_type, file_size_bytes)
    And binary data is not included in the list response

  @standard @ocr
  Scenario: OCR image bounding box is stored when available
    When an OCR image with bounding box coordinates is saved
    Then the row includes top_left_x, top_left_y, bottom_right_x, and bottom_right_y

  # --- File Content Blobs Table ---

  @critical @ocr
  Scenario: File content blob is stored
    When a file content blob is saved
    Then a row is created with id, binary data, mime_type, and file_size_bytes

  @standard @ocr
  Scenario: File content blob can be retrieved by ID
    Given a blob with id "blob-123"
    When the blob is retrieved
    Then the binary data and mime_type are returned

  @standard @ocr
  Scenario: File content blob can be deleted
    Given a blob with id "blob-123"
    When the blob is deleted
    Then the row is removed from the table

  # --- Best-Effort Writes ---

  @critical @error-handling
  Scenario: Interaction persistence never raises exceptions
    Given the database is unavailable
    When the orchestrator attempts to persist an interaction
    Then no exception is raised
    And the error is logged

  @standard @error-handling
  Scenario: Playbook loading failure returns empty list
    Given the database is unavailable
    When playbooks are loaded for a task
    Then an empty list is returned without error
