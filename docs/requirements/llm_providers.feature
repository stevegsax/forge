@llm-providers
Feature: LLM Provider Abstraction
  The orchestrator supports multiple LLM providers through a unified protocol.
  Providers are identified by "provider:model" syntax and cached as singletons.
  Features unsupported by a provider are silently degraded per the D63 policy.

  # --- Provider ID Syntax ---

  @critical @phase-11
  Scenario: Parse provider-prefixed model ID
    When the orchestrator parses the model ID "mistral:mistral-large-latest"
    Then the provider is "mistral" and the model is "mistral-large-latest"

  @critical @phase-11
  Scenario: Bare model name defaults to Anthropic
    When the orchestrator parses the model ID "claude-sonnet-4-5-20250929"
    Then the provider is "anthropic" and the model is "claude-sonnet-4-5-20250929"

  @standard @phase-11
  Scenario: Unknown provider raises an error
    When the orchestrator attempts to get a provider for "unknown:some-model"
    Then a ValueError is raised with message containing "Unknown LLM provider"

  # --- Provider Registry ---

  @critical @phase-11
  Scenario: Provider instances are cached as singletons
    When the orchestrator requests the Anthropic provider twice
    Then both calls return the same instance

  @standard @phase-11
  Scenario: Provider cache can be reset for test isolation
    Given the provider cache contains an Anthropic provider
    When the provider cache is reset
    Then requesting the Anthropic provider creates a new instance

  # --- Anthropic Provider ---

  @critical @phase-9
  Scenario: Anthropic provider supports prompt caching
    Given a message list with cache_control enabled on the system message
    When the Anthropic provider builds request parameters
    Then the system message includes a cache_control block

  @standard @phase-9
  Scenario: Anthropic provider tracks cache token usage
    When the Anthropic provider receives a response with cache tokens
    Then the response includes cache_creation_input_tokens and cache_read_input_tokens

  @critical @phase-12
  Scenario: Anthropic provider supports extended thinking
    Given a thinking budget of 10000 tokens for an Opus model
    When the Anthropic provider builds request parameters
    Then the parameters include a thinking block with type "enabled" and the budget

  @edge-case @phase-12
  Scenario: Extended thinking is not available for Haiku
    Given a thinking budget of 10000 tokens for a Haiku model
    When the Anthropic provider builds the thinking parameter
    Then no thinking block is included

  @critical @phase-14
  Scenario: Anthropic provider supports batch submission
    Given a list of batch requests
    When the Anthropic provider submits a batch
    Then a batch ID is returned from the Messages Batch API

  @standard @phase-14
  Scenario: Anthropic provider parses structured batch results
    Given a batch result with tool_use content blocks
    When the Anthropic provider parses the batch result
    Then the tool_input dict is extracted from the response

  @standard @phase-14
  Scenario: Anthropic provider parses text-only batch results
    Given a batch result with text content only
    When the Anthropic provider parses the batch result with no output type
    Then the text_content is extracted from the response

  # --- Mistral Provider ---

  @critical @llm-providers
  Scenario: Mistral provider uses tool-call format for structured output
    Given a pydantic output type for structured output
    When the Mistral provider builds request parameters
    Then the parameters include a tool definition with the schema and tool_choice "any"

  @standard @llm-providers
  Scenario: Mistral provider supports multimodal image content
    Given a message containing an image content block
    When the Mistral provider builds request parameters
    Then the image is converted to a data URI with type "image_url"

  @standard @llm-providers
  Scenario: Mistral provider supports multimodal document content
    Given a message containing a document content block
    When the Mistral provider builds request parameters
    Then the document is converted to a data URI with type "document_url"

  @critical @llm-providers @batch
  Scenario: Mistral provider uses file-based upload for OCR batch endpoint
    Given a batch of OCR requests targeting the "/v1/ocr" endpoint
    When the Mistral provider submits the batch
    Then it uploads a JSONL file with purpose "batch" and creates the batch job referencing that file

  @standard @llm-providers @batch
  Scenario: Mistral provider uses inline requests for non-OCR batch endpoints
    Given a batch of chat completion requests
    When the Mistral provider submits the batch
    Then the requests are submitted inline without file upload

  @standard @llm-providers @batch
  Scenario: Mistral batch poll maps native statuses to normalized statuses
    Given a Mistral batch job with status "SUCCESS"
    When the Mistral provider polls the batch
    Then the normalized status is "ended"

  @standard @llm-providers @batch
  Scenario: Mistral batch result prioritizes output file over error file
    Given a Mistral batch with both error_file and output_file entries for the same custom_id
    When the Mistral provider retrieves batch results
    Then the output_file entry takes priority and the error entry is removed

  # --- Feature Degradation (D63) ---

  @critical @llm-providers
  Scenario: Prompt caching is silently skipped for Mistral
    Given cache_instructions is enabled
    When the Mistral provider builds request parameters
    Then no cache control blocks are included and no error is raised

  @critical @llm-providers
  Scenario: Extended thinking is silently skipped for Mistral
    Given a thinking budget of 10000 tokens
    When the Mistral provider builds request parameters
    Then the thinking budget is ignored and no error is raised

  @standard @llm-providers
  Scenario: Mistral provider always reports zero cache tokens
    When the Mistral provider returns a response
    Then cache_creation_input_tokens is 0 and cache_read_input_tokens is 0

  # --- Error Handling ---

  @critical @error-handling
  Scenario Outline: Non-retryable LLM errors
    When the LLM provider raises a "<error_type>"
    Then the error is not retried by the orchestrator

    Examples:
      | error_type              |
      | BadRequestError         |
      | AuthenticationError     |
      | PermissionDeniedError   |
      | NotFoundError           |

  @standard @error-handling @retry
  Scenario: Retryable errors are retried up to 3 attempts
    When the LLM provider raises a transient network error
    Then the orchestrator retries the call up to 3 times
