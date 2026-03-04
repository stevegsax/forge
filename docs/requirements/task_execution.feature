@task-execution @phase-1
Feature: Task Execution
  Every operation follows the universal workflow step: construct message,
  send to LLM, receive response, serialize result, evaluate transition.
  The orchestrator supports single-step and planned execution modes with
  retry loops driven by validation outcomes.

  # --- Universal Workflow Step ---

  @critical
  Scenario: Single-step execution follows the universal workflow step
    Given a task in single-step mode
    When the orchestrator executes the task
    Then the execution follows: create worktree, assemble context, call LLM, write output, validate, evaluate transition

  @critical
  Scenario: LLM response is structured as document completion
    When the orchestrator calls the LLM
    Then the call is structured as a document completion for batch API compatibility

  # --- Transition Logic ---

  @critical @deterministic
  Scenario Outline: Transition signal is determined by validation results and attempt count
    Given validation results where all_passed is "<all_passed>"
    And the current attempt is <attempt> of <max_attempts>
    When the orchestrator evaluates the transition
    Then the transition signal is "<signal>"

    Examples:
      | all_passed | attempt | max_attempts | signal             |
      | true       | 1       | 2            | success            |
      | true       | 2       | 2            | success            |
      | false      | 1       | 2            | failure_retryable  |
      | false      | 2       | 2            | failure_terminal   |
      | false      | 1       | 1            | failure_terminal   |

  # --- Single-Step Mode ---

  @critical
  Scenario: Successful single-step execution commits and returns success
    Given a task in single-step mode
    And the LLM output passes all validation checks
    When the orchestrator executes the task
    Then the changes are committed with status "success"
    And the task result status is "success"

  @critical @retry
  Scenario: Failed validation triggers retry in single-step mode
    Given a task with max_attempts set to 2
    And the first attempt fails validation
    When the orchestrator evaluates the transition
    Then the signal is "failure_retryable"
    And the worktree is removed
    And a fresh worktree is created for the next attempt

  @critical
  Scenario: Terminal failure commits and returns failure
    Given a task on its final attempt
    And the validation fails
    When the orchestrator evaluates the transition
    Then the signal is "failure_terminal"
    And the changes are committed with status "failure"
    And the task result includes the validation errors

  @standard
  Scenario: Default max attempts is 2
    When a task is created with default settings
    Then the max_attempts is 2

  # --- Task Result Structure ---

  @standard
  Scenario: Successful result includes output files and context stats
    Given a successfully completed task
    When the result is returned
    Then it includes the task_id, status "success", list of output files, validation results, and context stats

  @standard
  Scenario: Failed result includes error details
    Given a task that fails terminally
    When the result is returned
    Then it includes the task_id, status "failure_terminal", validation results, and error message

  # --- Domain-Specific Defaults ---

  @critical
  Scenario Outline: Domain determines output format and validation
    Given a task in the "<domain>" domain
    When the orchestrator applies domain defaults
    Then the output format is "<output_format>"

    Examples:
      | domain          | output_format   |
      | code_generation | code files      |
      | research        | markdown files  |
      | code_review     | markdown files  |
      | documentation   | markdown files  |
      | generic         | markdown files  |

  @standard
  Scenario: Task domain is an enumerated type
    Then the valid task domains are "code_generation", "research", "code_review", "documentation", and "generic"

  # --- Retry with Error Context ---

  @critical @retry @phase-8
  Scenario: Retry attempt includes prior validation errors in context
    Given a task on attempt 2 after a validation failure
    When the orchestrator assembles context for the retry
    Then the system prompt includes a "Previous Attempt Errors" section
    And the section contains the failed check results from attempt 1

  @standard @retry @phase-8
  Scenario: Error section is placed last in system prompt for cache efficiency
    Given a retry attempt with prior errors
    When the orchestrator builds the system prompt
    Then the error section appears after all stable and task-specific content

  # --- Execution Mode Selection ---

  @standard
  Scenario: Plan mode disabled runs single-step execution
    Given a task with plan mode disabled
    When the orchestrator processes the task
    Then it runs single-step execution

  @standard
  Scenario: Plan mode enabled runs planned execution
    Given a task with plan mode enabled
    When the orchestrator processes the task
    Then it invokes the planner and executes steps sequentially
