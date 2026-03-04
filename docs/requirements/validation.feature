@validation @phase-1 @phase-8
Feature: Validation Pipeline
  The orchestrator validates LLM output through a fix-then-check pipeline.
  Auto-fix runs ruff lint fix and format before validation checks.
  Failed validations trigger error-aware retries with AST context enrichment.

  Background:
    Given a worktree with files written by the LLM

  # --- Fix-Then-Check Pipeline ---

  @critical
  Scenario: Auto-fix runs before validation checks
    Given validation config with auto_fix enabled
    When the orchestrator validates the output
    Then "ruff check --fix" runs before "ruff check --no-fix"
    And "ruff format" runs before "ruff format --check"

  @critical
  Scenario: Ruff lint check detects violations after auto-fix
    Given a file with a lint violation that auto-fix cannot resolve
    When the orchestrator runs validation
    Then the ruff_lint check returns passed as false
    And the summary describes the violation

  @standard
  Scenario: Ruff format check detects formatting issues after auto-format
    Given a file with formatting issues that auto-format cannot resolve
    When the orchestrator runs validation
    Then the ruff_format check returns passed as false

  @standard
  Scenario: Test execution runs custom test command
    Given validation config with run_tests enabled and test_command "pytest tests/ -x"
    When the orchestrator runs validation
    Then the custom test command is executed
    And the result indicates whether tests passed

  @standard
  Scenario: Test execution is disabled by default
    Given default validation config
    When the orchestrator runs validation
    Then no test command is executed

  # --- Domain-Specific Validation Defaults ---

  @critical
  Scenario: Code generation domain enables auto-fix and linting
    Given a task in the "code_generation" domain
    When the orchestrator applies domain validation defaults
    Then auto_fix is true, run_ruff_lint is true, run_ruff_format is true, and run_tests is false

  @standard
  Scenario Outline: Non-code domains disable all validation
    Given a task in the "<domain>" domain
    When the orchestrator applies domain validation defaults
    Then auto_fix is false, run_ruff_lint is false, run_ruff_format is false, and run_tests is false

    Examples:
      | domain          |
      | research        |
      | code_review     |
      | documentation   |
      | generic         |

  # --- Validation Result Structure ---

  @standard
  Scenario: Validation result includes check name and pass status
    When a validation check completes
    Then the result includes check_name, passed boolean, and summary

  @standard
  Scenario: Long validation output is truncated in summary
    Given a validation check that produces output exceeding 200 characters
    When the result is parsed
    Then the summary is truncated to 200 characters
    And the full output is preserved in the details field

  @standard
  Scenario: Short validation output has no details field
    Given a validation check that produces output under 200 characters
    When the result is parsed
    Then the summary contains the full output
    And the details field is null

  # --- Error-Aware Retries ---

  @critical @retry
  Scenario: Failed validation triggers retry with error context
    Given a task with max_attempts set to 2
    And the first attempt fails ruff lint validation
    When the orchestrator retries the task
    Then the retry prompt includes a "Previous Attempt Errors" section
    And the section shows "Attempt 1 of 2"

  @critical @retry
  Scenario: Error section includes AST context for ruff errors
    Given a ruff lint error at line 42 of "src/module.py"
    When the error section is built for the retry prompt
    Then it includes the enclosing function or class definition
    And an error marker "# <-- ERROR" is placed at the error line

  @standard @retry
  Scenario: AST context finds innermost enclosing scope
    Given a ruff lint error inside a nested function
    When the error section is built
    Then the snippet shows the innermost enclosing function, not the outer class

  @standard @retry
  Scenario: Ruff error lines are deduplicated by file and line
    Given multiple ruff errors at the same file and line
    When the error section is built
    Then only one snippet is included for that location

  @standard @retry
  Scenario: Error details are truncated to 2000 characters
    Given a validation failure with output exceeding 2000 characters
    When the error section is built
    Then the error details are truncated to 2000 characters

  @critical @retry
  Scenario: Worktree is reset between retry attempts
    Given a task on attempt 1 that fails validation
    When the orchestrator prepares for attempt 2
    Then the worktree from attempt 1 is removed
    And a fresh worktree is created for attempt 2

  # --- Subprocess ---

  @standard
  Scenario: Validation subprocess timeout
    When a validation subprocess exceeds 60 seconds
    Then the subprocess is killed and the check is marked as failed

  @standard
  Scenario: Validation activity sends heartbeats
    When the validation activity runs
    Then it sends heartbeats within the 120-second heartbeat interval
