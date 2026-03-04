@exploration @phase-7
Feature: LLM-Guided Exploration
  The orchestrator runs an iterative exploration loop where the LLM requests
  context from 12 providers before generating output. The loop continues until
  the LLM signals readiness or the round limit is reached.

  Background:
    Given a task with LLM-guided exploration enabled

  # --- Exploration Loop ---

  @critical
  Scenario: Exploration loop iterates until LLM signals readiness
    Given the LLM returns context requests for 3 rounds then an empty request list
    When the orchestrator runs the exploration loop
    Then the loop executes 3 rounds of provider fulfillment
    And exits when the LLM returns an empty request list

  @critical
  Scenario: Exploration loop respects maximum round limit
    Given a max exploration rounds limit of 10
    And the LLM keeps requesting context every round
    When the orchestrator runs the exploration loop
    Then the loop stops after 10 rounds

  @standard
  Scenario: Default exploration round limit is 10
    When a task is created with default settings
    Then the max exploration rounds is 10

  @standard
  Scenario: Exploration can be disabled
    Given a task with max_exploration_rounds set to 0
    When the orchestrator processes the task
    Then the exploration loop is skipped entirely

  # --- Context Accumulation ---

  @critical
  Scenario: Context accumulates across exploration rounds
    Given the LLM requests "read_file" in round 1 and "search_code" in round 2
    When the orchestrator runs the exploration loop
    Then round 2 receives the accumulated results from round 1

  @standard
  Scenario: Accumulated context is appended to the system prompt
    Given the exploration loop produces context results
    When the orchestrator proceeds to generation
    Then the exploration results are included in the system prompt

  # --- Context Providers ---

  @critical
  Scenario Outline: Provider fulfills context request
    When the LLM requests the "<provider>" provider
    Then the orchestrator dispatches the request to the correct handler
    And returns a context result with content

    Examples:
      | provider          |
      | read_file         |
      | search_code       |
      | symbol_list       |
      | import_graph      |
      | run_tests         |
      | lint_check        |
      | git_log           |
      | git_diff          |
      | repo_map          |
      | discover_context  |
      | past_runs         |
      | playbooks         |

  @standard
  Scenario: search_code provider limits results to 100 matches
    Given a search pattern that matches 200 files
    When the "search_code" provider is invoked
    Then the result contains at most 100 matches

  @standard
  Scenario: run_tests provider truncates output to 4000 characters
    Given a test run producing 10000 characters of output
    When the "run_tests" provider is invoked
    Then the result is truncated to 4000 characters

  @standard
  Scenario: lint_check provider truncates output to 4000 characters
    Given a lint check producing 10000 characters of output
    When the "lint_check" provider is invoked
    Then the result is truncated to 4000 characters

  # --- Exploration in Different Modes ---

  @standard
  Scenario: Single-step mode runs exploration after context assembly
    Given a task in single-step mode
    When the orchestrator processes the task
    Then the exploration loop runs after context assembly and before generation

  @standard
  Scenario: Planned mode runs exploration before planner call
    Given a task with planning enabled
    When the orchestrator processes the task
    Then the exploration loop runs before the planner is called

  # --- Readiness Signal ---

  @critical
  Scenario: Empty request list signals LLM readiness
    When the exploration LLM returns a response with an empty requests list
    Then the exploration loop exits
    And the orchestrator proceeds to generation
