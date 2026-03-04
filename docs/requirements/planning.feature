@planning @phase-2
Feature: Planning and Decomposition
  The orchestrator uses a planner LLM to decompose tasks into ordered steps.
  Each step is executed sequentially with its own retry loop. Sanity checks
  can validate progress and mutate the remaining plan at configurable intervals.

  Background:
    Given a task with planning enabled

  # --- Planner Decomposition ---

  @critical
  Scenario: Planner decomposes task into ordered steps
    When the orchestrator calls the planner
    Then the planner returns a plan with one or more steps
    And each step has a step_id, description, and target_files

  @critical
  Scenario: Planner uses the reasoning tier model
    When the orchestrator calls the planner
    Then it uses the model assigned to the "reasoning" capability tier

  @standard
  Scenario: Plan includes an explanation of decomposition strategy
    When the planner returns a plan
    Then the plan includes an explanation field describing the decomposition strategy

  @standard
  Scenario: Plan steps can specify optional context files
    When the planner creates a step with context_files
    Then those files are included in the step's context assembly

  @standard
  Scenario: Plan steps can specify optional capability tier override
    When the planner creates a step with capability_tier "reasoning"
    Then that step uses the reasoning tier model instead of the default generation tier

  @standard
  Scenario: Plan steps can include sub-tasks for fan-out
    When the planner creates a step with sub_tasks
    Then that step is dispatched as a fan-out step with parallel child workflows

  # --- Extended Thinking ---

  @critical @phase-12
  Scenario: Planner supports extended thinking
    Given a thinking budget of 10000 tokens
    When the orchestrator calls the planner
    Then the planner LLM call includes the thinking budget parameter

  @standard @phase-12
  Scenario: Extended thinking is disabled by default
    When a task is created with default settings
    Then the thinking budget is 0

  # --- Sequential Step Execution ---

  @critical
  Scenario: Steps execute sequentially in plan order
    Given a plan with steps "step-1", "step-2", "step-3"
    When the orchestrator executes the plan
    Then "step-1" completes before "step-2" starts
    And "step-2" completes before "step-3" starts

  @critical @retry
  Scenario: Each step has its own retry loop
    Given a plan step with max_step_attempts set to 2
    And the first attempt of the step fails validation
    When the orchestrator retries the step
    Then the worktree is reset and the step is retried with error context

  @standard
  Scenario: Default max step attempts is 2
    When a task is created with default settings
    Then the max_step_attempts is 2

  @critical
  Scenario: Step failure terminates plan execution
    Given a plan step that fails terminally after all retries
    When the orchestrator evaluates the step result
    Then the entire plan execution stops
    And the task result status is "failure_terminal"

  @standard
  Scenario: Successful step commits with step ID in message
    Given a plan step "step-1" that passes validation
    When the orchestrator commits the step
    Then the commit message includes the step ID

  @standard
  Scenario: Step context includes completed step history
    Given a plan with steps "step-1" and "step-2"
    And "step-1" has completed successfully
    When the orchestrator assembles context for "step-2"
    Then the context includes information about completed step "step-1"

  # --- Sanity Checks ---

  @critical
  Scenario: Sanity check is disabled by default
    When a task is created with default settings
    Then the sanity_check_interval is 0

  @critical
  Scenario Outline: Sanity check verdict determines next action
    Given a sanity check returns verdict "<verdict>"
    When the orchestrator processes the verdict
    Then the action is "<action>"

    Examples:
      | verdict  | action                                         |
      | continue | plan execution continues with the next step     |
      | revise   | remaining steps are replaced with revised steps  |
      | abort    | plan execution stops with failure_terminal       |

  @standard
  Scenario: Sanity check triggers at configured interval
    Given a sanity_check_interval of 3
    And a plan with 6 steps
    When the orchestrator completes step 3
    Then a sanity check is triggered

  @standard
  Scenario: Sanity check does not trigger on the last step
    Given a sanity_check_interval of 2
    And a plan with 4 steps
    When the orchestrator completes the 4th (last) step
    Then no sanity check is triggered

  # --- Plan Mutation ---

  @critical
  Scenario: REVISE verdict replaces remaining plan steps
    Given a plan with steps "step-1", "step-2", "step-3", "step-4"
    And the sanity check after "step-2" returns REVISE with revised steps "step-2b", "step-2c"
    When the orchestrator applies the mutation
    Then the plan becomes "step-1", "step-2", "step-2b", "step-2c"
    And execution continues from "step-2b"

  @standard
  Scenario: ABORT verdict returns terminal failure with explanation
    Given the sanity check returns ABORT with explanation "Fundamental design flaw"
    When the orchestrator processes the verdict
    Then the task result status is "failure_terminal"
    And the error message includes "Sanity check aborted: Fundamental design flaw"
