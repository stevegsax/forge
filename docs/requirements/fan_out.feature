@fan-out @phase-3
Feature: Fan-Out Execution
  The orchestrator supports parallel sub-task execution through fan-out steps.
  Sub-tasks run as child workflows in their own worktrees. Sub-task worktrees
  are always removed without committing. Only the parent workflow commits
  the gathered and merged results. Nested fan-out is depth-bounded.

  Background:
    Given a planned task with a fan-out step containing sub-tasks

  # --- Parallel Child Workflows ---

  @critical @temporal
  Scenario: Fan-out step spawns parallel child workflows
    Given a plan step with 3 sub-tasks
    When the orchestrator executes the fan-out step
    Then 3 child workflows are started in parallel

  @critical @temporal
  Scenario: Each child workflow gets its own worktree
    Given a fan-out step with sub-task "analyze"
    And the parent task ID is "my-task"
    When the child workflow runs
    Then it operates in a worktree with compound ID "my-task.sub.analyze"

  @critical @temporal
  Scenario: Child workflow ID includes compound ID
    Given a parent task "my-task" with sub-task "analyze"
    When the child workflow is started
    Then the Temporal workflow ID is "forge-subtask-my-task.sub.analyze"

  # --- No Sub-Task Commits (D16) ---

  @critical
  Scenario: Sub-task worktrees are removed without committing
    Given a sub-task that completes successfully
    When the child workflow finishes
    Then the output files are collected in memory
    And the worktree is removed without a commit

  @critical
  Scenario: Parent workflow commits gathered results
    Given all sub-tasks complete successfully
    And the gathered files have no conflicts
    When the parent workflow writes the merged files
    Then the parent worktree is committed with message containing "fan-out gather"

  # --- File Conflict Detection ---

  @critical
  Scenario: Non-conflicting files are merged directly
    Given sub-task "A" produces "src/utils.py" and sub-task "B" produces "src/helpers.py"
    When the orchestrator detects file conflicts
    Then both files are classified as non-conflicting
    And no conflict resolution is needed

  @critical
  Scenario: Same file from multiple sub-tasks is a conflict
    Given sub-task "A" and sub-task "B" both produce "src/models.py" with different content
    When the orchestrator detects file conflicts
    Then "src/models.py" is classified as a conflict with 2 versions

  @standard
  Scenario: Failed sub-tasks are excluded from conflict detection
    Given sub-task "A" succeeds and sub-task "B" fails
    When the orchestrator detects file conflicts
    Then only sub-task "A" output is considered

  @standard
  Scenario: Conflict includes original file content if it exists
    Given "src/models.py" exists in the parent worktree before fan-out
    And two sub-tasks produce conflicting versions
    When the orchestrator detects the conflict
    Then the conflict object includes the original_content from the parent worktree

  @standard
  Scenario: Conflict for a new file has no original content
    Given "src/new_module.py" does not exist in the parent worktree
    And two sub-tasks both create it with different content
    When the orchestrator detects the conflict
    Then the conflict object has original_content as null

  # --- LLM-Based Conflict Resolution ---

  @critical
  Scenario: Conflicts are resolved by LLM when enabled
    Given file conflicts exist and conflict resolution is enabled
    When the orchestrator resolves the conflicts
    Then the LLM receives all competing versions and the original content
    And it returns merged file contents for each conflicting path

  @standard
  Scenario: Conflict resolution uses the reasoning tier model
    When the orchestrator calls the conflict resolution LLM
    Then it uses the model assigned to the "reasoning" capability tier

  @standard
  Scenario: Resolved files are verified for completeness
    Given 2 conflicting files
    When the LLM resolves the conflicts
    Then the orchestrator verifies that resolved files cover all conflicting paths

  @standard
  Scenario: Conflict resolution can be disabled
    Given file conflicts exist and conflict resolution is disabled
    When the orchestrator processes the fan-out results
    Then the task fails with a terminal error about unresolved conflicts

  # --- Child Workflow Failure ---

  @critical @error-handling
  Scenario: Any child workflow failure fails the fan-out step
    Given a fan-out step with 3 sub-tasks
    And sub-task "B" fails terminally
    When the orchestrator gathers results
    Then the fan-out step fails immediately

  # --- Nested Fan-Out ---

  @critical
  Scenario: Default fan-out depth is 1
    When a task is created with default settings
    Then the max_fan_out_depth is 1

  @standard
  Scenario: Nested fan-out increments depth for children
    Given max_fan_out_depth is 2 and current depth is 0
    When a sub-task has its own sub-tasks
    Then children are spawned at depth 1 with max_depth 2

  @standard
  Scenario: Fan-out at max depth executes sub-tasks as leaf tasks
    Given max_fan_out_depth is 1 and current depth is 1
    When a sub-task has sub-tasks defined
    Then the sub-tasks are ignored and the sub-task runs as a single-step leaf task

  @standard
  Scenario: Child workflow timeout scales with remaining depth
    Given max_fan_out_depth is 2
    When a child is spawned at depth 0
    Then its timeout is 25 minutes (base 15 + 5 per remaining level)

  @standard @temporal
  Scenario: Nested fan-out worktrees are always removed
    Given a nested fan-out at depth 1
    When all grandchild workflows complete
    Then the nested parent worktree is removed without committing
