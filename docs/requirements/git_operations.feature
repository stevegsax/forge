@git @phase-1 @phase-3
Feature: Git Operations
  The orchestrator manages git worktrees for task isolation.
  Each task operates in its own worktree branched from a base branch.
  Sub-tasks use compound IDs for nested worktree paths.

  Background:
    Given a git repository at the project root

  # --- Task ID Validation ---

  @critical
  Scenario: Valid task ID is accepted
    When a task ID "my-task-123" is validated
    Then the task ID is accepted

  @critical
  Scenario Outline: Invalid task ID is rejected
    When a task ID "<task_id>" is validated
    Then a ValueError is raised

    Examples:
      | task_id       |
      |               |
      | .hidden       |
      | -leading      |
      | has spaces    |
      | has/slashes   |

  @standard
  Scenario: Task ID allows dots, hyphens, and underscores
    When a task ID "my.task_v2-beta" is validated
    Then the task ID is accepted

  # --- Branch and Path Naming ---

  @critical
  Scenario: Worktree path follows convention
    Given a task with id "my-task"
    When the worktree path is computed
    Then the path is "<repo_root>/.forge-worktrees/my-task"

  @critical
  Scenario: Branch name follows convention
    Given a task with id "my-task"
    When the branch name is computed
    Then the branch name is "forge/my-task"

  @standard
  Scenario: Commit message follows convention
    Given a task with id "my-task" and status "success"
    When the commit message is computed
    Then the message is "forge(my-task): success"

  # --- Worktree Lifecycle ---

  @critical
  Scenario: Create worktree from base branch
    Given a task with id "my-task"
    When the orchestrator creates a worktree from branch "main"
    Then a new worktree exists at the computed path
    And a branch "forge/my-task" is created

  @critical
  Scenario: Commit changes in worktree
    Given a worktree for task "my-task" with modified files
    When the orchestrator commits with status "success"
    Then all changes are staged and committed
    And the commit message is "forge(my-task): success"
    And the commit SHA is returned

  @standard
  Scenario: Commit specific files only
    Given a worktree for task "my-task" with multiple modified files
    When the orchestrator commits specific files with status "success"
    Then only the specified files are staged and committed

  @critical @error-handling
  Scenario: Commit with no changes raises error
    Given a worktree for task "my-task" with no modifications
    When the orchestrator attempts to commit
    Then a CommitError is raised with message containing "Nothing to commit"

  @critical
  Scenario: Reset worktree discards all changes
    Given a worktree for task "my-task" with uncommitted changes
    When the orchestrator resets the worktree
    Then all uncommitted changes are discarded
    And untracked files are removed

  @standard
  Scenario: Remove worktree and branch
    Given a worktree for task "my-task"
    When the orchestrator removes the worktree
    Then the worktree directory is deleted
    And the branch "forge/my-task" is deleted

  @standard
  Scenario: Force remove worktree ignores uncommitted changes
    Given a worktree for task "my-task" with uncommitted changes
    When the orchestrator force-removes the worktree
    Then the worktree is removed without error

  @standard
  Scenario: Check worktree existence
    Given a worktree for task "my-task"
    When the orchestrator checks if the worktree exists
    Then the result is true

  @standard
  Scenario: List active worktrees
    Given worktrees for tasks "task-a" and "task-b"
    When the orchestrator lists worktrees
    Then the result contains "task-a" and "task-b"

  # --- Sub-Task Compound IDs ---

  @critical @phase-3
  Scenario: Sub-task compound ID format
    Given a parent task "my-task" with a sub-task "analyze"
    When the compound ID is constructed
    Then the compound ID is "my-task.sub.analyze"

  @standard @phase-3
  Scenario: Nested sub-task compound ID
    Given a parent compound ID "my-task.sub.analyze" with a sub-task "validate"
    When the nested compound ID is constructed
    Then the compound ID is "my-task.sub.analyze.sub.validate"

  @standard @phase-3
  Scenario: Compound IDs are valid task IDs
    Given a compound ID "my-task.sub.analyze.sub.validate"
    When the compound ID is used as a task ID
    Then it passes task ID validation
    And worktree and branch paths are computed correctly

  # --- Error Handling ---

  @standard @error-handling
  Scenario: Git subprocess timeout
    When a git operation takes longer than 30 seconds
    Then the subprocess is killed and an error is raised

  @standard @error-handling
  Scenario: Repo discovery outside git repository
    Given a path that is not inside a git repository
    When the orchestrator attempts to discover the repo root
    Then a RepoDiscoveryError is raised
