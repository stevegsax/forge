@cli
Feature: CLI Commands
  The orchestrator provides a command-line interface with 7 commands for task
  execution, worker management, status inspection, knowledge extraction,
  playbook browsing, planner evaluation, and generic workflow launching.

  # --- Exit Codes ---

  @critical
  Scenario Outline: CLI exit codes indicate result category
    When a CLI command exits with code <exit_code>
    Then it indicates "<meaning>"

    Examples:
      | exit_code | meaning                        |
      | 0         | success                        |
      | 1         | task or validation failure      |
      | 3         | infrastructure or system error   |

  # --- forge run ---

  @critical @task-execution
  Scenario: Run a task with inline description
    Given the command "forge run --task-id my-task --description 'Implement feature X' --target-file src/feature.py"
    When the command executes
    Then a task workflow is started with the given parameters

  @standard @task-execution
  Scenario: Run a task from JSON file
    Given a task definition file at "task.json"
    When the user runs "forge run --task-file task.json"
    Then the task definition is loaded from the JSON file

  @standard @task-execution
  Scenario: Run with --no-wait returns workflow ID immediately
    When the user runs "forge run --task-id my-task --description 'Test' --no-wait"
    Then the workflow ID is printed and the command exits without waiting

  @standard @task-execution
  Scenario: Run with --json outputs TaskResult as JSON
    When the user runs "forge run --task-id my-task --description 'Test' --json"
    Then the output is a JSON-serialized TaskResult

  @standard @task-execution
  Scenario: Run with --plan enables planning mode
    When the user runs "forge run --task-id my-task --description 'Test' --plan"
    Then the planner is invoked before step execution

  @standard @task-execution
  Scenario: Run with --no-lint disables ruff lint validation
    When the user runs "forge run --task-id my-task --description 'Test' --no-lint"
    Then ruff lint checking is disabled

  @standard @task-execution
  Scenario: Run with --no-format disables ruff format validation
    When the user runs "forge run --task-id my-task --description 'Test' --no-format"
    Then ruff format checking is disabled

  @standard @task-execution
  Scenario: Run with --run-tests enables test execution
    When the user runs "forge run --task-id my-task --description 'Test' --run-tests --test-command 'pytest'"
    Then test execution is enabled with the given command

  @standard @task-execution
  Scenario: Run with --no-explore disables exploration
    When the user runs "forge run --task-id my-task --description 'Test' --no-explore"
    Then the LLM-guided exploration loop is skipped

  @standard @task-execution
  Scenario: Run with --no-auto-discover disables context discovery
    When the user runs "forge run --task-id my-task --description 'Test' --no-auto-discover"
    Then automatic import graph discovery is disabled

  @standard @model-routing
  Scenario: Run with model tier overrides
    When the user runs "forge run --task-id my-task --description 'Test' --reasoning-model custom:model"
    Then the reasoning tier is overridden to "custom:model"

  @standard @task-execution
  Scenario: Run exits with code 1 on task failure
    Given a task that fails validation
    When the command completes
    Then the exit code is 1

  @standard @task-execution
  Scenario: Run exits with code 3 on infrastructure error
    Given the Temporal server is unreachable
    When the user runs "forge run"
    Then the exit code is 3

  # --- forge worker ---

  @critical @temporal
  Scenario: Start Temporal worker
    When the user runs "forge worker"
    Then a Temporal worker is started on the default task queue

  @standard @temporal
  Scenario: Worker uses configurable Temporal address
    When the user runs "forge worker --temporal-address custom:7233"
    Then the worker connects to "custom:7233"

  @standard @temporal @batch
  Scenario: Worker batch poll interval is configurable
    When the user runs "forge worker --batch-poll-interval 300"
    Then the batch poller runs every 300 seconds

  # --- forge status ---

  @standard @observability
  Scenario: List recent runs
    When the user runs "forge status"
    Then recent task runs are displayed

  @standard @observability
  Scenario: Show details for specific workflow
    When the user runs "forge status --workflow-id wf-123"
    Then details for workflow "wf-123" are displayed

  @standard @observability
  Scenario: Status with --json outputs machine-readable format
    When the user runs "forge status --json"
    Then the output is JSON-formatted

  @standard @observability
  Scenario: Status exits with code 1 when no store exists
    Given no database file exists
    When the user runs "forge status"
    Then the exit code is 1

  # --- forge playbooks ---

  @standard @knowledge
  Scenario: List playbook entries
    When the user runs "forge playbooks"
    Then recent playbook entries are displayed

  @standard @knowledge
  Scenario: Filter playbooks by tag
    When the user runs "forge playbooks --tag python --tag validation"
    Then only entries matching those tags are displayed

  @standard @knowledge
  Scenario: Filter playbooks by source task
    When the user runs "forge playbooks --task-id my-task"
    Then only entries from that source task are displayed

  # --- forge eval-planner ---

  @standard
  Scenario: Evaluate planner against corpus
    Given an eval corpus directory with case files
    When the user runs "forge eval-planner --corpus-dir ./eval-cases"
    Then the planner is evaluated against each case

  @standard
  Scenario: Eval with --judge enables LLM scoring
    When the user runs "forge eval-planner --corpus-dir ./eval-cases --judge"
    Then LLM-as-judge scoring is applied to each evaluation result

  @standard
  Scenario: Eval with --dry-run lists cases without evaluating
    When the user runs "forge eval-planner --corpus-dir ./eval-cases --dry-run"
    Then the cases are listed without running evaluation

  # --- forge start ---

  @critical
  Scenario: Start a generic workflow by class name
    When the user runs "forge start OcrSubmitWorkflow '{"file_path": "/doc.pdf"}'"
    Then a Temporal workflow of type OcrSubmitWorkflow is started with the given input

  @standard
  Scenario: Start with --input-file reads JSON from file
    Given a file "input.json" with workflow arguments
    When the user runs "forge start OcrSubmitWorkflow --input-file input.json"
    Then the workflow input is loaded from the file

  @standard
  Scenario: Start with --wait blocks for result
    When the user runs "forge start OcrSubmitWorkflow '{}' --wait"
    Then the command blocks until the workflow completes and prints the result as JSON

  @standard
  Scenario: Start without --wait returns workflow ID
    When the user runs "forge start OcrSubmitWorkflow '{}'"
    Then the workflow ID is printed and the command exits immediately

  @standard
  Scenario: Start with --id sets custom workflow ID
    When the user runs "forge start OcrSubmitWorkflow '{}' --id my-custom-wf"
    Then the Temporal workflow ID is "my-custom-wf"

  @standard
  Scenario: Start auto-generates workflow ID when --id is omitted
    When the user runs "forge start OcrSubmitWorkflow '{}'"
    Then a workflow ID is auto-generated in the format "{workflow_lower}-{short_uuid}"

  @standard
  Scenario: Start with --timeout sets execution timeout
    When the user runs "forge start OcrSubmitWorkflow '{}' --timeout 24"
    Then the workflow execution timeout is 24 hours

  @standard
  Scenario: Start default timeout is 48 hours
    When the user runs "forge start OcrSubmitWorkflow '{}'"
    Then the workflow execution timeout is 48 hours

  @standard @error-handling
  Scenario: Start exits with code 3 on infrastructure error
    Given the Temporal server is unreachable
    When the user runs "forge start OcrSubmitWorkflow '{}'"
    Then the exit code is 3

  @standard
  Scenario: Providing both input argument and --input-file is an error
    When the user runs "forge start OcrSubmitWorkflow '{}' --input-file input.json"
    Then a UsageError is raised about mutually exclusive inputs
