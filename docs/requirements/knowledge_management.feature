@knowledge @phase-6
Feature: Knowledge Management
  The orchestrator extracts lessons from completed task runs into playbook entries.
  Playbooks are tagged, stored in the database, and injected into future task
  contexts at priority 5. Extraction runs on a configurable schedule.

  # --- Extraction Workflow ---

  @critical @temporal
  Scenario: Extraction workflow processes unextracted runs
    Given completed task runs that have not been extracted
    When the extraction workflow runs
    Then it fetches unextracted runs from the database
    And calls the LLM to extract playbook entries
    And saves the entries to the database

  @critical
  Scenario: Extraction uses the summarization tier model
    When the extraction workflow calls the LLM
    Then it uses the model assigned to the "summarization" capability tier

  @standard
  Scenario: Extraction workflow has three sequential activities
    When the extraction workflow runs
    Then it executes fetch_extraction_input, call_extraction_llm, and save_extraction_results in order

  # --- Playbook Entries ---

  @critical
  Scenario: Playbook entry contains structured fields
    When the LLM extracts a playbook entry
    Then it includes a title, content, tags list, source_task_id, and source_workflow_id

  @standard
  Scenario: Playbook content is actionable and concise
    When the LLM extracts lessons
    Then each entry content is 2-4 sentences of actionable guidance

  # --- Tag Inference ---

  @critical @deterministic
  Scenario: Tags are inferred from target file extensions
    Given target files including "src/module.py" and "src/component.tsx"
    When tags are inferred for the task
    Then the tags include "python" and "typescript"

  @standard @deterministic
  Scenario: Tags are inferred from task description keywords
    Given a task description containing "refactor the API validation"
    When tags are inferred for the task
    Then the tags include "refactoring", "api", and "validation"

  @standard @deterministic
  Scenario Outline: Keyword-to-tag mapping
    Given a task description containing "<keyword>"
    When tags are inferred
    Then the tags include "<tag>"

    Examples:
      | keyword   | tag           |
      | test      | test-writing  |
      | refactor  | refactoring   |
      | api       | api           |
      | database  | database      |
      | migration | migration     |
      | cli       | cli           |
      | validate  | validation    |
      | bug       | bug-fix       |
      | fix       | bug-fix       |

  @standard @deterministic
  Scenario: Tags fallback to domain name when no keywords match
    Given a task with no matching keywords in the description
    When tags are inferred
    Then the tags include the task domain name with hyphens

  @standard @deterministic
  Scenario: Inferred tags are sorted and deduplicated
    Given a task description with multiple matching keywords
    When tags are inferred
    Then the resulting tag list is sorted alphabetically with no duplicates

  # --- Playbook Retrieval ---

  @critical
  Scenario: Playbooks are retrieved by matching tags
    Given playbook entries tagged with "python" and "validation"
    When the orchestrator queries playbooks with tags "python"
    Then entries matching any of the queried tags are returned

  @standard
  Scenario: Playbook retrieval uses SQLite json_each for tag matching
    When playbooks are queried by tags
    Then the query matches any tag in the entry's tags list

  @standard
  Scenario: Playbook retrieval is limited to 5 entries per task
    Given more than 5 playbooks matching the task tags
    When playbooks are loaded for a task
    Then at most 5 entries are returned

  # --- Context Injection ---

  @critical
  Scenario: Playbooks are injected at priority 5 in context assembly
    Given matching playbooks exist for a task
    When the orchestrator assembles context
    Then playbook items are included at priority 5 with representation "playbook"

  @standard
  Scenario: Playbook context items are formatted with title and content
    Given a playbook entry with title "Handle Import Errors" and content "Always check..."
    When the entry is converted to a context item
    Then the file_path is "playbook:Handle Import Errors"
    And the content is formatted as "**Handle Import Errors**\nAlways check..."

  @standard
  Scenario: Playbook injection is best-effort
    Given the database is unavailable
    When the orchestrator attempts to load playbooks
    Then an empty list is returned without raising an error

  # --- Scheduled Execution ---

  @standard @temporal
  Scenario: Extraction runs on a configurable schedule
    Given the worker is started with extraction_interval of 14400 seconds
    When the worker schedules extraction
    Then extraction runs every 14400 seconds (4 hours)

  # --- Dry-Run Preview ---

  @standard @cli
  Scenario: Dry-run lists unextracted runs without processing
    Given completed runs that have not been extracted
    When the user runs "forge extract --dry-run"
    Then the unextracted runs are listed without starting the extraction workflow
