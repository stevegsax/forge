@context @phase-4
Feature: Context Assembly
  The orchestrator assembles context for each LLM call through automatic
  import graph discovery, PageRank-based ranking, token budget packing,
  and priority-ordered inclusion with graceful degradation.

  Background:
    Given a Python project with an import graph

  # --- Import Graph Discovery ---

  @critical
  Scenario: Import graph is built from the project package
    Given a project with package name "forge"
    When the orchestrator builds the import graph
    Then the graph contains module nodes and import edges

  @standard
  Scenario: File paths are converted to module names
    Given a file at "src/forge/models.py"
    When the file path is converted to a module name
    Then the result is "forge.models"

  @standard
  Scenario: Module names are converted back to file paths
    Given a module name "forge.models"
    When the module name is converted to a file path
    Then the result resolves to "src/forge/models.py"

  # --- PageRank Ranking ---

  @critical
  Scenario: Target files are ranked highest by PageRank
    Given target files "src/forge/models.py" and "src/forge/workflows.py"
    When the orchestrator ranks files by importance
    Then the target files have the highest importance scores

  @standard
  Scenario: Direct imports are classified as DIRECT_IMPORT
    Given a target file that imports "forge.utils"
    When the orchestrator ranks files
    Then "forge.utils" is classified with relationship "DIRECT_IMPORT"

  @standard
  Scenario: Transitive imports are classified as TRANSITIVE_IMPORT
    Given a target file that imports "forge.utils" which imports "forge.helpers"
    When the orchestrator ranks files with max_depth 2
    Then "forge.helpers" is classified with relationship "TRANSITIVE_IMPORT"

  @standard
  Scenario: Downstream modules are classified as DOWNSTREAM
    Given a module "forge.cli" that imports the target file
    When the orchestrator ranks files
    Then "forge.cli" is classified with relationship "DOWNSTREAM"

  @standard
  Scenario: BFS distance is computed from target files
    Given a target file with a direct import at distance 1 and a transitive import at distance 2
    When the orchestrator ranks files
    Then distances are correctly assigned

  # --- Token Budget Packing ---

  @critical
  Scenario: Context items are packed within token budget
    Given a token budget of 100000 tokens
    And context items totaling 80000 estimated tokens
    When the orchestrator packs the context
    Then all items fit within the budget

  @critical
  Scenario: Items exceeding budget are excluded
    Given a token budget of 50000 tokens
    And context items totaling 120000 estimated tokens
    When the orchestrator packs the context
    Then lower-priority items are excluded to fit the budget

  @critical
  Scenario: Full representations degrade to signatures when budget is tight
    Given a token budget that cannot fit all items as full representations
    When the orchestrator packs the context
    Then some items are degraded from "full" to "signatures" representation
    And the reduced count is tracked in the result

  @standard
  Scenario: Items that cannot fit even as signatures are truncated
    Given a token budget too small for any representation of a low-priority item
    When the orchestrator packs the context
    Then the item is excluded and the truncated count is tracked

  # --- Priority Ordering ---

  @critical
  Scenario: Priority ordering determines inclusion order
    When the orchestrator builds context items
    Then priority 2 items (target files) are included before priority 3 items (direct imports)
    And priority 3 items are included before priority 4 items (transitive imports)

  @standard
  Scenario Outline: Context items have correct priority assignments
    When the orchestrator assigns priorities to context items
    Then "<category>" items have priority <priority>

    Examples:
      | category           | priority |
      | target files       | 2        |
      | direct imports     | 3        |
      | transitive imports | 4        |
      | playbooks          | 5        |
      | repo map           | 5        |
      | manual context     | 6        |

  @standard
  Scenario: Within the same priority, higher PageRank importance is included first
    Given two direct import files with different PageRank scores
    When the orchestrator packs the context
    Then the file with higher importance is included first

  # --- Repo Map ---

  @standard
  Scenario: Repo map is generated with binary search for token budget
    Given a repo map token budget of 2048 tokens
    When the orchestrator generates the repo map
    Then the map fits within the budget
    And includes file paths with signature details for top-ranked files

  @standard
  Scenario: Repo map tracks file counts
    When the orchestrator generates the repo map
    Then the result includes files_with_signatures and files_path_only counts

  # --- Graceful Degradation ---

  @standard @edge-case
  Scenario: Auto-discovery disabled falls back to manual context
    Given a task with auto_discover set to false
    When the orchestrator assembles context
    Then only manually specified context files are included
    And no import graph is built

  @standard
  Scenario: Token estimation uses character-based heuristic
    Given a text string of 400 characters
    When the token count is estimated
    Then the estimate is approximately 100 tokens

  # --- Context Stats ---

  @standard
  Scenario: Context assembly returns statistics
    When the orchestrator assembles context
    Then the result includes files_discovered, files_included_full, files_included_signatures, files_truncated, total_estimated_tokens, and budget_utilization

  # --- System Prompt Ordering for Cache Efficiency ---

  @standard @phase-9
  Scenario: System prompt sections are ordered for cache stability
    When the orchestrator builds the system prompt
    Then stable content (role, instructions, repo map, playbooks) comes first
    And task-specific content comes next
    And volatile content (error sections) comes last
