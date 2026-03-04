@output @phase-1
Feature: Output Processing
  The orchestrator writes LLM output to files and applies search/replace edits.
  A 4-level edit matching fallback chain handles inexact matches from LLM output.
  Path traversal protection prevents writes outside the worktree.

  Background:
    Given a worktree at a known path

  # --- File Writing ---

  @critical
  Scenario: Write new files from LLM response
    Given an LLM response containing new file outputs
    When the orchestrator writes the output
    Then each file is created at its specified path within the worktree
    And parent directories are created as needed

  @standard
  Scenario: Overlapping file paths in files and edits are rejected
    Given an LLM response with the same file path in both files and edits
    When the orchestrator attempts to write the output
    Then an OutputWriteError is raised with a message about the overlap

  # --- Path Traversal Protection ---

  @critical
  Scenario: Absolute path is rejected
    Given an LLM response with file path "/etc/passwd"
    When the orchestrator attempts to resolve the path
    Then an OutputWriteError is raised

  @critical
  Scenario: Parent directory traversal is rejected
    Given an LLM response with file path "../outside/secret.py"
    When the orchestrator attempts to resolve the path
    Then an OutputWriteError is raised

  @critical @edge-case
  Scenario: Sneaky traversal via intermediate components is rejected
    Given an LLM response with file path "src/../../outside/secret.py"
    When the orchestrator attempts to resolve the path
    Then an OutputWriteError is raised

  @standard
  Scenario: Edit target file must exist
    Given an LLM response with edits for a file that does not exist
    When the orchestrator attempts to resolve the edit path
    Then an OutputWriteError is raised

  # --- Edit Matching: Level 1 — Exact Match ---

  @critical
  Scenario: Exact match applies edit successfully
    Given a file with content containing a unique search string
    When the orchestrator applies an edit with that exact search string
    Then the edit is applied at match level "exact"

  @critical
  Scenario: Exact match rejects ambiguous matches
    Given a file with content containing a search string that appears twice
    When the orchestrator applies an edit with that search string
    Then an EditApplicationError is raised mentioning ambiguity

  @standard
  Scenario: Exact match with zero occurrences falls through to next level
    Given a file with content that does not contain the search string
    When the orchestrator applies an edit with that search string
    Then the edit falls through to whitespace-normalized matching

  # --- Edit Matching: Level 2 — Whitespace-Normalized ---

  @critical
  Scenario: Whitespace-normalized match handles trailing whitespace differences
    Given a file where lines have trailing spaces
    And an LLM edit search string without trailing spaces
    When the orchestrator applies the edit
    Then the edit is applied at match level "whitespace"

  @standard
  Scenario: Whitespace-normalized match handles line-ending differences
    Given a file with lines ending in CRLF
    And an LLM edit search string with LF line endings
    When the orchestrator applies the edit
    Then the edit is applied at match level "whitespace"

  @standard @edge-case
  Scenario: Whitespace-normalized match rejects ambiguous matches
    Given a file where whitespace normalization produces two matches
    When the orchestrator applies the edit
    Then an EditApplicationError is raised mentioning ambiguity

  # --- Edit Matching: Level 3 — Indentation-Normalized ---

  @critical
  Scenario: Indentation-normalized match handles different indent levels
    Given a file with a code block indented at 8 spaces
    And an LLM edit search string for the same code indented at 0 spaces
    When the orchestrator applies the edit
    Then the edit is applied at match level "indentation"
    And the replacement preserves the 8-space indentation level

  @standard
  Scenario: Indentation-normalized match preserves internal structure
    Given a file with a nested code block at 8 spaces with inner lines at 12 spaces
    And an LLM edit search string for the same block at 0 spaces with inner lines at 4 spaces
    When the orchestrator applies the edit
    Then the replacement maintains the relative indentation offset

  @standard @edge-case
  Scenario: Indentation-normalized match rejects cross-level ambiguity
    Given a file where the search string matches at two different indentation levels
    When the orchestrator applies the edit
    Then an EditApplicationError is raised mentioning ambiguity

  # --- Edit Matching: Level 4 — Fuzzy Match ---

  @critical
  Scenario: Fuzzy match finds similar content above threshold
    Given a file with content similar to the search string at 0.7 similarity
    When the orchestrator applies the edit with a threshold of 0.6
    Then the edit is applied at match level "fuzzy"
    And the similarity score is included in the match result

  @critical @edge-case
  Scenario: Fuzzy match rejects ambiguous candidates
    Given a file with two regions similar to the search string
    And the best score is 0.75 and the second-best is 0.72
    When the orchestrator applies the edit
    Then an EditApplicationError is raised mentioning "fuzzy match is ambiguous"
    And the message includes both scores and the 0.05 gap requirement

  @standard
  Scenario: Fuzzy match below threshold fails
    Given a file with content at only 0.4 similarity to the search string
    When the orchestrator applies the edit with a threshold of 0.6
    Then an EditApplicationError is raised

  @standard
  Scenario: Fuzzy match uniqueness requires 0.05 gap between best and second-best
    Given a file with two regions at 0.80 and 0.76 similarity
    When the orchestrator applies the edit
    Then the edit is applied because the gap of 0.04 rounds up to the threshold

  # --- Sequential Edit Application ---

  @standard
  Scenario: Edits are applied sequentially
    Given a file with two non-overlapping regions to edit
    When the orchestrator applies both edits
    Then the second edit sees the result of the first edit

  @standard
  Scenario: Multiple edits on the same file use different match levels
    Given a file requiring one exact match edit and one fuzzy match edit
    When the orchestrator applies both edits
    Then each edit reports its own match level

  @critical
  Scenario: Empty search string is rejected
    Given an edit with an empty search string
    When the orchestrator attempts to apply the edit
    Then an EditApplicationError is raised
