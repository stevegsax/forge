# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- **Architectural Improvements for Maintainability and Testability:**
  - **Granular DTOs:** Refactored input models (`AssembleContextInput`, `AssembleStepContextInput`, `AssembleSanityCheckContextInput`, and `ExplorationInput`) in `src/forge/models.py` to use explicit fields (e.g., `task_id`, `description`, `target_files`, `context_config`) instead of the monolithic `TaskDefinition` object.
  - **Decoupled Activities:** Updated `assemble_context`, `assemble_step_context`, `assemble_planner_context`, `assemble_sanity_check_context`, and exploration logic to accept the more granular models. This makes the logic cleaner, reduces the blast radius of changes to `TaskDefinition`, and simplifies testing.
  - **Workflow Alignment:** Modified `ForgeTaskWorkflow` dispatch to correctly map `TaskDefinition` properties into the decoupled DTO structures during single-step, planned multi-step, and exploration loops.
  - **Test Suite Updates:** Updated unit tests (`test_activity_context.py`, `test_activity_exploration.py`, `test_activity_planner.py`, `test_model_routing.py`) to align with the new DTO shapes and validate logic appropriately.
