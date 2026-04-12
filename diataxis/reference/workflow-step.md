# Universal Workflow Step Reference

This document describes the activities, workflows, data models, and configuration that implement the universal workflow step. For background on why this pattern exists, see the [explanation](../explanation/workflow-step.md).


## Activities

Activities are organized by their role in the five-phase pattern: Construct, Send, Receive/Serialize, Validate, and Transition. Additional activities support planning, exploration, fan-out, and batch processing.

### Core pipeline activities

These activities implement the five phases of the universal workflow step.

| Activity | Input type | Output type | Timeout | Heartbeat | Retry policy |
|----------|-----------|-------------|---------|-----------|-------------|
| `assemble_context` | `AssembleContextInput` | `AssembledContext` | 30s | -- | 2 attempts |
| `call_llm` | `AssembledContext` | `LLMCallResult` | 5m | 60s | 3 attempts; non-retryable: `BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError` |
| `write_output` | `WriteOutputInput` | `WriteResult` | 30s | -- | 2 attempts; non-retryable: `OutputWriteError`, `EditApplicationError` |
| `validate_output` | `ValidateOutputInput` | `list[ValidationResult]` | 2m | 120s | 2 attempts |
| `evaluate_transition` | `TransitionInput` | `str` | 10s | -- | 2 attempts |

### Context assembly variants

These assemble prompts for specific execution modes. All share the same output type.

| Activity | Input type | Output type | Timeout | Retry policy |
|----------|-----------|-------------|---------|-------------|
| `assemble_context` | `AssembleContextInput` | `AssembledContext` | 30s | 2 attempts |
| `assemble_step_context` | `AssembleStepContextInput` | `AssembledContext` | 30s | 2 attempts |
| `assemble_sub_task_context` | `AssembleSubTaskContextInput` | `AssembledContext` | 30s | 2 attempts |

### Planning activities

| Activity | Input type | Output type | Timeout | Retry policy |
|----------|-----------|-------------|---------|-------------|
| `assemble_planner_context` | `AssembleContextInput` | `PlannerInput` | 30s | 2 attempts |
| `call_planner` | `PlannerInput` | `PlanCallResult` | 5m | 3 attempts; non-retryable: `BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError` |

### Exploration activities

| Activity | Input type | Output type | Timeout | Retry policy |
|----------|-----------|-------------|---------|-------------|
| `assemble_exploration_context` | `ExplorationInput` | `AssembledContext` | 30s | 2 attempts |
| `call_exploration_llm` | `ExplorationInput` | `ExplorationResponse` | 5m | 3 attempts; non-retryable: `BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError` |
| `fulfill_context_requests` | `FulfillContextInput` | `list[ContextResult]` | 2m | 2 attempts |

### Sanity check activities

| Activity | Input type | Output type | Timeout | Retry policy |
|----------|-----------|-------------|---------|-------------|
| `assemble_sanity_check_context` | `AssembleSanityCheckContextInput` | `AssembledContext` | 30s | 2 attempts |
| `call_sanity_check` | `SanityCheckInput` | `SanityCheckCallResult` | 5m | 3 attempts; non-retryable: `BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError` |

### Conflict resolution activities

| Activity | Input type | Output type | Timeout | Retry policy |
|----------|-----------|-------------|---------|-------------|
| `detect_file_conflicts_activity` | `DetectFileConflictsInput` | `DetectFileConflictsOutput` | 30s | 2 attempts |
| `assemble_conflict_resolution_context` | `ConflictResolutionInput` | `ConflictResolutionCallInput` | 30s | 2 attempts |
| `call_conflict_resolution` | `ConflictResolutionCallInput` | `ConflictResolutionCallResult` | 5m | 3 attempts; non-retryable: `BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError` |

### Git activities

| Activity | Input type | Output type | Timeout | Retry policy |
|----------|-----------|-------------|---------|-------------|
| `create_worktree_activity` | `CreateWorktreeInput` | `CreateWorktreeOutput` | 30s | 2 attempts; non-retryable: `CommitError`, `RepoDiscoveryError` |
| `remove_worktree_activity` | `RemoveWorktreeInput` | `None` | 30s | 2 attempts; non-retryable: `CommitError`, `RepoDiscoveryError` |
| `commit_changes_activity` | `CommitChangesInput` | `CommitChangesOutput` | 30s | 2 attempts; non-retryable: `CommitError`, `RepoDiscoveryError` |
| `reset_worktree_activity` | `ResetWorktreeInput` | `None` | 30s | 2 attempts; non-retryable: `CommitError`, `RepoDiscoveryError` |

### Batch processing activities

| Activity | Input type | Output type | Timeout | Retry policy |
|----------|-----------|-------------|---------|-------------|
| `submit_batch_request` | `BatchSubmitInput` | `BatchSubmitResult` | 5m | 3 attempts; non-retryable: `BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError` |
| `parse_llm_response` | `ParseResponseInput` | `ParsedLLMResponse` | 30s | 2 attempts |
| `poll_batch_results` | `BatchPollerInput` | `BatchPollerResult` | 5m | 3 attempts |

### File write activity

| Activity | Input type | Output type | Timeout | Retry policy |
|----------|-----------|-------------|---------|-------------|
| `write_files` | `WriteFilesInput` | `WriteResult` | 30s | 2 attempts; non-retryable: `OutputWriteError`, `EditApplicationError` |


## Workflows

### ForgeTaskWorkflow

The primary workflow. Accepts a `ForgeTaskInput` and returns a `TaskResult`.

| Property | Value |
|----------|-------|
| **Name** | `ForgeTaskWorkflow` |
| **Task queue** | `forge-task-queue` |
| **Input type** | `ForgeTaskInput` |
| **Output type** | `TaskResult` |
| **Signal** | `batch_result_received(BatchResult)` |

Dispatches between two execution paths based on `ForgeTaskInput.plan`:

- **`plan=False`** (single-step): Creates a worktree per attempt. Runs the five-phase pattern once per attempt. On `SUCCESS`, commits. On `FAILURE_RETRYABLE`, destroys the worktree and creates a fresh one. Bounded by `max_attempts`.

- **`plan=True`** (planned multi-step): Creates one worktree. Calls the planner to decompose the task into ordered steps. Executes each step sequentially. Each step runs the five-phase pattern. On step `SUCCESS`, commits. On step `FAILURE_RETRYABLE`, resets uncommitted changes and retries the step. Bounded by `max_step_attempts` per step.

    - **Fan-out** (steps with `sub_tasks`): Starts a child `ForgeTaskWorkflow` per sub-task. Each child runs in its own worktree. After all children complete, gathers results, detects file conflicts, optionally resolves conflicts via LLM, merges into the parent worktree, validates, and commits. Child workflow timeout scales by nesting depth: base 15 minutes plus 5 minutes per remaining level.


## TransitionSignal

| Value | Meaning | Action |
|-------|---------|--------|
| `SUCCESS` | All validation checks passed. | Commit changes; proceed to next step or return success. |
| `FAILURE_RETRYABLE` | Validation checks failed; retry attempts remain. | Reset worktree (planned) or destroy/recreate worktree (single-step); retry with error feedback. |
| `FAILURE_TERMINAL` | No retry attempts remain, or failure is unrecoverable. | Return failure result. |

Three additional signals are defined in the design document for future use:

| Value | Status | Purpose |
|-------|--------|---------|
| `new_tasks_discovered` | Deferred | Agent found work not in the plan; triggers re-planning. |
| `blocked_on_human` | Deferred | Task requires human input; pauses and escalates. |
| `blocked_on_sibling` | Deferred | Task depends on an in-flight sibling's output; re-evaluates ordering. |


## SanityCheckVerdict

Returned by the sanity check activity during planned multi-step execution.

| Value | Meaning |
|-------|---------|
| `CONTINUE` | Plan remains valid; proceed with remaining steps. |
| `REVISE` | Plan is stale; rewrite remaining steps. |
| `ABORT` | Plan is unrecoverable; stop execution. |


## Core data models

### ForgeTaskInput

Workflow input. Wraps a `TaskDefinition` with execution settings.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task` | `TaskDefinition` | Yes | The task to execute. |
| `repo_root` | `str` | Yes | Absolute path to the repository root. |
| `max_attempts` | `int` | No (default: 2) | Maximum retry attempts for single-step mode. |
| `plan` | `bool` | No (default: `False`) | Enable planning mode. |
| `max_step_attempts` | `int` | No (default: 2) | Maximum retry attempts per step in planning mode. |
| `max_sub_task_attempts` | `int` | No (default: 2) | Maximum retry attempts per sub-task in fan-out steps. |
| `max_exploration_rounds` | `int` | No (default: 10) | Maximum rounds of LLM-guided context exploration. 0 disables exploration. |
| `max_fan_out_depth` | `int` | No (default: 1) | Maximum recursive fan-out depth. 1 = flat fan-out only. |
| `sanity_check_interval` | `int` | No (default: 0) | Run sanity check every N steps. 0 disables. |
| `resolve_conflicts` | `bool` | No (default: `True`) | Attempt LLM-based conflict resolution for fan-out file conflicts. |
| `model_routing` | `ModelConfig` | No | Maps capability tiers to concrete model names. |
| `thinking` | `ThinkingConfig` | No (default: 10,000 tokens) | Extended thinking configuration for the planner. |
| `sync_mode` | `bool` | No (default: `False`) | Use synchronous Messages API instead of batch mode. |
| `log_messages` | `bool` | No (default: `False`) | Save full API request/response JSON to the worktree. |

### TaskDefinition

A single unit of work.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task_id` | `str` | Yes | Unique identifier for the task. |
| `description` | `str` | Yes | What the task should produce. |
| `domain` | `TaskDomain` | No (default: `code_generation`) | The kind of task: code generation, research, review, documentation, generic. |
| `target_files` | `list[str]` | No (default: `[]`) | Files to create or modify. Optional when planning. |
| `context_files` | `list[str]` | No (default: `[]`) | Files to include as context for the LLM. |
| `validation` | `ValidationConfig` | No | Validation check configuration. |
| `base_branch` | `str` | No (default: `"main"`) | Branch to create the worktree from. |
| `context` | `ContextConfig` | No | Context discovery configuration. |

### TaskResult

Final workflow output.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task_id` | `str` | Yes | Identifier of the executed task. |
| `status` | `TransitionSignal` | Yes | Final outcome: `SUCCESS`, `FAILURE_RETRYABLE`, or `FAILURE_TERMINAL`. |
| `output_files` | `dict[str, str]` | No (default: `{}`) | Mapping of file path to content for all written files. |
| `validation_results` | `list[ValidationResult]` | No (default: `[]`) | Results from validation checks. |
| `error` | `str \| None` | No | If the task failed, a concise explanation. |
| `worktree_path` | `str \| None` | No | Path to the worktree where work was done. |
| `worktree_branch` | `str \| None` | No | Branch name of the worktree. |
| `step_results` | `list[StepResult]` | No (default: `[]`) | Per-step results (planned mode only). |
| `plan` | `Plan \| None` | No | The plan produced by the planner (planned mode only). |
| `llm_stats` | `LLMStats \| None` | No | Token usage and latency for the generation call. |
| `planner_stats` | `LLMStats \| None` | No | Token usage and latency for the planner call. |
| `context_stats` | `ContextStats \| None` | No | Statistics from context assembly. |
| `sanity_check_count` | `int` | No (default: 0) | Number of sanity checks executed during planned mode. |

### StepResult

Outcome of a single plan step.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `step_id` | `str` | Yes | Identifier of the plan step. |
| `status` | `TransitionSignal` | Yes | Outcome of this step. |
| `output_files` | `dict[str, str]` | No (default: `{}`) | Files written by this step. |
| `validation_results` | `list[ValidationResult]` | No (default: `[]`) | Validation check results. |
| `commit_sha` | `str \| None` | No | Git commit SHA on success. |
| `error` | `str \| None` | No | Error description on failure. |
| `sub_task_results` | `list[SubTaskResult]` | No (default: `[]`) | Per-sub-task results (fan-out steps only). |
| `llm_stats` | `LLMStats \| None` | No | Token usage and latency. |
| `digest` | `str` | No (default: `""`) | Compact summary for sanity check consumption. |
| `conflict_resolution` | `ConflictResolutionCallResult \| None` | No | Conflict resolution outcome (fan-out steps only). |

### SubTaskResult

Outcome of a single sub-task within a fan-out step.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sub_task_id` | `str` | Yes | Identifier of the sub-task. |
| `status` | `TransitionSignal` | Yes | Outcome of this sub-task. |
| `output_files` | `dict[str, str]` | No (default: `{}`) | Files produced by this sub-task. |
| `validation_results` | `list[ValidationResult]` | No (default: `[]`) | Validation check results. |
| `digest` | `str` | No (default: `""`) | From `LLMResponse.explanation`. |
| `error` | `str \| None` | No | Error description on failure. |
| `llm_stats` | `LLMStats \| None` | No | Token usage and latency. |
| `sub_task_results` | `list[SubTaskResult]` | No (default: `[]`) | Nested sub-task results (recursive fan-out). |
| `conflict_resolution` | `ConflictResolutionCallResult \| None` | No | Conflict resolution outcome (nested fan-out only). |

### LLMCallResult

Output of `call_llm`. Extends `LLMStats` with the parsed response.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task_id` | `str` | Yes | Identifier of the task. |
| `response` | `LLMResponse` | Yes | The parsed structured response. |
| `model_name` | `str` | Yes | Model used for the call. Inherited from `LLMStats`. |
| `input_tokens` | `int` | Yes | Input tokens consumed. Inherited from `LLMStats`. |
| `output_tokens` | `int` | Yes | Output tokens consumed. Inherited from `LLMStats`. |
| `latency_ms` | `float` | Yes | Request latency in milliseconds. Inherited from `LLMStats`. |
| `cache_creation_input_tokens` | `int` | No (default: 0) | Tokens written to cache. Inherited from `LLMStats`. |
| `cache_read_input_tokens` | `int` | No (default: 0) | Tokens read from cache. Inherited from `LLMStats`. |

### LLMStats

Lightweight statistics carried on Temporal result payloads.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_name` | `str` | Yes | Model used for the call. |
| `input_tokens` | `int` | Yes | Input tokens consumed. |
| `output_tokens` | `int` | Yes | Output tokens consumed. |
| `latency_ms` | `float` | Yes | Request latency in milliseconds. |
| `cache_creation_input_tokens` | `int` | No (default: 0) | Tokens written to prompt cache. |
| `cache_read_input_tokens` | `int` | No (default: 0) | Tokens read from prompt cache. |

### AssembledContext

Output of context assembly, input to `call_llm`.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task_id` | `str` | Yes | Identifier of the task. |
| `system_prompt` | `str` | Yes | The fully assembled system prompt. |
| `user_prompt` | `str` | Yes | The user prompt. |
| `context_stats` | `ContextStats \| None` | No | Statistics from context assembly. |
| `step_id` | `str \| None` | No | Step identifier (planned mode). |
| `sub_task_id` | `str \| None` | No | Sub-task identifier (fan-out mode). |
| `model_name` | `str` | No (default: `""`) | Model to use for the call. |
| `log_messages` | `bool` | No (default: `False`) | Whether to save API messages to disk. |
| `worktree_path` | `str` | No (default: `""`) | Path to the worktree. |

### LLMResponse

Structured output from the LLM, validated by Pydantic.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `files` | `list[FileOutput]` | No (default: `[]`) | New files to create with complete content. |
| `edits` | `list[FileEdit]` | No (default: `[]`) | Search/replace edits for existing files. |
| `explanation` | `str` | Yes | Brief explanation of what was produced. |

A file path must not appear in both `files` and `edits`.

### ValidationResult

Output from a single validation check.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `check_name` | `str` | Yes | Name of the check (e.g., `ruff_lint`, `ruff_format`). |
| `passed` | `bool` | Yes | Whether the check passed. |
| `summary` | `str` | Yes | Concise summary of the result. |
| `details` | `str \| None` | No | Extended details. Not sent to the LLM by default. |


## Enums

### TaskDomain

| Value | Description |
|-------|-------------|
| `code_generation` | Code generation tasks. |
| `research` | Research and analysis tasks. |
| `code_review` | Code review tasks. |
| `documentation` | Documentation tasks. |
| `generic` | Tasks that do not fit a specific domain. |

### CapabilityTier

| Value | Default model | Use cases |
|-------|--------------|-----------|
| `reasoning` | `anthropic:claude-opus-4-6` | Planning, sanity checks, conflict resolution. |
| `generation` | `anthropic:claude-sonnet-4-5-20250929` | Code/content generation. |
| `summarization` | `anthropic:claude-sonnet-4-5-20250929` | Knowledge extraction. |
| `classification` | `anthropic:claude-haiku-4-5-20251001` | Exploration, transition evaluation. |

### MatchLevel

Edit matching fallback levels, in order of attempt.

| Value | Description |
|-------|-------------|
| `exact` | The search string appears exactly once in the file. |
| `whitespace` | Match after stripping trailing whitespace from each line. |
| `indentation` | Match after dedenting and re-indenting at each indentation level. |
| `fuzzy` | `difflib.SequenceMatcher` best match above 60% similarity with 5% uniqueness gap. |


## Configuration models

### ValidationConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auto_fix` | `bool` | `True` | Automatically apply ruff fixes before checking. |
| `run_ruff_lint` | `bool` | `True` | Run ruff linter. |
| `run_ruff_format` | `bool` | `True` | Run ruff format check. |
| `run_tests` | `bool` | `False` | Run test suite. |
| `test_command` | `str \| None` | `None` | Custom test command. Uses `pytest` if not set. |

### ContextConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auto_discover` | `bool` | `True` | Enable automatic context discovery via import graph. |
| `include_dependencies` | `bool` | `False` | Include direct import contents and transitive symbol signatures upfront. When `False`, the LLM pulls dependencies on demand via exploration. |
| `token_budget` | `int` | `100000` | Token budget for context. |
| `output_reserve` | `int` | `16000` | Tokens reserved for LLM output. |
| `max_import_depth` | `int` | `2` | How deep to trace imports. |
| `include_repo_map` | `bool` | `True` | Include the PageRank-ranked repository map. |
| `repo_map_tokens` | `int` | `2048` | Token budget for the repo map. |
| `package_name` | `str \| None` | `None` | Python package name for import graph. Auto-detected if not set. |

### ModelConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `reasoning` | `str` | `anthropic:claude-opus-4-6` | Model for the reasoning tier. |
| `generation` | `str` | `anthropic:claude-sonnet-4-5-20250929` | Model for the generation tier. |
| `summarization` | `str` | `anthropic:claude-sonnet-4-5-20250929` | Model for the summarization tier. |
| `classification` | `str` | `anthropic:claude-haiku-4-5-20251001` | Model for the classification tier. |

### ThinkingConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `budget_tokens` | `int` | `0` | Token budget for extended thinking. 0 disables. |
| `effort` | `ThinkingEffort` | `"high"` | Effort level for adaptive thinking: `low`, `medium`, `high`, `max`. |


## Source locations

| Component | File |
|-----------|------|
| Data models | `src/forge/models.py` |
| Workflow | `src/forge/workflows.py` |
| Workflow building blocks | `src/forge/workflow_blocks.py` |
| Context assembly | `src/forge/activities/context.py` |
| LLM call | `src/forge/activities/llm.py` |
| Output writing | `src/forge/activities/output.py` |
| Validation | `src/forge/activities/validate.py` |
| Transition evaluation | `src/forge/activities/transition.py` |
| Planning | `src/forge/activities/planner.py` |
| Exploration | `src/forge/activities/exploration.py` |
| Conflict resolution | `src/forge/activities/conflict_resolution.py` |
| Sanity check | `src/forge/activities/sanity_check.py` |
| Git operations | `src/forge/activities/git_activities.py` |
| Batch submission | `src/forge/activities/batch_submit.py` |
| Batch parsing | `src/forge/activities/batch_parse.py` |
| Batch polling | `src/forge/activities/batch_poll.py` |
