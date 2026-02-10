# Phase 7: LLM-Guided Context Exploration

## Overview

Phase 7 adds an exploration loop before code generation. Instead of the system deciding what context the LLM sees, the LLM requests context from a menu of providers until it has enough understanding to produce output.

## Execution Flow

Previous (single-step):

```
create_worktree -> assemble_context -> call_llm -> write -> validate -> commit
```

New (with exploration):

```
create_worktree -> assemble_initial_context
  -> [exploration loop]
    -> call_exploration_llm(context + provider menu) -> ExplorationResponse
    -> if requests empty: break
    -> fulfill_context_requests(requests) -> responses
    -> accumulate responses into context
    -> repeat (up to max_exploration_rounds)
  -> call_llm(accumulated context) -> write -> validate -> commit
```

## Context Providers

Each provider wraps an existing tool. No new dependencies.

| Provider | Function | Parameters |
|----------|----------|------------|
| `read_file` | Read file contents | `path` |
| `search_code` | Regex search across files | `pattern`, `glob` |
| `symbol_list` | Public API of a module | `file_path` |
| `import_graph` | Module import/imported-by relationships | `file_path` |
| `run_tests` | Execute pytest | `path`, `marker` |
| `lint_check` | Ruff lint issues | `files` |
| `git_log` | Recent commit history | `path`, `n` |
| `git_diff` | Diff against base branch | `base` |
| `repo_map` | Ranked project structure | (none) |
| `discover_context` | Full auto-discovery for target files (always includes dependencies) | `target_files` |
| `past_runs` | Recent workflow results | `limit` |
| `playbooks` | Relevant playbook entries | `tags` |

## Data Models

New models in `src/forge/models.py`:

- `ContextProviderSpec` -- Provider description shown to the LLM
- `ContextRequest` -- A request for context from a specific provider
- `ExplorationResponse` -- LLM output (list of requests, empty = ready to generate)
- `ContextResult` -- Result of fulfilling a context request
- `FulfillContextInput` -- Input to the fulfill activity
- `ExplorationInput` -- Input to the exploration LLM call

`ForgeTaskInput` gains `max_exploration_rounds: int = 10`.

## Activities

Two new activities in `src/forge/activities/exploration.py`:

- `call_exploration_llm` -- Calls pydantic-ai with `ExplorationResponse` output type
- `fulfill_context_requests` -- Dispatches requests to the provider registry

## Workflow Integration

- **Single-step mode**: Exploration loop runs between `assemble_context` and `call_llm`. Results are appended to the system prompt.
- **Planned mode**: Exploration loop runs before the planner call. Results are appended to the planner's system prompt.

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-exploration-rounds` | `10` | Max rounds of exploration (0 disables) |
| `--no-explore` | off | Disable exploration entirely |
| `--include-deps` | off | Include dependency contents in upfront context (see D49) |

## Key Files

- `src/forge/models.py` -- Exploration data models
- `src/forge/providers.py` -- Provider registry and handlers
- `src/forge/activities/exploration.py` -- Exploration activities
- `src/forge/workflows.py` -- Exploration loop integration
- `src/forge/worker.py` -- Activity registration
- `src/forge/cli.py` -- CLI options
- `tests/test_providers.py` -- Provider tests
- `tests/test_activity_exploration.py` -- Activity tests
- `tests/test_exploration_models.py` -- Model tests
