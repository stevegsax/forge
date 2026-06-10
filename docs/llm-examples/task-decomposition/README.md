# Worked Example: Task Decomposition Pipeline

This example traces Forge's 10-step task decomposition pipeline from
start to finish for a simple user request:

> "Write a python module that reads the files from the current directory
> and prints them to stdout"

The task is intentionally simple so the example focuses on pipeline
mechanics rather than complex decomposition. The result is a 3-leaf
`PlanDAG` with dependency edges.

## Pipeline Reference

The decomposition pipeline is defined in
[DECOMPOSITION.md](../../planning/task-management/DECOMPOSITION.md).

## Steps

| Step | Name | LLM? | Model Tier | PlanDAG Version |
|------|------|------|------------|-----------------|
| 1 | [Classify](step_0001/) | Yes | CLASSIFICATION | v1 (skeleton) |
| 2 | [Clarify](step_0002/) | Yes | GENERATION | -- (no mutation) |
| 3 | [Goal Statement](step_0003/) | Yes | GENERATION | v2 |
| 4 | [First Pass Decompose](step_0004/) | Yes | REASONING | v3 |
| 5 | [Recursive Split](step_0005/) | Yes | CLASSIFICATION | -- (no mutation) |
| 6 | [Dependency Analysis](step_0006/) | Yes | REASONING | v4 |
| 7 | [Acceptance Criteria](step_0007/) | Yes | GENERATION | v5 |
| 8 | [Deterministic Checks](step_0008/) | No | -- | -- (no mutation) |
| 9 | [Adversarial Review](step_0009/) | Yes | REASONING | -- (no mutation) |
| 10 | [User Approval](step_0010/) | No | -- | -- (status change) |

## Entities

| Entity | ID |
|--------|----|
| Plan | `plan-7a3f` |
| Root node | `node-root` |
| Create file_printer module | `node-001` |
| Add CLI entry point | `node-002` |
| Write unit tests | `node-003` |
| Root -> node-001 | `edge-pc-001` |
| Root -> node-002 | `edge-pc-002` |
| Root -> node-003 | `edge-pc-003` |
| node-002 depends on node-001 | `edge-dep-001` |
| node-003 depends on node-001 | `edge-dep-002` |

## Templates

The [templates/](templates/) directory contains 8 Jinja2 prompt
templates showing the `{% extends %}` inheritance pattern described in
DECOMPOSITION.md. Each template includes the JSON response schema so
readers can see what the LLM is asked to produce.

## How to Read

1. Start here for the overview and entity table.
2. Walk through `step_0001/` to `step_0010/` in order.
3. For each step, read the `README.md` first, then `llm_prompt.md` and
   `llm_result.md` to see the full LLM interaction.
4. Compare prompts against the source templates in `templates/` to see
   how variables are substituted.
