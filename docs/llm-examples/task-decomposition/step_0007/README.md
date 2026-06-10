# Step 7: Acceptance Criteria

- **Parent step:** step_0006
- **Model tier:** GENERATION -> `claude-sonnet-4-5-20250929`
- **LLM calls:** 3 (one per leaf node, in parallel)
- **PlanDAG version:** v5

## Action

Generate specific, testable acceptance criteria for each leaf node.
The software workflow template guides criteria toward tests, lint
checks, importability, and edge case handling.

Three parallel LLM calls are made, one per leaf:

1. **node-001** (Create file_printer module) — 5 criteria covering
   importability, text file reading, binary file handling, empty
   directory behavior, and file headers.
2. **node-002** (Add CLI entry point) — 4 criteria covering
   `python -m` invocation, default directory, custom directory argument,
   and help text.
3. **node-003** (Write unit tests) — 4 criteria covering test count,
   pytest compatibility, `tmp_path` isolation, and coverage of all
   specified scenarios.

## Decision

Each node receives 4-5 criteria that are specific enough to be
mechanically verifiable. Vague criteria like "works correctly" are
rejected in favor of precise conditions like "returns empty string
when directory contains no files."

## Outcome

- PlanDAG v5 created with `acceptance_criteria` populated on all 3
  leaf nodes
- All criteria are testable (can be evaluated by the deterministic
  check or LLM judge in the validation step)
- Pipeline advances to step 8 (Deterministic Checks)

## Files

- [llm_prompt.md](llm_prompt.md) — all 3 acceptance criteria prompts
- [llm_result.md](llm_result.md) — all 3 acceptance criteria results
