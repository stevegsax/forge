# Step 5: Recursive Split (Atomicity Check)

- **Parent step:** step_0004
- **Model tier:** CLASSIFICATION -> `claude-haiku-4-5-20251001`
- **LLM calls:** 3 (one per leaf node)
- **PlanDAG version:** -- (no mutation)

## Action

Check each leaf node for atomicity: can it be completed in a single
LLM call? The atomicity check uses a lightweight classification model
since it's a binary yes/no judgment per node.

Three parallel LLM calls are made, one for each leaf:

1. **node-001** (Create file_printer module) — single file, one
   function, simple logic. **Atomic.**
2. **node-002** (Add CLI entry point) — modifies one file, adds
   `__main__` block and argparse. **Atomic.**
3. **node-003** (Write unit tests) — single test file with 4 test
   cases using pytest. **Atomic.**

## Decision

All 3 nodes are confirmed atomic. No further splitting is needed.
The recursive split loop terminates after one iteration with zero
splits.

For a more complex task, non-atomic nodes would be split into children
and re-checked. The loop continues until every leaf passes the
atomicity check or the iteration budget (5 rounds) is exhausted.

## Outcome

- All 3 leaf nodes confirmed atomic
- PlanDAG is not mutated (no new nodes or edges)
- Pipeline advances to step 6 (Dependency Analysis)

## Files

- [llm_prompt.md](llm_prompt.md) — all 3 atomicity check prompts
- [llm_result.md](llm_result.md) — all 3 atomicity check results
