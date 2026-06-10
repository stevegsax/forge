# Step 9: Adversarial Review

- **Parent step:** step_0008
- **Model tier:** REASONING -> `claude-opus-4-6`
- **LLM calls:** 3 (one per judge, in parallel)
- **PlanDAG version:** -- (no mutation)

## Action

Three independent adversarial judges review the plan. Each judge has
a different persona and must argue AGAINST the plan before arguing FOR
it, then render a verdict.

For `software` workflow type, the default personas are:

1. **Expert Skeptic** — focuses on edge cases, failure modes, missing
   error handling
2. **Completeness Auditor** — focuses on coverage gaps, missing steps,
   overlooked requirements
3. **Dependency Critic** — focuses on ordering errors, hidden
   dependencies, parallelism opportunities

## Decision

All 3 judges vote APPROVE (3-of-3, unanimous). The consensus rule
requires 2-of-3 APPROVE for acceptance.

### Score Summary

| Dimension | Expert Skeptic | Completeness Auditor | Dependency Critic |
|-----------|---------------|---------------------|-------------------|
| COMPLETENESS | 4 | 4 | 4 |
| GRANULARITY | 5 | 5 | 5 |
| FEASIBILITY | 5 | 5 | 5 |
| DEPENDENCY_CORRECTNESS | 5 | 4 | 5 |
| ACCEPTANCE_CRITERIA_QUALITY | 4 | 4 | 4 |

Each judge identified minor weaknesses (e.g., no explicit criterion
for handling symlinks, no mention of subdirectory behavior) but
concluded these are acceptable for a "simple" complexity task with
clear scope boundaries.

## Outcome

- 3-of-3 APPROVE (unanimous)
- `judge_round = 1` (no revision needed)
- PlanDAG is not mutated
- Pipeline advances to step 10 (User Approval)

## Files

- [llm_prompt.md](llm_prompt.md) — all 3 judge prompts
- [llm_result.md](llm_result.md) — all 3 judge verdicts with scores
