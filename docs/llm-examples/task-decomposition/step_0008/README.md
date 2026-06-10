# Step 8: Deterministic Checks

- **Parent step:** step_0007
- **No LLM** — pure function
- **LLM calls:** 0
- **PlanDAG version:** -- (no mutation)

## Action

Run 12 structural validations against the PlanDAG. These are pure
functions with no LLM involvement.

## Checks Performed

| # | Check | Result |
|---|-------|--------|
| 1 | DAG is acyclic | PASS |
| 2 | All node IDs are unique | PASS |
| 3 | All edge references point to existing nodes | PASS |
| 4 | Every leaf has at least one acceptance criterion | PASS |
| 5 | Every leaf has an execution_type | PASS |
| 6 | Exactly one root node exists | PASS |
| 7 | Every non-root node is reachable from root via PARENT_CHILD edges | PASS |
| 8 | `children` lists consistent with PARENT_CHILD edges | PASS |
| 9 | No DEPENDS_ON edge connects a parent to its own descendant | PASS |
| 10 | Cross-workflow references are valid | PASS (none present) |
| 11 | Container nodes have at least 2 children | PASS (root has 3) |
| 12 | No circular parent-child relationships | PASS |

## Decision

All 12 checks pass. No structural repairs are needed.

If any check had failed, the system would route the failure to the
appropriate repair step based on the failure class:

- Structural / splitting failures -> step 5 (Recursive Split)
- Dependency-only failures -> step 6 (Dependency Analysis)
- Edge consistency failures -> step 5 (Recursive Split)

## Outcome

- All 12 deterministic checks pass
- PlanDAG is not mutated
- Pipeline advances to step 9 (Adversarial Review)
