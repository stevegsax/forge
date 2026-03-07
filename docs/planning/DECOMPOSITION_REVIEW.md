# Task Decomposition Plan Review

**Date:** 2026-03-07  
**Reviewer:** Codex

## Summary

The decomposition architecture is strong overall: it has clear transform boundaries, durable versioning, explicit human-in-the-loop checkpoints, and deterministic guardrails. The biggest issues are a few spec contradictions and missing invariants that will make implementation ambiguous if not resolved early.

## What Looks Strong

1. **Transform granularity is implementation-friendly.** The “many small transforms” strategy should make prompts and failure handling easier to debug.
2. **Auditability is first-class.** Versioned artifacts + persisted review/clarification records are exactly what this workflow needs.
3. **Adversarial review design is practical.** Persona-based 2-of-3 consensus adds quality control without requiring unanimity.
4. **Human interaction pattern is consistent.** Reusing one signal/wait protocol reduces workflow complexity.

## Critical Issues to Resolve Before Implementation

### 1) Revision-attempt counting is contradictory

- `DECOMPOSITION.md` says deterministic-check failures count toward the “3 allowed revision attempts.”
- `DECOMPOSITION_SCENARIOS.md` says deterministic-check failures do **not** count toward the 3-round adversarial-review limit.

**Recommendation:** Define **two independent counters**:
- `judge_round` (max 3, increments only after adversarial review consensus = reject)
- `repair_round` (optional safety cap for deterministic or structural repair loops)

Then update both docs to reference the same terms.

### 2) Root/top-level node semantics are undefined but required

The deterministic checks require every non-root node to have a parent edge, but the PlanDAG model and first-pass output never define a root node.

**Recommendation:** Add an explicit synthetic root node invariant:
- exactly one root node (`is_leaf=false`, no parent)
- all first-pass top-level tasks become children of root
- orphan checks are evaluated relative to that root

### 3) Recursive split termination condition is too strict

The scenario says each pass “reduces the number of non-leaf nodes,” which is not generally true when splitting one non-leaf into multiple children that may remain non-leaf.

**Recommendation:** Replace with a monotonic progress invariant such as:
- every loop iteration must either
  - mark at least one previously-unconfirmed node as atomic, or
  - reduce a complexity metric (e.g., sum of estimated complexities for unresolved nodes)
- abort/escalate if no progress after N iterations.

## High-Value Clarifications

### 4) `children` list and `PARENT_CHILD` edges duplicate structure

Both are useful, but there is no source-of-truth rule.

**Recommendation:** Declare one canonical representation (suggest: edges) and treat the other as derived/materialized for convenience, validated in deterministic checks.

### 5) Cross-workflow sub-plan completion contract is underspecified

The parent waits for child workflow completion, but it does not define how child acceptance criteria/status map back to the parent node.

**Recommendation:** Define parent-node state transitions:
- `blocked_on_subplan` → `subplan_approved` or `subplan_rejected`
- reject parent plan if required child sub-plan is rejected, unless user explicitly overrides.

### 6) Template inventory in docs is inconsistent

`_shared/decompose_base.prompt.j2` is used in examples but missing from the listed shared template tree.

**Recommendation:** Align the tree with actual expected shared templates and include required/optional template contract per workflow type.

## Suggested Acceptance Invariants (Add to Spec)

1. Exactly one root node exists.
2. Every node is reachable from root via `PARENT_CHILD` edges.
3. All leaves have non-empty acceptance criteria and execution type.
4. No `DEPENDS_ON` edge may cross between parent and descendant nodes (to avoid semantic conflicts).
5. Cross-workflow child nodes must reference a valid `subplan_id` once spawned.

## Implementation Sequencing Advice

1. Lock invariants/data model first (root semantics, counters, canonical parent-child representation).
2. Implement deterministic checks next (as executable spec).
3. Build transform activities around those checks.
4. Add adversarial review and revision loop orchestration after structural reliability is proven.

## Overall Assessment

The plan is viable and well-structured. Resolve the three critical issues first (revision counters, root semantics, recursive progress definition), then implementation risk drops significantly.
