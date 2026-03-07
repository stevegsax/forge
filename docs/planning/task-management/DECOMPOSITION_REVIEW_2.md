# Decomposition Plan Review

**Date:** 2026-03-07  
**Scope:** `DECOMPOSITION.md` and `DECOMPOSITION_SCENARIOS.md`

## Executive Summary

The decomposition design is strong on modularity, observability, and quality gates. The staged pipeline, versioned artifacts, and adversarial review process are all good foundations for reliable planning.

The main issues are not conceptual; they are **spec alignment and operability gaps** that could cause implementation churn:

1. A few control-flow rules conflict across docs (especially revision loops).
2. Core graph semantics (root node and edge direction conventions) need explicit clarification.
3. Versioning/persistence expectations are ahead of when a valid `PlanDAG` exists.
4. Human-in-the-loop timeouts and sub-plan approval behavior need clearer lifecycle/state handling.

## Strengths

- **Clear transform boundaries** with one-thing-per-call activities, which should make retries, tracing, and prompt iteration manageable.
- **Excellent audit posture** via immutable `plan_versions` and judge review persistence.
- **Deterministic + adversarial checks** combine hard validation and qualitative critique.
- **Template-driven workflow specialization** is scalable and keeps domain behavior explicit.

## Findings and Recommendations

### 1) Revision-loop inconsistency between docs

**Issue:** `DECOMPOSITION.md` says deterministic check failures loop to **step 5 (Recursive Split)**, while dependency-cycle scenarios state looping back to **analyze_dependencies**.

**Recommendation:** Split repair paths by failure class:

- Structural/splitting issues -> step 5
- Dependency-only issues (e.g., cycle in `DEPENDS_ON`) -> step 6

Then codify this in both docs as a single table to avoid divergence.

### 2) Versioning expectations before `PlanDAG` materialization

**Issue:** Scenarios require a `plan_version` at classification, but classification does not yet produce a full `PlanDAG` in the main design narrative.

**Recommendation:** Either:

- introduce a pre-plan artifact type for early transforms, or
- define a minimal `PlanDAG` skeleton created immediately after classification.

Without this, persistence behavior is underspecified.

### 3) Root semantics are undefined but required by checks

**Issue:** Deterministic checks require "no orphan nodes" with an exception for "root," but no canonical root-node model is defined.

**Recommendation:** Explicitly define one of:

- a synthetic root node always present, or
- root = node with no incoming `PARENT_CHILD` edges.

Make this definition normative and reuse it in checks and rendering.

### 4) Edge-direction conventions need one canonical rule block

**Issue:** `DEPENDS_ON` direction is defined, but `PARENT_CHILD` direction can be misread across prose and scenarios.

**Recommendation:** Add a short normative section with examples:

- `DEPENDS_ON`: `source` depends on `target`
- `PARENT_CHILD`: `source=child`, `target=parent` (if this is intended)

Then reference that section from scenarios.

### 5) Revision-attempt budget is ambiguous

**Issue:** Deterministic failures are said to count against the same 3-round revision budget used for judge rounds.

**Risk:** benign structural fixes could consume the budget before adversarial review converges.

**Recommendation:** Keep separate counters:

- structural repair attempts (steps 5-8)
- adversarial rounds (step 9)

Escalation rules should specify thresholds for each.

### 6) Human timeout lifecycle is only defined for one wait point

**Issue:** Timeout behavior is specified for user approval, but clarify/goal waits also block on user input.

**Recommendation:** Define timeout policy for **all** signal waits (clarify, goal confirm, final approval), including status transitions and resume behavior.

### 7) Cross-workflow sub-plan UX could stall parent progress

**Issue:** Parent waits on child full pipeline + child user approval; multiple cross-workflow nodes may create serial friction.

**Recommendation:** Specify batching/parallel rules for child workflows and how parent UI surfaces pending sub-plan approvals.

### 8) Scenario coverage is strong but misses negative cases for persistence integrity

**Issue:** Existing scenarios verify rows are written, but not transactional consistency across retries.

**Recommendation:** Add scenarios for:

- idempotent activity retry not duplicating version numbers,
- monotonic version guarantees under concurrent writes,
- partial failure rollback semantics.

## Suggested Spec Edits (High Priority)

1. Add a **"Failure Routing Matrix"** mapping each deterministic failure type to step 5 vs step 6.
2. Add a **"Graph Semantics"** section defining root behavior and edge direction with 2-3 concrete examples.
3. Add a **"Versioning Lifecycle"** section clarifying when `PlanDAG` first exists and what early transforms persist.
4. Add a **"User Wait Timeout Policy"** section covering all signal/wait points.

## Overall Assessment

The plan is implementable and thoughtfully designed. With the alignment fixes above, it should be significantly easier to build without semantic drift between workflow code, validation logic, and behavioral tests.
