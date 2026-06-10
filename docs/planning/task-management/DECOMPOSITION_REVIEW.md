# Decomposition Design Review

**Reviewer:** Claude
**Date:** 2026-03-07
**Documents:** [DECOMPOSITION.md](DECOMPOSITION.md), [DECOMPOSITION_SCENARIOS.md](DECOMPOSITION_SCENARIOS.md)

---

## Overall Assessment

This is a well-structured design that follows Forge's established principles. The multi-transform pipeline with adversarial review is a significant step up from the current single-call planner. The Gherkin scenarios are thorough and cover the happy path, error cases, and edge cases for every pipeline stage.

**Verdict:** Approve with required changes (see below).

---

## Strengths

1. **Principled decomposition of the decomposition.** "Many small transforms over few large ones" is the right call. The current planner does too much in one LLM call — classification, clarification, decomposition, and ordering all happen implicitly. Separating these into explicit pipeline stages makes each one testable, observable, and independently improvable.

2. **Adversarial review is well-designed.** The 3-judge, 2-of-3 consensus model with persona selection by workflow type is a pragmatic approach. Requiring judges to argue AGAINST before FOR is a good debiasing technique. The 3-round escalation limit prevents infinite loops.

3. **Versioning and auditability.** Every transform producing a versioned snapshot is consistent with Forge's observability-first approach. The separate `plans.db` is a clean separation of concerns from the interaction store.

4. **Temporal patterns are idiomatic.** Signal/wait for human interaction, child workflows for cross-workflow sub-plans, and activity boundaries with appropriate timeouts all follow Temporal best practices. The activity table with timeouts and retry counts is specific and reasonable.

5. **Template-driven extensibility.** Adding a new workflow type by dropping a directory of Jinja2 templates with no code changes is an elegant extension mechanism.

6. **Scenarios are comprehensive.** The Gherkin suite covers ~60 scenarios across 14 features. Good coverage of negative cases (cyclic deps, orphan nodes, single-child containers) and boundary behavior (timeouts, 3-round escalation).

---

## Required Changes

### R1. Clarify the relationship between `PlanNode.children` and `PARENT_CHILD` edges

The data model stores the parent-child relationship in two places: `PlanNode.children` (a list of child UUIDs) and `PlanEdge` with `edge_type=PARENT_CHILD`. This is a consistency hazard — every mutation must update both, and the deterministic checks must verify they agree.

**Recommendation:** Pick one as the source of truth. Since the edge list is the canonical representation for the DAG, remove `PlanNode.children` and derive it from edges. Or keep `children` as a denormalized convenience field and add an explicit deterministic check that `children` matches the PARENT_CHILD edges.

### R2. Define the root node(s) explicitly

The data model has no concept of a root node. The deterministic check says "every non-root node has a parent edge" but there's no way to identify which nodes are roots. The `PlanDAG` should either:

- Add a `root_ids: list[str]` field, or
- Define root nodes as those with no incoming `PARENT_CHILD` edges (and document this convention)

The second option is simpler and consistent with standard DAG semantics, but needs to be stated explicitly. The scenarios reference "root" without defining it.

### R3. Bound the recursive split depth

The recursive split (step 5) has no explicit depth limit. The termination condition ("all nodes are leaves or containers whose children are all leaves") is correct but doesn't prevent pathological cases where the LLM keeps splitting indefinitely. Add a `max_depth` parameter (suggested default: 4-5 levels) and escalate to the user if exceeded.

This aligns with the existing `max_sub_task_depth` in `ForgeTaskInput` and the architecture principle "halt when confused."

### R4. Handle the shared revision counter between steps 8 and 9

The doc says deterministic check failure (step 8) "counts as one of the 3 allowed revision attempts" and loops back to step 5. But the adversarial review (step 9) also has its own 3-round limit. The scenario doc contradicts this: "Deterministic check failure does not count toward judge revision limit."

**The scenario doc is correct** — these should be independent counters. A structural validation failure is a different class of problem than a judge rejection. Clarify in the main design doc that:

- Deterministic failures have their own retry limit (suggest 2-3 attempts)
- Judge rejection rounds are counted separately
- Both have escalation paths

### R5. Add a scenario for the "single child → convert to leaf" rule

DECOMPOSITION.md step 5 says "if only one child would result, the parent is converted to a leaf instead." The scenarios mention this in "Container nodes have at least 2 children" but don't cover the conversion behavior. Add a scenario:

```gherkin
Scenario: Single child result converts parent to leaf
    Given a non-leaf node "initialize database"
    When the split_node activity produces only 1 child
    Then the parent node is converted to a leaf
    And the single child is discarded
    And the parent's acceptance_criteria are generated
```

### R6. Specify what happens to judge scores when a plan is approved

The judge produces scores on 5 dimensions (1-5 each). These scores are persisted in `judge_reviews`, but the design doesn't specify whether they influence anything downstream. At minimum:

- Are scores surfaced to the user at approval time?
- Is there a minimum score threshold for approval (e.g., a judge can APPROVE but give GRANULARITY a 1)?
- Are aggregate scores tracked for plan quality metrics over time?

If scores are purely informational, state that. If they gate anything, specify the thresholds.

---

## Recommended Changes (Non-Blocking)

### N1. Consider batch mode for adversarial judges

Open question 3 asks about this. For an interactive planning workflow, the latency of 3 concurrent synchronous calls is acceptable (they're already parallelized). Batch mode would save ~50% on cost but add minutes of latency. **Recommendation:** Default to sync, add a `use_batch_for_judges: bool` option for cost-sensitive non-interactive use cases (e.g., CI/CD pipeline planning).

### N2. Add an estimated token budget per transform

The model routing table specifies capability tiers but not token budgets. The existing system has `max_tokens` and `thinking.budget_tokens` per call. Each transform should have a recommended `max_tokens` range — classification needs ~100 tokens, first-pass decomposition might need 4000+, and judges need enough for thorough argumentation. This prevents surprises when model routing changes.

### N3. Consider caching the classification result

If the same user request is resubmitted (e.g., after a rejected plan), the classification step will re-run. Since classification is deterministic for the same input, consider caching or short-circuiting when a plan with the same `user_request` already exists.

### N4. Add a "plan diff" for revision rounds

When judges reject and the plan is revised, the user (and the judges in the next round) see the full revised plan. A diff between plan versions would make it much easier to understand what changed. The versioning infrastructure already supports this — add a `plan_diff(v1, v2) -> str` function alongside `plan_to_dot()`.

### N5. Clarify template fallback semantics

The scenarios say "Missing template falls back to shared" but `_shared/` templates are bases meant to be extended, not used standalone. If `research/` lacks `clarify.prompt.j2`, does the system use `_shared/clarify_base.prompt.j2` directly (which has empty `{% block %}` sections)? Or is a workflow-specific override always required? If the base templates are valid standalone (with sensible defaults in blocks), document this. If not, make the absence of a required template a startup-time error.

### N6. Think about plan execution translation now (Open Question 2)

The design defers the `PlanNode` → `ForgeTaskWorkflow` execution translation. This is the highest-risk open question because it constrains what `PlanNode` needs to contain. Key mapping questions:

- `PlanNode.context` must contain enough for `assemble_step_context()` — at minimum `target_files` and `context_files`
- `PlanNode.estimated_complexity` should map to the existing `CapabilityTier`
- `PlanNode.acceptance_criteria` must translate to `ValidationConfig` parameters

Consider defining the `PlanNode.context` schema per workflow type now, even if the translation layer is deferred. Otherwise you risk discovering at implementation time that the decomposition pipeline doesn't produce the fields that execution needs.

### N7. Add a max total node count guard

Large plans with 50+ leaf nodes may be a sign of scope creep or over-splitting. Add a configurable `max_leaf_nodes` (suggested default: 30) as a deterministic check. The Scope Guardian persona partially covers this, but a hard limit is more reliable.

---

## Scenario Gap Analysis

The scenarios are thorough but missing a few cases:

| Gap | Suggested Scenario |
| ----- | ------------------- |
| **Classify activity fails/times out** | Temporal retries; after exhaustion, workflow terminates with error status |
| **Goal statement generation after multiple clarify rounds** | Verify all accumulated answers (not just latest round) are included |
| **Recursive split with cross-workflow AND further splitting** | A research sub-plan node itself needs splitting — does the child workflow handle this? (Yes, but test it) |
| **Plan with zero dependencies** | All leaves are independent — valid plan, maximally parallel |
| **All 3 judges reject on round 1** | Same as 2-of-3, but verify all 3 sets of required_changes are collected |
| **Concurrent clarification from split + main pipeline** | A split raises a clarification question while the main pipeline is already waiting for a different user response — how are signals disambiguated? |
| **Empty plan (goal requires no decomposition)** | User asks for something trivially simple — does the pipeline produce a single-leaf plan? |

---

## Compatibility with Existing System

The design correctly identifies the mapping from existing components to new ones. Key compatibility observations:

1. **`PlanStep` → `PlanNode` is not 1:1.** `PlanStep` has `target_files` and `context_files` at the top level; `PlanNode` puts these in `context: dict`. The translation layer needs to extract these. Consider making `target_files` and `context_files` first-class fields on `PlanNode` (at least for the `software` workflow type) rather than burying them in a generic dict.

2. **`SubTask` → child `PlanNode` is clean.** The existing `SubTask` model maps naturally to `PlanNode` children with `PARENT_CHILD` edges. The fan-out mechanism in `ForgeTaskWorkflow` doesn't need to change.

3. **Deterministic checks are a superset.** The 9 new checks include all existing checks from `eval/deterministic.py` plus additional structural validations. The existing check functions can be reused or adapted.

4. **Judge scoring dimensions changed.** The existing judge uses 6 criteria (COMPLETENESS, GRANULARITY, ORDERING, CONTEXT_QUALITY, FAN_OUT_APPROPRIATENESS, EXPLANATION_QUALITY). The new design uses 5 (COMPLETENESS, GRANULARITY, FEASIBILITY, DEPENDENCY_CORRECTNESS, ACCEPTANCE_CRITERIA_QUALITY). ORDERING → DEPENDENCY_CORRECTNESS is a natural evolution. CONTEXT_QUALITY and FAN_OUT_APPROPRIATENESS are dropped (reasonable since context assembly happens later). EXPLANATION_QUALITY is dropped (the goal statement serves this purpose). FEASIBILITY and ACCEPTANCE_CRITERIA_QUALITY are new and valuable additions. Document this migration.

5. **`DomainConfig` → workflow templates is a significant migration.** The current `DomainConfig` in `domains.py` includes prompt fragments, validation defaults, and template strings. The new Jinja2 template system replaces the prompt aspects, but validation defaults (`ValidationConfig`) still need a home. Consider embedding `validation_defaults` in `description.md` metadata or a `config.yaml` per workflow type.

---

## Summary of Action Items

| # | Type | Item | Priority |
| --- | ------ | ------ | ---------- |
| R1 | Required | Resolve `children` vs `PARENT_CHILD` edge redundancy | High |
| R2 | Required | Define root node identification | High |
| R3 | Required | Bound recursive split depth | High |
| R4 | Required | Separate deterministic and judge revision counters | High |
| R5 | Required | Add single-child-to-leaf conversion scenario | Medium |
| R6 | Required | Specify judge score semantics | Medium |
| N1 | Recommended | Batch option for judges | Low |
| N2 | Recommended | Token budgets per transform | Medium |
| N3 | Recommended | Classification caching | Low |
| N4 | Recommended | Plan diff for revision rounds | Medium |
| N5 | Recommended | Clarify template fallback semantics | Medium |
| N6 | Recommended | Define `PlanNode.context` schema per workflow type | High |
| N7 | Recommended | Max leaf node count guard | Low |
