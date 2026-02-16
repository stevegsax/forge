# Codex Architecture Review: Forge

## Scope and Method

I reviewed:

- `docs/DESIGN.md`
- `docs/ARCHITECTURE.md`

I also compared Forge's design against several public projects and ecosystems to calibrate what is strong vs. missing in 2025-era agent systems:

- OpenHands (SDK/CLI/cloud + benchmark emphasis)
- SWE-agent (SWE-bench-driven engineering agent)
- LangGraph (stateful graph orchestration)
- Aider/Cline style coding agents (tight human-in-the-loop diff + validation ergonomics)

---

## Executive Summary

Forge has a **clear and internally consistent architecture**. The strongest decisions are:

1. Keep control with the orchestrator, not the LLM.
2. Treat every LLM interaction as a stateless document completion.
3. Use deterministic validation and explicit transition signals.
4. Isolate work in git worktrees.

Those are good foundations.

However, the design still feels like a **well-structured prototype**, not yet a robust production platform. The main weaknesses are:

- It is overly optimistic about planner quality and prompt-centric reliability.
- It has no explicit cost/SLA governance model.
- Safety/compliance controls are under-specified.
- Evaluation strategy is planner-heavy and product-impact-light.
- Multi-tenant and long-lived operations concerns are mostly absent.

If I were re-architecting, I would keep the core pattern but add a stronger **policy/control plane**, **state and memory governance**, and **production reliability mechanisms** before adding more autonomous features.

---

## What I Like

### 1) Orchestrator-owned control loop
This is the best decision in the design. Keeping transition logic outside the model avoids a lot of common agent failure modes.

### 2) Universal workflow step abstraction
"Construct → Send → Receive/Serialize → Validate → Transition" is simple and composable. It creates a stable mental model and helps testing.

### 3) Deterministic checks first
Running lint/type/test before subjective review is right. It reduces false confidence and creates clear retry signals.

### 4) Git worktree isolation
Per-task isolation is practical and auditable. It maps naturally to code-review workflows.

### 5) Observability intent is strong
Prompt/response persistence + traces + execution journal is the right direction for debugging and trust.

---

## What I Dislike (Critical Feedback)

## A) The design is too prompt-centric and under-specified as a control system
The docs repeatedly say planning quality dominates outcomes. True, but the architecture currently treats this as a prompt/model problem more than a systems problem.

### Why this is risky
- Prompt quality is hard to version and reason about under drift.
- Planning can silently degrade with model/provider changes.
- The system lacks hard policy constraints beyond validation and boundaries text.

### What I would do differently
- Add a **formal policy layer** (machine-checkable constraints) before execution:
  - Allowed file/path patterns
  - Allowed command classes
  - Resource/time budgets
  - Data sensitivity policy
- Make planner output pass a **policy compiler** before being admitted.

---

## B) Cost and latency governance are not first-class
The docs discuss token budgets for prompts, but there is little on **global spend control**, deadline/SLA behavior, or adaptive degradation.

### Missing capabilities
- Per-run budget caps and hard stop behavior.
- Per-tenant quotas.
- Deadline-aware execution (e.g., "must finish in 10 minutes").
- Dynamic model downgrades/upgrades based on remaining budget.

### What I would do differently
Introduce a scheduler that optimizes for a target objective: `quality x success probability / cost / latency`, with clear abort/degrade policies.

---

## C) Safety/compliance model is too weak for real environments
There is little explicit treatment of secret handling, PII boundaries, data residency, tool-level least privilege, or tenant isolation.

### Why this matters
As soon as this moves beyond personal repos, this becomes blocking.

### What I would add
- Policy-enforced redaction and secret scanning at context assembly.
- Tool permissions scoped per task and per tenant.
- Immutable audit logs for all external actions.
- Optional "approval gates" for high-risk actions.

---

## D) Retry semantics are simplistic and potentially expensive
"Fresh worktree and retry" is clean, but likely wasteful and can repeat mistakes.

### Problems
- Same failure may recur with marginal prompt changes.
- Full retry can throw away useful partial progress.
- No explicit root-cause classification pipeline.

### Improvements
- Failure classifier with buckets: deterministic bug, missing context, flaky tests, environment issue, policy violation.
- Retry strategy per class (not one-size-fits-all).
- Patch-level rollback/replay instead of always full reset.

---

## E) Conflict resolution via LLM is underspecified for correctness
Using an LLM to merge conflicting child outputs is useful but dangerous without stronger guards.

### Risks
- Semantic regressions despite syntactic validity.
- Hidden loss of one branch's intent.

### Improvements
- Structured merge protocol:
  - Extract intents per conflicting change.
  - Verify each intent survives in merged file.
  - Run targeted regression tests for conflict regions.
- Treat merge conflicts as a distinct validation domain.

---

## F) The exploration loop could become noisy and expensive
The pre-generation exploration loop is clever, but it can create context bloat and weak signal if not tightly constrained.

### Missing controls
- Provider-level budgets and call limits.
- Quality scoring of retrieved context.
- Stopping criteria beyond round count/empty requests.

### Improvements
- Add exploration utility scoring and pruning.
- Cache provider responses with invalidation rules.
- Penalize repeated low-value requests.

---

## G) Evaluation strategy is too narrow
Planner eval exists (good), but there is limited mention of end-to-end acceptance metrics tied to real outcomes.

### Missing metrics
- Task success on realistic benchmark suites.
- Human review burden (time-to-approve, rework rate).
- Regression rate after merge.
- Cost per successful task.

### What I would do differently
Adopt a multi-layer eval stack:
1. Unit evals (activities/components)
2. Scenario simulations (synthetic workflow DAGs)
3. Repository-scale benchmark tasks
4. Production shadow mode scoring before rollout

---

## H) Architecture claims domain agnosticism, but implementation is heavily code-centric
The docs claim broad task applicability, but many concrete mechanisms are Python/code-generation specific (`ast`, ruff, import graphs, file edits).

### Concern
This mismatch can confuse users and create roadmap debt.

### Recommendation
Either:
- Narrow the product claim to "code-first orchestrator" for release 1, or
- Define explicit domain adapters/interfaces now (context sources, validators, transition policies) so non-code domains are real pluggable modules.

---

## I) Human-in-the-loop is present but not operationally complete
The docs mention escalation and merge gating but not practical triage UX and queueing behavior.

### Missing operational features
- Escalation prioritization and assignment.
- Batched review views by risk/confidence.
- One-click replay with modified constraints.
- Reviewer feedback capture that closes the loop into policy and prompts.

---

## Key Missing Features (High Priority)

1. **Policy/control plane** for constraints and permissions.
2. **Budget and SLA manager** with hard caps and degrade policies.
3. **Task memory model** with explicit retention/expiry and provenance scoring.
4. **Benchmark-driven release gates** (not just planner eval).
5. **Multi-tenant security model** (RBAC, auditability, isolation).
6. **Flakiness and environment diagnosis layer** (to avoid pointless retries).
7. **Confidence scoring** and risk-based human review routing.
8. **Canary/shadow deployment path** for prompt/model changes.

---

## Comparison Notes from Similar Projects

### OpenHands
OpenHands emphasizes deployability across CLI/SDK/cloud and public benchmark visibility. Forge has stronger workflow formalism, but weaker productized operating model (tenancy, integration surface, operational UX).

### SWE-agent
SWE-agent's public positioning is benchmark-first (SWE-bench). Forge should borrow this discipline with repeatable benchmark suites and release gates that prevent regressions when prompts/models change.

### LangGraph
LangGraph's "long-running stateful graph" framing is very mature for orchestration ergonomics. Forge's Temporal design is strong, but state contracts and interrupt/resume semantics could be made more explicit and developer-facing.

### Aider/Cline-style workflows
These tools excel at developer ergonomics around diffs, iterative correction, and local feedback loops. Forge is stronger in backend orchestration, but could improve human-facing loop efficiency (review affordances, quick reruns, confidence annotations).

---

## Suggested Re-Architecture (If Starting Fresh)

## Layer 1: Deterministic control plane (new)
- Policy compiler + admission control
- Budget/SLA manager
- Permission broker for tools/providers

## Layer 2: Orchestration engine (keep Temporal)
- Universal workflow step remains
- Fan-out remains
- Retry engine becomes strategy-based (failure-class aware)

## Layer 3: Intelligence services (refactor)
- Planner, exploration, generation, reviewer as interchangeable services
- Strong versioning and A/B support per service

## Layer 4: Validation and assurance (expand)
- Deterministic + semantic + regression checks
- Confidence model and risk scoring
- Merge assurance for conflict-resolved artifacts

## Layer 5: Operations and governance (expand)
- Tenant controls, RBAC, audit trails
- Benchmark and release gating
- Drift monitoring for prompts/models/providers

---

## Concrete Next Steps (90-day plan)

1. Define and ship a minimal policy DSL and enforcement layer.
2. Add run-level budget caps and deadline semantics.
3. Implement failure classification and strategy-based retries.
4. Build end-to-end benchmark harness with pass/fail release gates.
5. Add risk/confidence scores to every step result.
6. Implement reviewer UX primitives in CLI (queueing, replay, selective re-run).

---

## Bottom Line

Forge's architecture is better than most early agent systems in one important way: it treats orchestration as a software system, not a prompt script.

But it still lacks core production muscles: policy governance, cost/SLA controls, robust evaluation gates, and operational human-loop tooling. Without those, autonomy features will scale failure faster than value.

If those gaps are closed, the design could become a genuinely strong foundation for reliable agentic execution.
