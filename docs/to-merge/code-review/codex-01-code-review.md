# CODEX Architecture Review: Forge

## Scope and method

I reviewed `docs/DESIGN.md` and `docs/ARCHITECTURE.md` as requested, then spot-checked implementation structure in `src/forge/` to see whether architecture claims look implementable in practice.

I also looked up similar open-source orchestration/agent frameworks on the web (GitHub metadata) to compare Forge’s design choices against common patterns in the ecosystem.

---

## Executive summary

Forge has a strong core idea: **document-completion-first orchestration with deterministic control in Temporal**. That is a credible and differentiated direction, especially for batch APIs and reproducibility.

That said, the current architecture is over-optimistic in several places and under-specified in a few risk-critical areas:

1. **Reliability and safety boundaries are not strict enough** for a system that can edit code, run tests, and execute repository commands.
2. **Planner and transition semantics are underspecified**, making failure handling and re-planning behavior ambiguous under real load.
3. **Git/worktree as universal storage abstraction is stretched too far** for non-code domains.
4. **Observability is deep but governance is shallow** (no strong audit/compliance model, no robust policy engine, no explicit blast-radius controls).
5. **Cost/latency controls are weakly defined** for recursive fan-out + exploration + retries.

If I were redesigning this, I would keep the universal workflow step and Temporal backbone, but add a stricter policy layer, an explicit state model, stronger task contracts, and tighter resource governance.

---

## What is good (and should be preserved)

- **Orchestrator owns control loop** (not the LLM): excellent for determinism, retries, and postmortem analysis.
- **Structured output schema (`files` + `edits`)**: practical and safer than pure free-form patch text.
- **Separation of activity boundaries**: aligns with Temporal best practices and supports independent timeouts/retries.
- **Built-in validation-first posture**: lint/format/tests before transition signals is the right default.
- **Fan-out/gather as first-class primitive**: good fit for independent sub-tasks.
- **Observability store + OTel**: this is substantially better than most agent prototypes.

---

## Critical architectural concerns

## 1) Safety model is implied, not enforced

The docs describe powerful context providers (`run_tests`, `git_diff`, etc.) and broad repository mutation, but there is no clearly specified **policy engine** for command allowlists/denylists, filesystem boundaries, network egress policy, secret redaction, or tenant isolation.

### Why this matters
- A planning or exploration error can trigger high-risk side effects.
- “Human-gated merges” is not enough; risk happens before merge (e.g., data exfiltration, destructive scripts, dependency tampering).

### What I would do differently
- Add a **policy layer as a mandatory pre-execution gate** for every provider/tool invocation.
- Introduce **capability-scoped execution tokens** (task can only call provider subset explicitly granted by plan/policy).
- Enforce **sandbox profiles** per task type (read-only, test-only, write-limited, network-off-by-default).

---

## 2) Transition vocabulary is too small for real operational states

Current transition states are elegant but too coarse (`success`, retryable, terminal, blocked, new tasks, sibling dependency). In practice, you need richer state to avoid accidental loops and silent partial failure.

### Missing states I would add
- `partial_success`
- `validation_inconclusive`
- `policy_violation`
- `budget_exhausted`
- `timeout_degraded`
- `requires_replan`
- `artifact_conflict_unresolved`

### Why this matters
Without these, many non-happy-path outcomes get collapsed into retry/terminal buckets, which reduces diagnosability and can waste expensive model calls.

---

## 3) Plan contract is underpowered for conflict prevention

The planner defines boundaries and dependencies, but there is no hard formalism for **write-set declarations**, **resource locks**, or **conflict prediction confidence**.

### What I dislike
- “What not to touch” as plain text is too weak.
- Conflict handling appears reactive (resolve after conflict), not proactive.

### What I would do differently
- Require each plan step to include:
  - declared write paths/globs,
  - expected read set,
  - required providers,
  - estimated token/runtime budget,
  - confidence score and fallback strategy.
- Add a **preflight conflict analyzer** that can reject unsafe parallelization before child workflows start.

---

## 4) Edit-application strategy may be too permissive

The architecture allows fuzzy matching fallback for edits. That improves success rate but increases risk of wrong-location edits when context is close but not identical.

### Concern
For code generation systems, a “successful fuzzy edit” can be more dangerous than a hard failure.

### Recommendation
- Make fuzzy matching opt-in by policy tier.
- Require post-edit structural checks (AST parse / semantic diff guardrails) before accepting fuzzy-applied edits.
- Emit explicit “precision level used” telemetry and gate low-precision writes behind stricter validation.

---

## 5) Git/worktree as universal datastore is philosophically elegant but practically leaky

This approach is strong for code changes, but less natural for research/analysis/content tasks where artifacts are not branch-friendly, are large, or need metadata indexing/queryability beyond files.

### What I would change
- Keep git/worktrees for code workflows.
- Add a **separate artifact store abstraction** (object store + metadata index) for non-code outputs and large intermediate artifacts.
- Model git as one execution backend, not the only general-purpose persistence layer.

---

## 6) Cost and latency governance need a first-class budget framework

The design combines planning, exploration rounds, retries, fan-out recursion, conflict resolution, and review. This can explode cost and wall-clock time.

### Missing features
- Global budget envelopes (token, dollars, wall-clock) at run level.
- Real-time budget burn-down and adaptive degradation policy.
- Automatic “quality mode fallback” (e.g., reduce exploration rounds when budget is tight).

### Recommendation
Implement **hierarchical budgets**:
- run budget,
- per-step budget,
- per-subtask budget,
- emergency reserve for final repair/review.

---

## 7) Evaluation strategy is not yet rigorous enough for autonomous operation claims

There is an eval module, but docs understate requirements for reliable autonomy.

### Missing capabilities
- Regression suite for planner quality over time.
- Golden-task replay with deterministic fixtures.
- Adversarial tests (prompt injection via repository files, deceptive imports, malformed diffs).
- Reliability SLOs (task success rate, false-success rate, mean retries, cost per successful task).

### What I would do differently
Treat eval as a **release gate**, not an auxiliary workflow.

---

## 8) Human escalation flow is sensible, but collaboration UX is underdesigned

“Pause and escalate” is right, but the operator experience seems minimally specified.

### Missing features
- Rich triage UI with:
  - causal timeline,
  - failed assumptions,
  - policy decisions,
  - recommended operator actions.
- Structured operator interventions (not just free-text corrections).
- Ability to checkpoint and branch “what-if” remediation paths.

---

## 9) Multi-provider strategy is too Anthropic-centric in implementation details

Docs list multiple providers, but architecture details focus heavily on Anthropic tool use and batch semantics.

### Risk
Provider lock-in can creep into prompt schema, parsing semantics, and retry assumptions.

### Recommendation
Define a **provider-neutral response contract and adapter test suite** that validates parity across providers (OpenAI, Anthropic, local models).

---

## 10) Missing product-level architecture concerns

The documents are strong on execution mechanics but weaker on product/platform concerns:

- Multi-tenancy model.
- RBAC/authorization boundaries.
- Secrets lifecycle and redaction policy.
- Data retention and deletion guarantees.
- Compliance/audit requirements.
- Disaster recovery strategy (Temporal + SQLite + git state alignment).

These are mandatory if Forge is intended for real organizational deployment.

---

## Comparison to similar projects (web scan)

I reviewed public GitHub metadata/descriptions for commonly used orchestration/agent systems:

- `langchain-ai/langgraph`
- `microsoft/autogen`
- `crewAIInc/crewAI`
- `OpenHands/OpenHands`
- `Significant-Gravitas/AutoGPT`
- also broader orchestration entries via GitHub search (e.g., Haystack, txtai, AGiXT).

## Key comparative takeaways

1. **Forge is stronger than many agent frameworks on deterministic orchestration** (Temporal + explicit activities).
2. **Forge is weaker than mature workflow systems on policy/governance hardening**.
3. **Forge’s doc-completion approach is cleaner than chat-loop agents for batchability**, but needs stronger guarantees around edit precision and budget control.
4. **Ecosystem trend**: successful projects increasingly emphasize memory, tool policy, eval harnesses, and operator UX—not just agent loop mechanics.

---

## Concrete redesign proposal (if starting from scratch)

1. **Core state machine v2**
   - Expand transition taxonomy.
   - Add explicit terminal reasons + machine-readable remediation hints.

2. **Policy engine first**
   - Every provider call passes through policy checks.
   - Capability-scoped permissions attached to each step/subtask.

3. **Contract-heavy planner**
   - Enforce structured read/write sets and budgets.
   - Reject plans that exceed risk thresholds before execution.

4. **Dual persistence model**
   - Git backend for code artifacts.
   - Artifact store backend for non-code outputs and telemetry attachments.

5. **Budget-aware runtime**
   - Dynamic throttling/fallback based on budget burn.
   - Hard stops for budget overrun with clear escalation state.

6. **Evaluation-as-gate**
   - Planner regression benchmark.
   - Golden replay corpus.
   - Security/adversarial scenarios.

7. **Operator control plane**
   - Review dashboard for halted runs.
   - One-click interventions with structured patch/decision schemas.

---

## Missing features checklist

- [ ] Policy-as-code for providers and execution environments.
- [ ] Strong secret scanning/redaction in prompts, traces, and store.
- [ ] First-class budgets and adaptive degradation.
- [ ] Provider-neutral abstraction compliance tests.
- [ ] Richer transition/state model.
- [ ] Planner quality benchmark with trend tracking.
- [ ] Adversarial evaluation suite.
- [ ] Multi-tenant isolation model.
- [ ] Operator UI/control plane.
- [ ] Disaster recovery playbook and data consistency checks.

---

## Final verdict

Forge’s architectural core is promising and more disciplined than most agent prototypes. But in its current form, it still reads like a strong **engineering experiment** rather than a production-hardened autonomous execution platform.

The biggest gap is not intelligence—it is **governance under uncertainty**: policy, budgets, state semantics, and operator controls. Fix those, and Forge becomes much more credible as a reliable system rather than an ambitious orchestration demo.
