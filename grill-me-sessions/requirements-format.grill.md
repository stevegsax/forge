# Grill Session: requirements-format

Started: 2026-05-23
Last updated: 2026-05-23
Status: in-progress
Domain: software / requirements-engineering for the Forge LLM build pipeline

## Summary

Stress-test of the most useful format for writing requirements UP FRONT, so an architect agent
(→ typed contract → code) and a test-design agent (→ oracle) both get complete information.
User's prior: stick with Gherkin + step files unless there's a compelling reason to switch. Real
goal: a format that GUIDES toward good specs and makes BAD specs hard to write. Depends on the
interaction-mode session: contracts are machine-readable typed artifacts (Gherkin alone can't be
the contract), and oracle independence (test and code derive from DIFFERENT sources).

DESIGN SO FAR: A new PRE-CONDITION stage runs two FIREWALLED reviewer conversations (architecture-
flavored, test-flavored) that share only a human-reviewed REQUIREMENT CORE. Refinement = DIVERGENT
(push back, suggest). Build = CONVERGENT (work the frozen spec). Build agents get only the
distilled spec, never the transcript. Human merges invariants. Gherkin demoted to ONE INPUT
(behavioral examples), not the full spec. Invariant notation resolved in detail (see below).

## Homework Findings (2026-05-23)

- `docs/requirements/` has 18 `.feature` files w/ tag taxonomy. Declarative, well-formed, but
  documentation of ALREADY-SHIPPED behavior + FUTURE pytest-bdd; no step files exist yet. Weak
  evidence Gherkin works for FORWARD specs.
- Decomposition pipeline ALREADY generates acceptance criteria (step 7, LLM) as free-text strings.

## Branches

1. Dual-consumer split (SPINE) — RESOLVED
2/3. The core's FORMAT — RESOLVED (invariant layer, structure, gap-list completeness)
4. "Make bad specs hard to write" — guardrails/linter (largely covered incrementally; see note)
5. Pipeline entry point — OPEN (genuine collisions: step-7 overlap, core granularity, 2 human gates)
6. Oracle-independence paradox — RESOLVED (no contract-derived tests) + residual logged

## Decision Log

### RESOLVED: Branch 1 — firewalled pre-condition stage (2026-05-23)
- New pre-condition stage before build architect + test-design agents.
- Two firewalled reviewer conversations (arch, test) share only a human-reviewed requirement CORE.
  Firewall persists into build: architect sees {core + arch-enriched}, test-design sees {core +
  test-enriched}; neither sees the other's enriched spec or any transcript.
- Refinement = divergent; build = convergent. "No pollution" delivered at HANDOFF (distill +
  discard transcript), not by the firewall. Invariants MUST live in the core; human integrates.

### DEFERRED: Refinement-conversation autonomy vs human throughput (2026-05-23)
- Open: human DRIVES each conversation vs only adjudicates/promotes? Risk: 2 firewalled
  conversations/feature + reconciliation could make the human the binding constraint at ~10
  features/day & ~2 human-hrs/day, denting interaction-mode's "human gate non-binding." Bounded if
  conversations run mostly autonomously + human only ratifies. Needs promotion rule + autonomy target.

### DECIDED: (A) Invariant notation = typed, quantified, pattern-tagged predicates (2026-05-23)
- Notation: `∀ <typed vars>: <precondition> ⟹ <postcondition over named schema fields>`. A
  declarative SPEC over the schema, not prose (avoids lossy induction) and not shared executable
  code (avoids zero-independence). Architect realizes it as a runtime contract (icontract /
  Pydantic validator); test-design realizes it as a Hypothesis property — INDEPENDENTLY, neither
  derived from the other.
- Each invariant tagged with a PROPERTY PATTERN from the canonical taxonomy (invariant, round-trip/
  invertibility, idempotence, commutativity, structural induction, easy-verification, model/oracle-
  comparison). The tag is BOTH a near-mechanical Hypothesis compiler AND an omission detector
  (a schema entity lacking a property of its type's implied patterns = loud flag).
- EARS controlled syntax for NON-invariant requirement sentences (esp. unwanted-behavior template
  = error taxonomy). Gherkin keeps only the behavioral-example job.

### DECIDED: Bundled invariant record + deterministic accept/reject precondition (2026-05-23)
- Record = {intent (prose) | predicate (∀) | pattern | witnesses (must satisfy) | counter-examples
  (must be rejected)}. Refinement agent PROPOSES the bundle; human RATIFIES against the
  witnesses/counter-examples (NOT by reading the ∀ — defeats automation-bias rubber-stamping).
- Precondition for entering the core: predicate must accept all witnesses + reject all
  counter-examples (deterministic, lintable). Doubles as NON-VACUITY guard (trivially-true
  predicate fails its counters) and seeds the test-design agent's example scenarios.
- Rationale: the agent's formalization is the single point of correlated failure; examples are the
  robust human ratification surface; empty bundle slots are conspicuous (guides toward good specs).

### DECIDED: Hard rule — property test is NEVER generated from the contract (2026-05-23)
- No `icontract-hypothesis` (or any contract→test generation) in the gate. Per Hillel Wayne, a
  contract-as-oracle makes the test verify code against the contract, both descending from the
  architect — the mutation gate would grade the architect's homework with the architect's key.
  Independence requires the property test descend from the requirement invariant, via a different agent.

### DECIDED: Core = structured sidecar, source of truth, ID-reconciled with .feature (2026-05-23)
- `.feature` keeps ONLY behavioral examples; a structured sidecar (YAML/typed-Markdown) linked by
  stable requirement ID carries schema, EARS statements, invariant bundles, non-functionals,
  non-goals, glossary. Sidecar is source of truth. Linter FAILS the build if a `.feature` or
  sidecar is orphaned or IDs don't reconcile. (Honors the Gherkin prior: stop overloading, don't drop.)

### DECIDED: All six core sections mandatory-present, explicit justified-N/A allowed (2026-05-23)
- Sections: (1) typed schema, (2) invariant bundles, (3) error taxonomy (EARS unwanted-behavior),
  (4) non-functional constraints, (5) non-goals, (6) glossary. Each must be PRESENT; any may be
  `N/A — none, because …`. Silence is never absence. Missing or unjustified-empty section → spec
  rejected before fan-out. Converts silent omission (the most common defect) into a visible assertion.

### LOGGED (residual, not blocking): same-model independence is partly cosmetic
- Both reviewers likely share a base model → shared priors = interaction-mode's accepted
  "domain/semantic omissions are IRREDUCIBLE." Carry forward.

## Open Threads

### Branch 2/3 remainder: overall core STRUCTURE + gap-list completeness (ACTIVE)
- The FORK (overload Gherkin via tags/docstrings vs. `.feature` for examples + structured SIDECAR
  linked by stable requirement ID) is now ripe — the 5-field invariant bundle obviously can't live
  in Gherkin. Need the decision + the overall file/record layout.
- Gap list items: (1) typed data shapes/schema, (2) invariants [DONE], (3) error taxonomy [EARS],
  (4) non-functional constraints (ordering/idempotency/perf), (5) non-goals/out-of-scope,
  (6) domain glossary. OPEN: which are MANDATORY (spec rejected if absent) vs optional.

### Branch 4: guardrails/linter (PENDING)
### Branch 5: pipeline entry point vs existing decomposition + PM gate (PENDING)

## Parking Lot

(none yet)
