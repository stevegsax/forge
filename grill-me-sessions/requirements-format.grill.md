# Grill Session: requirements-format

Started: 2026-05-23
Last updated: 2026-05-23
Status: complete
Domain: software / requirements-engineering for the Forge LLM build pipeline

## Summary

Stress-test of the most useful format for writing requirements UP FRONT, so an architect agent
(→ typed contract → code) and a test-design agent (→ oracle) both get complete information.
Prior: keep Gherkin + step files unless there's a compelling reason to switch. Real goal: a format
that GUIDES toward good specs and makes BAD specs hard to write. Builds directly on the
interaction-mode session (contracts = typed artifacts; oracle independence = test and code from
DIFFERENT sources).

CONCLUSION. Keep Gherkin for the ONE thing it's good at — behavioral examples — and stop
overloading it. A new PRE-CONDITION stage (the existing decomposition pipeline's step 7, now
replaced) runs two FIREWALLED reviewer conversations (architecture-flavored, test-flavored) that
share only a human-reviewed REQUIREMENT CORE, per component. The core is a structured SIDECAR
(source of truth, ID-reconciled with its `.feature`) with six mandatory sections. Invariants are
typed/quantified/pattern-tagged predicates delivered as ratified bundles (intent + predicate +
pattern + witnesses + counter-examples), realized INDEPENDENTLY as a runtime contract (architect)
and a Hypothesis property (test-design) — never generated from each other. Agents guess standard
assumptions; the human corrects exceptions through a single consolidated, exception-driven gate.
Throughput is governed by guess quality (correction rate), not component count.

## Homework Findings

- `docs/requirements/` has 18 well-formed `.feature` files, but they DOCUMENT shipped behavior
  (@phase-1..14) and there are NO step files yet (pytest-bdd is "future"). Weak evidence Gherkin
  works for FORWARD specs.
- The decomposition pipeline ALREADY LLM-generates acceptance criteria at step 7 (free-text
  strings). This was the latent duplication the session resolved.

## Branch Status

1. Dual-consumer split — RESOLVED
2/3. The core's FORMAT (invariant notation, structure, gap-list completeness) — RESOLVED
4. "Make bad specs hard to write" — RESOLVED in principle (guardrails decided; mechanical lint
   layer is implementation detail, carried forward)
5. Pipeline entry point — RESOLVED
6. Oracle-independence paradox — RESOLVED

## Decision Log

### RESOLVED: Branch 1 — firewalled pre-condition stage
- Two firewalled reviewer conversations (arch, test) share only a human-reviewed requirement CORE.
  Firewall persists into the build phase: architect sees {core + arch-enriched}, test-design sees
  {core + test-enriched}; neither sees the other's enriched spec or any transcript.
- Refinement = divergent (suggest/connect/push back); build = convergent (work the frozen spec,
  don't redesign). "No pollution" is delivered at the HANDOFF (distill to a clean artifact, discard
  the transcript), not by the firewall. Invariants MUST live in the core; the human integrates them.

### DECIDED: (A) Invariant notation = typed, quantified, pattern-tagged predicates
- `∀ <typed vars>: <precondition> ⟹ <postcondition over named schema fields>`. A declarative SPEC
  over the schema — not prose (avoids lossy induction), not shared executable code (avoids
  zero-independence). Architect realizes it as a runtime contract (icontract / Pydantic validator);
  test-design realizes it as a Hypothesis property — independently, neither derived from the other.
- Each invariant tagged with a PROPERTY PATTERN (invariant, round-trip/invertibility, idempotence,
  commutativity, structural induction, easy-verification, model/oracle-comparison). The tag is BOTH
  a near-mechanical Hypothesis compiler AND an omission detector (a schema entity lacking a property
  of its type's implied patterns = loud flag).
- EARS controlled syntax for non-invariant sentences (esp. unwanted-behavior = error taxonomy).

### DECIDED: Bundled invariant record + deterministic accept/reject precondition
- Record = {intent (prose) | predicate (∀) | pattern | witnesses (must satisfy) | counter-examples
  (must be rejected)}. Refinement agent PROPOSES; human RATIFIES against witnesses/counters (NOT by
  reading the ∀ — defeats automation-bias rubber-stamping).
- Entry precondition: predicate must accept all witnesses + reject all counter-examples
  (deterministic, lintable, AUTOMATIC). Doubles as a non-vacuity guard and seeds test-design's
  example scenarios. Rationale: the agent's formalization is the single point of correlated failure;
  examples are the robust human ratification surface; empty bundle slots are conspicuous.

### DECIDED: Hard rule — property test is NEVER generated from the contract
- No `icontract-hypothesis` / contract→test generation in the gate. Per Hillel Wayne, a
  contract-as-oracle makes the test verify code against the contract, both descending from the
  architect — the mutation gate would grade the architect's homework with the architect's own key.

### DECIDED: Core = structured sidecar, source of truth, ID-reconciled with .feature
- `.feature` keeps ONLY behavioral examples; a structured sidecar (YAML / typed-Markdown) linked by
  stable requirement ID carries everything else. Sidecar is source of truth. A linter FAILS the
  build if a `.feature` or sidecar is orphaned or the IDs don't reconcile. (Honors the Gherkin
  prior: stop overloading, don't drop.)

### DECIDED: All six core sections mandatory-present, explicit justified-N/A allowed
- (1) typed schema, (2) invariant bundles, (3) error taxonomy (EARS unwanted-behavior),
  (4) non-functional constraints (ordering/idempotency/perf), (5) non-goals, (6) glossary. Each
  must be PRESENT; any may be `N/A — none, because …`. Silence is never absence. Missing or
  unjustified-empty section → spec rejected before fan-out. Converts silent omission (the most
  common defect) into a visible, reviewable assertion.

### DECIDED: Branch 5 — replace decomposition step 7 with the ratified core
- The ratified core IS a superior acceptance-criteria artifact, so it REPLACES the old LLM-generated
  step 7. This kills the requirement-duplication that opened the session.
- Granularity: core is per-COMPONENT (PlanDAG leaf); method-level fan-out children INHERIT the
  component contract and get no refinement cycle of their own (this bounds human load).
- Sequencing: decomposition (steps 1–6) runs on the lightweight goal/clarify artifacts, NOT a
  ratified core; the rich core is produced AT step 7; the format linter becomes step 8 (the existing
  "deterministic checks" slot, now with teeth).
- Posture: agents GUESS standard assumptions; the human REFINES/CORRECTS. Throughput is governed by
  GUESS QUALITY (correction rate), not component count.
- Gate consolidation: the three potential human gates (step-7 ratification, step-10 plan approval,
  interaction-mode's PM gate) collapse into ONE exception-driven gate — agents guess everywhere,
  emit a single consolidated report flagging only what needs a human, human corrects exceptions.
- Independence/throughput reconciliation: the deterministic witness/counter/non-vacuity check runs
  on EVERY invariant AUTOMATICALLY (carries the bulk of correlation-safety). The human is pulled in
  only on agent-flagged low confidence or when the mechanical check can't adjudicate (e.g., the
  agent couldn't produce a counter-example — itself the red flag), plus an optional random spot-audit.

## Deferred / Carried-Forward Risks

### CARRIED: Correction-rate is unmeasured (the throughput linchpin)
- The whole "human non-binding" claim now rests on agents guessing well enough that the human rarely
  corrects. The correction rate is unmeasured — instrument it. If it's high, the consolidated gate
  becomes the binding constraint at ~10 features/day, ~2 human-hrs/day. (Mirrors interaction-mode's
  "experiment will tell" + "ceiling raised by spec quality.")

### CARRIED: Mechanical lint layer not fully specified
- Beyond ID reconciliation + mandatory sections + non-vaciousness: EARS template conformance,
  schema type-validation, and referential integrity (every invariant/error clause names a real
  schema entity). Implementation detail; doesn't change the design.

### CARRIED (residual, accepted): correlated semantic error survives
- (1) Both reviewers likely share a base model → shared priors. (2) A predicate that passes its own
  agent-written witnesses/counters can still be jointly wrong. Both are interaction-mode's accepted
  IRREDUCIBLE semantic omission; spot-audit is the only further mitigation. Not blocking.

## Parking Lot

(none)
