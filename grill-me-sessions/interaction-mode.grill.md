# Grill Session: interaction-mode

Started: 2026-05-23
Last updated: 2026-05-23
Status: in-progress
Domain: software development (LLM-systems architecture within it)

## Summary

Stress-testing whether Forge should keep single-shot / document-completion request
formatting (batch) vs adopt "conversation mode." Resolved early: at the wire level there is
no such choice — batch items are identical Messages API bodies; "single-shot vs conversation"
reduces to "turns + tool round-trips." The real plan under examination is a decompose-and-
fan-out build pipeline (domain → components(∥) → methods(∥) → integrate → accept), enabled
by FCIS, with model routing (cheap writes, expensive designs/reviews) and Temporal as the
loop harness. Conversation mode is now judged DEAD for this design. Active focus: the
freeze/backtracking risk in the fan-out and the cost model.

## Decision Log

### RESOLVED: Does batch require a different message structure than sync?
- No. Batch item = `{custom_id, params}` with `params` a standard `messages.create` body.
  Single-shot is a design choice, not a batch constraint. (2026-05-23)

### RESOLVED (tentative): Is conversation/multi-turn mode needed for this design?
- No. Every node in the build breakdown is a single-shot document completion. Even the
  fix-loops (c.4.3/c.6/d) are "rebuild prompt with prior attempt + failure as DATA, resubmit"
  — Forge's error-aware-retry pattern, not append-turn conversation. True in-context
  multi-turn only beats this when verbatim prior assistant/thinking turns must be preserved
  (unpredictable agentic loops, interleaved thinking, prefill) — none present here.
- Answers user Q1 (single-shot not obsolete; it IS the architecture) and Q2 (only verbatim
  turn preservation is unique to multi-turn; design never needs it). (2026-05-23)

### RESOLVED: #3 test quality + #2 contract reviewer + #4 escalation ladder
- Tests get highest thinking + strictest review (oracle for all loops). Reviewer agent gates
  contracts vs requirements. Retry-count → escalate-to-smarter-model → halt. All sound, with
  residues tracked below. (2026-05-23)

### DECIDED: Contract richness — rich, invariant-bearing contracts; re-run safe superset
- **Decision**: Contracts carry invariants (not just data types). Invalidation re-runs the
  conservative type-level superset. User accepts higher API + wall-clock cost; will tune if real.
- **Rationale**: Semantic bugs are the expensive ones; trustworthy semantic-change detection
  is worth the upfront design weight and the over-invalidation waste.
- **Date**: 2026-05-23

### RESOLVED: Local models (Gemma/Qwen/Granite) driver
- Driver = privacy + control (not cost). User accepts no batch discount locally. Nothing
  PREVENTS local use; caveats are mechanism-level (mandatory chat template, Gemma has no
  system role, grammar/guided-decoding instead of tool_choice, model-specific vision). (2026-05-23)

## Open Threads

### RESOLVED: ORACLE INDEPENDENCE (moved from open — see Decision Log additions below)
- (a) DECIDED: mutation survival (cosmic-ray) + property-based tests (Hypothesis) are a HARD
  "done" gate — the mechanical defense for structural omissions, independent of who authored
  the test.
- (b) DECIDED: domain/semantic omissions accepted as IRREDUCIBLE. CAVEAT recorded: user's
  reason ("addressed before design") relocates the residual to the requirements-input boundary
  — must be an EXPLICIT precondition ("Forge assumes domain-complete requirements; it does not
  discover missing business rules"). Failure mode = polished garbage: incomplete requirements
  built flawlessly with all gates green. Negotiation HARDENS requirements (ambiguity); it does
  NOT manufacture missing domain knowledge.

#### (history) ORACLE INDEPENDENCE
Rich contracts risk turning the test into a transcription of the contract, collapsing the two
independent error-detectors (contract understanding, test understanding) into one. User's
proposed precondition: a multi-party requirements negotiation (product owner + pedantic test
author + pedantic contract/design author) until requirements are specific enough that tests
and design can be authored independently.
- Refinement: this negotiation is conversation-SHAPED but flattenable to Temporal-orchestrated
  rounds (transcript as data) — does NOT reopen API-level conversation mode. Build nodes stay
  single-shot.
- Roles confirmed: product owner + pedantic test author + pedantic contract/design author.
  Negotiation workflow (human-in-loop, possibly synchronous, new UI) TABLED as separate work;
  assumed to satisfy the independence requirement.
- User primarily defends AMBIGUITY; expects stated assumptions to also surface omissions.
- Omission taxonomy established (the sharpening of "spot omissions together"):
  - AMBIGUITY → conversation solves outright. ✓
  - STRUCTURAL omissions (empty/null/boundary/overflow/order) → pedantic test author catches IF
    truly adversarial; mechanical backstop = mutation survival (cosmic-ray) + property-based
    (Hypothesis), which don't depend on anyone remembering.
  - DOMAIN/SEMANTIC omissions (unstated business rule) → IRREDUCIBLE. No automated oracle can
    catch a rule never expressed; conversation fails under common-mode blind spots. Carried to
    acceptance/production; mitigated only by product-owner diligence.
- TWO DECISIONS PENDING to close thread:
  (a) Mutation survival + property-based tests as a HARD "done" gate — yes/no? (only mechanical
      defense for structural omissions)
  (b) Conscious ACCEPTANCE of domain-omission as irreducible residual (not assumed away)?

### FREEZE / BACKTRACKING (core resolved via contract-richness DECIDED; residue below)
User's model: nested loops; on failure, ask parent to retry WITH a post-mortem (not from
scratch); optimistic that a contract change affects only a SUBSET of methods; experimentation
will validate. Switch batch→sync when few repairs; prefer one expensive redesign over many
cheap repairs when many. Aggressive (nested) caching for cost control. Invited alternatives.

Challenges raised (awaiting user):
1. BLAST-RADIUS INVERSION: FCIS decouples BEHAVIOR but methods stay coupled through shared
   DATA SHAPES. Private-contract change = small blast radius; SHARED-type change = large by
   construction — and shared/foundational types are the most likely to be wrong early and the
   most expensive to change. "Subset" optimism holds for trivial changes, fails for the ones
   that matter.
   - SUGGESTED: (a) two-tier freeze — lock shared domain types first (expensive model + test-
     grade review), then per-method contracts, THEN fan out. (b) Compute the invalidation set
     from the type-dependency graph instead of hoping it's small — turns optimism into a bound.
2. SIBLING INVALIDATION is horizontal; the loop model is vertical (child→parent). A late
   contract change makes already-PASSED sibling methods stale. Same type-graph answers it.
3. CHEAP FEASIBILITY PROBE before the expensive fan-out: each method sketches its impl and
   flags "can't satisfy this contract / missing input" — early warning at the freeze boundary,
   also catches compose-failures the local contract reviewer (#2) misses.
4. GLOBAL ESCALATION BUDGET: per-node retry caps (#4) don't bound the TREE. Nested loops can
   cascade (method fail → component redesign → re-fan-out → fail). Need per-component
   depth/spend ceiling that halts the subtree to a human.

### COST MODEL (active)
- CACHING CONTRADICTION: user excluded caching in opening msg, now relies on it. And caching
  + batch is the weakest combo — default TTL 5 min (since Mar 2026), batch scheduling
  non-deterministic up to 24h, so cache written in a batch likely expires before the follow-up
  runs. The two cost levers (batch discount, caching) partly fight: the discount comes from
  relaxing time; relaxing time blows the cache. Caching pays best in SYNCHRONOUS tight loops.
- BATCH/SYNC SWITCH is a good adaptive instinct, but needs (a) an estimator to decide BEFORE
  repairs are done (failing-test count? reviewer severity? diff size?) and (b) it dovetails
  with where caching actually works (sync, sub-5-min loops).
- COST INVERSION (carried from earlier): #3 puts expensive model on test authoring (high-
  cardinality node); #4 escalations scale with node count. "Cheap does the bulk" still
  unmeasured. Need a sketched token mix.

## Parking Lot
- Model routing by task type: experiment-driven; deferred to measurement.
- Cross-component (global) contract composition review vs local review.
- Property-based tests (Hypothesis) as a stronger oracle for pure functions than examples.
