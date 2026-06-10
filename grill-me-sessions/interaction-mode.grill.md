# Grill Session: interaction-mode

Started: 2026-05-23
Last updated: 2026-05-23
Status: complete
Domain: software development (LLM-systems architecture within it)

## Summary

Research-only stress-test of whether Forge should format LLM requests as single-shot /
document-completion (batch) vs "conversation mode." Core finding: at the wire level there is
no such choice — a batch item is an identical Messages API body; "single-shot vs conversation"
reduces to "how many turns + do you do tool round-trips." For Forge's decompose-and-fan-out
build design, single-shot document completion IS the architecture, and the THROUGHPUT goal the
user confirmed VINDICATES the batch + single-shot + fan-out foundation. The session then
hardened the operational design: rich invariant-bearing contracts with type-graph invalidation,
an independent test oracle backed by a mutation-survival gate, and a front-of-pipeline
requirements step where a PM agent triages, agents emit a consolidated report, and a human
gives selective (exception-only) sign-off before the expensive fan-out. At ~10 features/day
with ~2 human-hours/day the human gate is comfortably non-binding; the throughput ceiling sits
around ~25-30 features/day and is raised by spec quality (lower flag rate).

### Answers to the three original questions

1. **Is single-shot advantageous, or obsolete thinking?** Not obsolete — for decompose-and-
   fan-out it IS the architecture; conversation mode would re-couple nodes you worked to keep
   independent. The genuinely obsolete belief was "batch needs a different message structure"
   — it doesn't.
2. **Anything native prompts can do that conversation can't?** The only thing in-context
   multi-turn uniquely offers is verbatim preservation of the model's own prior assistant/
   thinking turns. The design never needs it. Conversation mode is dead for build nodes; the
   requirements negotiation is conversation-SHAPED but flattenable to orchestrated rounds
   (transcript as data).
3. **Local LLM (Gemma/Qwen/Granite) single-shot advantage?** No single-shot advantage per se.
   Nothing prevents local use; driver is privacy/control; you forgo the batch discount.
   Mechanism caveats: mandatory chat template, Gemma has no system role, grammar/guided
   decoding instead of tool_choice, model-specific vision.

## Decision Log

### RESOLVED: Batch requires no different message structure than sync

Batch item = `{custom_id, params}`, params a standard `messages.create` body. Single-shot is a
design choice, not a batch constraint. (2026-05-23)

### RESOLVED: Conversation/multi-turn mode is not needed for this design

Every build node is a single-shot document completion; fix-loops rebuild the prompt with prior
attempt + failure as DATA (error-aware-retry), not append-turn conversation. (2026-05-23)

### RESOLVED: Local-model driver = privacy + control (not cost); no hard blocker. (2026-05-23)

### DECIDED: Rich invariant-bearing contracts; re-run conservative (type-level) superset

Semantic bugs are the expensive ones; trustworthy semantic-change detection is worth the
upfront weight and over-invalidation waste. PRECONDITIONS: contracts must be machine-readable
typed artifacts (so invalidation is computed deterministically, not by an LLM); semantics
encoded testably. (2026-05-23)

### DECIDED: Oracle independence

(a) Mutation survival (cosmic-ray) + property-based tests (Hypothesis) are a HARD "done" gate
— the mechanical defense for structural omissions, independent of who authored the test.
(b) Domain/semantic omissions accepted as IRREDUCIBLE (carried to acceptance/production).
Test and code must derive from DIFFERENT sources (requirements vs contract) to stay
uncorrelated. (2026-05-23)

### DECIDED: Optimize throughput-across-features (not latency-per-feature)

Vindicates batch + single-shot + fan-out; Amdahl does not bind. Governing law = Theory of
Constraints (throughput = slowest shared stage). (2026-05-23)

### DECIDED: Front-of-pipeline requirements gate

PM agent triages each point: (1) resolvable from spec → auto; (2) not-in-spec/low-risk →
documented assumption, proceed optimistically, async human review, veto → type-graph-bounded
rework; (3) genuine domain Q / high-risk / deadlock → human. Agents emit a CONSOLIDATED,
deduped report (assumptions / decisions / where-they-need-help / lingering questions). Human
gives SELECTIVE sign-off (blocks only on bucket-3) but ALL features are reported (audit trail).
Gate sits AFTER design, BEFORE fan-out — doubles as the fail-fast / WIP-control mechanism.
Arithmetic at ~10 features/day, ~2 human-hours/day: human is non-binding; ceiling ~25-30/day,
raised by spec quality. (2026-05-23)

## Deferred / Open Risks (carry forward)

### DEFERRED: Cheap feasibility probe / contract dry-run before fan-out

- **Risk if ignored**: Promoted from "nice" to effectively REQUIRED by the throughput choice
  (high WIP means a doomed feature must fail fast before the queue burns money). The
  human-review-before-fan-out gate partially covers this, but a cheap automated "can each
  method satisfy its contract?" probe catches contract gaps the human report may miss.

### DEFERRED: Global escalation budget / depth ceiling

- **Open**: Per-node retry caps exist (#4) but nothing bounds the TREE. Nested loops can
  cascade (method fail → contract redesign → re-fan-out). Needs a per-component spend/depth
  ceiling that halts a subtree to a human. **Risk**: a pathological component re-runs subtrees
  and burns budget without surfacing.

### DEFERRED: Cross-component (global) contract composition review

- **Open**: The contract reviewer checks one component's contracts vs requirements (LOCAL). It
  does not verify all contracts COMPOSE (A emits a shape B won't accept). **Risk**: composition
  mismatch surfaces only at integration (d), after maximum parallel spend.

### DEFERRED: Cost model / token mix unmeasured

- **Open**: "Cheap model does the bulk" is still faith. Rich contracts + #3 (expensive model on
  high-cardinality test authoring) + #4 escalation tail may pull expensive compute toward the
  wide parts of the tree. **Risk**: cost savings smaller than hoped. User: experiment will tell.

### DEFERRED: Caching is the weakest lever in batch

- **Open**: Default cache TTL is 5 min (since Mar 2026); batch scheduling is non-deterministic
  up to 24h, so a prefix cached for a batch likely expires before the follow-up runs. Caching
  pays in SYNCHRONOUS tight loops — aligns with the batch↔sync switching instinct. **Risk**:
  counting on caching to control batch cost will disappoint.

### DEFERRED: Batch↔sync switching estimator

- **Open**: Good adaptive instinct (few repairs → sync; many → one expensive redesign), but the
  decision needs a pre-repair signal (failing-test count? reviewer severity? diff size?).

### DEFERRED: Imperative shell sizing

- **Open**: The shell is the un-parallelizable, holistic, critical-path node per component
  (designed after c.3, built by a separate workflow). Its difficulty scales with component
  complexity and gets your most expensive model. **Risk**: under-sized in the plan; a large
  shell erodes a component's parallelism benefit.

### DEFERRED: API rate-limit sizing of fan-out width

- **Open**: Non-binding at 10 features/day, but fan-out width × WIP eventually collides with
  org tokens/min + requests/min. Size intended width to actual tier before scaling throughput.

### TABLED: Requirements-negotiation workflow (human-in-loop UI)

- Separate build with its own UI; assumed to satisfy the oracle-independence requirement.
