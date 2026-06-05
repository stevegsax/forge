# Autonomous-Agent Requirements Standard

## Purpose

This standard defines the minimum specification package required before handing work to an
autonomous implementation agent. The goal is to make missing information visible before execution,
push correctness checks as far left as possible, and maximize the amount of behavior that can be
encoded statically rather than discovered at runtime.

This standard assumes:

- the implementation agent has read-write access to the code repository
- the implementation agent can read, but not modify, tests in a separate repository
- the implementation agent can run commands and modify files locally
- network access is restricted to an approved allowlist
- a human performs a requirements review before the implementation handoff

## Core Principles

1. Behavioral examples are necessary but not sufficient. Gherkin stays, but only for examples.
2. The requirement core is the source of truth. Feature files are paired behavioral views.
3. Every important rule must be expressed at the strongest practical enforcement layer.
4. Prefer illegal states to be unrepresentable over merely detectable.
5. Runtime checks are a fallback after representation, construction, and transition constraints.
6. Business logic belongs in a functional core; stateful orchestration belongs in an imperative
   shell.
7. Tests and implementation must descend independently from the reviewed requirements package.
8. Silence is never acceptable. Every mandatory section must be present, even when the answer is
   `N/A — none, because ...`.

## Required Artifact Set

Each capability, component, or PlanDAG leaf handed to the implementation agent must have exactly
these artifacts:

1. A behavioral example file: `<requirement-id>.feature`
2. A structured requirement core: `<requirement-id>.core.md`
3. A completed review record based on [REVIEW_CHECKLIST.md](REVIEW_CHECKLIST.md)

The `.core.md` file is the source of truth. The `.feature` file contains only examples, scenarios,
and concrete user-visible behavior. A requirement package is invalid if either file is missing or
if the requirement IDs do not match.

## Naming And Metadata

Each requirement core must begin with a metadata block using this exact field set:

```yaml
requirement_id: planning
title: Planning and decomposition
status: draft | in-review | approved | superseded
owner: team-or-person
reviewers:
  - person-a
code_repo: forge
test_repo: forge-tests
paired_feature: docs/requirements/planning.feature
last_updated: YYYY-MM-DD
language_target: python
architecture_pattern: functional-core-imperative-shell
```

Rules:

- `requirement_id` is stable and file-name-safe.
- `status: approved` is required before agent handoff.
- `paired_feature` must point to the matching `.feature` file.
- `language_target` must be `python` unless an exception is approved.
- `architecture_pattern` must be `functional-core-imperative-shell` unless an exception is
  approved in the review record.

## Mandatory Sections In `<requirement-id>.core.md`

Sections must appear in this order.

### 1. Scope And Outcome

State:

- the business goal
- the system boundary
- the inputs the component owns
- the outputs the component owns
- explicit out-of-scope items
- external dependencies the shell may touch

This section answers: "What is this component responsible for, and what is it not responsible for?"

### 2. Domain Algebra

Define the domain using typed, reviewable structures:

- entities and value objects
- fields and types
- legal variants and enums
- state machines or lifecycle states
- units, bounds, and cardinality constraints
- identity and referential relationships

Authoring rules:

- prefer sum types, enums, and distinct value objects over strings and booleans
- model mutually exclusive states as separate variants, not flag combinations
- call out fields that are derived rather than stored
- describe which states must be impossible to represent

### 3. Behavioral Examples

Link the paired `.feature` file and summarize the scenario groups. The `.core.md` file must not
duplicate the full Gherkin text; it should explain what the examples cover and where intentional
gaps remain.

The `.feature` file should focus on:

- happy paths
- edge cases
- user-visible failures
- ordering-sensitive flows
- concurrency-sensitive flows when behavior is externally observable

### 4. Contract Map

Every meaningful rule must appear in a contract table. Each row is one rule.

Required columns:

| Field | Meaning |
|------|---------|
| `rule_id` | Stable identifier such as `PLAN-RULE-001` |
| `intent` | Plain-English explanation of the rule |
| `formal_statement` | Structured predicate, transition, or assume/guarantee statement |
| `kind` | `representation`, `construction`, `transition`, `operation`, `global`, or `capability` |
| `strongest_layer` | Where the rule should be enforced first |
| `static_encoding_candidate` | Proposed type, variant, smart constructor, protocol, or state machine |
| `runtime_check` | Residual runtime check, or `none` |
| `witnesses` | Concrete examples that must satisfy the rule |
| `counterexamples` | Concrete examples that must be rejected |
| `failure_mode` | Exact rejection behavior if violated |

Authoring rules:

- write the strongest practical layer, not just the easiest one
- if `kind = capability`, specify the exact effect interface the core may use
- if a rule cannot be made static, explain why in `static_encoding_candidate`
- witnesses and counterexamples are mandatory for every rule
- do not generate tests from runtime contracts; both derive independently from this table

### 5. Functional Core / Imperative Shell Split

State the decomposition explicitly.

Required subsections:

- `Pure core responsibilities`
- `Imperative shell responsibilities`
- `Ports and capabilities`
- `Determinism and injected dependencies`

Authoring rules:

- pure core contains calculations, invariants, decisions, and state transitions
- shell contains persistence, I/O, retries, logging, time, randomness, and orchestration
- every side effect must be behind a named port or capability
- if a business rule remains in the shell, justify why it cannot live in the core

### 6. Error Taxonomy

List the failure modes the system must distinguish. Include:

- validation failures
- invariant violations
- unsupported transitions
- external dependency failures
- timeout and retry cases
- operator-visible and user-visible error surfaces

Use controlled, repeatable language. For conditional behaviors, prefer EARS-style phrasing:

- `When <trigger>, the system shall <response>.`
- `If <precondition> and <event>, the system shall <response>.`
- `Where <constraint is violated>, the system shall reject with <error>.`

### 7. Non-Functional Constraints

This section must cover, where relevant:

- idempotency
- ordering guarantees
- concurrency assumptions
- performance and latency bounds
- durability expectations
- observability and auditability
- security boundaries
- resource ceilings

If a topic does not apply, mark it `N/A — none, because ...`.

### 8. External Interfaces And Assume/Guarantee Contracts

For every API, CLI surface, queue message, file format, or persistence boundary, specify:

- `Requires`
- `Guarantees`
- `Fails with`
- `Preserves`

This section exists to keep interface contracts reviewable without digging through prose.

### 9. Non-Goals

List the behaviors the implementation agent must not infer, optimize for, or silently add.

Examples:

- unsupported workflows
- deferred scalability work
- UI or API affordances that are intentionally omitted
- "nice to have" ideas not part of the current contract

### 10. Glossary And Assumptions

Define domain terms and assumptions that a reviewer should verify before approval.

Allowed assumption categories:

- stable business terminology
- infrastructure assumptions
- sequencing assumptions
- data-quality assumptions

If an assumption is uncertain, it must be promoted into an explicit review question.

## Strongest-Layer Rule

For every rule in the contract map, choose the strongest practical layer in this order:

1. `Representation`
2. `Construction`
3. `Transition`
4. `Operation`
5. `Global`
6. `Capability`

Use the first layer that can reasonably enforce the rule. Do not skip directly to runtime
validation when the rule can be captured by types, state variants, constructors, or explicit
capabilities.

Examples:

- "An order is either draft or submitted" belongs in `representation`
- "Only parsed UUIDs may enter the core" belongs in `construction`
- "Submitted orders cannot return to draft" belongs in `transition`
- "This function requires normalized inputs" belongs in `operation`
- "Total debits equal total credits" may remain `global`
- "Only the shell may touch the filesystem" belongs in `capability`

## Python Realization Policy

This is a requirements standard, but the target implementation language is Python. Requirement
authors must therefore write rules that map cleanly to a constrained Python subset.

Preferred implementation targets:

- immutable domain values using `@dataclass(frozen=True, slots=True, kw_only=True)`
- distinct domain identities using `NewType`
- explicit variants using `Enum`, `Literal`, or tagged unions
- effect boundaries expressed as `Protocol`
- boundary parsing and coercion isolated to Pydantic models or equivalent validators
- exhaustive branching via `match` plus `assert_never` where practical

Anti-patterns that must be called out during review:

- business objects modeled as `dict[str, Any]`
- free-form status strings where a finite variant set exists
- logic hidden in ORM models, CLI entrypoints, HTTP handlers, or adapters
- unchecked `Any`, broad `cast`, or unreviewed `type: ignore`
- exceptions as the primary domain-control-flow mechanism inside the pure core

## Review And Handoff Gate

A requirement package is ready for implementation handoff only when:

1. The `.feature` and `.core.md` files both exist and reconcile by ID.
2. Every mandatory section is present.
3. Every contract rule has witnesses and counterexamples.
4. Every important business rule is assigned to its strongest practical enforcement layer.
5. The functional-core / imperative-shell split is explicit.
6. External effects are enumerated as capabilities or ports.
7. All open questions are resolved or elevated into explicit assumptions approved by the reviewer.
8. The review record is completed and the status is set to `approved`.

## Separation Of Duties

To preserve oracle independence:

- requirements authors maintain the reviewed specification package
- test authors or a test-generation agent derive tests independently from the approved package
- the implementation agent may read the test repository but may not modify it
- the implementation agent may not change reviewed requirements as part of delivery

The implementation agent is not allowed to weaken the contract by editing requirements, removing
examples, or mutating tests.

## Definition Of Done For The Standard

This standard is being followed only if new or revised requirement packages include:

- a paired `.feature` file
- a paired `.core.md` file
- a completed review checklist
- explicit contract mapping
- an explicit functional core / imperative shell split

Without those artifacts, a requirement may still be useful for discussion, but it is not ready for
autonomous implementation.
