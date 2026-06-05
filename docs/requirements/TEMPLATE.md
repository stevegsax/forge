# Requirement Core Template

Use this file as the starting point for `<requirement-id>.core.md`.

```md
---
requirement_id: <requirement-id>
title: <human-readable title>
status: draft
owner: <team-or-person>
reviewers:
  - <reviewer>
code_repo: <repo-name>
test_repo: <read-only-test-repo-name>
paired_feature: docs/requirements/<requirement-id>.feature
last_updated: YYYY-MM-DD
language_target: python
architecture_pattern: functional-core-imperative-shell
---

# <Title>

## 1. Scope And Outcome

### Goal

<What business outcome does this component produce?>

### In Scope

- <owned input or responsibility>
- <owned output or responsibility>

### Out Of Scope

- <explicit non-goal>

### External Dependencies

- <dependency the shell may touch>

## 2. Domain Algebra

### Entities And Value Objects

| Name | Kind | Definition | Illegal States Made Unrepresentable |
|------|------|------------|-------------------------------------|
| `<type>` | `entity \| value-object \| enum \| state` | `<shape or meaning>` | `<what cannot exist>` |

### Derived Fields

| Field | Derived From | Why It Is Not Stored Independently |
|------|---------------|------------------------------------|
| `<field>` | `<formula or source>` | `<reason>` |

### State Machine

| From | Event | To | Notes |
|------|-------|----|-------|
| `<state>` | `<event>` | `<state>` | `<constraints>` |

## 3. Behavioral Examples

- Paired feature file: `docs/requirements/<requirement-id>.feature`
- Scenario groups covered:
  - <group>
  - <group>
- Intentional gaps:
  - <gap or `none`>

## 4. Contract Map

| rule_id | intent | formal_statement | kind | strongest_layer | static_encoding_candidate | runtime_check | witnesses | counterexamples | failure_mode |
|---------|--------|------------------|------|-----------------|---------------------------|---------------|-----------|-----------------|--------------|
| `<id>` | `<plain English>` | `<predicate or transition>` | `representation` | `representation` | `<type or variant>` | `none` | `<must pass>` | `<must fail>` | `<rejection behavior>` |

Guidance:

- `formal_statement` should be structured enough that two different implementers would encode the
  same rule.
- `witnesses` and `counterexamples` should use domain language, not vague prose.
- `runtime_check` should be `none` unless a residual dynamic check is actually required.

## 5. Functional Core / Imperative Shell Split

### Pure Core Responsibilities

- <pure decision>
- <state transition>

### Imperative Shell Responsibilities

- <load or persist state>
- <perform external I/O>

### Ports And Capabilities

| Name | Kind | Used By | Purpose | Allowed Operations |
|------|------|---------|---------|--------------------|
| `<port>` | `protocol \| adapter \| capability` | `<shell or core caller>` | `<why it exists>` | `<operations>` |

### Determinism And Injected Dependencies

| Dependency | Why It Must Be Injected | Injection Shape |
|------------|--------------------------|-----------------|
| `<clock>` | `<determinism reason>` | `<callable or protocol>` |

## 6. Error Taxonomy

| Error ID | Trigger | Detection Layer | User/Operator Surface | Retryable | Notes |
|----------|---------|-----------------|-----------------------|-----------|-------|
| `<ERR-001>` | `<condition>` | `<layer>` | `<message or channel>` | `yes/no` | `<notes>` |

## 7. Non-Functional Constraints

| Constraint | Requirement |
|------------|-------------|
| Idempotency | `<statement or N/A — none, because ...>` |
| Ordering | `<statement or N/A — none, because ...>` |
| Concurrency | `<statement or N/A — none, because ...>` |
| Performance | `<statement or N/A — none, because ...>` |
| Durability | `<statement or N/A — none, because ...>` |
| Observability | `<statement or N/A — none, because ...>` |
| Security | `<statement or N/A — none, because ...>` |
| Resource ceilings | `<statement or N/A — none, because ...>` |

## 8. External Interfaces And Assume/Guarantee Contracts

### <Interface Name>

- Requires: <caller obligations>
- Guarantees: <component promises>
- Fails with: <error taxonomy references>
- Preserves: <invariants that remain true>

## 9. Non-Goals

- <thing the implementation must not infer>
- <deferred behavior>

## 10. Glossary And Assumptions

### Glossary

| Term | Meaning |
|------|---------|
| `<term>` | `<definition>` |

### Assumptions

| Assumption | Why It Is Safe | What Changes If False |
|------------|----------------|-----------------------|
| `<assumption>` | `<reason>` | `<impact>` |
```

## Authoring Notes

- Keep examples in the paired `.feature` file; keep contracts and structure in the `.core.md` file.
- Prefer exact field names, transition names, and error IDs over descriptive paragraphs.
- If a section is empty, write `N/A — none, because ...`; do not omit the section.
- If a rule could live in the pure core but is left in the shell, justify that decision explicitly.
