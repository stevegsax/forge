# Requirement Review Checklist

Use this checklist before changing a requirement core from `in-review` to `approved`.

## Package Integrity

- [ ] The package contains both `<requirement-id>.feature` and `<requirement-id>.core.md`.
- [ ] The requirement ID matches across metadata, filenames, and links.
- [ ] The paired feature file contains behavioral examples only, not source-of-truth contracts.
- [ ] The requirement core status, owner, reviewers, and last-updated date are filled in.

## Scope And Clarity

- [ ] The business goal is explicit.
- [ ] The system boundary is explicit.
- [ ] Inputs, outputs, and owned responsibilities are explicit.
- [ ] Out-of-scope items are written down.
- [ ] External dependencies the shell may touch are enumerated.

## Static-First Contracting

- [ ] Every important rule appears in the contract map.
- [ ] Every contract rule has a stable `rule_id`.
- [ ] Every contract rule has at least one witness.
- [ ] Every contract rule has at least one counterexample.
- [ ] Each rule is assigned to the strongest practical enforcement layer.
- [ ] The spec prefers illegal states to be unrepresentable where practical.
- [ ] Residual runtime checks are justified rather than used by default.
- [ ] Rules that cannot be made static explain why not.

## Functional Core / Imperative Shell

- [ ] Pure core responsibilities are separated from shell responsibilities.
- [ ] Business policy lives in the pure core unless a reviewed exception says otherwise.
- [ ] Filesystem, database, network, clock, randomness, subprocess, and secret access are listed as
      capabilities or ports when relevant.
- [ ] The shell is described as orchestration, not as a place where domain decisions happen.
- [ ] Deterministic behavior and injected dependencies are explicit.

## Domain Modeling

- [ ] The domain algebra uses distinct types, variants, or states instead of vague strings and
      booleans wherever practical.
- [ ] Derived fields are identified.
- [ ] Lifecycle transitions are explicit.
- [ ] Illegal or impossible states are named explicitly.

## Errors And Constraints

- [ ] The error taxonomy distinguishes validation, transition, external, and retry-related failures.
- [ ] User-visible and operator-visible failure surfaces are called out.
- [ ] Non-functional constraints cover idempotency, ordering, concurrency, performance, durability,
      observability, security, and resource ceilings, or explicitly mark them `N/A — none, because
      ...`.
- [ ] Interface-level `Requires`, `Guarantees`, `Fails with`, and `Preserves` statements exist for
      each external boundary.

## Python Readiness

- [ ] The rules can be implemented in a constrained Python subset without leaning on untyped
      dictionaries or broad `Any`.
- [ ] Domain identities, variants, and protocols are specific enough to map to Python types.
- [ ] Boundary validation is separated from internal domain logic.
- [ ] Nothing in the spec pressures the implementation toward mixing business logic into handlers,
      ORM models, or adapters.

## Independence And Handoff

- [ ] The requirement package is strong enough that an independent test author could derive tests
      without asking the implementation author for clarification.
- [ ] The implementation agent does not need permission to rewrite requirements or tests to
      succeed.
- [ ] Open questions have been resolved or promoted into reviewed assumptions.
- [ ] Non-goals are concrete enough to prevent speculative implementation.

## Approval Record

- [ ] Reviewer believes the package is ready for autonomous implementation.
- [ ] Reviewer believes the package is ready for independent test derivation.
- [ ] Status has been changed to `approved`.
- [ ] Approval date and reviewer name have been recorded in the review system used by the team.
