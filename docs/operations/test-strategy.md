# Forge Testing Strategy

This guide is for new engineers who need a practical, repeatable way to write Python tests for Forge. It focuses on predictability, speed, and signal quality so we can ship changes with confidence.

## 1) General approach
- Favour the testing pyramid: most coverage from fast, isolated unit tests; a thinner layer of integration tests for cross-module seams; a small set of end-to-end or workflow checks for critical user paths.
- Keep the suite green and quick; slow or flaky tests erode trust and are ignored.
- Match test depth to risk: higher change rate, higher business impact, or complex logic deserves deeper tests and more scenarios; stable glue code can rely on lightweight checks.
- Make tests the executable version of the acceptance criteria—write them while refining requirements, not after the feature lands.
- Prefer deterministic, single-threaded execution. When testing async code (common in Forge), control the event loop with `pytest-asyncio` and explicit awaits.

## 2) Are we testing the right things?
- Trace from goals → behaviours → interfaces: start with the user-visible behaviour (docs/PHASE*.md), map to public functions and CLI paths, then to edge cases (errors, timeouts, missing inputs).
- Cover contracts and invariants: idempotency of workflows, immutability of committed worktrees, ordering guarantees in fan-out/gather, error-handling and rollback paths.
- Guard high-risk integrations: Temporal workflow boundaries, git worktree operations, and context assembly heuristics should each have at least one integration test hitting the real component, plus unit tests with fakes.
- Validate negative space: what must never happen (e.g., double commits, unvalidated LLM output) deserves tests that assert we refuse or sanitize it.

## 3) Documentation that enables good tests
- Write acceptance criteria as Given/When/Then examples; include unhappy paths and boundary values.
- Record interfaces: input schemas, expected side effects (files touched, git commits), and external calls (Temporal activities, network). The clearer the contract, the smaller the fixture setup.
- Capture timing and ordering rules explicitly (e.g., "child workflows must finish before parent commits"), so tests can assert sequence.
- Note observability hooks (logs, metrics, traces) that tests can assert on without coupling to internals.

## 4) Hallmarks of a good test
- Follows the F.I.R.S.T./FIRST-U heuristic: Fast, Isolated, Repeatable, Self-validating, Timely/Understandable.
- Has one clear reason to fail; asserts a single behaviour or invariant.
- Uses minimal, explicit fixtures; avoids hidden global state.
- Names read as documentation: `test_context_pack_falls_back_on_manual_files_when_budget_exceeded`.
- Deterministic data and clocks: fixed seeds, frozen time, stable tmp paths.

## 5) How to spot a bad test
- Flaky (sometimes passes, sometimes fails) — usually relies on uncontrolled state, time, or external services.
- Over-specified: asserts every internal call instead of observable outcomes, making refactors painful.
- Over-mocked: mocks so many collaborators that the behaviour under test is no longer realistic; prefer fakes for complex collaborators.
- Broad and slow: covers multiple behaviours, hits network or disk unnecessarily, or needs long sleeps to "stabilize".
- Opaque failures: multiple assertions with unclear messages ("assert False"), or data-less snapshots that hide intent.

## 6) What to do
- Start with unit tests for pure logic; add contract tests for service boundaries; add a small number of scenario tests across the CLI to protect the golden paths.
- Use `conftest.py` fixtures for shared setup (e.g., temp git repo, ruff config) and keep them focused.
- Mock or fake external systems (Temporal server, network) to keep tests fast and isolated; prefer fakes over deep mocking when behaviour matters.
- Parametrize edge cases (empty inputs, large payloads, invalid configs) to increase coverage with minimal code.
- Keep assertions high-level (results, side effects, emitted files) and supplement with trace/log checks where appropriate.
- Run `pytest -q` locally before pushing; use `ruff` to catch style and lint drift early.

## 7) What to avoid
- Hitting live network, real clocks, or random seeds without control — primary sources of flakiness.
- Long-running end-to-end UI-style tests as a substitute for missing unit coverage; keep top-of-pyramid tests minimal.
- Test order dependencies or shared mutable globals.
- Asserting internal implementation details that are free to change (private helpers, exact log wording) unless the contract demands it.

## 8) Tools for repeatable, reliable tests
- **pytest** with **pytest-asyncio** (already configured) for async-friendly, expressive tests.
- **Pytest fixtures** for controlled, reusable setup; scope them appropriately (function by default; session for expensive resources).
- **Ruff** for linting to keep tests tidy and consistent.
- **uv** for locked, reproducible environments; prefer `uv run pytest` to ensure deps match `uv.lock`.
- Optional add-ons (use as needed): `pytest-xdist` for parallel runs, `pytest-rerunfailures` to detect flakiness (investigate root causes, don’t hide them), and `hypothesis` for property-based edge coverage.

## Quick checklist
- [ ] Test names state behaviour, not method names.
- [ ] Every test controls time, randomness, and IO it touches.
- [ ] One assertion clause (logical) per test; failures are readable.
- [ ] Fixtures are explicit and minimal; no implicit state.
- [ ] Fast locally (< 1s per unit test); CI runtime dominated by a few integration/e2e cases.
