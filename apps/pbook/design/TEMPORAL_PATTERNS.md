# SAX Temporal Patterns

House rules for Temporal services (pbook, ocr, forge). Adopted for new work as of June 2026;
existing code migrates per [REVIEW-2026-06.md](REVIEW-2026-06.md). Where current code
contradicts a rule, the rule wins for anything new.

## 1. Temporal earns its keep, or it isn't used

A workflow is justified only for multi-minute, multi-step jobs with external side effects
whose partial progress must survive a crash — transcript ingestion, curation/consolidation,
provider-batch lifecycles. CRUD reads and writes, retrieval, and export are synchronous
library calls against Postgres: the database is already the durable store, and a failed read
is just rerun. Workflow-as-RPC (one-activity wrapper workflows) is banned.

## 2. One composition root per process; no module globals

Each worker or CLI entrypoint builds its dependencies once — frozen Settings
(pydantic-settings; the only place env is read), engine, LLM/embedding clients — and passes
them inward. Activities are methods on frozen dataclass classes constructed with their
dependencies; `Worker(activities=[acts.method, ...])` registers bound methods. No
module-level providers, clients, or engine caches; tests construct the same classes with
fakes instead of monkeypatching.

## 3. Typed activities; no string-keyed registries

An activity's input and output are Pydantic models named in its signature, so mypy checks the
contract and no name registry can drift. A generic LLM activity, where genuinely needed,
carries `model_json_schema()` in its input payload; the calling workflow validates the result
with its own model class. Worker-startup name registration is not a pattern.

## 4. Deterministic idempotency at every durable write

Retries are at-least-once; every durable write needs a deterministic natural key plus
`ON CONFLICT` (pbook: `origin_hash` = sha256 of session_id + experience_hash + normalized
title; ocr: uuid5 image ids). Similarity thresholds are policy (match-or-attach decisions),
never retry protection — embeddings drift.

## 5. Isolate per-item failure inside batch workflows

A workflow that loops over items wraps each item in try/except and records a per-item outcome.
One bad item must not abort the batch; the workflow result reports
`{succeeded, failed, errors}`.

## 6. Centralized retry/timeout presets

One `policies.py` (or the platform `temporal.retry` module) per service holds frozen, named
presets — `LLM_RETRY`, `DB_RETRY`, `IO_RETRY`, with timeout and heartbeat constants. Activity
call sites import presets; ad-hoc per-file RetryPolicy literals are not allowed. Auth/config
errors are classified non-retryable (typed SDK exceptions first, message markers as fallback)
so a missing key fails the workflow instead of hanging it.

## 7. Payload limits: claim-check beyond 256 KB

Large payloads (transcripts, batch results) never enter workflow history. Same-host: pass
`{path, id}` and read in the activity. Cross-host: S3 via the platform contracts helpers.
Temporal's hard limits (2 MB/payload, 4 MB/gRPC message, 50 MB/51.2k-event history) are
designed for, not discovered.

## 8. Waiting on external completion: timer loop, not shared poller

A workflow waiting on an external job (provider batch, long-running service) polls with an
in-workflow timer loop — `workflow.sleep(60–300s)` plus a cheap status activity. A shared
poller workflow signaling waiters is reserved for true fan-in of child workflows (ocr's chunk
gather). The few redundant status calls are cheaper than a coordination subsystem.

**Amended 2026-06-10 (merged platform plan) — scoping.** This pattern was adopted
platform-wide: forge and ocr batch waiting also use the timer loop, and the shared
`BatchPollerWorkflow` + result-signal subsystem is deleted. Scope: poll interval ≥ 300s.
History budget: a poll cycle costs ~11 history events, so a 25h wait at 600s is ≈ 1,650
events — about 3% of the 51.2k-event cap — and the worst observed case (~30 waits at 300s)
is ≈ 24k events, still safe. If wait counts grow beyond that, continue-as-new is the
documented escape hatch.

**Amended 2026-07-20 (Phase 4 shipped) — shared implementation.** The timer loop is now a
platform primitive: `sax_platform.temporal.polling` supplies `wait_batch_ended` (the
deadline/sleep/status-dispatch skeleton), the 25h `BATCH_WAIT_CEILING`, and two pure poll
schedules — `FixedInterval` (forge's steady cadence, default 600s, floor 300s) and
`BackoffSchedule` (300s for the first hour, then doubling to an 1800s cap with per-waiter
jitter, for burst waiter counts such as ocr's one-batch-per-chunk submissions). forge and ocr
are the two consumers, and no shared poller remains anywhere — ocr's chunk gather became
parent-awaited children. The scaling envelope above is unchanged: the ≥300s floor and the
~11-events-per-poll history budget still bound a single wait, and continue-as-new stays the
escape hatch if concurrent wait counts ever outgrow the 51.2k-event cap.

## 9. Cross-service boundaries

Services do not call each other's activities or child workflows cross-queue, and do not import
each other's code into workflow definitions. If one service must trigger another, it starts a
workflow by string name with wire models pinned in the platform contracts module. Promote to
Temporal Nexus only when a second consumer or a namespace split appears.

## 10. Sessions and schedules

Lifecycle rows (e.g. ingestion sessions) are owned by the workflow that does the work: the
first activity writes `running`, an error handler writes `error`, and a scheduled sweep flips
rows stuck `running` past a deadline — never seeded by a CLI that then exits. Recurring jobs
are Temporal Schedules created idempotently at worker startup (fixed schedule id;
already-exists ignored).

## 11. Determinism: replay tests over worker versioning

No build-ID worker versioning while there is no deployed base — deploys are drain-and-restart.
The determinism gate is replay tests: committed event histories under `tests/replay/`, run
through `Replayer` in pytest, regenerated by a single documented command whenever a workflow
changes. Workflow code uses `workflow.now()`, `workflow.uuid4()`, `workflow.random()` — never
the stdlib equivalents.

## 12. Testing

Multi-step workflows are tested under `WorkflowEnvironment.start_time_skipping()` with real
activity implementations against the test Postgres where practical, fakes where not. Pure
logic (ranking, packing, prompt building, state transitions) is never tested through a
workflow — it is imported and called directly.
