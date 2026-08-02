# Proposal: database change requests (forge → sax-datastores)

**Status:** DRAFT for operator review — 2026-08-02
**From:** forge (consumer), written by its planning session
**To:** the sax-datastores operator
**Scope:** how a consumer requests **schema changes** (DDL) and their
extensions after initial provisioning. Provisioning-level asks (new
slugs, stacks, buckets, databases) stay in the existing
`datastore-registration.yaml` flow; this proposal does not touch it.

This document is standalone: the operator session that reads it has no
access to the conversation that produced it, so everything it relies
on is stated below as an assumption to approve or correct.

## Assumptions

- **A1 — the operating model.** sax-datastores owns database strategy
  and operates the databases; consumers ask for the layout, schema,
  and extensions they want and never perform the setup. A request is a
  ticket against the operator; the operator checks for conflicts and
  performs the structural updates, or pushes back. (Owner statement,
  2026-08-02.)
- **A2 — §21 direction.** Per sax-datastores `rationale.md` §21 as
  ruled in outline: products hold no DDL rights; application
  credentials are consolidated to a small set (at minimum an app/DML
  credential, an operator-held DDL credential, and a read-only
  credential per environment — final names and count are the
  operator's). Structural change happens only under the operator's
  DDL credential.
- **A3 — storage layout is not assumed.** Whether per-product
  databases remain (as provisioned today) or storage consolidates
  further is the operator's open design. This proposal works under
  either; if databases consolidate, the request artifact gains a
  namespace/schema field and nothing else changes.
- **A4 — current consumer mechanics.** The forge monorepo carries
  three Alembic chains (forge's, ocr's `alembic_version_ocr` — both
  in forge's database by design — and pbook's
  `pbk_alembic_version`). All three are applied **at worker startup**
  today. Forge accepts that this must end under A1/A2; see Position.
- **A5 — the trust direction persists.** As in the registration flow:
  consumers commit request artifacts in their own repos (the commit
  hash is the request); the operator reads consumer repos and applies;
  neither side ever writes into the other's repo.
- **A6 — one human, two hats today.** The same owner currently
  operates sax-datastores and owns forge. The process should still
  work when the hats separate, so nothing below relies on shared
  memory between them.
- **A7 — CI stays self-sufficient.** Consumer CI builds schemas in
  ephemeral containers with no operator involvement. Whatever artifact
  the request uses must remain executable by the consumer's own test
  tooling.
- **A8 — DDL vs DML split.** Schema structure is the operator's;
  data is the consumer's. Backfills and data migrations are DML,
  executed by the consumer under its app credential, coordinated with
  the schema request only when ordering requires it.

## Position

Forge endorses operator-owned structure without reservation — the
credential scope-back prevents by construction the accident classes
already observed (cross-repo write-backs, agent reads of secrets).
Three consequences follow, stated bluntly because they are the parts
that cost forge work:

1. **Startup migrations must die.** Racing worker instances, DDL locks
   during boot, and deploys coupled to migration duration are the
   standard failure modes — industry guidance is explicit that
   migrations should never run as a side effect of an instance
   starting, because a failed migration crash-loops the fleet. Under
   A2 the app credential cannot run DDL anyway, so the startup path
   becomes **verify-only**: compare `alembic_version` against the
   code's expected head and fail closed with a named error
   ("schema behind — a change request is pending/needed"), never DDL.
2. **Schema-apply must decouple from code-deploy.** With an operator
   gate in the path, same-instant coordination would make every
   release wait on a human. Forge therefore adopts the
   expand/contract compatibility contract: every schema change is
   backward-compatible with the currently deployed code (expand →
   migrate → contract; a breaking change is three requests, not one),
   so the operator can apply whenever convenient and forge deploys
   afterward, in that order, without a shared clock.
3. **The request must be an exact artifact, not prose.** "Operator
   performs the structural updates" should mean *reviews and executes
   a reviewed script*, never *re-derives DDL from a description* —
   transcription is the failure mode, and it would also require the
   operator to learn every product's ORM.

## Requirements

What any accepted process must provide the consumer. R1–R5 are hard;
R6–R10 are strongly wanted.

- **R1 — committed, machine-readable request**, pinned by commit hash,
  exactly like the registration flow.
- **R2 — consumer authors, operator applies, no consumer code runs
  under operator credentials.** The consumer knows its own data model;
  the operator should not execute consumer Python to apply it. (The
  mechanism below achieves this with plain SQL.)
- **R3 — machine-checkable completion.** The applied change must be
  visible to the consumer's own credential (the `alembic_version`
  stamp suffices) so verify-at-startup and deploy gating need no
  human signal — plus a durable operator-side record for audit.
- **R4 — CI parity** (assumption A7 as a requirement): the migration
  chain remains runnable by the consumer's tests against throwaway
  databases.
- **R5 — defined review scope.** The operator's conflict check needs a
  published checklist: lock profile, duration vs table size,
  cross-product resource conflicts, shared-surface changes
  (extensions), naming. Interior design of a consumer's own tables
  stays the consumer's judgment — otherwise the gate becomes a full
  design review and the operator the bottleneck-owner of every data
  model.
- **R6 — turnaround expectation.** Even an informal SLO ("non-risky
  requests same-day; risky ones get a reviewed slot") — a single
  human gate serializes all products. Batch application of several
  pending requests in one act should be normal.
- **R7 — emergency path defined in advance.** When production is
  broken and the fix is structural (an index, a constraint drop),
  there must be a pre-agreed fast lane — still operator-executed,
  but not waiting on the normal cycle.
- **R8 — a reverse channel.** The model defines consumer→operator
  only; three live cases already run the other way (an endpoint move
  that requires a lockstep consumer edit; a credential rotation that
  gates a consumer act; §21's own rollout touching every product's
  `DATABASE_URL`). Operator-initiated changes should arrive as change
  notices that create entries in the consumer's queue — the mirror of
  the ticket rule.
- **R9 — bounded dev friction.** Recommendation, needs a ruling: keep
  `dev`/`test` self-service (the consumer's dev credential may run
  DDL there, or dev keeps a permissive role) and gate `prod` only.
  The dev instance is disposable by design; the blast-radius argument
  for the prod gate does not apply, and a ticket round-trip per
  model-iteration would move development back onto SQLite-shaped
  shortcuts. If the operator rules all-environments-gated, forge
  complies — but wants the cost acknowledged.
- **R10 — backfills classified as DML** (assumption A8 as a
  requirement): the request declares any companion backfill and its
  ordering, but the backfill itself runs consumer-side under the app
  credential.

## Recommended mechanism (v1)

Modeled on the registration flow, which has already run end-to-end
once. Six steps:

1. **Author.** Forge writes Alembic revisions as today (they remain
   the CI schema source, satisfying R4).
2. **Generate.** Forge produces the exact SQL with Alembic's offline
   mode — `alembic upgrade <from>:<to> --sql` — which exists
   precisely for organizations where DDL access is restricted and
   scripts are handed to a DBA. Offline mode cannot execute
   data-dependent Python (no live SELECTs), which is a feature here:
   it structurally separates DDL (in the request) from DML backfills
   (consumer-run, R10).
3. **Request.** Forge commits both under `datastore-changes/` in its
   own repo and notifies the operator (same channel as registration
   applies):

   ```text
   datastore-changes/
   └── 0001-<slug>/
       ├── request.md    summary + database/chain + from→to revisions
       │                 + risk notes (locks, est. duration, table
       │                 sizes) + backfill plan + rollback stance
       └── change.sql    generated offline SQL; header comment states
                         the expected current alembic_version
   ```

   The commit hash is the request (R1).
4. **Review.** The operator checks the SQL against their checklist
   (seed below) and the pinned commit. Push-back happens here, as
   review comments or a refusal with reasons — the "request changes"
   half of the model.
5. **Apply.** The operator verifies the database's current
   `alembic_version` equals the request's stated from-revision
   (refuse otherwise — raw SQL is not idempotent, and this pre-check
   is what makes double-apply impossible), then executes `change.sql`
   under the DDL credential. The script's own `UPDATE alembic_version`
   is the completion signal (R3). The operator records
   {request id, pinned commit, applied date, outcome} wherever they
   keep approval records — the record's shape is theirs.
6. **Deploy.** Forge ships the code that uses the new schema whenever
   ready — the compatibility contract (Position 2) makes the ordering
   safe without coordination.

### Operator checklist seed (offered; the operator owns the final)

- Lock profile: any `ACCESS EXCLUSIVE` on large or hot tables?
  `CREATE INDEX CONCURRENTLY` where applicable? `lock_timeout` /
  `statement_timeout` set in the script?
- Duration estimate vs table size stated in `request.md`?
- Cross-product conflicts: shared-instance resources, naming
  collisions; any extension addition is a shared-surface change (per
  the registration flow) and gets extra scrutiny.
- Reversibility stance stated (forward-fix is acceptable; silence is
  not).
- Expand/contract compliance: does currently-deployed consumer code
  keep working against the post-change schema?

### Forge-side changes queued once this is approved

- Replace `run_migrations` at all three worker startups with
  verify-only + fail-closed named error (Position 1).
- Adopt the compatibility contract into forge's process records
  (PROCESS.md/SDLC.md) so it binds future change-sets.
- Redesign the owed DB-URL guard around the operator's final
  credential naming (A2 leaves names open).
- CI/testcontainers move to the canonical image (already queued in
  T10.1) so the schemas CI builds match what the operator runs.

## Alternatives considered

- **Prose tickets; operator writes the DDL.** Rejected: transcription
  errors, and the operator must internalize every product's model.
- **Operator executes the consumer's Alembic chain directly.**
  Workable and simpler, but it runs consumer Python under a
  privileged credential. Kept as a fallback for operations offline
  SQL cannot express; the default stays reviewed plain SQL.
- **Consumer keeps a DDL credential behind an approval gate.**
  Rejected: defeats §21's credential scope-back — the accident
  surface this model exists to remove.
- **Fully declarative desired-state schema** (Atlas/Skeema-style: the
  consumer commits target schema; operator tooling diffs live state
  and plans the change). Attractive end-state and philosophically the
  same as the registration flow; adoptable later without breaking
  v1 — the `change.sql` artifact simply becomes generated from the
  declaration instead of from Alembic. Not proposed now because the
  Alembic chains are already embedded and CI-load-bearing.

## Open questions for the operator

1. Dev/test self-service or all environments gated (R9)?
2. Storage end-state: per-product databases or consolidation (A3)?
   Affects only the request's namespace field.
3. Turnaround expectation (R6)?
4. Emergency path shape (R7)?
5. Notification form for step 3 — is the verbal "please apply" of the
   registration flow enough, or does the operator want a standing
   inbox (issues, a requests file)?
6. Who stamps the operator-side record, and does a validator (like
   `check-registrations.sh`) diff records against applied state?
7. Does sax-temporal adopt the same request/record/validator
   discipline for namespaces? Its side of the model currently has no
   intake or completion artifact at all, and its ledger went stale
   within days of the last change — same review moment, same fix
   shape.

## Sources

Industry practice consulted for this proposal:

- [Liquibase — database change management best practices](https://www.liquibase.com/blog/database-change-management-best-practices)
  and [Bytebase — what is database change management](https://www.bytebase.com/blog/what-is-database-change-management/) —
  changes as reviewable artifacts tied to justification; risk-based
  approval routing; DBA reviewers for structural change, developers
  for data migrations.
- [GitLab's database review guidelines](https://git.math.duke.edu/gitlab/help/development/database_review.md) —
  the published-checklist precedent for operator review.
- [Expand/contract pattern](https://www.tim-wellhausen.de/papers/ExpandAndContract/ExpandAndContract.html)
  ([practical guide](https://blogs.reliablepenguin.com/2025/11/16/database-migrations-without-drama-expand-contract-in-practice),
  [zero-downtime overview](https://www.harness.io/blog/zero-downtime-database-migrations-safe-schema-changes)) —
  never change schema and dependent code in one step; decouple
  migrations from app boot ("a failed migration crash-loops your
  fleet").
- [Alembic offline mode](https://alembic.sqlalchemy.org/en/latest/offline.html) —
  `upgrade <rev> --sql` generates hand-off SQL for restricted-DDL
  organizations; offline scripts cannot depend on live queries, which
  enforces this proposal's DDL/DML split.

Repo precedents: sax-datastores `docs/registration.md` (commit-as-
request, one-way trust, validator), `rationale.md` §17/§21/§22, and
the applied forge/pbook registrations of 2026-07-30.
