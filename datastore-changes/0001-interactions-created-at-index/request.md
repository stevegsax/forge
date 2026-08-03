# Change request: `interactions-created-at-index`

| Field | Value |
| --- | --- |
| Product | `forge` |
| Id | `0001` |
| Database | `forge_prod` (prod) |
| Schema | `public` |
| Version table | `alembic_version_forge` |
| From → to | `003` → `004` (linear; single head verified) |
| Tier claimed | 1 (additive, no rewrite, no shared surface) |
| Stacks | `prod` only — dev already applied self-service (evidence below) |

## What and why

One concurrent index: `ix_interactions_created_at` on
`interactions (created_at)`. The interactions table is forge's
authoritative spend record (forge D97) — terminal results carry
surviving-call totals only, so every cost or audit question ("what ran
in the last hour", "what did today cost") resolves to a time-windowed
read of this table, and forge's queued T7.3 (honest token accounting)
plans its calibration reads over recent windows of the same table.
Nothing indexes `created_at` today, so each such query is a sequential
scan. This is also, deliberately, forge's **maiden request** under the
schema-change process (issue #2): small, additive, single-phase.

## Phases

| File | From → to | Transactional | statement_timeout |
| --- | --- | --- | --- |
| `change-1.sql` | `003` → `004` | no (concurrent index) | 0 |

Generated `alembic upgrade 003:004 --sql` (postgres dialect, offline);
`BEGIN`/`COMMIT` lines stripped per the process; the per-phase stamp
(`UPDATE alembic_version_forge …`) is the final statement. The
concurrent build is `IF NOT EXISTS` — resumable, as a
non-transactional phase must be.

## Risk notes (evidence, not intentions)

- `interactions` on `forge_prod`: **0 rows, 32 kB total relation
  size** (derived 2026-08-02 via forge's own credential — the store
  was rebuilt empty on 2026-07-31 and no prod tasks have run since).
  The concurrent build completes in milliseconds at this size.
- Lock profile: `CREATE INDEX CONCURRENTLY` takes
  `SHARE UPDATE EXCLUSIVE` — never blocks DML; blocks only other
  schema changes on the table, of which there are none pending.
- The stamp `UPDATE` touches the one-row version table.
- `CONCURRENTLY` is habit and lint-cleanliness at this size, not
  present need — chosen so the maiden request exercises the phased
  apply path exactly as a production-scale request would.

## Compatibility (expand/contract)

Purely additive; no deployed code reads or requires the index, and no
code change ships with this request. Currently-deployed code is
unaffected in both directions.

## Backfill

None.

## Rollback stance

Forward-fix; the down revision exists
(`DROP INDEX CONCURRENTLY IF EXISTS`) if the operator ever wants it.

## Lint

Squawk (`squawk-cli` latest, run 2026-08-02 over `change-1.sql`):
**2 findings, both justified — and both structural to this process.**

```text
warning[require-lock-timeout]: Missing `set lock_timeout` before
  potentially slow SHARE UPDATE EXCLUSIVE lock operations
warning[require-statement-timeout]: Missing `set statement_timeout`
  before potentially slow operations
```

Justification: the schema-change process deliberately forbids session
settings in the artifact — the apply runbook **injects**
`lock_timeout` (default 5s) and a per-phase `statement_timeout`
(`apply-change.sh`), and refuses artifacts containing transaction
control. These two rules will therefore fire on **every** request
under this process. Suggestion for the operator: exclude
`require-lock-timeout` and `require-statement-timeout` in the
recommended Squawk configuration so "clean lint" stays a meaningful
Tier-1 signal; until then, treat this pair as permanently justified.

## Dev evidence

`forge_dev` applied self-service 2026-08-02 (R9): the revision reached
dev via the still-extant startup-migration path at a dev-worker
restart; verified `alembic_version_forge = 004` and
`ix_interactions_created_at` present, via forge's own dev credential.
Note: verify-only startup is queued (T10.1 remaining item 1) — this is
the last request expected to reach dev through startup migration
rather than an explicit `alembic upgrade`.
