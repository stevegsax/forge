# Handoff — Monorepo consolidation + local-first deployment

**Date:** 2026-07-16
**Status:** Complete and shipped. All work is merged to `main` and pushed; the
deployment is live. Nothing is in flight.

This is a state-of-the-world note for the next session. It does not restate the
work queue ([TASKS.md](TASKS.md)), the decisions ([../docs/DECISIONS.md](../docs/DECISIONS.md)),
or the status-of-record ([../docs/OVERVIEW.md](../docs/OVERVIEW.md)) — it says
what changed, what to know before touching anything, and where to start.

## What changed

Two decisions were made and executed end-to-end.

**D98 — forge is the monorepo** (supersedes part of D87: no new `sax` repo).
T2.1 was rewritten as an incremental in-place absorption and is now **COMPLETE**:

| Increment | Landed | Notes |
| --- | --- | --- |
| 1 — pbook → `apps/pbook` | tag `pbook-v0.2.2` | Root becomes a uv workspace |
| 2 — sax-llm + forge-contracts → `libs/` | tags `sax-llm-v0.1.2`, `forge-contracts-v0.1.2` | Closes Finding A (below); workspace becomes self-contained |
| 3 — ocr → `apps/ocr` | tags `ocr-v0.1.1`, `ocr-v0.1.2` | Consolidation complete |
| Python 3.14 bump | `590b217` | Gated on a wheel-matrix re-check; zero lock churn |

Each import used git-filter-repo with a rehearsal on a throwaway clone; full
history and blame survive (`git log --follow` works across the rewrites). Tags
were renamed into per-package series (`v*` → `<pkg>-v*`). The five predecessor
repos (`pbook`, `sax-llm`, `forge-contracts`, `forge-ocr`) are pushed, given
pointer-READMEs, and **archived on GitHub**.

**D99 — EC2 retired, local-first deployment.** T0.7 was rewritten from "deploy
hardening" to "retire EC2" and is complete: Temporal self-hosts in the podman
stack persisting to that stack's Postgres; workers run under launchd on the
always-on desktop; Supabase + S3 remain the managed stores; Terraform, the SSM
bootstrap, the systemd units, the cert tooling, and the mTLS gateway are
deleted. Remote access is deliberately out of scope (may return).

## What to know before touching anything

- **The ambient shell env points at production.** `FORGE_DB_URL` → Supabase,
  `AWS_*` → real S3. Override before any local DB/blob command. The local
  stack's Postgres is on **5434** on this machine (5433 was taken; the override
  lives in the gitignored `deploy/local-stack/.env`).
- **The deployment is live.** Three launchd agents are running and polling:
  `forge-worker-1/2` (`forge-task-queue`) and `ocr-worker` (`ocr-task-queue`).
  The pbook worker is *not* installed (`install.sh --with-pbook` when wanted).
  Logs: `$XDG_STATE_HOME/forge/logs/`. The workers migrate the **production**
  Supabase schema on startup — that is by design, and it already applied
  T1.6a's pending migration on first boot.
- **`uv sync` is not enough.** `apps/ocr` is a member nothing depends on, so a
  bare (exact) sync prunes it. Setup is `uv sync --all-packages`; the ocr CLI is
  `uv run --package ocr ocr <cmd>`. `uv run` syncs inexactly, so the live
  workers are never stripped by a member-scoped run.
- **Suites run per-package, from each package's own directory** (forge 1233,
  pbook 345, sax-llm 165, forge-contracts 10, ocr 48 — all green on 3.14).
  Postgres-marker suites need `DOCKER_HOST` pointed at the podman socket or
  they silently **skip**:
  `DOCKER_HOST="unix://$(podman machine inspect --format '{{.ConnectionInfo.PodmanSocket.Path}}')" TESTCONTAINERS_RYUK_DISABLED=true uv run pytest -m postgres`

## Findings from this work (recorded, not all fixed)

- **Finding A** (design gate, D98) — *closed*. The pbook skill's documented
  end-state uvx invocation couldn't resolve while the root had `../` path
  sources; the skill ran an interim pin against the archived pbook repo until
  increment 2 made the workspace self-contained. It now pins
  `git+…forge.git@pbook-v0.2.3#subdirectory=apps/pbook`, verified against
  GitHub. **When cutting a new `pbook-v*` tag, bump the pin in the
  `skill-pbook` repo.**
- **Concurrent migrations raced** — *fixed* (`20003e8`). Both workers migrate at
  startup and Alembic takes no lock; the first launchd boot had worker-2 die on
  `DuplicateColumn` (KeepAlive recovered it). `run_migrations` now takes a
  session-level `pg_advisory_lock`; regression test in
  `tests/test_migrations_postgres.py`.
- **launchd bootout is asynchronous** — *fixed* (`296c330`). Re-running the
  installer over live agents failed with EIO and left an agent uninstalled; the
  installer now waits for teardown before bootstrapping.
- **Two live deploy defects** — *fixed with the EC2 deletion*: bootstrap never
  cloned forge-contracts (fresh-instance `uv sync --frozen` had been broken
  since T1.0), and the pbook unit still set a SQLite-era `PBOOK_DB_PATH`.
- **Still open** — see the "Tooling & ops debt" table in
  [../docs/OVERVIEW.md](../docs/OVERVIEW.md): no CI, uneven gates, four standing
  lint/type findings, the dead `batch-status` skill, the dead skill-pbook eval
  harness, and `pmset` not applied (owner parked it).

## Where to start

**T2.2 — root gates** is the natural next task and the reason most of the debt
above is worth reading first: it turns the by-hand `uv run` gates into one
workspace-wide CI gate, and it must decide what to do about the four standing
findings, the missing coverage gates on `apps/ocr` / `libs/forge-contracts`, and
the import-DAG rule D87 specified (apps never import apps; libs never import
apps; `contracts` imports no SDKs). **T2.3a–d** then roll mypy strict across the
members. Those two close Phase 2, after which the ordering gate opens onto
Phase 3 (`sax_platform` consolidation + structured outputs).

Phase 0 (T0.1–T0.6, T0.8) remains independent of everything and can land anytime.
