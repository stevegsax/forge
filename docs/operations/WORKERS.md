# Workers

## Overview

A worker is a long-running process that polls a Temporal task queue and executes the activities of the workflows scheduled on it. Workers are stateless — all workflow state lives in the Temporal server — so a crashed worker loses no work, and the server, not the host, is the source of truth for what is running.

Three workers exist, one per queue:

| Worker | Command | Task queue | Owns |
| --- | --- | --- | --- |
| forge | `uv run forge worker` | `forge-task-queue` | task/sub-task workflows, LLM + batch activities, the `batch_jobs` ledger |
| ocr | `uv run --package ocr ocr worker` | `ocr-task-queue` | Mistral OCR batches, the `ocr-batch-tracker` Schedule |
| pbook | `uv run pbook worker` | `pbook-task-queue` | knowledge extraction and retrieval |

All three connect to the **same namespace** — the namespace is derived from the environment and the `forge` product slug (`sax_platform.contracts.constants.PRODUCT_SLUG`), not from the worker, so a lane is `forge-dev` or `forge-prod` for all of them. Queues, not namespaces, separate the three.

## Where workers run

Forge owns no infrastructure (D104). Temporal comes from the shared `sax-temporal` stacks and Postgres from the shared `sax-datastores` stacks; the workers are the only forge-owned processes.

| Lane | Supervisor | Code it runs | Temporal | Postgres |
| --- | --- | --- | --- | --- |
| production | launchd (`deploy/launchd/install.sh`) | the pinned checkout `~/repos-sax/forge-prod` (D103) | `127.0.0.1:7243`, namespace `forge-prod` | `127.0.0.1:5442` — `forge_prod`, `pbook_prod` |
| staging (dev) | tmux (`make dev-worker`) | the live working tree | `127.0.0.1:7236`, namespace `forge-dev` | `127.0.0.1:5432` — `forge_dev`, `pbook_dev`; blobs in the dev MinIO on `:9000` |

The ports are the isolation: production is reached only by naming `:7243` and `:5442`, both non-default, so a tool-default connection lands on dev or on nothing.

## Startup sequence

Each worker main is a composition root (T3.6): it reads a frozen settings object **once**, then builds one database engine, one Anthropic SDK client, and one S3 blob client for the whole process and injects them into the class-based activities. Nothing downstream reads the environment. The order of the startup checks is load-bearing — each one fails before the next can do damage:

1. **`resolve_forge_env(os.environ)`** — the declared environment, or exit 78 (D102).
2. **`require_clean_prod_code(env)`** — on prod only, refuse a dirty or unverifiable checkout, exit 78 (D103).
3. **Settings** — `ForgeSettings()` / `OcrSettings` / `PbookSettings`; an unset required variable raises here, before any connection.
4. **`resolve_temporal_target(env, address_override=…)`** — the address and namespace, derived together.
5. **Schema verification** — read the chain's version table, refuse to start when it is behind (see below). No DDL.
6. **`stamped_worker_identity(identity)`** — the launch-time git version stamped onto the base identity.
7. **Connect and register**, then poll.

Fail-fast configuration, so a misconfiguration is a startup failure rather than a mid-workflow surprise:

- The **forge worker** fails if `FORGE_DB_URL` is unset (`DbSettings.url` has no default).
- The **ocr worker** fails if `FORGE_DB_URL`, `FORGE_OCR_S3_BUCKET`, or `MISTRAL_API_KEY` is unset. It submits and polls its own Mistral batches (T4.2); the forge worker never reads `MISTRAL_API_KEY` — its batch transport is anthropic-only.
- The **pbook worker** treats an unset `PBOOK_DATABASE_URL` as "store disabled": it logs a warning, skips the schema check, and runs.

The forge and ocr workers also drain gracefully: `graceful_shutdown_timeout` is 5 minutes, long enough that a SIGTERM never cancels an in-flight LLM call.

## Environment guard

Every worker and CLI is fronted by an explicit-environment guard (D102): it reads `FORGE_ENV` (`prod` / `dev` / `test`) and **fails if it is unset** — there is no default, and a guard failure exits **78** (`EX_CONFIG`, distinct from every command's own exit codes). "Which database am I about to touch" is therefore never a question, and reaching production is a deliberate act rather than an unset variable falling through to a default.

- **Workers** resolve `FORGE_ENV` before any settings or store setup and log it (`env=prod`) at startup. The launchd plists declare `FORGE_ENV=prod` and `FORGE_PROD_ACK=yes` in `EnvironmentVariables` — the plists are the explicit production act — while `run-worker.sh` sources `deploy/launchd/load-env.sh`, which loads `~/.config/forge/envs/$FORGE_ENV.env`, refuses a file that is not chmod 600, parses it line-by-line without ever shell-evaluating a value, and exits 78 if the profile's `FORGE_ENV_TAG` disagrees with `FORGE_ENV`.
- **Production** additionally requires `FORGE_PROD_ACK=yes`, which no profile file sets — only the plist or an interactive shell — so sourcing a profile can never by itself grant production access.
- **Interactive** commands must declare their environment. Sourcing a profile needs `set -a` so the values (and the tag) are exported; a plain `source` does not export, and the guard rejects an unexported tag by design:

  ```bash
  set -a; source ~/.config/forge/envs/prod.env; set +a
  export FORGE_ENV=prod FORGE_PROD_ACK=yes
  ```

- **Non-prod work** uses the `dev` profile (`~/.config/forge/envs/dev.env`, `FORGE_ENV_TAG=dev`, no ack): `set -a; source …/envs/dev.env; set +a; export FORGE_ENV=dev`. It points at `forge_dev` on the shared dev Postgres (`:5432`) and at the dev MinIO (`:9000`).
- **Production workers additionally require a clean, committed checkout** (D103). After resolving `FORGE_ENV=prod` and before touching anything, each worker checks its checkout's git version and exits **78** if it is `-dirty` or cannot be determined, naming the directory and pointing at `deploy/prod-deploy.sh`. The mechanism it defends: a worker `exec`s `uv run` inside its checkout, so that tree's contents at launch *are* the running code, and an uncommitted edit would ship with nothing able to say afterwards what ran. Dev and test are deliberately exempt — editing a checkout under a running dev worker is the point of the staging lane. Deploy production with `make prod-deploy REF=<ref>` (see [DEPLOYMENT.md](DEPLOYMENT.md#the-production-checkout-d103)).
- **The `--env` flag** loads a profile without a manual `set -a; source`: every CLI accepts `--env <name-or-path>` in **either position** — before or after the subcommand (`ocr tracker-status --env dev` and `ocr --env dev tracker-status` are equivalent; given at both levels, the value after the subcommand wins). A bare name resolves to `$XDG_CONFIG_HOME/forge/envs/<name>.env` (and sets `FORGE_ENV` to that name); a path — or any value ending `.env` — is read verbatim and takes `FORGE_ENV` from the file's `FORGE_ENV_TAG`. The profile parser tolerates a leading `export`, strips one surrounding quote pair, and expands `${VAR}` references (braced form only, against the current environment) — never any command execution. It never supplies `FORGE_PROD_ACK`, so `--env prod` still fails unless the ack is exported separately; a missing file, malformed line, or path-form profile without a `FORGE_ENV_TAG` exits 78.
- **`deploy/prod-ocr <cmd>`** wraps the whole prod ceremony for ocr CLI commands: it exports `FORGE_ENV=prod` and the ack, sources the profile through the same `load-env.sh`, and `exec`s `uv run --package ocr ocr <cmd>` **from the pinned checkout**, so the client runs the same commit as the workers serving it. It refuses (exit 78) rather than falling back to the live tree if `~/repos-sax/forge-prod` is missing.

## Schema verification (workers never migrate)

Since 2026-08-03 (`d77761f`), worker startup applies **no DDL**. It reads the chain's version table, compares it against the head the running code declares, and refuses to start when they are incompatible. Two reasons: under the sax-datastores operator model the application credential cannot run DDL at all, and a migration that runs as a side effect of a process starting turns one failed migration into a crash-looping fleet.

| Chain | Version table | Self-service apply |
| --- | --- | --- |
| forge | `alembic_version_forge` | `uv run forge migrate` |
| ocr | `alembic_version_ocr` | `uv run --package ocr ocr migrate` |
| pbook | `pbook.pbk_alembic_version` | `uv run pbook migrate` |

The verdicts (`sax_platform.db.verify`):

- **at head** — proceed, one INFO line naming the revision and the masked database URL.
- **ahead** — the database carries a revision this code's chain has never heard of. **Allowed**, with a WARNING. Under the binding expand/contract contract the schema change lands *before* the code that uses it, so a worker restart inside that window must not brick the lane. If no change is in flight, that warning means the deployment is running stale code.
- **behind**, **uninitialized**, **ambiguous stamp** (more than one row), **broken chain** (the deployed code resolves to more than one head) — `SchemaVersionError`, and the worker exits. The message names the chain, both revisions, the masked database, and the fix.

Applying a chain is self-service on dev and test. **Production schema changes go through the sax-datastores change-request process** — offline SQL artifacts reviewed and applied by the operator, never by forge: [PROCESS.md](../../development-plans/PROCESS.md#schema-changes-operator-requests), and `sax-datastores/docs/schema-changes.md`. Generate a request with `make db-change CHAIN=forge|ocr|pbook FROM=<rev> [TO=<rev>] TITLE=<kebab-title>`.

## Staging lane (dev)

Environments are separated by **server**: dev is `127.0.0.1:7236`, production is `127.0.0.1:7243` (org servers owned by sax-temporal). Inside each, a namespace scopes task queues, schedules, and workflow ids, so a worker or CLI in one cannot see — or poll — another's work. Dev runs in `forge-dev`, production in `forge-prod`.

Neither value is configured. `resolve_temporal_target` derives both from `FORGE_ENV` immediately before every connect and returns them as one frozen `TemporalTarget`, so the address and the namespace cannot disagree. `FORGE_TEMPORAL_NAMESPACE` no longer exists — a line for it in a profile is silently ignored. `FORGE_TEMPORAL_ADDRESS` survives only as an override, and dev/prod refuse any value that is not their own server: pointing a dev process at `:7243` raises `ForgeEnvError` with a message naming the fix, rather than connecting. (`FORGE_ENV=test` is the exception: its server is an ephemeral per-job container on an arbitrary port, so it *must* supply an address, and any address is accepted.)

The namespace name is the backstop for everything the code cannot reach — the `temporal` CLI, one-off scripts, an operator in a shell. Names are `<slug>-<env>`, and the bare slug and `default` exist on no server, so a hand-typed `-n forge` fails with "namespace not found" everywhere instead of landing in production. Convention and ledger: `sax-temporal/docs/namespaces.md`.

Namespaces are created once per server, out of band by sax-temporal (`make namespace ENV=<env> NS=<slug>-<env>`, which applies the right retention per environment). Both already exist.

The dev profile (`~/.config/forge/envs/dev.env`, from `deploy/launchd/envs/dev.env.example`) declares `FORGE_ENV_TAG=dev` and nothing Temporal-related — the target follows from it:

```bash
make dev-worker WORKER=forge     # forge worker, forge-dev
make dev-worker WORKER=ocr       # ocr worker, forge-dev (WORKER defaults to ocr)
make dev-worker WORKER=pbook     # pbook worker, forge-dev
```

`make dev-worker` starts a detached, crash-safe tmux session named `dev-<worker>-worker`: `remain-on-exit` is on, so a crashed worker leaves a dead pane holding its final output instead of vaporizing the session, and the pane is tee'd to `$XDG_STATE_HOME/forge/logs/dev-<worker>-worker.log`. It refuses to clobber a crashed session — use `make dev-worker-restart WORKER=<name>`, which kills the session (dead pane included) and starts fresh.

**A functional lane needs the forge worker even for pure-OCR flows.** ocr's ledger writes (`persist_block` → `forge-task-queue`) are cross-queue activity calls, so with only a dev ocr worker running, an OCR submission blocks at its first ledger persist and the tracker cannot route hints (the live job has no routable `batch_jobs` row — the tracker logs a warning and skips it). Start the pair.

The dev ocr worker installs its own `ocr-batch-tracker` Schedule inside `forge-dev`, separate from production's in `forge-prod`. A dev CLI submits into `forge-dev` too, so it only ever reaches the dev workers.

Commands that never connect to Temporal — `ocr tracker-status`, the `migrate` commands, `make db-change` — are unaffected by any of this.

## Deployed state (verified 2026-08-03)

Re-derive rather than trusting this section; the commands are in the next one.

- **Production is up and runs `104c14b`.** Three pollers — `prod-forge-worker-1@104c14b` and `prod-forge-worker-2@104c14b` on `forge-task-queue`, `prod-ocr-worker@104c14b` on `ocr-task-queue` — all on `:7243` in `forge-prod`. `~/repos-sax/forge-prod` is clean at `104c14b`, matching the stamps. The `ocr-batch-tracker` Schedule fires every 120s and its runs complete.
- **There is no pbook poller on either lane.** The `com.saxcapital.pbook-worker` label is installed but `disabled` (`launchctl print-disabled gui/$UID`), pending the operator's password rotation for the pbook credential. `pbook-task-queue` exists on both servers with an empty poller list.
- **The dev lane is down.** No poller on any dev queue. Its `ocr-batch-tracker` Schedule keeps firing anyway — schedules live on the server, not in the worker — so its runs time out and `ocr-task-queue` carries a growing workflow backlog (~700 as of this writing). Restarting the pair drains it: `make dev-worker WORKER=forge` and `make dev-worker WORKER=ocr`.

## Checking whether workers are running

Workers write no PID file and keep no local state, so the **Temporal server is the source of truth**. Every `temporal` invocation needs an explicit `--address` and `--namespace`: the CLI's defaults (`127.0.0.1:7233`, namespace `default`) match nothing here, and a bare command either fails to connect or asks a server about a namespace that does not exist on it.

```bash
# dev lane
temporal task-queue describe --task-queue forge-task-queue --address 127.0.0.1:7236 --namespace forge-dev
temporal task-queue describe --task-queue ocr-task-queue   --address 127.0.0.1:7236 --namespace forge-dev
temporal task-queue describe --task-queue pbook-task-queue --address 127.0.0.1:7236 --namespace forge-dev

# production — same commands with the prod pair
temporal task-queue describe --task-queue forge-task-queue --address 127.0.0.1:7243 --namespace forge-prod
```

The output lists each poller's identity, task-queue type (workflow / activity), and last access time, plus the backlog counts. An empty `Pollers` section means no worker is serving that queue.

**Web UI.** The dev stack ships one at `http://localhost:8236` (`sax-temporal-dev-ui`). The prod stack deliberately ships none — inspect `:7243` with the `temporal` CLI, or point a UI at it yourself.

**Local processes.** `make workers-status` prints both lanes: `launchctl list | grep com.saxcapital` for production, and the tmux `dev-*` sessions for staging, distinguishing a running session from a crashed one whose dead pane still holds the final output.

## Worker identity

Each worker reports an identity string to the Temporal server; it appears in workflow history events and in the poller list, so it can answer both "which lane is this?" and "which code is it running?".

**The base names the lane**, so the identity agrees with the other two things that already encode the lane — `FORGE_ENV` and the namespace — instead of quietly contradicting them:

| Lane | Started by | Base identity | `FORGE_ENV` | Namespace |
| --- | --- | --- | --- | --- |
| production | launchd (`install.sh`) | `prod-forge-worker-1`, `prod-forge-worker-2`, `prod-ocr-worker`, `prod-pbook-worker` | `prod` | `forge-prod` |
| staging | tmux (`make dev-worker`) | `dev-forge-worker`, `dev-ocr-worker`, `dev-pbook-worker` | `dev` | `forge-dev` |

The dev base is also the tmux session name and the row `make workers-status` prints, so one string identifies a dev worker everywhere you might meet it. Set a base by hand with `--worker-identity` (on `forge worker`, `ocr worker`, and `pbook worker`) or `FORGE_WORKER_IDENTITY`; when omitted, the SDK default `{pid}@{hostname}` is used. Either way the identity is the *base* — the version is appended next.

### Which code is a worker running?

Each worker stamps the git version of the tree it was launched from onto its identity, giving `<base>@<version>`:

| Worker | Identity |
| --- | --- |
| production forge worker 1 (launchd) | `prod-forge-worker-1@104c14b` |
| staging ocr worker (tmux) | `dev-ocr-worker@104c14b` |
| a worker with no base set | `12345@buchla@104c14b` |
| any worker launched from a modified tree | `prod-forge-worker-1@104c14b-dirty` |

This exists because a Python worker binds its code at import: the modules loaded when the process started are the modules it runs until it is restarted, while the checkout keeps moving — a worker `exec`s `uv run` straight out of that checkout. So the stamp records the tree **as it was at that worker's launch**. It does not change when you commit, pull, or edit; a worker picks up new code, and a new stamp, only at its next restart (`make prod-deploy REF=<ref>` for production, `make dev-worker-restart WORKER=<name>` for the dev lane). A worker started before this mechanism existed carries no version suffix at all.

A `-dirty` suffix means `git status --porcelain` was non-empty at launch — modified tracked files or untracked files. The commit alone then does not describe the running code: reproducing that worker's behavior needs the commit *plus* those local changes.

**A production identity should always be a clean SHA.** Since D103 the prod workers run the pinned `~/repos-sax/forge-prod` checkout and refuse to start on a dirty or unverifiable one (exit 78), so `prod-forge-worker-1@104c14b` is the expected shape and `-dirty` should be unreachable there. If you ever see one, something started a worker outside `prod-deploy.sh` — treat it as an incident and capture `git -C ~/repos-sax/forge-prod status` before cleaning, since that tree is the only record of what actually ran.

If the version cannot be determined (no `git` on `PATH`, not a repository, the command fails or times out), nothing is stamped and the identity is exactly the base. Version discovery is best-effort by design and can never keep a worker from starting. The production *guard* takes the opposite reading of the same `None`: an unverifiable checkout is not evidence of a clean one, so it refuses.

## Restarting and deploying

| Goal | Command | Mechanism |
| --- | --- | --- |
| deploy new production code | `make prod-deploy REF=<ref>` | pins `~/repos-sax/forge-prod` to the ref, syncs, restarts (D103) |
| restart production workers on current disk code | `make workers-restart` | resolves each launchd label to its pid and SIGTERMs *that pid*; the worker drains and launchd's `KeepAlive` relaunches it |
| restart one dev worker | `make dev-worker-restart WORKER=<name>` | kills and recreates the tmux session |

`make workers-restart` is **production-only by construction**. It signals launchd-resolved pids, never command-line patterns: the dev tmux workers run byte-identical command lines (the lane lives in environment variables, invisible to `pkill`), so the old pattern-matching restart took the staging lane down with production — observed 2026-07-24.

Changing a *base identity* is a different operation from deploying code. Production bases live in each launchd agent's `ProgramArguments`, and a restart relaunches from the job definition launchd already has loaded — so a rewritten plist is adopted only when `deploy/launchd/install.sh` (run from the prod checkout) boots the agents out and bootstraps them again. The command sequence, the opt-in-flag trap, and the drain caveat are in [deploy/launchd/README.md](../../deploy/launchd/README.md#changing-a-plist-worker-identities-environment-keepalive). The dev lane has no such step: `make dev-worker-restart` rebuilds the tmux command line from the Makefile.

`install.sh` generates agents for `forge-worker-1` and `forge-worker-2` always, plus `pbook-worker` with `--with-pbook` and `ocr-worker` with `--with-ocr`. It is safe to run again post-D104: the retired `forge-stack` and `db-backup` agents are gone from its list, so it can no longer recreate the discarded containers.

### The `launchctl` disabled-label hazard

A teardown that ran `launchctl disable` leaves the label in a disabled state that survives reinstall, and `bootstrap` then fails with a bare `Input/output error` that names nothing:

```bash
launchctl print-disabled gui/$UID          # names the cause
launchctl enable gui/$UID/<label>          # the fix
```

Three labels are disabled today: `com.saxcapital.pbook-worker` (deliberate, pending the pbook credential rotation) and the retired `com.saxcapital.forge-stack` and `com.saxcapital.db-backup` (D104).

## The OCR tracker

The ocr worker installs and reconciles the `ocr-batch-tracker` Temporal Schedule at startup — one `OcrBatchTrackerWorkflow` run every 120s, `overlap=SKIP`, each run bounded by a 5-minute execution timeout — **before** it serves work, and aborts startup if it cannot. Without the Schedule the store children never receive their status hints, so the worker would be unable to finish a batch (T4.4).

```bash
temporal schedule list --address 127.0.0.1:7236 --namespace forge-dev
temporal schedule list --address 127.0.0.1:7243 --namespace forge-prod
```

`uv run --package ocr ocr tracker-status` is the health probe, and it is deliberately **Temporal-free** — a direct read of `FORGE_DB_URL`, so it answers even when Temporal or the workers are down. It prints `checked_at_gmt` first, then the `ocr_tracker_heartbeat` fields, the current live-job count, and a `fresh`/`stale`/`never-ran` verdict. Exit codes are script-friendly:

| Code | Meaning |
| --- | --- |
| 0 | fresh |
| 1 | stale or never-ran, no live jobs |
| 2 | stale or never-ran **with** live jobs waiting — work is queued but the tracker is not completing cycles; check the worker and the Mistral API |
| 3 | probe error (`FORGE_DB_URL` unset/invalid, or the store unreachable) — prints `status: error`, reason on stderr; never collides with the verdict codes |

Staleness threshold: `--stale-after` seconds, default 300 (2–3 tracker cycles). Like every command it needs a declared environment — `ocr tracker-status --env dev`, or `deploy/prod-ocr tracker-status` for production.

## Scaling

Multiple workers can poll the same task queue; the Temporal server distributes work across them. Workers are stateless, so scaling out is starting more processes on the same queue in the same namespace. Production runs two forge workers for redundancy and one ocr worker.

## Further reading

- [DEPLOYMENT.md](DEPLOYMENT.md) — topology, the production checkout, the full environment-variable reference.
- [DEBUGGING.md](DEBUGGING.md) — logs, the observability store, and `temporal` CLI recipes.
- [deploy/launchd/README.md](../../deploy/launchd/README.md) — installing, operating, and changing the launchd agents.
- [What is a Temporal Worker?](https://docs.temporal.io/workers) — worker concepts, identity, and configuration.
- [Worker deployment and performance](https://docs.temporal.io/best-practices/worker) — production deployment, monitoring metrics, and task queue separation.
- [Temporal Best Practices](https://docs.temporal.io/best-practices) — design patterns, testing, and production readiness.
