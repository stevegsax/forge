# Workers

## Overview

A Forge worker is a long-running process that polls the Temporal server for queued workflows and executes activities (LLM calls, context assembly, validation, git operations). The worker itself is stateless — all workflow state lives in the Temporal server. This means multiple workers can run on different machines, and if a worker crashes, another can pick up where it left off.

Start a worker:

```bash
forge worker
forge worker --temporal-address temporal.example.com:7233
```

The worker polls the `forge-task-queue` task queue. All Forge workflows and activities are registered on this single queue.

## Startup and Configuration

Each worker main is a composition root (T3.6): it reads a frozen settings object **once** at startup, then builds one database engine, one Anthropic SDK client, and one S3 blob client for the whole process, injecting them into the class-based activities. The **ocr worker** additionally builds one required Mistral OCR client; the forge worker builds no Mistral client — its batch transport is anthropic-only (T4.2). Environment variables are read only through those settings, so a misconfiguration is a **startup failure, not a mid-workflow surprise**:

- The **forge worker** fails fast if `FORGE_DB_URL` is unset.
- The **ocr worker** fails fast if `FORGE_DB_URL` or `FORGE_OCR_S3_BUCKET` is unset (the bucket check moved to startup in T3.6; it used to surface at first OCR use).
- `MISTRAL_API_KEY` is a **fail-fast for the ocr worker** — it submits and polls its own Mistral batches, so an unset key aborts startup (T4.2). The forge worker never reads it (anthropic-only transport).

The **ocr worker** additionally installs and reconciles the `ocr-batch-tracker` Temporal Schedule at startup — one `OcrBatchTrackerWorkflow` run every 120s (overlap=SKIP) — before it serves work, and **aborts startup if it cannot** install it. Without the Schedule the store children never receive their status hints, so the worker would be unable to finish a batch (T4.4).

To check tracker health, run `uv run --package ocr ocr tracker-status`: a Temporal-free DB probe that prints the current GMT time, the `ocr_tracker_heartbeat` fields, the current live-job count, and a `fresh`/`stale`/`never-ran` verdict. Exit codes are script-friendly: 0 fresh, 1 stale or never-ran with no live jobs, 2 stale **with** live jobs waiting (work is queued but the tracker is not completing cycles — check the worker and the Mistral API), 3 probe error (missing `FORGE_DB_URL` or unreachable database — the probe could not answer; never collides with the verdict codes). Staleness threshold: `--stale-after` seconds, default 300 (2–3 tracker cycles). It reads whatever `FORGE_DB_URL` points at; a bare interactive shell carries neither that variable nor `FORGE_ENV` — source a profile with the [environment guard](#environment-guard) pattern first, or point `FORGE_ENV=dev` at the local stack's dev database on port 5434.

The full environment-variable reference is in [DEPLOYMENT.md](DEPLOYMENT.md#configuration).

### Environment guard

Every worker and CLI is fronted by an explicit-environment guard (D102): it reads `FORGE_ENV` (`prod` / `dev` / `test`) and **fails if it is unset** — there is no default, and a guard failure exits **78** (`EX_CONFIG`, distinct from every command's own exit codes). This makes "which database am I about to touch" unambiguous, and reaching production a deliberate act rather than the result of an unset variable falling through to a default.

- **Workers** resolve `FORGE_ENV` before any settings or store setup and log the resolved env (`env=prod`) at startup. The launchd plists declare `FORGE_ENV=prod` and `FORGE_PROD_ACK=yes` in `EnvironmentVariables` — the plists are the explicit production act — while `run-worker.sh` loads the matching profile `~/.config/forge/envs/prod.env`, aborting if the profile's `FORGE_ENV_TAG` disagrees or the file is not chmod 600.
- **Production** additionally requires `FORGE_PROD_ACK=yes`, which the profile files never set — only the plist or an interactive shell — so sourcing a profile can never by itself grant production access.
- **Interactive** commands must declare their environment. Sourcing a profile needs `set -a` so the values (and the tag) are exported — a plain `source` does not export, and the guard rejects an unexported tag by design:

  ```bash
  set -a; source ~/.config/forge/envs/prod.env; set +a
  export FORGE_ENV=prod FORGE_PROD_ACK=yes
  ```

- **Non-prod work** uses the `dev` profile (`~/.config/forge/envs/dev.env`, `FORGE_ENV_TAG=dev`, no ack): `set -a; source …/envs/dev.env; set +a; export FORGE_ENV=dev`. It points at the local `forge_dev` database and MinIO, so no interactive command reaches production without `FORGE_ENV=prod` and the ack.
- **Production workers additionally require a clean, committed checkout** (D103). After resolving `FORGE_ENV=prod` and before touching anything, each worker checks its checkout's git version and exits **78** if it is `-dirty` or cannot be determined, naming the directory and pointing at `deploy/prod-deploy.sh`. The mechanism it defends: a worker execs `uv run` inside its checkout, so that tree's contents at launch *are* the running code, and an uncommitted edit would ship with nothing able to say afterwards what ran. Dev and test are deliberately exempt — editing a checkout under a running dev worker is the point of the staging lane. Deploy production with `make prod-deploy REF=<ref>` (see [DEPLOYMENT.md](DEPLOYMENT.md#the-production-checkout-d103)).
- **The `--env` flag** loads a profile without a manual `set -a; source`: every CLI accepts `--env <name-or-path>` in **either position** — before or after the subcommand (`uv run --package ocr ocr tracker-status --env dev` and `uv run --package ocr ocr --env dev tracker-status` are equivalent; given at both levels, the value after the subcommand wins). A bare shell with no `--env` fails the guard when the command runs. A bare name resolves to `$XDG_CONFIG_HOME/forge/envs/<name>.env` (and sets `FORGE_ENV` to that name); a path — or any value ending `.env` — is read verbatim and takes `FORGE_ENV` from the file's `FORGE_ENV_TAG`. The profile parser tolerates a leading `export`, strips one surrounding quote pair, and expands `${VAR}` references (braced form only, against the current environment) — never any command execution. It never supplies `FORGE_PROD_ACK`, so `--env prod` still fails unless the ack is exported separately; a missing file, malformed line, or path-form profile without a `FORGE_ENV_TAG` exits 78.

### Staging lane (dev namespace)

Environments are separated by **server**: dev is `127.0.0.1:7236`, production is `127.0.0.1:7243` (org servers owned by sax-temporal). Inside each, a namespace scopes task queues, schedules, and workflow ids, so a worker or CLI in one cannot see — or poll — another's work. Dev runs in `forge-dev`, production in `forge-prod`.

Neither value is configured. `resolve_temporal_target` derives both from `FORGE_ENV` immediately before every connect and returns them as one frozen `TemporalTarget`, so the address and the namespace cannot disagree. `FORGE_TEMPORAL_NAMESPACE` no longer exists. `FORGE_TEMPORAL_ADDRESS` survives only as an override, and dev/prod refuse any value that is not their own server — pointing a dev process at `:7243` exits 78 with a message naming the fix, rather than connecting. (`FORGE_ENV=test` is the exception: its server is an ephemeral per-job container on an arbitrary port, so it *must* supply an address, and any address is accepted.)

The namespace name is the backstop for everything the code cannot reach — the `temporal` CLI, one-off scripts, an operator in a shell. Names are `<slug>-<env>`, and the bare slug and `default` exist on no server, so a hand-typed `-n forge` fails with "namespace not found" everywhere instead of landing in production. Convention: `sax-temporal/docs/namespaces.md`.

Commands that never connect to Temporal (`tracker-status`, `ocr migrate`, pbook's direct-DB commands) are unaffected by any of this.

> **Deployed state lags the code.** Production and the dev lane still run a pre-cutover commit against the legacy `:7233` server in the `default` / `forge-dev` namespaces. The cutover is operational work — create `forge-dev` on `:7236` and `forge-prod` on `:7243`, update the deployed profiles under `~/.config/forge/envs/`, drain `:7233`, deploy. Deploying the new code before those namespaces exist fails every worker closed at startup. Ledger: `sax-temporal/docs/namespaces.md`, "Migration status".

Each namespace is created once per server, out of band:

```bash
temporal operator namespace create --retention 168h -n forge-dev  --address 127.0.0.1:7236
temporal operator namespace create --retention 720h -n forge-prod --address 127.0.0.1:7243
```

(sax-temporal's `make namespace ENV=<env> NS=<slug>-<env>` does this with the right retention per environment.)

The dev profile (`~/.config/forge/envs/dev.env`, from `deploy/launchd/envs/dev.env.example`) declares `FORGE_ENV_TAG=dev` and nothing Temporal-related — the target follows from it. Start a dev worker the same way as any other — `--env dev` or a sourced profile — and it connects to `:7236` in `forge-dev`:

```bash
uv run forge worker --env dev            # forge worker in forge-dev
uv run --package ocr ocr worker --env dev  # ocr worker in forge-dev
```

The dev ocr worker installs its own `ocr-batch-tracker` Schedule inside `forge-dev`, separate from production's in `forge-prod`. A dev CLI submits into `forge-dev` too, so it only ever reaches the dev workers.

**A functional lane needs the forge worker even for pure-OCR flows.** ocr's ledger writes (`persist_block` → `forge-task-queue`) are cross-queue activity calls, so with only a dev ocr worker running, an OCR submission blocks at its first ledger persist and the tracker cannot route hints (the live job has no routable `batch_jobs` row — the tracker logs a warning and skips it). Start the pair: `make dev-worker` and `make dev-worker WORKER=forge` (add `WORKER=pbook` if exercising ingestion). Restart one with `make dev-worker-restart WORKER=<name>` — the dev lane restarts independently of production, and `make workers-restart` is production-only by construction (it signals launchd-resolved pids, never command-line patterns, so it cannot touch the tmux workers).

## Checking Whether Workers Are Running

Workers do not write a PID file or store local state. Because workers can run on any machine that can reach the Temporal server, the Temporal server itself is the source of truth.

### Temporal CLI

List active pollers on the Forge task queue:

```bash
temporal task-queue describe --task-queue forge-task-queue
```

This shows all workers currently polling the queue, including their identity, last access time, and the workflow/activity types they handle. If the list is empty, no workers are running.

### Temporal Web UI

The Temporal Web UI (default `http://localhost:8233`) shows active workers under the task queue view. Navigate to the `forge-task-queue` task queue to see connected pollers.

### Local process check

If you only need to check the local machine:

```bash
pgrep -f "forge worker"
```

## Worker Identity

Each worker reports an identity string to the Temporal server. By default, the Python SDK sets this to `{pid}@{hostname}`. This identity appears in workflow history events and in the task queue poller list, so you can trace which worker executed a given activity.

**The base names the lane.** Every supervised worker is given a base identity that says which lane it belongs to, so the identity agrees with the other two things that already encode the lane — `FORGE_ENV` and the Temporal namespace — instead of quietly contradicting them:

| Lane | Started by | Base identity | `FORGE_ENV` | Namespace |
| --- | --- | --- | --- | --- |
| production | launchd (`install.sh`) | `prod-forge-worker-1`, `prod-forge-worker-2`, `prod-ocr-worker`, `prod-pbook-worker` | `prod` | `default` (→ `forge-prod` at cutover) |
| staging | tmux (`make dev-worker`) | `dev-forge-worker`, `dev-ocr-worker`, `dev-pbook-worker` | `dev` | `forge-dev` |

The dev base is also the tmux session name and the row `make workers-status` prints, so one string identifies a dev worker everywhere you might meet it. (These bases replaced a `desktop-` prefix that dated from the D99 EC2 retirement: once the desktop became the only host, "desktop" distinguished nothing, while the lane distinguishes the thing an operator actually needs to know before acting.)

Set a base identity by hand via the `--worker-identity` flag (present on `forge worker`, `ocr worker`, and `pbook worker`) or the `FORGE_WORKER_IDENTITY` environment variable:

```bash
forge worker --worker-identity "prod-forge-worker-3"
FORGE_WORKER_IDENTITY="dev-forge-worker" forge worker --env dev
```

When omitted, the SDK default (`{pid}@{hostname}`) is used. Either way the identity is the *base* — every worker appends the git version it was launched from, as described next.

### Which code is a worker running?

Each worker stamps the git version of the tree it was launched from onto its identity, giving `<base>@<commit>`:

| Worker | Identity |
| --- | --- |
| production forge worker 1 (launchd) | `prod-forge-worker-1@bb64d88` |
| staging ocr worker (tmux) | `dev-ocr-worker@bb64d88` |
| a worker with no base set | `12345@buchla.local@bb64d88` |
| any worker launched from a modified tree | `prod-forge-worker-1@bb64d88-dirty` |

The server is the authority — ask it which code is polling a queue:

```bash
temporal task-queue describe --task-queue forge-task-queue
temporal task-queue describe --task-queue ocr-task-queue
temporal task-queue describe --task-queue pbook-task-queue
temporal task-queue describe --task-queue forge-task-queue --namespace forge-dev   # dev lane
```

This exists because a Python worker binds its code at import: the modules loaded when the process started are the modules it runs until it is restarted, while the checkout it was launched from keeps moving — a worker `exec`s `uv run` straight out of that checkout (D99). So the stamp records the tree **as it was at that worker's launch**. It does not change when you commit, pull, or edit; a worker picks up new code, and a new stamp, only at its next restart (`make prod-deploy REF=<ref>` for production, `make dev-worker-restart WORKER=<name>` for the dev lane). A worker started before this change carries no version suffix at all.

**A production identity should always be a clean SHA.** Since D103 the prod workers run the pinned `~/repos-sax/forge-prod` checkout and refuse to start on a dirty or unverifiable one (exit 78), so `prod-forge-worker-1@bb64d88` is the expected shape and `prod-forge-worker-1@bb64d88-dirty` should be unreachable. If you ever see one, something started a worker outside `prod-deploy.sh` — treat it as an incident: capture `git -C ~/repos-sax/forge-prod status` before cleaning, since that tree is the only record of what actually ran.

Changing a *base* identity is a different operation from deploying code. The production bases live in each launchd agent's `ProgramArguments`, and a restart relaunches from the job definition launchd already has loaded — so a rewritten plist is adopted only when `deploy/launchd/install.sh` (run from the prod checkout) boots the agents out and bootstraps them again. The command sequence, the opt-in-flag trap, and the drain caveat are in [deploy/launchd/README.md](../../deploy/launchd/README.md#changing-a-plist-worker-identities-environment-keepalive). The dev lane has no such step: `make dev-worker-restart WORKER=<name>` rebuilds the tmux command line from the Makefile.

A `-dirty` suffix means the tree had uncommitted changes at launch — modified tracked files or untracked files (`git status --porcelain` was non-empty). Since the worker runs the live tree, the commit alone then does not describe the running code: `bb64d88` names the last commit and whatever was uncommitted at launch is loaded on top of it. Reproducing that worker's behavior needs the commit plus those local changes, so treat a `-dirty` production poller as a signal to restart it from a clean tree once the work is committed.

If the version cannot be determined (no `git` on `PATH`, the directory is not a repository, the command fails or times out), nothing is stamped and the identity is exactly the base. Version discovery is best-effort by design and can never keep a worker from starting.

## Scaling

Multiple workers can poll the same task queue from different machines. The Temporal server distributes work across them automatically. Workers are stateless, so scaling out is as simple as starting more `forge worker` processes pointed at the same Temporal server.

Deploy at least two workers per task queue for redundancy.

## Further Reading

- [What is a Temporal Worker?](https://docs.temporal.io/workers) — Worker concepts, identity, and configuration.
- [Worker deployment and performance](https://docs.temporal.io/best-practices/worker) — Production deployment, monitoring metrics, and task queue separation.
- [Worker Versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning) — Pinning workflow versions to specific worker builds for safe rollouts.
- [Temporal Best Practices](https://docs.temporal.io/best-practices) — General best practices covering design patterns, testing, and production readiness.
