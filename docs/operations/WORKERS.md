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
- **The `--env` flag** loads a profile without a manual `set -a; source`: every CLI accepts `--env <name-or-path>` in **either position** — before or after the subcommand (`uv run --package ocr ocr tracker-status --env dev` and `uv run --package ocr ocr --env dev tracker-status` are equivalent; given at both levels, the value after the subcommand wins). A bare shell with no `--env` fails the guard when the command runs. A bare name resolves to `$XDG_CONFIG_HOME/forge/envs/<name>.env` (and sets `FORGE_ENV` to that name); a path — or any value ending `.env` — is read verbatim and takes `FORGE_ENV` from the file's `FORGE_ENV_TAG`. The profile parser tolerates a leading `export`, strips one surrounding quote pair, and expands `${VAR}` references (braced form only, against the current environment) — never any command execution. It never supplies `FORGE_PROD_ACK`, so `--env prod` still fails unless the ack is exported separately; a missing file, malformed line, or path-form profile without a `FORGE_ENV_TAG` exits 78.

### Staging lane (dev namespace)

Production and dev share one local Temporal server, so isolation between them is by **namespace**: a Temporal namespace scopes task queues, schedules, and workflow ids, so a worker or CLI that connects to a different namespace cannot see — or poll — another namespace's work. Production runs in the `default` namespace (unchanged); dev runs in `forge-dev`.

The pairing is enforced, not just conventional. `FORGE_TEMPORAL_NAMESPACE` (read by `TemporalSettings`, defaulting to `default`) is checked against `FORGE_ENV` by `require_namespace_coherence` immediately before every connect: `FORGE_ENV=prod` **must** use `default`, and `FORGE_ENV=dev`/`test` **must not** — an incoherent pairing fails fast with an actionable message (workers crash at startup before touching a database; CLIs exit 78). So a dev process can never silently poll production's queues, and prod stays zero-config (it sets nothing and gets `default`). Commands that never connect to Temporal (`tracker-status`, `ocr migrate`, pbook's direct-DB commands) are unaffected by the namespace entirely.

The `forge-dev` namespace is created once per server, out of band (already done on this machine, with 72h retention):

```bash
temporal operator namespace create --retention 72h -n forge-dev
```

The dev profile (`~/.config/forge/envs/dev.env`, from `deploy/launchd/envs/dev.env.example`) sets `FORGE_TEMPORAL_NAMESPACE=forge-dev` alongside `FORGE_ENV_TAG=dev`. Start a dev worker the same way as any other — `--env dev` or a sourced profile — and it connects into `forge-dev`:

```bash
uv run forge worker --env dev            # forge worker in forge-dev
uv run --package ocr ocr worker --env dev  # ocr worker in forge-dev
```

The dev ocr worker installs its own `ocr-batch-tracker` Schedule inside `forge-dev`, separate from production's in `default`. A dev CLI submits into `forge-dev` too, so it only ever reaches the dev workers.

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

The default identity is adequate for single-machine development. In multi-machine or containerized deployments, the default is often unhelpful (container PIDs are always `1`, cloud hostnames are random strings). In those environments, set a custom identity that maps back to the machine or deployment unit (e.g., ECS task ID, k8s pod name).

Set a custom identity via the `--worker-identity` flag or `FORGE_WORKER_IDENTITY` environment variable:

```bash
forge worker --worker-identity "worker-us-east-1a-01"
FORGE_WORKER_IDENTITY="pod-abc123" forge worker
```

When omitted, the SDK default (`{pid}@{hostname}`) is used. Either way the identity is the *base* — every worker appends the git version it was launched from, as described next.

### Which code is a worker running?

Each worker stamps the git version of the tree it was launched from onto its identity, giving `<base>@<commit>`:

| Worker | Identity |
| --- | --- |
| forge (launchd) | `desktop-forge-worker-1@bb64d88` |
| ocr / pbook (no base set) | `12345@desktop@bb64d88` |
| any worker launched from a modified tree | `desktop-forge-worker-1@bb64d88-dirty` |

The server is the authority — ask it which code is polling a queue:

```bash
temporal task-queue describe --task-queue forge-task-queue
temporal task-queue describe --task-queue ocr-task-queue
temporal task-queue describe --task-queue pbook-task-queue
temporal task-queue describe --task-queue forge-task-queue --namespace forge-dev   # dev lane
```

This exists because a Python worker binds its code at import: the modules loaded when the process started are the modules it runs until it is restarted, while the working tree it was launched from keeps moving — the launchd and tmux workers `exec uv run` straight out of the live repo (D99). So the stamp records the tree **as it was at that worker's launch**. It does not change when you commit, pull, or edit; a worker picks up new code, and a new stamp, only at its next restart (`make workers-restart` for production, `make dev-worker-restart WORKER=<name>` for the dev lane). A worker started before this change carries no version suffix at all.

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
