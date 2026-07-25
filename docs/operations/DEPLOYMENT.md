# Deployment: Local-First Forge on an Always-On Desktop

Forge deploys to a single always-on macOS desktop (D99, D102). Temporal is
**self-hosted in the local podman stack** with persistence in that stack's
Postgres; the forge and pbook **workers run as launchd-supervised host
processes** from the repo checkout; Forge's and pbook's application state
lives in **local Postgres** — the `forge` and `pbook` databases in that same
podman stack, beside Temporal's (D102, rehomed off Supabase 2026-07-22);
OCR/batch blobs live in **S3** with only references in the database. The
desktop holds every database of record plus disposable work products
(worktrees, cloned repos, logs); offsite durability comes from a nightly
`pg_dump` → S3 and from versioned S3 blobs.

> Scope: single-operator, low-volume, one machine. There is **no remote
> access**: Temporal binds to loopback only. The EC2/mTLS deployment this
> replaces was removed by D99 (its Terraform, SSM bootstrap, gateway, and
> cert tooling live in git history; remote access may return later).
> High availability is explicitly out of scope — see
> [Always-on and availability](#always-on-and-availability).

## Architecture

```text
┌────────────────────── always-on macOS desktop ───────────────────────┐
│                                                                       │
│  podman (deploy/local-stack)          launchd (deploy/launchd)        │
│  ┌──────────────────────────┐   ┌───────────────────────────────────┐ │
│  │ temporal :7233 (loopback)│◄──┤ forge-worker-1 / -2 → forge worker │ │
│  │   persistence ─┐         │   │   FORGE_TEMPORAL_ADDRESS=          │ │
│  │ temporal-ui :8233        │   │     127.0.0.1:7233                 │ │
│  │ postgres :5433 ◄┘        │   │ pbook-worker → pbook worker (opt)  │ │
│  │   forge + pbook +        │   │ ocr-worker → ocr worker (opt)      │ │
│  │   forge_dev + temporal + │   │   env: envs/$FORGE_ENV.env         │ │
│  │   temporal_visibility    │   └───────────────┬───────────────────┘ │
│  │ minio :9002/:9003 (dev)  │                   │                     │
│  └──────────────────────────┘                   │                     │
│  worktrees, cloned repos, logs (disposable)     │ outbound HTTPS      │
└─────────────────────────────────────────────────┼─────────────────────┘
                                                  ▼
                              S3 (OCR/batch blobs) · Anthropic · Mistral
```

**Why this shape.** The operator's desktop is always on, so the EC2 box
that existed to host Temporal and the workers was pure overhead (D99).
Temporal persists locally because with the engine local, remote
persistence would put the internet inside every workflow tick; the
application databases sit in that same local Postgres (D102) — their
low-rate business writes don't carry the per-tick latency cost, and a
nightly `pg_dump` → S3 supplies the offsite durability that keeping them
managed used to buy. Blobs stay on S3, which is already offsite.

### What runs where

| Component | Where | Lifetime | Notes |
| --- | --- | --- | --- |
| Temporal server + UI | podman (`deploy/local-stack`) | Always up | Persistence → the stack's Postgres (`temporal`, `temporal_visibility`) |
| Stack Postgres | podman, host port 5433 | Always up | Temporal (`temporal`, `temporal_visibility`) + the `forge`/`pbook` app databases (D102) + the `forge_dev` dev database |
| MinIO | podman, host ports 9002/9003 | Dev only | Local S3 surface for dev; production blobs go to real S3 |
| `forge worker` (×2) | launchd host processes | Always up | Poll `forge-task-queue`; need git/uv/ruff and repos on disk |
| `pbook worker` | launchd host process | Optional | Only if transcript ingestion is used; polls `pbook-task-queue` |
| `ocr worker` | launchd host process | Optional | Only if OCR is used (`install.sh --with-ocr`); polls `ocr-task-queue` |
| `forge` / `pbook` CLIs | Host shell | On demand | Connect to `127.0.0.1:7233` |
| Forge store | local Postgres (`forge` database) | Always up | interactions, runs, playbooks, batch_jobs, OCR records; forge + ocr Alembic chains in `public` |
| pbook store | local Postgres (`pbook` database) | Optional | `PBOOK_DATABASE_URL` (set in the prod profile since D102); Postgres-only |
| Blobs | S3 | Always up | Image/file bytes; DB holds the S3 key; lifecycle policy in `deploy/s3/` |

The workers run on the **host** (not in a container) because the forge
worker is a build agent: it needs `git`, the target repositories, `uv`,
and `ruff`/test tooling on a writable filesystem. (Containerizing them
became *possible* when the workspace went self-contained in T2.1
increment 2, but the build-agent rationale keeps them on the host.)

## External dependencies

| Dependency | Purpose | Requirement |
| --- | --- | --- |
| S3 bucket | OCR/batch blobs + nightly DB backups | Creds via `~/.aws` or the env file; lifecycle policy: [deploy/s3/](../../deploy/s3/) |
| Anthropic API | All Forge LLM calls; pbook extraction/review | `ANTHROPIC_API_KEY` |
| Mistral API | OCR (ocr app only) | `MISTRAL_API_KEY` — **required by the ocr worker** (fail-fast at startup, T4.2); the forge worker never reads it |
| OpenAI API | pbook embeddings | `OPENAI_API_KEY` (only if pbook used) |
| GitHub | Clone/push the repos Forge operates on | The operator's normal git credentials |

## Packaging

The forge repo root is a uv workspace (D98) and is **self-contained**:
every internal package is a workspace member, so one clone and one
`uv sync` produce the whole runtime — no sibling checkouts.

```toml
[tool.uv.workspace]
members = ["apps/pbook", "apps/ocr", "libs/sax-platform"]

[tool.uv.sources]
pbook = { workspace = true }
sax-platform = { workspace = true }
```

```text
forge/                     # the workspace root — this is the whole deployment
├── apps/pbook/            # knowledge playbook service
├── apps/ocr/              # document OCR app (member, not a forge dependency)
└── libs/sax-platform/     # model-tier registry, LLM client, Mistral OCR,
                            # shared wire contracts + platform primitives
                            # (absorbed libs/forge-contracts at T3.4; sax-llm
                            # deleted at T3.5 — four packages now)
```

```bash
git clone <forge> && cd forge
uv sync --all-packages          # installs forge + all members (ocr included)
uv run forge --version
uv run pbook --help             # the same venv serves every worker
uv run --package ocr ocr --help # ocr is not a forge dependency — use --package
```

Pin deployments to a known-good commit or tag of this one repo.

## Deployment process

### 1. Bring up the stack

```bash
podman machine start          # first time: podman machine init
make stack-up                 # Postgres + Temporal + UI + MinIO
```

Temporal's auto-setup creates its databases in the stack's Postgres on
first boot. UI: `http://localhost:8233`. Details, ports, and overrides:
[deploy/local-stack/README.md](../../deploy/local-stack/README.md).

### 2. Configure the worker environment

Environment is selected by `FORGE_ENV` (`prod` / `dev` / `test`, **no
default**) and loaded from a matching per-environment profile (D102):

```bash
mkdir -p ~/.config/forge/envs
cp deploy/launchd/envs/prod.env.example ~/.config/forge/envs/prod.env
chmod 600 ~/.config/forge/envs/prod.env   # then fill in the CHANGEMEs
```

The profile replaces the EC2-era SSM plumbing: `FORGE_DB_URL` (the local
`forge` database), `PBOOK_DATABASE_URL` (the local `pbook` database),
`ANTHROPIC_API_KEY`, `FORGE_OCR_S3_BUCKET`, `FORGE_BACKUP_S3_BUCKET`, and
friends, plus its own `FORGE_ENV_TAG=prod`. The launchd wrapper parses it
without shell evaluation (so URLs with `&` are safe) and refuses to start
if the tag disagrees with `FORGE_ENV` or the file is not chmod 600.
Production additionally requires `FORGE_PROD_ACK=yes` — set only by the
plist (or an interactive shell), never by a profile — so sourcing a
profile can never by itself grant production access. The guard aborts with
exit **78** if `FORGE_ENV` is unset. See
[WORKERS.md](WORKERS.md#environment-guard) for the interactive sourcing
pattern.

### 3. Migrations

The forge worker runs its own Alembic migrations at startup (advisory-locked
against the shared database); the ocr worker likewise applies its own `ocr_*`
chain at startup. The pbook worker applies its own chain (custom
`pbk_alembic_version` table) to head at startup too — but only when
`PBOOK_DATABASE_URL` is set (unset → store disabled, migration skipped; the
prod profile sets it since D102). Run `uv run pbook migrate` manually only to
migrate without starting the worker.

### 4. Install the launchd agents

```bash
deploy/launchd/install.sh             # add --with-pbook for ingestion
```

Installs the stack agent (RunAtLoad: podman machine + `stack-up` at
login) and the KeepAlive worker agents. Operation, logs, and restart
commands: [deploy/launchd/README.md](../../deploy/launchd/README.md).

### 5. Verify

```bash
podman ps                                            # 4 containers healthy
tail -f ~/.local/state/forge/logs/forge-worker-1.log # "worker started", polling
uv run forge status --limit 3                        # CLI → Temporal + store
open http://localhost:8233                           # workers visible under Workers
```

### 6. Apply the S3 lifecycle policy (once)

```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket "$FORGE_OCR_S3_BUCKET" \
  --lifecycle-configuration file://deploy/s3/lifecycle.json
```

Rationale and what it deliberately does not expire:
[deploy/s3/README.md](../../deploy/s3/README.md).

## Configuration

Worker/CLI environment (the launchd agents read these from
`~/.config/forge/envs/$FORGE_ENV.env`, selected by `FORGE_ENV`):

| Variable | Purpose | Production value |
| --- | --- | --- |
| `FORGE_ENV` | **Required, no default.** Selects the profile and gates every command and worker; unset → exit 78 (D102) | `prod` |
| `FORGE_PROD_ACK` | **Required for `prod`.** The explicit production acknowledgement; set by the plist (or interactive shell), never by a profile | `yes` |
| `FORGE_ENV_TAG` | Declared **inside** each profile; the loader aborts if it disagrees with `FORGE_ENV` | `prod` |
| `FORGE_TEMPORAL_ADDRESS` | Temporal frontend | `127.0.0.1:7233` |
| `FORGE_TEMPORAL_NAMESPACE` | Temporal namespace; coherence-checked against `FORGE_ENV` before every connect (prod must use `default`, dev/test must not — the staging lane's isolation; see [WORKERS.md](WORKERS.md#staging-lane-dev-namespace)) | `default` (prod, by omission) / `forge-dev` (dev profile) |
| `FORGE_DB_URL` | **Required.** Forge store (local `forge` database). Unset → hard error | `postgresql+psycopg://forge:…@127.0.0.1:5433/forge` |
| `FORGE_OCR_S3_BUCKET` | S3 bucket for blobs. The **ocr worker fails fast at startup if unset** (T3.6; previously a first-use error); forge needs it for OCR/batch-blob work | bucket name |
| `FORGE_OCR_S3_PREFIX` | Optional key prefix for blobs | e.g. `ocr/` |
| `FORGE_LOG_DIR` | App log directory (empty = no file logging) | `$XDG_STATE_HOME/forge/logs` |
| `FORGE_OTEL_EXPORTER` | `console`/`otlp_grpc`/`otlp_http`/`none` (code default `console`) | `none` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint for the `otlp_*` exporters — the standard OpenTelemetry SDK var (forge's own `FORGE_OTEL_ENDPOINT` was deleted at T3.6) | unset (exporter is `none`) |
| `FORGE_WORKER_IDENTITY` | *Base* worker identity in Temporal (the launch-time git version is appended); read by all three workers' `--worker-identity` option | set per lane: launchd agents `prod-forge-worker-1/2`, `prod-ocr-worker`, `prod-pbook-worker`; `make dev-worker` sets `dev-<app>-worker` |
| `ANTHROPIC_API_KEY` | Anthropic SDK auth | key |
| `MISTRAL_API_KEY` | OCR (ocr app). **Required by the ocr worker** — it submits and polls its own Mistral batches and fails fast at startup without it (T4.2). The forge worker never reads it (anthropic-only transport) | key |
| `OPENAI_API_KEY` | pbook embeddings | key (if pbook used) |
| `PBOOK_DATABASE_URL` | pbook Postgres store (local `pbook` database); set in the prod profile since D102 | `postgresql+psycopg://forge:…@127.0.0.1:5433/pbook` |
| `FORGE_BACKUP_S3_BUCKET` | S3 bucket for the nightly `pg_dump` (D102) | bucket name (currently the blobs bucket) |
| `AWS_*` | S3 auth if not using `~/.aws` | keys/region |

The pbook worker takes no Temporal env var — it connects to its `localhost:7233`
default (the same loopback frontend), so there is nothing to set for it here.

For the **dev** stack surfaces (local `forge_dev` Postgres, MinIO), see
[deploy/local-stack/.env.example](../../deploy/local-stack/.env.example)
— and note its footgun warning: the dev `AWS_*`/`FORGE_DB_URL` exports
override production values in the same shell.

## Supabase (retired/frozen)

Supabase was the application store of record until 2026-07-22; D102 rehomed
`forge` and `pbook` onto the local stack and retired it. The final
pre-cutover dump is retained — the migration dumps and their
Supabase-specific restore transforms (`extensions.vector` → `public.vector`,
dropped RLS toggles, dropped the PG17-only `SET transaction_timeout`) are
kept as the historical record in the T0.9 task file — and the Supabase
project is left frozen as a fallback until the owner decommissions it.
Nothing in the running system reads it.

## Always-on and availability

The desktop is the availability story, accepted by D99. Batch polling
(D88's timer loops) stalls while the machine sleeps — sleep silently
pauses every in-flight workflow until wake. System sleep is therefore
disabled on AC power (applied 2026-07-16):

```bash
sudo pmset -c sleep 0 displaysleep 10 disksleep 0
```

Verify with `pmset -g custom` — under `AC Power`, `sleep` must be `0`.
That is the load-bearing setting; `displaysleep` and `disksleep` only
affect the screen and (on this NVMe hardware) latency, not uptime.
Residual exposure is reboots, unplugging, and hardware — nothing here
protects against those.

After a reboot: log in once — the launchd agents bring up the podman
machine, the stack, and the workers without manual steps.

## Backup

- **App databases** (`forge`/`pbook`, state of record): the nightly
  `com.saxcapital.db-backup` launchd agent (03:30) `pg_dump`s both to
  `s3://$FORGE_BACKUP_S3_BUCKET/db-backups/` (D102), replacing Supabase's
  managed backups. Run on demand with `make backup-app-dbs`.
- **Stack Postgres** (Temporal workflow histories, plus the app databases'
  data directory): rides the machine's backup discipline (Time Machine
  covers `$XDG_DATA_HOME/forge/postgres`); losing it loses in-flight
  workflow state, not records of completed work.
- **S3**: versioned; noncurrent versions expire per the lifecycle policy.

## History

This replaced the EC2 + mTLS deployment (D99, 2026-07-16). That design —
Terraform, SSM secret bootstrap, an nginx mutual-TLS gateway for remote
CLIs, and Supabase-backed Temporal — survives in git history
(`deploy/terraform`, `deploy/scripts`, `deploy/certs`, `deploy/client`,
`docs/operations/SECURE-REMOTE-ACCESS.md` before this change). The
client-side TLS env handling — now in `sax_platform.temporal.client`
(absorbed from forge-contracts at T3.4) — remains in the code, dormant,
should remote access return.
