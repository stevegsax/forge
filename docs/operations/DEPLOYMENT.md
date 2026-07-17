# Deployment: Local-First Forge on an Always-On Desktop

Forge deploys to a single always-on macOS desktop (D99). Temporal is
**self-hosted in the local podman stack** with persistence in that stack's
Postgres; the forge and pbook **workers run as launchd-supervised host
processes** from the repo checkout; Forge's and pbook's application state
lives in **Supabase Postgres**; OCR/batch blobs live in **S3** with only
references in the database. The desktop holds Temporal's workflow
histories and disposable work products (worktrees, cloned repos, logs) —
application state of record stays managed and offsite.

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
│  │   forge_dev + temporal + │   │   env: ~/.config/forge/forge.env   │ │
│  │   temporal_visibility    │   └───────────────┬───────────────────┘ │
│  │ minio :9002/:9003 (dev)  │                   │                     │
│  └──────────────────────────┘                   │ outbound HTTPS      │
│  worktrees, cloned repos, logs (disposable)     │                     │
└─────────────────────────────────────────────────┼─────────────────────┘
                                                  ▼
                              Supabase Postgres (forge + pbook state)
                              S3 (OCR/batch blobs) · Anthropic · Mistral
```

**Why this shape.** The operator's desktop is always on, so the EC2 box
that existed to host Temporal and the workers was pure overhead (D99).
Temporal persists locally because with the engine local, Supabase
persistence would put the internet inside every workflow tick; the
application stores stay on Supabase/S3 because offsite durability matters
for the state of record and buys nothing for replayable orchestration
state a desktop backup can cover.

### What runs where

| Component | Where | Lifetime | Notes |
| --- | --- | --- | --- |
| Temporal server + UI | podman (`deploy/local-stack`) | Always up | Persistence → the stack's Postgres (`temporal`, `temporal_visibility`) |
| Stack Postgres | podman, host port 5433 | Always up | Temporal persistence + the `forge_dev` dev database |
| MinIO | podman, host ports 9002/9003 | Dev only | Local S3 surface for dev; production blobs go to real S3 |
| `forge worker` (×2) | launchd host processes | Always up | Poll `forge-task-queue`; need git/uv/ruff and repos on disk |
| `pbook worker` | launchd host process | Optional | Only if transcript ingestion is used; polls `pbook-task-queue` |
| `ocr worker` | launchd host process | Optional | Only if OCR is used (`install.sh --with-ocr`); polls `ocr-task-queue` |
| `forge` / `pbook` CLIs | Host shell | On demand | Connect to `127.0.0.1:7233` |
| Forge store | Supabase (`forge` database) | Always up | interactions, runs, playbooks, batch_jobs, OCR records |
| pbook store | Supabase (`pbook` schema) | Optional | `PBOOK_DATABASE_URL`; Postgres-only |
| Blobs | S3 | Always up | Image/file bytes; DB holds the S3 key; lifecycle policy in `deploy/s3/` |

The workers run on the **host** (not in a container) because the forge
worker is a build agent: it needs `git`, the target repositories, `uv`,
and `ruff`/test tooling on a writable filesystem. (Containerizing them
became *possible* when the workspace went self-contained in T2.1
increment 2, but the build-agent rationale keeps them on the host.)

## External dependencies

| Dependency | Purpose | Requirement |
| --- | --- | --- |
| Supabase Postgres | Forge store (and pbook store if used) | `FORGE_DB_URL`; direct (non-transaction-pooled) connection |
| S3 bucket | OCR/batch blobs | Creds via `~/.aws` or the env file; lifecycle policy: [deploy/s3/](../../deploy/s3/) |
| Anthropic API | All Forge LLM calls; pbook extraction/review | `ANTHROPIC_API_KEY` |
| Mistral API | OCR pipeline | `MISTRAL_API_KEY` (required by the ocr worker) |
| OpenAI API | pbook embeddings | `OPENAI_API_KEY` (only if pbook used) |
| GitHub | Clone/push the repos Forge operates on | The operator's normal git credentials |

## Packaging

The forge repo root is a uv workspace (D98) and is **self-contained**:
every internal package is a workspace member, so one clone and one
`uv sync` produce the whole runtime — no sibling checkouts.

```toml
[tool.uv.workspace]
members = ["apps/pbook", "libs/sax-llm", "apps/ocr", "libs/sax-platform"]

[tool.uv.sources]
sax-llm = { workspace = true }
pbook = { workspace = true }
sax-platform = { workspace = true }
```

```text
forge/                     # the workspace root — this is the whole deployment
├── apps/pbook/            # knowledge playbook service
├── apps/ocr/              # document OCR app (member, not a forge dependency)
├── libs/sax-llm/          # LLM provider abstraction
└── libs/sax-platform/     # model-tier registry, LLM client, Mistral OCR,
                            # shared wire contracts + platform primitives
                            # (absorbed libs/forge-contracts at T3.4)
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

```bash
mkdir -p ~/.config/forge
cp deploy/launchd/forge.env.example ~/.config/forge/forge.env
chmod 600 ~/.config/forge/forge.env   # then fill in the CHANGEMEs
```

One file replaces the EC2-era SSM plumbing: `FORGE_DB_URL` (Supabase),
`ANTHROPIC_API_KEY`, `FORGE_OCR_S3_BUCKET`, and friends. The launchd
wrapper parses it without shell evaluation, so URLs with `&` are safe.

### 3. Migrations

The forge worker runs its own Alembic migrations at startup (advisory-locked
against the shared database); the ocr worker likewise applies its own `ocr_*`
chain at startup. pbook does **not** auto-migrate: if ingestion is in scope,
run `uv run pbook migrate` once (and after upgrades that ship pbook
migrations).

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
`~/.config/forge/forge.env`):

| Variable | Purpose | Production value |
| --- | --- | --- |
| `FORGE_TEMPORAL_ADDRESS` | Temporal frontend | `127.0.0.1:7233` |
| `FORGE_DB_URL` | **Required.** Forge store. Unset → hard error | `postgresql+psycopg2://…supabase…/forge?sslmode=require` |
| `FORGE_OCR_S3_BUCKET` | S3 bucket for blobs; required for OCR work | bucket name |
| `FORGE_OCR_S3_PREFIX` | Optional key prefix for blobs | e.g. `ocr/` |
| `FORGE_LOG_DIR` | App log directory (empty = no file logging) | `$XDG_STATE_HOME/forge/logs` |
| `FORGE_OTEL_EXPORTER` | `console`/`otlp_grpc`/`otlp_http`/`none` (code default `console`) | `none` |
| `FORGE_OTEL_ENDPOINT` | OTel endpoint; only read for the `otlp_*` exporters | unset (exporter is `none`) |
| `FORGE_WORKER_IDENTITY` | Worker identity in Temporal | set by the launchd agents (`desktop-forge-worker-1/2`) |
| `ANTHROPIC_API_KEY` | Anthropic SDK auth | key |
| `MISTRAL_API_KEY` | **Required by the ocr worker.** Read at startup; the worker fails fast if unset | key (ocr worker only) |
| `OPENAI_API_KEY` | pbook embeddings | key (if pbook used) |
| `PBOOK_DATABASE_URL` | pbook Postgres store | Supabase URL (if pbook used) |
| `AWS_*` | S3 auth if not using `~/.aws` | keys/region |

The pbook worker takes no Temporal env var — it connects to its `localhost:7233`
default (the same loopback frontend), so there is nothing to set for it here.

For the **dev** stack surfaces (local `forge_dev` Postgres, MinIO), see
[deploy/local-stack/.env.example](../../deploy/local-stack/.env.example)
— and note its footgun warning: the dev `AWS_*`/`FORGE_DB_URL` exports
override production values in the same shell.

## Supabase gotchas

1. **One database now.** Only the `forge` database (plus pbook's schema
   if used) lives in Supabase — Temporal's `temporal`/`temporal_visibility`
   moved into the local stack (D99).
2. **No transaction-mode pooling.** The Supavisor pooler in transaction
   mode (6543) breaks prepared statements. Use the direct connection
   (5432) or session-mode pooler in `FORGE_DB_URL`.
3. **IPv6.** Supabase direct connections are IPv6-only without the IPv4
   add-on; from a residential/office network this usually just works,
   but if connections hang, check IPv6 egress or use the session pooler.

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

- **Supabase** (forge/pbook state of record): Supabase's own backups.
- **Stack Postgres** (Temporal workflow histories): rides the machine's
  backup discipline (Time Machine covers
  `$XDG_DATA_HOME/forge/postgres`); losing it loses in-flight workflow
  state, not records of completed work.
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
