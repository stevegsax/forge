# Deployment: Self-Hosted Forge on AWS EC2

This document describes how to deploy Forge to a single AWS EC2 instance for a
small team at low volume. Temporal is **self-hosted** on the instance with its
persistence backed by a **Supabase-hosted PostgreSQL** database. Forge's own
store (`forge.db`) is **also** backed by Supabase Postgres, and OCR image/file
blobs are stored in **S3** with only references kept in the database. The result
is an instance that holds almost no durable state of its own.

It covers the target architecture, the code changes required to externalize the
Forge store, how to package Forge and its sibling dependencies, the step-by-step
deployment process, configuration, and the gotchas that bite specifically with
the Temporal + Supabase + S3 combination.

> Scope: single-instance, small-team, low-volume. High availability and
> multi-region are out of scope. Multi-host workers become *possible* once the
> store is on Postgres (see [Scaling](#scaling)), but the steps below describe a
> single instance.

## Architecture

```
┌──────────────────────── EC2 (Amazon Linux 2023, t3.large) ─────────────────────────┐
│                                                                                     │
│  Docker Compose                          systemd                                    │
│  ┌─────────────────────────┐   ┌──────────────────────────────────────────────┐    │
│  │ temporal (auto-setup)   │   │ forge-worker@1 / @2   →  forge worker         │    │
│  │   :7233 (gRPC, local)   │◄──┤   FORGE_TEMPORAL_ADDRESS=127.0.0.1:7233       │    │
│  │ temporal-ui :8080       │   │   FORGE_DB_URL=postgresql+psycopg2://…/forge  │    │
│  └────────────┬────────────┘   │   FORGE_OCR_S3_BUCKET=…  (IAM role for S3)    │    │
│               │                 │ pbook-worker          →  pbook worker         │    │
│               │ TLS (5432)      │   (only if `forge ingest` is used)            │    │
│               │                 └───────────────┬──────────────────────────────┘    │
│  /data (EBS): in-progress worktrees, cloned repos, logs  (re-clonable / disposable) │
└───────────────┼─────────────────────────────────┼──────────────────────────────────┘
                │                                  │ outbound HTTPS / AWS APIs
                ▼                                  ▼
        Supabase PostgreSQL                Anthropic · Mistral · GitHub · S3
        ├─ temporal            (Temporal durable state)
        ├─ temporal_visibility (Temporal visibility)
        └─ forge               (Forge store: interactions, runs, playbooks,
                                batch_jobs, OCR records + S3 references)
                                                   │
                                                   ▼
                                          S3 bucket (OCR image/file blobs)
```

**Why this shape.** Both halves of Forge's durable state now live off the
instance: Temporal's orchestration state and Forge's own application store are in
Supabase, and large OCR blobs are in S3. The EC2 box keeps only **in-progress git
worktrees** and **cloned target repos** on its EBS volume — both disposable
(completed output is committed and pushed to the target repo; repos are
re-clonable). You can terminate and recreate the instance and lose nothing
durable. The worker is still a *build agent* that needs git, repos, `uv`, and
`ruff` on local disk to do its work — that local disk is just no longer where the
records of that work are kept.

### What runs where

| Component | Where | Lifetime | Notes |
|---|---|---|---|
| Temporal server | Docker on EC2 | Always up | Persistence → Supabase (`temporal`, `temporal_visibility`) over TLS |
| Temporal Web UI | Docker on EC2 | Always up | Bind to localhost; reach via SSM tunnel |
| `forge worker` (×2) | systemd on EC2 host | Always up | Polls `forge-task-queue`; needs repos + git + ruff on host |
| `pbook worker` | systemd on EC2 host | Optional | Only if transcript ingestion is used; polls `pbook-task-queue` |
| `forge run` / CLI | On the instance (via SSM) | On demand | Connects to `127.0.0.1:7233` |
| Forge store | Supabase (`forge` database) | Always up | interactions, runs, playbooks, batch_jobs, OCR records |
| OCR blobs | S3 | Always up | Image/file bytes; database holds the S3 key + `ocr-image://` URI |

The worker runs on the **host** (not in a container) because it is a build agent:
it needs `git`, the target repositories, `uv`, and `ruff`/test tooling on a
writable filesystem to create worktrees and validate output. Temporal runs in
Docker because it is a self-contained service with no filesystem coupling.

## External dependencies

| Dependency | Purpose | Requirement |
|---|---|---|
| Supabase PostgreSQL | Temporal state **and** Forge store | Postgres 12+; three databases; TLS; direct (non-transaction-pooled) connection |
| S3 bucket | OCR image/file blobs | IAM instance-role access (no static keys) |
| Anthropic API | All Forge LLM calls; pbook extraction/review | `ANTHROPIC_API_KEY`, outbound HTTPS |
| Mistral API | OCR pipeline | `MISTRAL_API_KEY`, outbound HTTPS (only if OCR used) |
| OpenAI API | pbook embeddings (semantic search / dedup) | `OPENAI_API_KEY`, outbound HTTPS (only if pbook used) |
| GitHub | Clone/push the repos Forge operates on | `SAX_GITHUB_TOKEN`, outbound HTTPS |
| pbook + sax-llm repos | Sibling Python packages | Present at build/deploy time (see Packaging) |

## Prerequisite code changes

The current code stores everything in local SQLite (`forge.db`) and keeps OCR
blobs as `LargeBinary` columns. This deployment requires two contained changes
that must ship **before** it is possible. Both are well-scoped because store
access is centralized.

### A. Back the Forge store with Postgres

All store access funnels through `get_db_path()` → `get_engine(db_path)` (~10 call
sites in activities, context providers, and the CLI), so the change is central:

1. **Connection resolution** — resolve `FORGE_DB_URL` (e.g.
   `postgresql+psycopg2://user:pwd@host:5432/forge?sslmode=require`) and fall back
   to the SQLite path when unset. `get_engine` builds the engine from the URL.
   Keep SQLite working for local dev and the test suite.
2. **WAL pragma** — the `PRAGMA journal_mode=WAL` listener in `get_engine`
   (`src/forge/store.py:327`) is SQLite-only; gate it on `engine.dialect.name == "sqlite"`.
3. **Alembic** — `run_migrations` already overrides `sqlalchemy.url`
   programmatically, so it works against Postgres as-is; just pass the URL. The
   hardcoded `sqlite:///forge.db` in `alembic/alembic.ini` is an unused default.
   Verify migration `008` (`batch_alter_table`) applies on Postgres — batch mode
   degrades to a native `ALTER COLUMN` there.
4. **Driver** — add `psycopg2-binary` to `pyproject.toml`. The store is
   **synchronous** SQLAlchemy, and sync `psycopg2` does not issue server-side
   prepared statements, which keeps it compatible with Supabase's pooler.
5. **Pool sizing** — set a small `pool_size`/`max_overflow` to respect Supabase
   connection caps.

Schema portability is already fine: every column is `sa.Text`/`sa.Integer`/
`sa.LargeBinary`/`UTCDateTime` — no SQLite-only types. All store access happens in
activities and CLI code (never in workflow code), so this does not affect Temporal
determinism.

### B. Move OCR blobs to S3, keep references in Postgres

Today `file_content_blobs.data` and `ocr_images.data` are `LargeBinary`
(`src/forge/store.py:199,215`). Replace the bytes with an S3 reference:

1. **Schema migration** — add `s3_key` (and keep `mime_type`, `file_size_bytes`,
   the `ocr-image://` URI); drop the `data` column.
2. **Write path** — `save_file_content` / `save_ocr_result` (and the OCR store
   activity, `src/forge/ocr/activities.py`) upload bytes to S3 under a
   deterministic key and persist the key + metadata.
3. **Read path** — `get_file_content` and the `ocr-image://` resolver fetch from
   S3 by key.
4. **Client + auth** — add `boto3`; access S3 via the **EC2 instance role** (no
   static keys). Config: `FORGE_OCR_S3_BUCKET`, optional `FORGE_OCR_S3_PREFIX`,
   region from the instance.
5. **Lifecycle** — S3 lifecycle rules can expire old blobs; the database row is
   the index of record.

S3 is the only OCR blob store — there is no inline-in-DB fallback. `FORGE_OCR_S3_BUCKET`
unset or S3 unreachable fails the OCR *task* (the worker keeps running); non-OCR work
needs no bucket. Tests mock S3 with `moto` rather than storing bytes in SQLite.

> **Implemented (Phase A + B)** in
> [development-plans/externalize-store-postgres-s3.md](../../development-plans/externalize-store-postgres-s3.md):
> the store is configured by a required `FORGE_DB_URL` and OCR blobs live in S3
> (`s3_key` references; migration `014`). Phase C (survivable writes) is pending.

## Packaging Forge and pbook for deployment

Forge declares its sibling dependencies as **editable path sources** in
`pyproject.toml`:

```toml
[tool.uv.sources]
sax-llm = { path = "../sax-llm", editable = true }
pbook  = { path = "../pbook",  editable = true }
```

Neither `sax-llm` nor `pbook` is published to a package index, so a plain
`pip install forge` cannot resolve them. The dependency graph is:

```
sax-llm  (anthropic, mistralai, pydantic)
   └── pbook  (+ temporalio, sqlalchemy, alembic, openai, numpy)
         └── forge  (+ opentelemetry, grimp, networkx, scipy, pymupdf, click,
                       psycopg2-binary, boto3)   ← last two added by the changes above
```

Any packaging approach must bring **all three** repositories along. Three options,
in increasing isolation:

### Option A — Source tree + `uv sync` (recommended for low volume)

Reproduce the dev layout on the instance and let `uv` build the editable install.
Simplest, matches development exactly, no extra build step.

```
/srv/forge-app/
├── forge/      # this repo
├── sax-llm/    # sibling — referenced as ../sax-llm
└── pbook/      # sibling — referenced as ../pbook
```

```bash
cd /srv/forge-app/forge
uv sync --frozen          # installs forge + sax-llm + pbook from uv.lock into .venv
uv run forge --version    # smoke test
```

Pin to a known-good commit per repo (record the three commit SHAs in your release
notes). `--frozen` ensures the deployed dependency set matches `uv.lock`.

### Option B — Wheel bundle (cleaner artifact, no source at runtime)

Build wheels for all three and install them into a venv. The runtime needs no
source tree, only the wheels.

```bash
uv build --wheel ../sax-llm -o dist/   # sax_llm-0.1.0-*.whl
uv build --wheel ../pbook   -o dist/   # pbook-0.1.0-*.whl
uv build --wheel .          -o dist/   # forge-0.1.0-*.whl

# On the instance:
uv venv && uv pip install dist/sax_llm-*.whl dist/pbook-*.whl dist/forge-*.whl
```

A wheel's metadata requires `sax-llm`/`pbook` *by name* (the path sources are
dev-time only), so install the three wheels together — `pip` won't find the
siblings on an index.

### Option C — Container image (most reproducible)

Build one image with all three repos copied in and `uv sync` baked into the layer.
Best when you later move workers to ECS/containers. The worker image still needs
`git` and any test tooling for the target repos, and must mount a volume for
worktrees and the cloned repos.

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && pip install uv && rm -rf /var/lib/apt/lists/*
WORKDIR /srv/forge-app
COPY sax-llm/ ./sax-llm/
COPY pbook/   ./pbook/
COPY forge/   ./forge/
WORKDIR /srv/forge-app/forge
RUN uv sync --frozen
ENTRYPOINT ["uv", "run", "forge", "worker"]
```

**Recommendation for this deployment:** Option A. It is the least friction for a
single instance and lets you `git pull` + `uv sync` to upgrade.

### pbook is a parallel deployment

If transcript ingestion (`forge ingest`) is in scope, pbook deploys alongside
Forge. Verified specifics from the pbook repo:

- **Its own worker** — `pbook worker --temporal-address 127.0.0.1:7233`, polling
  `pbook-task-queue` (separate from Forge's queue).
- **Migrations are explicit** — unlike the Forge worker, the pbook worker does
  **not** auto-run migrations on startup. Run `pbook migrate` once at deploy time
  (and after upgrades).
- **An extra secret: `OPENAI_API_KEY`** — pbook generates embeddings via OpenAI
  (`text-embedding-3-small`) for semantic search and de-duplication, in addition
  to using Anthropic for extraction/review. Add it to SSM if pbook is in scope.
- **Store path** — `PBOOK_DB_PATH` → `$XDG_STATE_HOME/pbook/pbook.db` (same XDG
  pattern as Forge).
- **sax-llm source differs** — pbook's own `pyproject.toml` pulls sax-llm from
  GitHub (`rev = v0.1.0`), while Forge pins both siblings to local editable paths.
  In the Forge workspace resolution Forge's local sources win, so a Forge `uv sync`
  uses the local sax-llm/pbook. A *standalone* pbook build needs GitHub access.

**Externalizing pbook's store to Postgres is a *larger* change than Forge's.**
pbook's tag query uses SQLite-specific `json_each(tags_json)` and relies on
`PRAGMA foreign_keys=ON` for cascade deletes — neither runs on Postgres. Moving
pbook's store would require rewriting that query (e.g. `jsonb_array_elements_text`)
and the FK strategy on top of the connection/driver work. For low volume, the
pragmatic choice is to **keep pbook on local SQLite** on the EBS volume (its
embeddings are small float32 blobs, not S3-worthy) and revisit if it becomes a
scaling concern.

If ingestion is **not** in scope, omit the pbook worker entirely; the Forge worker
logs a warning and skips ingestion workflows, and `forge ingest` exits with a clear
error.

## Deployment process

### 1. Provision Supabase Postgres

1. Create a Supabase project in a **region close to your intended EC2 region** —
   both Temporal and now the Forge store are chatty with this database.
2. Create the three databases. Connect with the **direct connection** (port 5432)
   as the project's `postgres` role:

   ```sql
   CREATE DATABASE temporal;
   CREATE DATABASE temporal_visibility;
   CREATE DATABASE forge;
   ```

   If your Supabase plan/role cannot `CREATE DATABASE`, fall back to additional
   Supabase projects. See [Supabase gotchas](#supabase-specific-gotchas).
3. Record connection details for both Temporal (`temporal`/`temporal_visibility`)
   and Forge (`forge`). **Do not** use the transaction-mode pooler (port 6543).

### 2. Provision the S3 bucket

1. Create a private S3 bucket for OCR blobs (e.g. `forge-ocr-blobs-<acct>`), same
   region as EC2. Block all public access.
2. Optionally set a lifecycle rule to expire blobs after N days.
3. The EC2 instance role (below) gets `s3:GetObject`/`PutObject`/`DeleteObject` on
   `arn:aws:s3:::forge-ocr-blobs-<acct>/*` — and nothing broader.

### 3. Provision the EC2 instance

- **Instance type:** `t3.large` (2 vCPU / 8 GB) is comfortable; `t3.medium` works
  at very low volume. Avoid spot — workflows run long (48 h default timeout) and a
  batch poller fires every 10 min.
- **EBS:** a `gp3` volume mounted at `/data` for worktrees, cloned repos, and
  logs. This volume now holds **no durable records** — it can be smaller and
  snapshots are optional (nice-to-have for faster instance rebuilds, not for data
  safety).
- **Security group:** **no inbound** rules. Allow all egress. Manage via **SSM
  Session Manager** — no SSH key, no open ports.
- **IAM instance role:** read on the SSM/Secrets Manager parameters holding the
  keys + Supabase password; S3 access scoped to the OCR bucket;
  `AmazonSSMManagedInstanceCore` for Session Manager.
- **Egress note:** Supabase direct connections are **IPv6-only** without the IPv4
  add-on. Ensure VPC IPv6 egress, enable the Supabase IPv4 add-on, or use the
  session-mode pooler (port 5432, IPv4).

### 4. Store secrets

Put each secret in **SSM Parameter Store (SecureString)** (or Secrets Manager):

- `/forge/ANTHROPIC_API_KEY`
- `/forge/MISTRAL_API_KEY` (only if OCR used)
- `/forge/OPENAI_API_KEY` (only if pbook ingestion used — for embeddings)
- `/forge/SAX_GITHUB_TOKEN`
- `/forge/SUPABASE_TEMPORAL_PWD` and `/forge/SUPABASE_FORGE_DB_URL`

Note S3 needs **no secret** — the instance role handles it. Load secrets into an
`EnvironmentFile` (`/etc/forge/forge.env`, `chmod 600`) on boot via cloud-init
running `aws ssm get-parameter --with-decryption`. Never bake secrets into the AMI
or repo.

### 5. Run Temporal (Docker) against Supabase

Install Docker + Compose. Run Temporal with SQL persistence pointed at Supabase.
Use the `auto-setup` image for the **first** boot (it creates and versions the
Temporal schema in `temporal`/`temporal_visibility`), then set
`SKIP_SCHEMA_SETUP=true` (or switch to `temporalio/server`) for steady state.

```yaml
# /srv/forge-app/temporal/compose.yaml (illustrative — fill from forge.env)
services:
  temporal:
    image: temporalio/auto-setup:1.25.0
    ports:
      - "127.0.0.1:7233:7233"          # gRPC, host-local only
    environment:
      DB: postgres12
      POSTGRES_SEEDS: ${SUPABASE_HOST}  # db.<ref>.supabase.co
      DB_PORT: "5432"                   # direct connection, NOT 6543
      POSTGRES_USER: ${SUPABASE_USER}
      POSTGRES_PWD: ${SUPABASE_TEMPORAL_PWD}
      DBNAME: temporal
      VISIBILITY_DBNAME: temporal_visibility
      SQL_TLS_ENABLED: "true"
      ENABLE_ES: "false"                # SQL-based advanced visibility, no Elasticsearch
      SQL_MAX_CONNS: "10"
      # SKIP_SCHEMA_SETUP: "true"       # set after first successful boot
  temporal-ui:
    image: temporalio/ui:2.31.0
    depends_on: [temporal]
    ports:
      - "127.0.0.1:8080:8080"
    environment:
      TEMPORAL_ADDRESS: temporal:7233
```

```bash
docker compose -f /srv/forge-app/temporal/compose.yaml up -d
temporal operator namespace list --address 127.0.0.1:7233   # 'default' should exist
```

### 6. Install the Forge runtime and code

```bash
sudo dnf install -y git
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv (brings managed Python 3.12)

sudo mkdir -p /srv/forge-app && cd /srv/forge-app
git clone <forge>   forge
git clone <sax-llm> sax-llm
git clone <pbook>   pbook       # only if ingestion in scope
cd forge && uv sync --frozen

sudo mkdir -p /data/repos && git clone <target-repo> /data/repos/<name>
```

The worker runs Forge's Alembic migrations against the `forge` Postgres database
automatically on startup (`_init_store()` in `src/forge/worker.py`).

### 7. systemd units for the workers

```ini
# /etc/systemd/system/forge-worker@.service
[Unit]
Description=Forge worker %i
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/srv/forge-app/forge
EnvironmentFile=/etc/forge/forge.env
Environment=FORGE_TEMPORAL_ADDRESS=127.0.0.1:7233
Environment=FORGE_DB_URL=${SUPABASE_FORGE_DB_URL}
Environment=FORGE_OCR_S3_BUCKET=forge-ocr-blobs-<acct>
Environment=FORGE_LOG_DIR=/data/forge/logs
Environment=FORGE_OTEL_EXPORTER=none
Environment=FORGE_WORKER_IDENTITY=ec2-forge-%i
ExecStart=/usr/bin/env uv run forge worker
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now forge-worker@1 forge-worker@2
```

Run **two** workers for redundancy. They now share the same Postgres store, so
their observability data is coherent regardless of host (this is the property the
store change unlocks). If pbook ingestion is in scope, add a `pbook-worker.service`
pointed at the same Temporal address (polls `pbook-task-queue`).

### 8. Verify end to end

```bash
temporal task-queue describe --task-queue forge-task-queue --address 127.0.0.1:7233

cd /srv/forge-app/forge
uv run forge run \
  --task-id smoke-test \
  --description "Add a docstring to the module" \
  --target-file /data/repos/<name>/<some_file>.py

uv run forge status --limit 5   # reads from the Supabase forge store
```

Confirm an OCR run writes a blob to S3 and a reference row to Postgres:

```bash
uv run forge start OcrSyncWorkflow '{"file_path": "/data/repos/<name>/sample.pdf"}' --wait
aws s3 ls s3://forge-ocr-blobs-<acct>/   # blob present
uv run forge ocr-jobs --limit 5          # reference row present
```

Browse the Temporal UI via SSM port forwarding:

```bash
aws ssm start-session --target <instance-id> \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["8080"],"localPortNumber":["8080"]}'
# then open http://localhost:8080
```

To submit tasks from a laptop without exposing 7233, port-forward 7233 the same
way and set `FORGE_TEMPORAL_ADDRESS=127.0.0.1:7233` locally.

## Configuration reference

| Variable | Purpose | Production value |
|---|---|---|
| `FORGE_TEMPORAL_ADDRESS` | Temporal frontend address | `127.0.0.1:7233` |
| `FORGE_DB_URL` | **Required.** Forge store connection: `sqlite:///<path>` (dev/tests) or `postgresql+psycopg2://…` (prod). Unset → hard error; no disable-store mode | `postgresql+psycopg2://…/forge?sslmode=require` |
| `FORGE_OCR_S3_BUCKET` | S3 bucket for OCR blobs. Required for OCR work; unset or unreachable → the OCR task fails (no inline-in-DB fallback) | `forge-ocr-blobs-<acct>` |
| `FORGE_OCR_S3_PREFIX` | Optional key prefix for OCR blobs | e.g. `ocr/` |
| `FORGE_LOG_DIR` | Log directory (empty = disable file logging) | `/data/forge/logs` |
| `FORGE_OTEL_EXPORTER` | Trace exporter: `console`/`otlp_grpc`/`otlp_http`/`none` | `none` (default `console` is noisy) |
| `FORGE_OTEL_ENDPOINT` | OTel collector endpoint | set only if exporting traces |
| `FORGE_WORKER_IDENTITY` | Worker identity in Temporal | `ec2-forge-1`, `ec2-forge-2`, … |
| `ANTHROPIC_API_KEY` | Anthropic SDK auth | from SSM |
| `MISTRAL_API_KEY` | Mistral OCR auth | from SSM (if OCR used) |
| `OPENAI_API_KEY` | pbook embeddings auth | from SSM (if pbook used) |
| `PBOOK_DB_PATH` | pbook SQLite store (if ingestion used) | `/data/pbook/pbook.db` |
| `SAX_GITHUB_TOKEN` | Git access to private repos | from SSM |

`FORGE_DB_URL` and `FORGE_OCR_S3_BUCKET` are introduced by the
[prerequisite code changes](#prerequisite-code-changes). The worker connects to
Temporal with only a data converter and optional identity — **no TLS** — which is
correct for a co-located Temporal at `localhost:7233` and needs no code change.
(The Supabase TLS is on the Postgres connections, configured via `FORGE_DB_URL`'s
`sslmode=require` and Temporal's `SQL_TLS_ENABLED`.)

## Supabase-specific gotchas

1. **Three databases now.** `temporal`, `temporal_visibility`, **and** `forge`.
   A Supabase project ships a single `postgres` database; create the others via
   `CREATE DATABASE` on the direct connection, or use additional projects.
2. **No transaction-mode pooling.** The Supavisor pooler in transaction mode
   (6543) breaks both Temporal's and (for async drivers) Postgres prepared
   statements. Use the **direct connection (5432)** or **session-mode pooler**.
   Forge's store uses sync `psycopg2`, which is tolerant, but keep it consistent.
3. **IPv6 / IPv4.** Supabase direct connections are IPv6-only without the IPv4
   add-on. Ensure VPC IPv6 egress, buy the add-on, or use the session-mode pooler.
4. **TLS is mandatory.** `SQL_TLS_ENABLED=true` for Temporal; `sslmode=require` in
   `FORGE_DB_URL`.
5. **Connection caps.** Temporal pools per service, and now Forge's workers each
   open a pool against `forge`. Keep both small (`SQL_MAX_CONNS=10`; small
   SQLAlchemy `pool_size`) and prefer a paid tier for headroom.
6. **Latency and cost.** Every Temporal transition and every store write is a DB
   round trip to Supabase; co-locate regions. OCR blobs are in S3, so the large
   payloads stay out of Postgres — this is why S3 matters at any non-trivial OCR
   volume.
7. **`auto-setup` re-runs schema on every start.** Use it for the first boot only.

## Persistence and backups

Durable state now lives in three managed stores; the EC2 disk holds none of it:

- **Temporal state (Supabase `temporal`/`temporal_visibility`):** workflow history,
  schedules, task queues. Enable Supabase automated backups / PITR.
- **Forge store (Supabase `forge`):** interactions, runs, playbooks, batch jobs,
  OCR reference rows. Same Supabase backup covers it.
- **OCR blobs (S3):** enable versioning and/or a lifecycle policy; S3 is durable by
  design.
- **EBS volume:** only in-progress worktrees + cloned repos + logs. **No backup
  required for data safety** — completed output is committed to its target repo and
  pushed; worktrees are disposable; repos are re-clonable. Snapshot only if you
  want faster instance rebuilds.

## Scaling

The SQLite-on-local-disk constraint that previously prevented multi-host workers
is removed by the store-on-Postgres change: workers on different hosts share one
`forge` database, so observability and playbook data stay coherent. Temporal state
is already external. OCR blobs are in shared S3.

The remaining per-host requirement is the **build-agent filesystem**: each worker
host needs the target repos checked out and disk for worktrees. To add a second
worker host you provision another instance with the same runtime, repos, and
`forge.env`, pointed at the same Temporal and the same `forge` database. Watch
Supabase connection caps as worker count grows.

## Security checklist

- No inbound ports; manage via SSM Session Manager; reach Temporal UI/gRPC via SSM
  port forwarding only. Temporal OSS has **no built-in authentication** — never
  expose 7233 publicly.
- API keys and the Supabase password live in SSM/Secrets Manager, injected at
  boot; the instance role grants read on only those parameters.
- S3 access is via the instance role, scoped to the OCR bucket only — no static S3
  keys anywhere.
- `forge.env` is `chmod 600`, owned by the worker user.
- Postgres connections use TLS; the `forge` database password is a role scoped to
  only that database.

## Open questions to resolve before go-live

- **Prerequisite code changes (A) and (B)** implemented, tested (SQLite path still
  green for the test suite), and merged.
- **Supabase `CREATE DATABASE` privilege** — confirm, or plan for extra projects.
- **IPv4 vs IPv6 egress** path from the VPC to Supabase.
- **pbook scope** — is `forge ingest` part of this deployment? If so: provision
  `OPENAI_API_KEY`, run `pbook migrate`, and stand up the pbook worker. Note
  pbook's store is **not** Postgres-portable as-is (SQLite `json_each`/`PRAGMA
  foreign_keys`), so plan to keep it on local SQLite unless you invest in rewriting
  that query.
- **Pinned commits** for forge, sax-llm, pbook recorded in release notes.
- **Backup/restore drill** for Supabase (and S3 versioning) tested once.
