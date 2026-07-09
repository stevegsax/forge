# Local development stack (Postgres + MinIO)

A podman-managed PostgreSQL + MinIO for running Forge locally against the same
surfaces as production (Supabase Postgres for the store, AWS S3 for blobs). It
is **not** required by the test suite: the default `uv run pytest` run uses
temp-file SQLite and mocks S3 in-process with moto, and the opt-in
`uv run pytest -m postgres` suite starts its own throwaway testcontainer.

## Contents

```text
deploy/local-database/
├── compose.yaml            postgres (pgvector/pgvector:pg16) + minio + minio-init
├── docker/initdb/
│   └── 01-extensions.sql   enables the `vector` extension on first init
├── .env.example            FORGE_DB_URL, AWS_* / MinIO, and *_PORT / *_DATA overrides
└── README.md
```

The `make db-*` targets in the repo-root `Makefile` wrap this stack; `make
db-up` starts everything (Postgres + MinIO + bucket creation).

## Prerequisites

- A running podman machine: `podman machine start` (first time: `podman machine init`).
- A compose provider for `podman compose` (e.g. `uv tool install podman-compose`).

## Usage

```bash
export FORGE_DB_URL=postgresql+psycopg2://forge:forge@localhost:5433/forge_dev
make db-up        # start Postgres
make db-migrate   # create Forge's tables (interactions, runs, batch_jobs, playbooks)
make db-psql      # psql shell;  \dx shows the vector extension
make db-logs      # tail logs
make db-down      # stop and remove the container (data survives — see below)
```

Forge reads `FORGE_DB_URL` from the environment (there is no default and no
`.env` auto-load); set it via direnv/keychain or a manual `export`. The
username/password/database above are the throwaway literals in `compose.yaml`.

### Host port

The container publishes host port `FORGE_PG_PORT`, defaulting to `5433` (not
`5432`) so it coexists with another local Postgres already on `5432` — e.g. a
sibling project's dev DB — with no override. If `5433` is *also* taken, podman
refuses to start with `"proxy already running"`; pick a free port and match it
in the URL:

```bash
export FORGE_PG_PORT=5434
export FORGE_DB_URL=postgresql+psycopg2://forge:forge@localhost:5434/forge_dev
make db-up
```

## Object storage (MinIO)

Forge's blob path (`forge_contracts.s3_blobs`, keyed under
`FORGE_OCR_S3_BUCKET`) uses `boto3.client("s3")`. botocore honors
`AWS_ENDPOINT_URL_S3` and uses path-style addressing, so pointing that at MinIO
routes Forge's blob I/O to the local container **with no code change**. `make
db-up` also creates the bucket (via the `minio-init` one-shot).

```bash
export AWS_ENDPOINT_URL_S3=http://localhost:9002
export AWS_ACCESS_KEY_ID=forge
export AWS_SECRET_ACCESS_KEY=forge-minio-secret
export AWS_DEFAULT_REGION=us-east-2
export FORGE_OCR_S3_BUCKET=saxcapital-forge-blobs-dev
make db-up
```

The web console is at `http://localhost:9003` (log in with the creds above).
Host ports default to `9002` (S3 API) / `9003` (console) — set via
`FORGE_MINIO_PORT` / `FORGE_MINIO_CONSOLE_PORT` — so this coexists with a
sibling project's MinIO already on `9000`; keep `AWS_ENDPOINT_URL_S3` in sync
with `FORGE_MINIO_PORT`.

> **Footgun:** these `AWS_*` vars override any real-AWS creds/endpoint in your
> shell (same shape as `FORGE_DB_URL` → Supabase). Set all of them together so
> you never half-point at real S3.

## Data persistence

Both data sets are bind-mounted to host directories so they survive
`db-down`/rebuilds. `make db-up` puts them at `$XDG_DATA_HOME/forge/postgres`
and `$XDG_DATA_HOME/forge/minio` (falling back to `~/.local/share/forge/…`);
override with `FORGE_PG_DATA` / `FORGE_MINIO_DATA`. Running `podman compose up`
directly (without make) falls back to `./data/postgres` and `./data/minio`
here, which are gitignored. To start clean, `make db-down` then delete those
directories.

## Images

- `pgvector/pgvector:pg16` — the Postgres 16 engine the migration testcontainer
  validates against, plus pgvector so one local database forward-fits the
  migration's Phase 6 (pbook needs the `vector` extension). Forge's own schema
  uses no extensions, so plain `postgres:16` would also serve Forge alone.
- `quay.io/minio/minio` + `quay.io/minio/mc` (pinned release tags) — the S3 API
  and the one-shot bucket creator. Credentials are throwaway literals used only
  by the container.
