# Local stack (Postgres + MinIO + Temporal)

The podman-managed stack this deployment runs on (D99). It has two roles:

- **Temporal is production.** The workflow engine self-hosts here on the
  always-on desktop, persisting to this stack's Postgres (databases
  `temporal` + `temporal_visibility` beside `forge_dev`). The forge and
  pbook workers run on the host and connect to `127.0.0.1:7233`; the web
  UI is at `http://localhost:8233`.
- **Postgres (`forge_dev`) + MinIO are dev counterparts** of the managed
  app stores (production forge/pbook state is Supabase Postgres; blobs
  are AWS S3 — see
  [../../docs/operations/DEPLOYMENT.md](../../docs/operations/DEPLOYMENT.md)).

It is **not** required by the test suite: the default `uv run pytest` run
uses temp-file SQLite and mocks S3 in-process with moto, and the opt-in
`uv run pytest -m postgres` suite starts its own throwaway testcontainer.

## Contents

```text
deploy/local-stack/
├── compose.yaml            postgres (pgvector:pg16) + temporal + temporal-ui
│                           + minio + minio-init — all loopback-only
├── docker/initdb/
│   └── 01-extensions.sql   enables the `vector` extension on first init
├── .env.example            FORGE_DB_URL, Temporal addresses, AWS_*/MinIO,
│                           and *_PORT / *_DATA overrides
└── README.md
```

The `make stack-*` targets in the repo-root `Makefile` wrap this stack;
`make stack-up` starts everything (Postgres + Temporal + UI + MinIO +
bucket creation).

## Prerequisites

- A running podman machine: `podman machine start` (first time: `podman
  machine init`).
- A compose provider for `podman compose` (e.g. `uv tool install
  podman-compose`).

## Usage

```bash
make stack-up     # start everything (needs a running podman machine)
make db-migrate   # create Forge's tables in forge_dev (needs FORGE_DB_URL set)
make stack-psql   # psql shell;  \dx shows the vector extension;  \l lists
                  # forge_dev + temporal + temporal_visibility
make stack-logs   # tail all containers' logs
make stack-down   # stop and remove the containers (data survives — see below)
```

Forge reads `FORGE_DB_URL` from the environment (there is no default and no
`.env` auto-load); set it via direnv/keychain or a manual `export`. The
username/password/database literals are the throwaway values in
`compose.yaml`.

For surviving reboots on the always-on desktop (podman machine start +
`stack-up` at login), see the launchd agents in
[../launchd/](../launchd/).

## Ports

All loopback-only; defaults chosen to coexist with sibling projects'
stacks (override via the `FORGE_*_PORT` variables in `.env.example`):

| Service | Host port | Purpose |
| --- | --- | --- |
| postgres | 5433 | `forge_dev` + Temporal persistence |
| temporal | 7233 | Temporal frontend (workers, CLIs) |
| temporal-ui | 8233 | Temporal web UI |
| minio | 9002 / 9003 | S3 API / web console |

If a port is taken, podman refuses to start with `"proxy already
running"`; pick a free port, export the matching `FORGE_*_PORT`, and keep
`FORGE_DB_URL` / `AWS_ENDPOINT_URL_S3` in sync.

## Temporal persistence

`temporalio/auto-setup` creates the `temporal` and `temporal_visibility`
databases and schemas on first boot (the `forge` user is this instance's
superuser). Re-running setup on later restarts is harmless against this
local Postgres; once boots are stable you can set `SKIP_SCHEMA_SETUP=true`
in `compose.yaml` to skip it. Temporal's state lives inside Postgres, so
backup discipline for the Postgres data directory covers workflow
histories too.

## pbook against this stack

pbook's tables live in their own `pbook` schema with their own Alembic
version table, so it shares `forge_dev` cleanly:

```bash
export PBOOK_DATABASE_URL=postgresql+psycopg://forge:forge@localhost:5433/forge_dev
uv run pbook migrate
```

## Object storage (MinIO)

Forge's blob path (`forge_contracts.s3_blobs`, keyed under
`FORGE_OCR_S3_BUCKET`) uses `boto3.client("s3")`. botocore honors
`AWS_ENDPOINT_URL_S3` and uses path-style addressing, so pointing that at
MinIO routes Forge's blob I/O to the local container **with no code
change**. `make stack-up` also creates the bucket (via the `minio-init`
one-shot).

```bash
export AWS_ENDPOINT_URL_S3=http://localhost:9002
export AWS_ACCESS_KEY_ID=forge
export AWS_SECRET_ACCESS_KEY=forge-minio-secret
export AWS_DEFAULT_REGION=us-east-2
export FORGE_OCR_S3_BUCKET=saxcapital-forge-blobs-dev
```

The web console is at `http://localhost:9003` (log in with the creds
above).

> **Footgun:** these `AWS_*` vars override any real-AWS creds/endpoint in
> your shell (same shape as `FORGE_DB_URL` → Supabase). Set all of them
> together so you never half-point at real S3.

## Data persistence

Data sets are bind-mounted to host directories so they survive
`stack-down`/rebuilds. `make stack-up` puts them at
`$XDG_DATA_HOME/forge/postgres` and `$XDG_DATA_HOME/forge/minio` (falling
back to `~/.local/share/forge/…`); override with `FORGE_PG_DATA` /
`FORGE_MINIO_DATA`. Running `podman compose up` directly (without make)
falls back to `./data/postgres` and `./data/minio` here, which are
gitignored. Temporal has no volume of its own — its state is rows in
Postgres. To start clean, `make stack-down` then delete those
directories.

> Renamed from `deploy/local-database/` when Temporal joined the stack
> (D99). The compose project name changed (`forge-postgres` →
> `forge-stack`): if the old project's containers are still running, stop
> them once with `podman stop forge-postgres forge-minio` before the
> first `make stack-up`. Data is unaffected (bind mounts).

## Images

- `pgvector/pgvector:pg16` — the Postgres 16 engine the migration
  testcontainer validates against, plus pgvector for pbook's embeddings.
  Forge's own schema uses no extensions.
- `temporalio/auto-setup:1.25.2` + `temporalio/ui:2.31.0` — pinned
  Temporal server (with first-boot schema setup) and web UI.
- `quay.io/minio/minio` + `quay.io/minio/mc` (pinned release tags) — the
  S3 API and the one-shot bucket creator. Credentials are throwaway
  literals used only by the container.
