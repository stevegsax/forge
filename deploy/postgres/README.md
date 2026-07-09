# Local development Postgres

A podman-managed PostgreSQL for running Forge locally against the same engine
as production (Supabase Postgres). It is **not** required by the test suite:
the default `uv run pytest` run uses temp-file SQLite, and the opt-in
`uv run pytest -m postgres` suite starts its own throwaway testcontainer.

## Contents

```text
deploy/postgres/
├── compose.yaml            pgvector/pgvector:pg16, loopback-only on :5433
├── docker/initdb/
│   └── 01-extensions.sql   enables the `vector` extension on first init
├── .env.example            FORGE_DB_URL + FORGE_PG_PORT / FORGE_PG_DATA overrides
└── README.md
```

The `make db-*` targets in the repo-root `Makefile` wrap this stack.

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

## Data persistence

Data is bind-mounted to a host directory so it survives `db-down`/rebuilds.
`make db-up` puts it at `$XDG_DATA_HOME/forge/postgres` (falling back to
`~/.local/share/forge/postgres`); override with `FORGE_PG_DATA`. Running
`podman compose up` directly (without make) falls back to `./data/postgres`
here, which is gitignored. To start clean, `make db-down` then delete that
directory.

## Image

`pgvector/pgvector:pg16` — the Postgres 16 engine the migration testcontainer
validates against, plus pgvector so one local database forward-fits the
migration's Phase 6 (pbook needs the `vector` extension). Forge's own schema
uses no extensions, so plain `postgres:16` would also serve Forge alone.
